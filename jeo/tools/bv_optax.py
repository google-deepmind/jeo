# Copyright 2024 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Gradient transformations and other optax utilities.

Based on
https://github.com/google-research/big_vision/blob/main/big_vision/utils.py
"""

import operator
import jax
import jax.numpy as jnp
from jeo import train_utils
from jeo.tools import tree_utils
import optax


def find_states(opt_state, cls):
  leaves = jax.tree.leaves(
      opt_state, is_leaf=lambda node: isinstance(node, cls))
  return [leaf for leaf in leaves if isinstance(leaf, cls)]


def get_count(opt_state, jittable=False):
  """Returns `ScaleByScheduleState.count` from `opt_state` as an integer."""
  counts = [
      state.count
      for state in find_states(opt_state, optax.ScaleByScheduleState)
  ]
  if jittable:
    return counts[0]
  else:
    counts = {int(c) for c in counts}
    assert len(counts) == 1, f"Expected exactly 1 ScaleByScheduleState:{counts}"
    return next(iter(counts))


def replace_frozen(schedule, pytree, replacement, log=None):
  """Replaces values matching frozen params in `pytree` with `replacement`."""
  if not isinstance(schedule, (list, tuple)):
    return pytree
  masks, scheds = _make_mask_trees(pytree, schedule, log=log)
  frozen_mask, _, _ = tree_utils.split_frozen(masks, scheds)
  return jax.tree.map(
      lambda v, f: replacement if f else v, pytree, frozen_mask)


def clip_by_per_example_global_norm(
    max_norm: float,
) -> optax.GradientTransformation:
  """Clips the norm of per-example gradients."""

  def init_fn(params):
    del params
    return optax.EmptyState()

  def update_fn(updates, state, params=None):
    del params
    grads_flat, grads_treedef = jax.tree_util.tree_flatten(updates)
    batch_size = grads_flat[0].shape[0]
    clipped, _ = optax.per_example_global_norm_clip(grads_flat, max_norm)
    grads_sum = jax.tree_util.tree_unflatten(grads_treedef, clipped)
    grads_mean = jax.tree_util.tree_map(lambda x: x / batch_size, grads_sum)
    return grads_mean, state

  return optax.GradientTransformation(init_fn, update_fn)


def make(config, params, *, sched_kw):
  """Returns gradient transform and learning rate functions."""

  # Global schedule. No schedule means frozen.
  schedule = config.get("schedule", {})
  if not isinstance(schedule, (tuple, list)):
    schedule = [(".*", schedule)]
  masks, scheds = _make_mask_trees(params, schedule, "config.schedule")
  frozen_mask, masks, scheds = tree_utils.split_frozen(masks, scheds)
  not_frozen_mask = jax.tree.map(operator.not_, frozen_mask)
  def create_schedule(mult=1.0, **kw):
    assert "base" not in kw, kw
    return create_learning_rate_schedule(base=mult, **kw)
  schedule_fns = [create_schedule(**sched_kw, **sched) for sched in scheds]
  schedule_txs = [
      optax.masked(optax.scale_by_schedule(schedule_fn), mask)
      for schedule_fn, mask in zip(schedule_fns, masks)
  ] + [
      # Removes weight decay updates. Note that weight decay already has an
      # independent mask (which cannot be combined easily with a second mask),
      # so instead we multiply updates for frozen params with zero.
      optax.masked(optax.set_to_zero(), frozen_mask)
  ]

  # Gradient clipping.
  if clip_norm := config.get("grad_clip_norm"):
    if config.get("grad_clip_per_example"):
      clip_tx = clip_by_per_example_global_norm(clip_norm)
    else:
      clip_tx = optax.clip_by_global_norm(clip_norm)
    grad_clip_norm_tx = optax.masked(clip_tx, not_frozen_mask)
  else:
    grad_clip_norm_tx = optax.identity()

  # Optimizer updates.
  tx_func = operator.attrgetter(config.optax_name)(optax)
  opt_txs = [optax.masked(tx_func(**config.get("optax", {})), not_frozen_mask)]
  assert "optim" not in config, "Deprecated option, use config.optax."

  # Learning rate multipliers. Defaults to 1.0.
  lr_mult_txs = [optax.scale(config.lr)]
  if config.get("lr_mults"):
    masks, mults = _make_mask_trees(params, config.lr_mults, "config.lr_mults")
    assert all(mult > 0 for mult in mults), (
        f"Use schedule=None for parameter freezing instead of lr_mults={mults}")
    lr_mult_txs += [
        optax.masked(optax.scale(mult), mask)
        for mult, mask in zip(mults, masks)
    ]

  # Weight decay. Defaults to 0.0.
  # Weight decay is not gradient-based but instead uses "params side-input".
  # Hence, weight decay is additive and independent of previous gradient-based
  # updates.
  assert "weight_decay" not in config, "Deprecated option. Use wd and schedule."
  assert config.get("weight_decay_decouple", True), (
      "Coupled weight decay not supported anymore.")
  if config.get("wd"):
    wd_mults = config.get("wd_mults", [(".*/kernel$", 1.0)])
    masks, mults = _make_mask_trees(params, wd_mults, "config.wd_mults")
    weight_decay_txs = [
        optax.add_decayed_weights(config.wd * mult, mask)
        for mult, mask in zip(mults, masks)
    ]
  else:
    weight_decay_txs = []

  # Combine gradient updates and learning rate schedules.
  return optax.chain(
      grad_clip_norm_tx,
      *opt_txs,
      *lr_mult_txs,
      *weight_decay_txs,
      *schedule_txs,
      optax.scale(-1.0)), schedule_fns


def _make_mask_trees(params, patterns_values, log):
  patterns, values = zip(*patterns_values)
  masks = tree_utils.make_mask_trees(params, patterns, log=log)
  return masks, values


def scale_by_adafactor(min_dim_size_to_factor=32,
                       decay_rate=0.8, decay_offset=0,
                       beta2_cap=0.999,
                       clipping_threshold=None,
                       momentum=0.9, dtype_momentum=jnp.bfloat16,
                       eps=1e-30):
  """The BigVision variant of Adafactor optimizer."""

  def _decay_rate_pow(i, exponent):
    """Second-order moment decay schedule."""
    t = jnp.array(i, jnp.float32) + 1.0
    return jnp.minimum(beta2_cap, 1.0 - t**(-exponent))

  scale_by_rms = optax.scale_by_factored_rms(
      factored=True,
      decay_rate=decay_rate,
      step_offset=decay_offset,
      min_dim_size_to_factor=min_dim_size_to_factor,
      epsilon=eps,
      decay_rate_fn=_decay_rate_pow)

  clip = (optax.clip_by_block_rms(clipping_threshold) if clipping_threshold
          else optax.identity())

  mom = (optax.ema(momentum, debias=False, accumulator_dtype=dtype_momentum)
         if momentum else optax.identity())

  return optax.chain(scale_by_rms, clip, mom)


def momentum_hp(momentum=0.9, dtype=jnp.bfloat16, nesterov=False):
  """SGD-Momentum with half-precision accumulator."""
  return optax.trace(decay=momentum, accumulator_dtype=dtype, nesterov=nesterov)


# Aliases for custom optimizers.
# A fake object to allow for foo.bar access syntax, see
# https://stackoverflow.com/a/19476841/2366315
optax.big_vision = type("", (), {})()
optax.big_vision.scale_by_adafactor = scale_by_adafactor  # pytype: disable=module-attr
optax.big_vision.momentum_hp = momentum_hp  # pytype: disable=module-attr
optax.big_vision.sgd = optax.identity  # pytype: disable=module-attr


def create_learning_rate_schedule(
    total_steps,
    batch_size=None,
    data_size=None,
    base=1.0,
    decay_type="stair",
    scale_with_batchsize=False,
    **kw,
):
  """Creates learning rate schedule.

  Args:
    total_steps: The total number of steps to run.
    batch_size: The global batch-size optionally used for scaling.
    data_size: Number of examples in the training data (for epoch conversion).
    base: The starting learning-rate (without warmup).
    decay_type: 'linear' or 'cosine', 'rsqrt', 'stair'.
    scale_with_batchsize: Whether or not to scale lr automatically.
    **kw: extra arguments specific to individual decay_types. Also contains
      declaration of `{warmup,cooldown}_{steps,epochs,examples}` that applies on
      top of any/all decay_type.

  Returns:
    A function learning_rate(step): float -> {"learning_rate": float}.
  """

  def to_steps(name, default=0):
    return train_utils.steps(name, kw, data_size, batch_size, total_steps,
                             default=default)

  warmup_steps = to_steps("warmup")
  cooldown_steps = to_steps("cooldown")
  frozen_steps = to_steps("frozen")

  # Early catch hard to backtrack errors due to warmup_steps >= total_steps,
  # but let it run for 0 and 1 steps used to eval and debug runs.
  assert (total_steps <= 1) or (
      warmup_steps < total_steps
  ), "warmup_steps is >= total_steps"

  def step_fn(step):
    """Step to learning rate function."""
    lr = base

    # This implements the linear scaling rule following
    # Goyal et al. at arxiv.org/abs/1706.02677.
    # The reference batch size in literature is 256, so we scale the lr to
    # adjust to the literature lr when bach_size changes.
    if scale_with_batchsize:
      lr = lr * batch_size / 256.0

    progress = (step - warmup_steps - frozen_steps) / (
        total_steps - warmup_steps - frozen_steps
    )
    progress = jnp.clip(progress, 0.0, 1.0)
    if decay_type in ("linear", "polynomial"):
      power = kw.get("power", 1)
      zero = kw.get("end", kw.get("linear_end", 0))
      lr = zero + (lr - zero) * (1.0 - progress) ** power
    elif decay_type == "cosine":
      lr = lr * 0.5 * (1.0 + jnp.cos(jnp.pi * progress))
    elif decay_type == "rsqrt":
      # See (internal link) for details, especially how to set timescale
      # and shift in order to continue smoothly when changing batch-size.
      t = to_steps("timescale", default=kw.get("timescale", 10_000))
      shift = to_steps("shift", default=kw.get("shift", 0))
      lr = jnp.where(
          warmup_steps <= step,
          lr / jnp.sqrt(1 + (step + shift - warmup_steps) / t),  # In decay
          lr / jnp.sqrt(1 + shift / t),
      )  # In warmup.
    elif decay_type == "stair":
      i = jnp.searchsorted(jnp.array(kw.get("steps", [])), step + 1)
      lr = lr * jnp.take(jnp.array([1.0] + list(kw.get("mults", []))), i)
    else:
      raise ValueError(f"Unknown lr type {decay_type}")

    if warmup_steps:
      lr = lr * jnp.minimum(1.0, (step - frozen_steps) / warmup_steps)
    if cooldown_steps:
      lr = lr * jnp.minimum(1.0, (total_steps - step) / cooldown_steps)
    lr = jnp.where(step < frozen_steps, 0.0, lr)

    return jnp.asarray(lr, dtype=jnp.float32)

  return step_fn
