# Copyright 2025 DeepMind Technologies Limited.
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

"""Simple trainer."""

from functools import partial  # pylint: disable=g-importing-member
import multiprocessing.pool
import os
import time

from absl import app
from absl import flags
from absl import logging
from clu import parameter_overview
import flax
import jax
from jax.experimental import multihost_utils
import jax.numpy as jnp
from jeo import input_pipeline
from jeo import train_utils
from jeo.components import early_stopping
from jeo.components import mixup
from jeo.evaluators import builder as eval_builder
from jeo.tasks import task_builder
from jeo.tools import bv_optax
from jeo.tools import checkpointing
from jeo.tools import inspect
from jeo.tools import metric_writers
from ml_collections import config_flags
import numpy as np
import optax

from tensorflow.io import gfile

config_flags.DEFINE_config_file(
    "config", None, "Training configuration.", lock_config=True
)
flags.DEFINE_string("workdir", default=None, help="Work unit directory.")
flags.DEFINE_boolean(
    "cleanup",
    default=False,
    help="Delete workdir (only) after successful completion.",
)
flags.DEFINE_string(
    "file_group",
    default=None,
    help="Permission group to use for file storage, e.g. checkpointing.",
)
flags.DEFINE_string(
    "data_service_address",
    None,
    "The address of the tf.data service (currently only for training).",
)


FLAGS = flags.FLAGS

# Adds jax flags to the program.
jax.config.parse_flags_with_absl()
# Fixes design flaw in jax.random that may cause unnecessary device-to-device
# communications, making some operations faster.
jax.config.update("jax_threefry_partitionable", True)


def _main(_):
  """Runs the training loop end-to-end."""
  # ## Trainer section 1: Setup.

  config = FLAGS.config
  workdir = FLAGS.workdir
  logging.info(
      "\u001b[33mHello from process %s holding %s/%s devices and writing to "
      "workdir %s.\u001b[0m",
      jax.process_index(),
      jax.local_device_count(),
      jax.device_count(),
      workdir,
  )
  train_utils.validate_and_update_config_inplace(config)
  os.umask(0o022); gfile.makedirs(workdir)

  # The pool is used to perform misc operations such as logging in async way.
  pool = multiprocessing.pool.ThreadPool()

  # Register preprocessing ops from modules listed on `pp_modules`. Note that
  # the most common modules are automatically registered in input_pipeline:
  # pp_ops, pp_image_ops, decode_ops, bv_ops, sat_ops.
  for m in config.get("pp_modules", []):
    train_utils.import_module(m, "pp")

  # This seed makes the Jax part of things (like model init) deterministic.
  # However, full training still won't be deterministic, for example due to the
  # tf.data pipeline not being deterministic even if we would set TF seed.
  # See (internal link) for a fun read on what it takes.
  rng = jax.random.PRNGKey(config.get("seed", 0))

  xid, wid = -1, -1
  fillin = lambda s: s
  def write_note(note):
    if jax.process_index() == 0:
      logging.info("\u001b[33mNOTE\u001b[0m: %s", note)

  last_time_set_notes = 0
  notes_wait_time = config.get("notes_wait_time", 0)

  def write_note(note):
    nonlocal last_time_set_notes
    if jax.process_index() == 0:
      # Don't write notes more than once every notes_wait_time seconds.
      # For experiments that run very fast, one could exhaust quota for analysis
      # context service.
      if time.time() - last_time_set_notes >= notes_wait_time:
        last_time_set_notes = time.time()
        pool.apply_async(lambda note=note: xm_wu.set_notes(note))
      logging.info("NOTE: %s", note)

  write_note("Initializing...")

  batch_size = config.batch_size
  local_batch_size = batch_size // jax.process_count()
  if batch_size % jax.device_count() != 0:
    raise ValueError(
        f"Batch sizes ({batch_size} must "
        f"be divisible by device number ({jax.device_count()})"
    )
  logging.info(
      "Global batch size %d on %d hosts results in %d local batch size. "
      "With %d dev per host (%d dev total), that's a %d per-device batch size.",
      batch_size,
      jax.process_count(),
      local_batch_size,
      jax.local_device_count(),
      jax.device_count(),
      local_batch_size // jax.local_device_count(),
  )

  mw = metric_writers.MetricWriter(
      xid,
      wid,
  )
  chrono = train_utils.Chrono()

  # ## Trainer section 2: Datasets initialization.
  write_note("Initializing train dataset...")
  train_ds, ntrain_img = input_pipeline.get_data(
      train=True,
      dataset=config.get("dataset"),
      split=config.get("train_split", "train"),
      data_dir=fillin(config.get("dataset_dir")),
      dataset_module=config.get("dataset_module"),
      **config.get("dataset_kwargs", {}),
      batch_size=local_batch_size,
      preprocess_fn=config.pp_train,
      filter_fn=config.get("filter_fn", None),
      filter_final_fn=config.get("filter_final_fn", None),
      shuffle_buffer_size=config.get("shuffle_buffer_size"),
      prefetch=config.get("prefetch_to_host", 2),
      cache_raw=config.get("cache_raw", False),
      skip_decode=config.get("skip_decode", ("image",)),
      download_and_prepare=config.get("download_and_prepare", False),
      wid=wid,
      data_service_address=FLAGS.data_service_address,
  )
  # Start prefetching already.
  train_iter = input_pipeline.start_input_pipeline(
      train_ds, config.get("prefetch_to_device", 1)
  )
  steps_per_epoch = ntrain_img / batch_size

  # Get specific steps (configured as steps, epochs, examples or percent).
  total_steps = train_utils.steps("total", config, ntrain_img, batch_size)

  def get_steps(name, default=ValueError, cfg=config):
    return train_utils.steps(
        name, cfg, ntrain_img, batch_size, total_steps, default
    )

  log_training_steps = get_steps("log_training")
  keep_ckpt_steps = get_steps("keep_ckpt", None)
  ckpt_steps = get_steps("ckpt", None)
  save_ckpt_path = None
  if ckpt_steps:
    save_ckpt_path = os.path.join(workdir, "checkpoint.npz")

  logging.info(
      "Running train for %d steps, that means %f epochs and %f steps per epoch",
      total_steps,
      total_steps * batch_size / ntrain_img,
      steps_per_epoch,
  )
  logging.info("train_ds element_spec: %s", train_ds.element_spec)
  logging.info("ntrain_img: %s", ntrain_img)

  if total_steps < 1:
    raise ValueError(
        f"total_steps=={total_steps} will not do any training or evaluation."
    )

  # ## Trainer section 3: Model, task, and optimizer initialization.
  write_note(f"Initializing {config.model_name} model...")
  model_mod = train_utils.import_module(config.model_name, "models")
  model = model_mod.Model(**config.get("model", {}))

  task = task_builder.from_config(config)

  # We want all parameters to be created in host RAM, not on any device, they'll
  # be sent there later as needed, otherwise we already encountered two
  # situations where we allocate them twice.
  @partial(jax.jit, backend="cpu")
  def init(rng):
    batch = jax.tree.map(
        lambda x: jnp.zeros(
            (local_batch_size,) + x.shape[1:],
            x.dtype.as_numpy_dtype,
        ),
        train_ds.element_spec,
    )
    inputs = task.model_inputs(batch)
    variables = model.init(
        rng,
        *inputs,
        train=config.get("train_at_init") or config.get("train_always"),
    )
    model_state, params = flax.core.pop(variables, "params")
    params = flax.core.unfreeze(params)
    # Set bias in the head to a low value, such that loss is small initially.
    if "init_head_bias" in config:
      params["head"]["bias"] = jnp.full_like(
          params["head"]["bias"], config["init_head_bias"]
      )
    return params, model_state

  eval_rng_names = {"stochastic"}
  # noisy_labels layers require RNGS at init and eval. The eval is stochastic.
  if config.model.get("head_temperature"):  # For noisy labels.
    eval_rng_names |= {"diag_noise_samples", "standard_norm_noise_samples"}
  if config.get("train_at_init") or config.get("train_always"):
    eval_rng_names |= {"dropout", "mask", "masking"}
  init_rng_names = {"params"} | eval_rng_names
  train_rng_names = init_rng_names | {"dropout", "mask", "masking"}
  # IMPORTANT: To make sure that the order of rngs is deterministic we convert
  # them to lists and sort them. For example operation "|" on dicts is not
  # deterministic on different hosts.
  init_rng_names = sorted(list(init_rng_names))
  eval_rng_names = sorted(list(eval_rng_names))
  train_rng_names = sorted(list(train_rng_names))
  # Keeping eval RNGs constant across all evals to avoid fluctuations.
  eval_rngs = (
      None
      if not eval_rng_names
      else dict(zip(eval_rng_names, jax.random.split(rng, len(eval_rng_names))))
  )
  rng, *rng_init = jax.random.split(rng, 1 + len(init_rng_names))
  rng_init = dict(zip(init_rng_names, rng_init))
  multihost_utils.assert_equal(eval_rngs, "non deterministic")
  multihost_utils.assert_equal(rng_init, "non deterministic")

  params_cpu, state_cpu = init(rng_init)
  multihost_utils.assert_equal(state_cpu, "non deterministic")
  multihost_utils.assert_equal(params_cpu, "non deterministic")

  if jax.process_index() == 0:
    num_params = sum(p.size for p in jax.tree.leaves(params_cpu))
    parameter_overview.log_parameter_overview(params_cpu, msg="init params")
    if state_cpu:
      parameter_overview.log_parameter_overview(state_cpu, msg="init_state")
    mw.measure("num_params", num_params)
    num_freeze, num_learn = train_utils.log_frozen(params_cpu, config.schedule)
    mw.measure("num_arrays_frozen", num_freeze)
    mw.measure("num_arrays_learned", num_learn)

  write_note(f"Initializing {config.optax_name} optimizer...")
  tx, sched_fns = bv_optax.make(
      config,
      params_cpu,
      sched_kw=dict(
          batch_size=batch_size, total_steps=total_steps, data_size=ntrain_img
      ),
  )

  # We jit this, such that the arrays are created on the CPU, not device[0].
  opt_cpu = jax.jit(tx.init, backend="cpu")(params_cpu)
  sched_fns_cpu = [jax.jit(sched_fn, backend="cpu") for sched_fn in sched_fns]

  # ## Trainer section 4: Train step update function.
  @partial(jax.pmap, axis_name="batch", donate_argnums=(0, 1))
  def update_fn(params, state, opt, rng, batch):
    """Update step."""
    measurements = {}
    # Get device-specific loss rng.
    rng, rng_model = jax.random.split(rng, 2)
    rng_local = jax.random.fold_in(rng_model, jax.lax.axis_index("batch"))
    # Use Mix-augmentation for segmentation.
    if "mixup" in config:
      rng_local, batch = mixup.get_mixup(rng_local, batch, **config.mixup)
    rngs_local = dict(
        zip(train_rng_names, jax.random.split(rng_local, len(train_rng_names)))
    )

    def loss_fn(params, state, batch):
      model_inputs = task.model_inputs(batch)
      model_outputs, mutated_state = model.apply(
          {"params": flax.core.freeze(params), **state},
          *model_inputs,
          train=True,
          mutable=list(state.keys()),
          rngs=rngs_local,
      )
      loss, aux = task.get_loss_and_aux(model_outputs, batch, train=True)
      aux["state"] = mutated_state
      return loss, aux

    (l, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        params, state, batch
    )
    measurements["l2_grads_raw"] = optax.global_norm(grads)
    l, aux, grads = jax.lax.pmean((l, aux, grads), axis_name="batch")
    updates, opt = tx.update(grads, opt, params)
    params = optax.apply_updates(params, updates)
    state = aux.pop("state")

    grads_nonfrozen = bv_optax.replace_frozen(
        config.schedule, grads, jnp.zeros(1)
    )
    measurements["l2_grads"] = optax.global_norm(grads_nonfrozen)
    measurements["l2_params"] = optax.global_norm(params)
    measurements["l2_updates"] = optax.global_norm(updates)
    measurements["l2_state"] = optax.global_norm(state)
    for k, v in aux.items():
      measurements[f"train_{k}"] = v
    return params, state, opt, rng, l, measurements

  # ## Trainer section 5: Initialize model from checkpoint.
  # Decide how to initialize training. The order is important.
  # 1. Always resumes from the existing checkpoint, e.g. resumes a finetune job.
  # 2. Resume from a previous checkpoint, e.g. start a cooldown training job.
  # 3. Initialize model from something, e.g., start a fine-tuning job.
  # 4. Train from scratch.
  resume_ckpt_path = None
  if save_ckpt_path and gfile.exists(save_ckpt_path):
    resume_ckpt_path = save_ckpt_path
  elif config.get("resume"):
    resume_ckpt_path = fillin(config.resume)
  if resume_ckpt_path:
    write_note("Resume training from checkpoint...")
    checkpoint = {
        "params": params_cpu,
        "opt": opt_cpu,
        "chrono": chrono.save(),
        **state_cpu,
    }
    ckpt_tree = jax.tree.structure(checkpoint)
    loaded = checkpointing.load_checkpoint(resume_ckpt_path, ckpt_tree)
    # bfloat16 type gets lost when data is saved to disk, so we recover it.
    checkpoint = jax.tree.map(checkpointing.recover_dtype, loaded)
    params_cpu, opt_cpu = checkpoint["params"], checkpoint["opt"]
    state_cpu = {k: v for k, v in checkpoint.items() if k in state_cpu}
    chrono.load(checkpoint["chrono"])
  elif config.get("model_init"):
    write_note(f"Initialize model from {config.model_init}...")
    params_cpu, state_cpu = train_utils.load(
        model_mod.load,
        params_cpu,
        config.model_init,
        config.get("model"),
        **config.get("model_load", {}),
        init_state=state_cpu,
    )
    if jax.process_index() == 0:
      parameter_overview.log_parameter_overview(
          params_cpu, msg="restored params"
      )

  # ## Trainer section 6: Pre-loop misc stuff.
  write_note("Kicking off misc stuff...")
  first_step = bv_optax.get_count(opt_cpu)
  chrono.inform(
      first_step=first_step,
      total_steps=total_steps,
      global_bs=batch_size,
      steps_per_epoch=steps_per_epoch,
      measure=mw.measure,
      write_note=write_note,
  )
  prof = None  # Keeps track of start/stop of profiler state.

  write_note(f"Replicating...\n{chrono.note}")
  params_repl = flax.jax_utils.replicate(params_cpu)
  state_repl = flax.jax_utils.replicate(state_cpu)
  opt_repl = flax.jax_utils.replicate(opt_cpu)

  # We do not jit/pmap this function, because it is passed to evaluator that
  # does it later. We output as many intermediate tensors as possible for
  # maximal flexibility. Later `jit` will prune out things that are not needed.
  predict_fn = task.get_predict_fn(
      model, rngs=eval_rngs, train=config.get("train_always")
  )
  evaluators = eval_builder.from_config(
      config,
      predict_fn,
      lambda s: write_note(f"Init evaluator: {s}...\n{chrono.note}"),
      get_steps,
      workdir,
  )
  early_stopper = early_stopping.from_config(config, steps_per_epoch)

  _, rng_loop = jax.random.split(rng, 2)
  rngs_loop = flax.jax_utils.replicate(rng_loop)
  ckpt_writer = None

  def run_evaluators(step, stop_training=False):
    saved_metrics = {}
    for name, evaluator, log_steps, prefix in evaluators:
      if (
          train_utils.itstime(step, log_steps, total_steps, first=False)
          or stop_training
      ):
        chrono.pause(wait_for=(params_repl, state_repl))
        write_note(f"{name} evaluation...\n{chrono.note}")
        for key, value in evaluator.run({"params": params_repl, **state_repl},
                                        train_step=step):
          mw.measure(f"{prefix}{key}", value)
          stop_training |= early_stopper.should_stop(
              step, f"{prefix}{key}", value
          )
          saved_metrics[f"{prefix}{key}"] = value
        chrono.resume()
    if saved_metrics and not jax.process_index():
      if stop_training or step == total_steps:  # Always at the end.
        pool.apply_async(train_utils.save_metrics, (workdir, saved_metrics))
      elif config.get("save_metrics_at_each_eval"):  # At each eval.
        pool.apply_async(
            train_utils.save_metrics, (workdir, saved_metrics, step)
        )

  if first_step == 0 and config.get("eval_first_step", True):
    mw.step_start(first_step)
    run_evaluators(first_step)
    mw.step_end()

  # If we just want to run the evaluators on the given checkpoint, we specify
  # total_steps == 1 (ie. no training and only the evaluation is performed).
  do_train = total_steps > 1  # Don't train if only running eval for 1 step.

  # ## Trainer section 7: Train loop.
  write_note(f"First step compilations...\n{chrono.note}")
  error = None  # For exiting with an error after cleanup. Avoids indentation.
  stop_training = False  # For early stopping.
  train_metrics = []
  # Using a python integer for step here, because opt.state.step is allocated
  # on TPU during replication.
  for step, train_batch in zip(
      range(first_step + 1, total_steps + 1), train_iter
  ):
    mw.step_start(step)

    if do_train:
      with jax.profiler.StepTraceAnnotation("train_step", step_num=step):
        (
            params_repl,
            state_repl,
            opt_repl,
            rngs_loop,
            loss_value,
            measurements,
        ) = update_fn(params_repl, state_repl, opt_repl, rngs_loop, train_batch)
      if jax.process_index() == 0:
        train_metrics.append({"training_loss": loss_value, **measurements})

      # On the first host, let's always profile a handful of early steps.
      if jax.process_index() == 0 and config.get("xprof", True):
        prof = train_utils.startstop_prof(
            prof, step, first_step, log_training_steps
        )

    # Checkpoint saving.
    if (do_train and save_ckpt_path and train_utils.itstime(
        step, ckpt_steps, total_steps, host=0, first=False)):  # fmt: skip
      chrono.pause(wait_for=(params_repl, opt_repl, state_repl))
      train_utils.checkpointing_timeout(
          ckpt_writer, config.get("ckpt_timeout", 1)
      )
      # We need to transfer the weights over now or else we risk keeping them
      # alive while they'll be updated in a future step, creating hard to debug
      # memory errors (see (internal link)). Also, takes device 0's params only.
      params_cpu = jax.tree.map(lambda x: np.array(x[0]), params_repl)
      opt_cpu, state_cpu = jax.tree.map(
          lambda x: np.array(x[0]), (opt_repl, state_repl)
      )
      # Check whether we want to keep a copy of the current checkpoint.
      copy_step = None
      if train_utils.itstime(step, keep_ckpt_steps, total_steps):
        copy_step = step
      checkpoint = {
          "params": params_cpu,
          "opt": opt_cpu,
          "chrono": chrono.save(),
          **state_cpu,
      }
      ckpt_writer = pool.apply_async(
          checkpointing.save_checkpoint_oss,
          (checkpoint, save_ckpt_path, copy_step, FLAGS.file_group),
      )
      chrono.resume()
    # Report training progress.
    if (
        do_train
        and train_utils.itstime(step, log_training_steps, total_steps, host=0)
        or chrono.warmup
        and jax.process_index() == 0
    ):
      for i, sched_fn_cpu in enumerate(sched_fns_cpu):
        mw.measure(f"global_schedule{i if i else ''}", sched_fn_cpu(step - 1))
      l = mw.measure("training_loss", loss_value[0])  # pylint: disable=undefined-variable
      for name, value in measurements.items():  # pylint: disable=undefined-variable
        mw.measure(name, value[0])
        stop_training |= early_stopper.should_stop(step, name, value[0])
      # Report training metrics aggregated between the logging events.
      for name, value in inspect.pytree_list_to_dict(train_metrics).items():
        mw.measure(f"{name}_agg", value.mean())
      train_metrics = []
      chrono.tick(step, mw.measure, write_note)
      if not np.isfinite(l):
        error = (
            "The loss became nan or inf somewhere within steps "
            f"[{step - log_training_steps}, {step}]"
        )
        break

    run_evaluators(step, stop_training)
    mw.step_end()
    if stop_training:
      write_note(f"Early stopping triggered at step {step}...\n{chrono.note}")
      break

  # ## Trainer section 8: Post-loop finalize.
  # Always give a chance to stop the profiler, no matter how things ended.
  if jax.process_index() == 0 and prof is not None:
    train_utils.startstop_prof(prof)

  # Last note needs to happen before the pool is closed.
  if not stop_training:
    if not error:
      write_note(f"Done!\n{chrono.note}")
    else:
      write_note(f"Failed!\n{error}\n{chrono.note}")

  pool.close()
  pool.join()
  mw.close()
  train_utils.finalize_and_cleanup(workdir, FLAGS.cleanup, error)


def main(_):
  _main(_)


if __name__ == "__main__":
  jax.config.parse_flags_with_absl()  # Adds jax flags to the program.
  app.run(main)
