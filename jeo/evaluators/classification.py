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

"""Classification evaluator."""
import functools
from typing import Any, Callable

import flax
import jax
import jax.numpy as jnp
from jeo import losses
from jeo.evaluators import builder as eval_builder
from jeo.evaluators import utils


DEFAULT_METRICS = ("acc", "f1", "aucpr", "prec", "recall", "loss")


def onehot(labels, num_classes, on_value=1.0, off_value=0.0) -> jnp.ndarray:
  x = labels[..., None] == jnp.arange(num_classes)[None]
  x = jax.lax.select(
      x, jnp.full(x.shape, on_value), jnp.full(x.shape, off_value)
  )
  return x.astype(jnp.float32)


def _postprocess_logits(logits: jnp.ndarray,
                        class_mapping: dict[int, int]) -> jnp.ndarray:
  predictions = jax.nn.softmax(logits)
  new_predictions = jnp.zeros_like(predictions)
  for old_class, new_class in class_mapping.items():
    new_predictions = new_predictions.at[..., new_class].set(
        new_predictions[..., new_class] + predictions[..., old_class])
  return jax.nn.one_hot(new_predictions.argmax(-1),
                        num_classes=logits.shape[-1])


class Evaluator(eval_builder.EvaluatorBase):
  """Classification (binary, multi-class, multi-label) evaluator."""

  def __init__(
      self,
      predict_fn,
      batch_size,
      loss_name="sigmoid_xent",
      multilabel=False,
      metrics=DEFAULT_METRICS,
      subsplit=None,
      subsplit_key="subsplit",
      multihead=False,
      loss_kw=None,
      per_class_metrics=False,
      label_map=None,
      exclude_background_class=False,
      class_mapping=None,
      logits_filter_key=None,
      logits_postprocess_fn: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
      **cfg,
  ):
    self._setup_metrics_and_eval_iter(metrics, predict_fn)
    self.cfg = {"batch_size": batch_size, **cfg}
    self.loss_fn = losses.get_loss_fn(loss_name, **(loss_kw or {}))
    self.logits_to_pred_fn = jax.nn.sigmoid if multilabel else jax.nn.softmax
    self.subsplit = subsplit
    self.multihead = multihead
    self.subsplit_key = subsplit_key
    self.per_class_metrics = per_class_metrics  # Bool or list.
    self.logits_filter_key = logits_filter_key
    self.logits_postprocess_fn = logits_postprocess_fn
    self.label_map = label_map
    self.exclude_background_class = exclude_background_class
    self.class_mapping = class_mapping

  def _get_eval_fn(self, predict_fn):
    """Produces eval function, also applies pmap."""

    @functools.partial(jax.pmap, axis_name="batch")
    def _eval_fn(params, batch):
      logits, *_ = predict_fn(params, **batch)  # logits: (B,N)
      if self.logits_postprocess_fn:
        logits = self.logits_postprocess_fn(logits)
      # TODO: Move into logits_postprocess_fn.
      if self.class_mapping:
        logits = _postprocess_logits(logits, self.class_mapping)
      if self.logits_filter_key:
        logits = logits + batch[self.logits_filter_key]
      labels = batch["labels"]
      if self.multihead:
        # logits (B,num_task_heads,N)
        head = batch["head"]
        head = jnp.expand_dims(head, [1, 2])
        logits = jnp.take_along_axis(logits, head, axis=1)
        logits = jnp.squeeze(logits, axis=1)
      # If the underlying model is doing segmentation, we need to preaggregate
      # the logits to make it look like classification.
      if len(logits.shape) == 4:  # (B,H,W,N)
        assert len(batch["segmentation_mask"].shape) == 3
        assert batch["segmentation_mask"].shape[0] == labels.shape[0]
        # Convert to probs so we could add them up.
        logits = jax.nn.softmax(logits)
        # Only use the pixels that correspond to batch["labels"]
        label_mask = (jnp.expand_dims(jnp.expand_dims(labels, -1), -1) ==
                      batch["segmentation_mask"])
        logits = logits * jnp.expand_dims(label_mask, axis=-1)
        # Sum over the H and W dimensions.
        logits = logits.sum(axis=-2).sum(axis=-2)
        # Labels need to be onehot encoded.
        labels = onehot(labels, logits.shape[-1], 1, 0)

      if logits.shape != labels.shape:  # one_hot encoded (B,N)
        raise ValueError(
            f"Logits shape {logits.shape} does not match batch "
            f" labels shape {labels.shape}"
        )

      if self.exclude_background_class:
        weights = (jnp.argmax(labels, axis=-1) > 0).astype("float32")
      else:
        weights = jnp.ones(labels.shape[:-1], "float32")

      loss = self.loss_fn(
          logits=logits, labels=labels, reduction=False, weights=weights
      )
      outputs = {"logits": logits,  # (B,N)
                 "predictions": self.logits_to_pred_fn(logits),}  # (B,N)
      inputs = {
          "labels": labels,  # (B,N)
          "label_weights": weights,  # (B,)
          "mask": self._get_mask(batch),  # (B,)
      }
      extra = {"exclude_background_class": self.exclude_background_class}
      metrics_updates = self.metrics_fn(loss, outputs, inputs, extra, "batch")
      return metrics_updates, (logits,)
    return _eval_fn

  def _get_mask(self, batch):
    if self.subsplit is None:
      return batch["_mask"]
    batch_subsplits = batch[self.subsplit_key]
    subsplit = jnp.array(self.subsplit)
    return jnp.logical_and(batch["_mask"], jnp.isin(batch_subsplits, subsplit))

  def run(self, params, train_step):
    """Computes all metrics."""
    del train_step

    if not hasattr(self, "data_iter"):
      self._setup_dataset(**self.cfg)

    self.metrics.reset_states()
    for _, batch in zip(range(self.steps), self.data_iter):
      update, model_outputs = self.eval_fn(params, batch)
      self.metrics.update_state(flax.jax_utils.unreplicate(update))
    for k, v in self.metrics.result().items():
      if k == "confusion_matrix" and self.per_class_metrics:
        for name, value in utils.get_per_class_metrics(
            v, self.label_map, metrics=self.per_class_metrics):
          yield (name, value)
      yield (k, v)
