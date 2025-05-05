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

"""Semantic segmentation evaluator."""
import functools
from typing import Any

import flax
import jax
import jax.numpy as jnp
from jeo import losses
from jeo.evaluators import builder as eval_builder
from jeo.evaluators import utils


DEFAULT_METRICS = ("acc", "loss", "miou")


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
  """Semantic segmentation evaluator."""

  def __init__(self, predict_fn, batch_size,
               loss_name="generalized_softmax_xent",
               loss_kw=None,
               exclude_background_class=False,
               sample_proportion=1., false_label_ind=0, true_label_ind=1,
               metrics=DEFAULT_METRICS,
               labels_key="segmentation_mask", mask_key=None,
               per_class_metrics=False,
               label_map=None, lumascope: dict[str, Any] | None = None,
               strata_key=None, strata_weights=None,
               class_mapping=None,
               logits_filter_key=None,
               **data_config):
    self._setup_metrics_and_eval_iter(metrics, predict_fn)
    self.data_config = {"batch_size": batch_size, **data_config}
    self.metric_inputs_fn = self._get_metric_inputs()
    self.loss_fn = losses.get_loss_fn(loss_name, **(loss_kw or {}))
    self.logits_to_pred_fn = jax.nn.softmax
    self.exclude_background_class = exclude_background_class
    self.sample_proportion = sample_proportion
    self.false_label_ind = false_label_ind
    self.true_label_ind = true_label_ind
    self.labels_key = labels_key
    self.mask_key = mask_key if mask_key is not None else f"{labels_key}_mask"
    self.per_class_metrics = per_class_metrics  # Bool or list.
    self.label_map = label_map
    self.lumascope = self._get_lumascope(lumascope, "semantic")
    self.strata_key = strata_key
    self.strata_weights = strata_weights
    self.logits_filter_key = logits_filter_key
    self.class_mapping = class_mapping

  def _get_metric_inputs(self):

    def _f(batch, model_outputs):
      logits = model_outputs[0]  # (B,H,W,N)
      if self.class_mapping:
        logits = _postprocess_logits(logits, self.class_mapping)
      if self.logits_filter_key:
        logits = logits + batch[self.logits_filter_key]

      labels = batch[self.labels_key]  # (B,H,W) or (B,H,W,1) or (B,H,W,N)
      if labels.shape[-1] == 1:
        labels = labels[..., 0]  # (B,H,W)
      int_labels = labels.argmax(-1) if labels.shape == logits.shape else labels

      if self.exclude_background_class:
        weights = (int_labels > 0).astype("float32")
      else:
        weights = jnp.ones(int_labels.shape, "float32")

      # Exclude non-valid pixels.
      if self.mask_key in batch.keys():
        valid_mask = batch[self.mask_key]
        if valid_mask.shape[-1] == 1:
          valid_mask = jnp.squeeze(valid_mask, -1)
        weights *= valid_mask
      loss = self.loss_fn(
          logits=logits, labels=labels, reduction=False, weights=weights)

      extra = {}
      if isinstance(loss, tuple):
        assert len(loss) == 2 and isinstance(loss[1], dict)
        loss, extra = loss

      # TODO: Extend ConfusionMatrix metrics to dense predictions.
      outputs = {
          "logits": logits,  # (B,H,W,N)
          "predictions": self.logits_to_pred_fn(logits),  # (B,H,W,N)
      }
      # We ensure to pass integer labels to the metrics.
      inputs = {
          "labels": int_labels,  # (B,H,W)
          "label_weights": weights,  # (B,H,W)
          "mask": (batch["_mask"])  # (B,)
      }
      extra |= {"exclude_background_class": self.exclude_background_class,
                "sample_proportion": self.sample_proportion,
                "false_label_ind": self.false_label_ind,
                "true_label_ind": self.true_label_ind}
      if self.strata_key is not None:
        extra |= {"strata": batch[self.strata_key],
                  "strata_size": len(self.strata_weights)}
      return loss, outputs, inputs, extra

    return _f

  def _get_eval_fn(self, predict_fn):
    """Produces eval function, also applies pmap."""

    @functools.partial(jax.pmap, axis_name="batch")
    def _eval_fn(params, batch):
      model_outputs = predict_fn(params, **batch)
      if not isinstance(model_outputs, (list, tuple)):
        model_outputs = (model_outputs,)

      loss, outputs, inputs, extra = self.metric_inputs_fn(batch, model_outputs)
      metrics_updates = self.metrics_fn(loss, outputs, inputs, extra, "batch")
      return metrics_updates, model_outputs

    return _eval_fn

  def run(self, params):
    """Computes all metrics."""
    if not hasattr(self, "data_iter"):
      self._setup_dataset(**self.data_config)

    self.metrics.reset_states()
    for _, batch in zip(range(self.steps), self.data_iter):
      update, model_outputs = self.eval_fn(params, batch)
      self.metrics.update_state(flax.jax_utils.unreplicate(update))
      self.lumascope.add(model_outputs, batch)
    self.lumascope.finish()

    for k, v in self.metrics.result().items():
      if k == "confusion_matrix" and self.per_class_metrics:
        for name, value in utils.get_per_class_metrics(
            v, self.label_map, metrics=self.per_class_metrics):
          yield (name, value)
      yield (k, v)
      if k == "strata_binary_confusion_matrix" and self.per_class_metrics:
        for name, value in utils.get_stratified_metrics(
            v, self.strata_weights):
          yield (name, value)
