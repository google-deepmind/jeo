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

"""Classification evaluator."""
import functools

import flax
import jax
import jax.numpy as jnp
from jeo import losses
from jeo.evaluators import builder as eval_builder


DEFAULT_METRICS = ("acc", "f1", "aucpr", "prec", "recall", "loss")


class Evaluator(eval_builder.EvaluatorBase):
  """Classification (binary, multi-class, multi-label) evaluator."""

  def __init__(self, predict_fn, batch_size, loss_name="sigmoid_xent",
               multilabel=False, metrics=DEFAULT_METRICS, subsplit=None,
               subsplit_key="subsplit", multihead=False, loss_kw=None,
               **data_config):
    self._setup_metrics_and_eval_iter(metrics, predict_fn)
    self.data_config = {"batch_size": batch_size, **data_config}
    self.loss_fn = losses.get_loss_fn(loss_name, **(loss_kw or {}))
    self.logits_to_pred_fn = jax.nn.sigmoid if multilabel else jax.nn.softmax
    self.subsplit = subsplit
    self.multihead = multihead
    self.subsplit_key = subsplit_key

  def _get_eval_fn(self, predict_fn):
    """Produces eval function, also applies pmap."""

    @functools.partial(jax.pmap, axis_name="batch")
    def _eval_fn(params, batch):
      logits, *_ = predict_fn(params, **batch)  # logits: (B,N)
      labels = batch["labels"]
      if self.multihead:
        # logits (B,num_task_heads,N)
        head = batch["head"]
        head = jnp.expand_dims(head, [1, 2])
        logits = jnp.take_along_axis(logits, head, axis=1)
        logits = jnp.squeeze(logits, axis=1)
      if logits.shape != labels.shape:  # one_hot encoded (B,N)
        raise ValueError(
            f"Logits shape {logits.shape} does not match batch "
            f" labels shape {labels.shape}"
        )
      loss = self.loss_fn(logits=logits, labels=labels, reduction=False)
      outputs = {"logits": logits,  # (B,N)
                 "predictions": self.logits_to_pred_fn(logits),}  # (B,N)
      inputs = {
          "labels": labels,  # (B,N)
          "mask": self._get_mask(batch),  # (B,)
      }
      metrics_updates = self.metrics_fn(loss, outputs, inputs, {}, "batch")
      return metrics_updates
    return _eval_fn

  def _get_mask(self, batch):
    if self.subsplit is None:
      return batch["_mask"]
    batch_subsplits = batch[self.subsplit_key]
    subsplit = jnp.array(self.subsplit)
    return jnp.logical_and(batch["_mask"], jnp.isin(batch_subsplits, subsplit))

  def run(self, params):
    """Computes all metrics."""
    if not hasattr(self, "data_iter"):
      self._setup_dataset(**self.data_config)

    self.metrics.reset_states()
    for _, batch in zip(range(self.steps), self.data_iter):
      update = self.eval_fn(params, batch)
      self.metrics.update_state(flax.jax_utils.unreplicate(update))
    for k, v in self.metrics.result().items():
      yield (k, v)
