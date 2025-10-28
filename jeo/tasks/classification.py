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

"""Classification tasks."""
from typing import Any

import jax
import jax.numpy as jnp
from jeo.tasks import task_builder


class ClassificationTask(task_builder.TaskBase):
  """Classification task."""

  def __init__(
      self,
      weights_key=None,
      exclude_background_class=False,
      **kwargs):
    super().__init__(**kwargs)
    self.weights_key = weights_key
    self.exclude_background_class = exclude_background_class

  def get_loss_and_aux(self, model_outputs, *batches, train=False):
    """Returns loss and aux dict."""
    batch = batches[0]
    if not isinstance(model_outputs, (list, tuple)):
      model_outputs = (model_outputs,)
    logits = model_outputs[0]  # (B,N)
    if logits.shape != batch["labels"].shape:  # one_hot encoded (B,N)
      raise ValueError(f"Logits shape {logits.shape} does not match batch "
                       f"labels shape {batch['labels'].shape}")
    pred = jnp.argmax(logits, axis=1)  # top1_idx (B,)
    # NOTE: that acc is not weighted, only loss is.
    aux = {"acc": jnp.take_along_axis(
        batch["labels"], pred[:, None], axis=1)[:, 0]}
    weights = _get_weights(batch, self.exclude_background_class,
                           weights_key=self.weights_key)
    loss = self.loss_fn(logits=logits, labels=batch["labels"], reduction=train,
                        weights=weights)
    if train:
      aux = jax.tree.map(jnp.mean, aux)
    return loss, aux


class MultiHeadClassificationTask(task_builder.TaskBase):
  """Classification task."""

  def get_loss_and_aux(self, model_outputs, *batches, train=False):
    """Returns loss and aux dict."""
    batch = batches[0]
    if not isinstance(model_outputs, (list, tuple)):
      model_outputs = (model_outputs,)
    logits = model_outputs[0]  # (B,num_task_heads,N)
    head = batch["head"]
    head = jnp.expand_dims(head, [1, 2])
    logits = jnp.take_along_axis(logits, head, axis=1)
    logits = jnp.squeeze(logits, axis=1)
    if logits.shape != batch["labels"].shape:  # one_hot encoded (B,N)
      raise ValueError(f"Logits shape {logits.shape} does not match batch "
                       f"labels shape {batch['labels'].shape}")
    pred = jnp.argmax(logits, axis=1)  # top1_idx (B,)
    aux = {"acc": jnp.take_along_axis(
        batch["labels"], pred[:, None], axis=1)[:, 0]}
    loss = self.loss_fn(logits=logits, labels=batch["labels"], reduction=train)
    if train:
      aux = jax.tree.map(jnp.mean, aux)
    return loss, aux

  def get_infer_postprocessing_fn(self):
    """Returns a function to postprocess model outputs for inference."""

    def fn(model_outputs):
      if not isinstance(model_outputs, (list, tuple)):
        model_outputs = (model_outputs,)
      logits, _ = model_outputs  # pytype: disable=bad-unpacking
      predictions = jnp.argmax(logits, axis=-1)
      probs = (250 * jax.nn.softmax(logits)).astype(jnp.uint8)
      return {"preds": predictions, "probs": probs}

    return fn


class MultitaskClassificationTask(task_builder.TaskBase):
  """Multitask classification task."""

  def get_loss_and_aux(self, model_outputs, *batches, train=False):
    """Returns loss and aux dict."""
    batch = batches[0]
    if not isinstance(model_outputs, (list, tuple)):
      model_outputs = (model_outputs,)
    aux = {}
    task_losses, task_accuracies = [], []
    for task_name, task_outputs in model_outputs[0].items():
      logits = task_outputs["logits"]  # (B,N)
      labels = batch[f"{task_name}_labels"]  # (B,N)
      assert logits.shape == labels.shape, "Assume one_hot encoded."
      pred = jnp.argmax(logits, axis=1)  # (B,)
      aux[f"acc_{task_name}"] = jnp.take_along_axis(
          labels, pred[:, None], axis=1)[:, 0]
      loss = self.loss_fn(labels=labels, logits=logits, reduction=train)
      aux[f"loss_{task_name}"] = loss
      task_losses.append(loss)
      task_accuracies.append(aux[f"acc_{task_name}"])
    # Loss aggregation. Simple sum of multi-task losses for now.
    loss = jnp.sum(jnp.asarray(task_losses), axis=0)
    aux["acc"] = jnp.mean(jnp.asarray(task_accuracies), axis=0)
    if train:
      aux = jax.tree.map(jnp.mean, aux)
    return loss, aux


def _get_weights(
    batch: dict[str, Any],
    exclude_background_class: bool,
    weights_key: str | None) -> jnp.ndarray | None:
  """Returns weights for loss."""
  weights = None
  if weights_key:
    if weights_key not in batch.keys():
      raise ValueError(f"Weights key {weights_key} not in batch.")
    weights = batch[weights_key]
  if exclude_background_class:
    labels = batch["labels"].argmax(-1)  # Get class from onehot encoded.
    if weights is None:
      weights = (labels > 0).astype("float32")
    else:
      weights = weights * (labels > 0).astype("float32")
  return weights
