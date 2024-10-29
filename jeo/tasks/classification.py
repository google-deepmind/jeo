# Copyright 2024 The jeo Authors.
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

import functools

import jax
import jax.numpy as jnp
from jeo.tasks import task_builder


class ClassificationTask(task_builder.TaskBase):
  """Classification task."""

  def __init__(self, weights_key=None, **kwargs):
    super().__init__(**kwargs)
    self.weights_key = weights_key

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
    loss_fn = self.loss_fn
    if self.weights_key:
      loss_fn = functools.partial(loss_fn, weights=batch[self.weights_key])
    loss = loss_fn(logits=logits, labels=batch["labels"], reduction=train)
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
    # Loss aggreation. Simple sum of multi-task losses for now.
    loss = jnp.sum(jnp.asarray(task_losses), axis=0)
    aux["acc"] = jnp.mean(jnp.asarray(task_accuracies), axis=0)
    if train:
      aux = jax.tree.map(jnp.mean, aux)
    return loss, aux
