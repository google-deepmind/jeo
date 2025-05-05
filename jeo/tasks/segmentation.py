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

"""Segmentation tasks."""
import jax
import jax.numpy as jnp
from jeo.tasks import task_builder
import numpy as np


class SegmentationTask(task_builder.TaskBase):
  """Semantic segmentation task."""

  def __init__(
      self,
      loss_name="generalized_softmax_xent",
      exclude_background_class=False,
      labels_key="segmentation_mask",
      weights_key=None,
      mask_key=None,
      model_aux_metrics=(),
      **kwargs,
  ):
    super().__init__(loss_name, **kwargs)
    self.exclude_background_class = exclude_background_class
    self.labels_key = labels_key
    self.weights_key = weights_key
    self.mask_key = mask_key if mask_key is not None else f"{labels_key}_mask"
    self.model_aux_metrics = model_aux_metrics

  def get_loss_and_aux(self, model_outputs, *batches, train=False):
    """Returns loss and aux dict."""
    batch = batches[0]
    if not isinstance(model_outputs, (list, tuple)):
      model_outputs = (model_outputs, {})
    logits, model_aux = model_outputs  # logits shape (B,H,W,N_CLASSES)
    labels = batch[self.labels_key]  # (B,H,W,1) or (B,H,W)
    if labels.shape[-1] == 1:
      labels = labels[..., -1]  # (B,H,W)
    weights = _get_weights(logits, labels, self.exclude_background_class,
                           batch=batch, weights_key=self.weights_key)

    # Exclude non-valid pixels.
    if self.mask_key in batch.keys():
      valid_mask = batch[self.mask_key]
      if valid_mask.shape[-1] == 1:
        valid_mask = jnp.squeeze(valid_mask, -1)
      weights *= valid_mask

    loss = self.loss_fn(logits=logits, labels=labels, reduction=train,
                        weights=weights)
    aux = {}
    if isinstance(loss, tuple):
      assert len(loss) == 2 and isinstance(loss[1], dict)
      loss, aux = loss[0], aux | loss[1]
    for k in self.model_aux_metrics:
      aux[k] = model_aux.get(k, 0.0)

    return loss, aux


class RegressionSegmentationTask(task_builder.TaskBase):
  """Regression segmentation task."""

  def __init__(self, loss_name="l2_loss", labels_key="labels",
               mask_key="valid_label_mask",
               **kwargs):
    super().__init__(loss_name, **kwargs)
    self.labels_key = labels_key
    self.mask_key = mask_key

  def get_loss_and_aux(self, model_outputs, *batches, train=False):
    """Returns loss and aux dict."""
    batch = batches[0]
    if not isinstance(model_outputs, (list, tuple)):
      model_outputs = (model_outputs,)
    logits = model_outputs[0]  # (B,H,W,1) or (B,H,W)
    if logits.shape[-1] == 1:
      logits = jnp.squeeze(logits, -1)  # (B,H,W)
    labels = batch[self.labels_key]  # (B,H,W,1) or (B,H,W); float.
    if labels.shape[-1] == 1:
      labels = jnp.squeeze(labels, -1)  # (B,H,W)
    weights = None
    if self.mask_key in batch.keys():
      weights = batch[self.mask_key]
      if weights.shape[-1] == 1:
        weights = jnp.squeeze(weights, -1)

    loss = self.loss_fn(logits=logits, labels=labels, reduction=train,
                        weights=weights)
    aux = {}
    if isinstance(loss, tuple):
      assert len(loss) == 2 and isinstance(loss[1], dict)
      loss, aux = loss
    return loss, aux


class PanopticSegmentationTask(task_builder.TaskBase):
  """Panoptic segmentation task."""

  def __init__(self, loss_name="generalized_softmax_xent",
               exclude_background_class=False, min_fraction=0.0, **kwargs):
    super().__init__(loss_name, **kwargs)
    self.exclude_background_class = exclude_background_class
    self.min_fraction = min_fraction

  def get_loss_and_aux(self, model_outputs, *batches, train=False):
    """Returns loss and aux dict."""
    batch = batches[0]
    if not isinstance(model_outputs, (list, tuple)):
      model_outputs = (model_outputs,)
    logits = self._panoptic_predictions_from_logits(model_outputs[0])
    labels_dict = dict(semantics=batch["segmentation_mask"],  # (B,H,W,1)
                       instances=batch["instances"])  # (B,H,W,1)
    aux = {}
    for key, labels in labels_dict.items():
      if labels.shape[-1] == 1:
        labels = labels[..., 0]  # (B,H,W)
      weights = _get_weights(logits[key], labels, self.exclude_background_class)
      # NOTE: The reduction is applied later on all aux variables when training.
      aux[f"loss_{key}"] = self.loss_fn(
          logits=logits[key], labels=labels, weights=weights, reduction=False)
    if train:
      aux = jax.tree.map(jnp.mean, aux)
    loss = aux["loss_semantics"] + aux["loss_instances"]
    return loss, aux

  def _panoptic_predictions_from_logits(self, logits):
    """Make panoptic prediction from logits."""
    # Based on third_party/py/big_vision/trainers/proj/uvim/panoptic_task.py
    semantics, instances = logits["sematics"], logits["instances"]
    ins = jnp.argmax(instances, axis=-1)
    # Note: Make sure each instance has all pixels annotated with same label.
    # Otherwise they are further split into more instances and greatly affect
    # the number of unmatched predicted segments (FP) and RQ.
    masks = jax.nn.one_hot(ins, instances.shape[-1], dtype=jnp.int32)
    label = jnp.argmax(jnp.einsum("bhwk,bhwn->bnk", semantics, masks), axis=-1)
    sem = jnp.einsum("bhwn,bn->bhw", masks, label)
    # Filter out small objects
    fraction = (jnp.sum(masks, axis=(1, 2), keepdims=True)
                / np.prod(ins.shape[1:3]))
    mask_big = (fraction > self.min_fraction).astype("int32")
    mask_big_spatial = jnp.sum(masks * mask_big, axis=-1, keepdims=False) > 0
    sem = sem * mask_big_spatial.astype("int32")
    ins = ins * mask_big_spatial.astype("int32")
    return {"semantics": sem, "instances": ins}


def _get_weights(logits, labels, exclude_background: bool, batch=None,
                 weights_key=None):
  """Returns weights for the loss."""
  if weights_key:
    if weights_key not in batch.keys():
      raise ValueError(f"Weights key {weights_key} not in batch.")
    weights = batch[weights_key]
  else:
    weights = jnp.ones(logits.shape[:-1], "float32")

  if exclude_background:
    if labels.shape == logits.shape:
      labels = labels.argmax(-1)
    weights *= (labels > 0).astype("float32")
  return weights
