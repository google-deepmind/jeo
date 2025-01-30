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

"""Task builder."""
import abc
from collections.abc import Callable, Sequence
import functools
from typing import Any

from absl import logging
import flax.linen as nn
import jax.numpy as jnp
from jeo import train_utils
from jeo.losses import losses


ArrayDict = dict[str, jnp.ndarray]
Mapping = dict[str, Any]
FloatOrArr = float | jnp.ndarray
PredictFn = Callable[[Mapping, Mapping], tuple[Any] | Mapping]
ArrayTupleOrDict = tuple[ArrayDict | jnp.ndarray, ...]


class TaskBase(abc.ABC):
  """Task interface."""

  def __init__(
      self,
      loss_name: str = "sigmoid_xent",
      loss_kw: dict[str, Any] | None = None,
      modalities: Sequence[str] | str = ("image",),
      input_as_dict: bool = False,
  ):
    self.loss_fn = losses.get_loss_fn(loss_name)
    if loss_kw is not None:
      self.loss_fn = functools.partial(self.loss_fn, **loss_kw)
    if isinstance(modalities, str):
      modalities = (modalities,)
    self.modalities = modalities
    self.input_as_dict = input_as_dict

  def model_inputs(self, *batches: Mapping) -> ArrayTupleOrDict:
    """Processes the model inputs.

    By default, only a single batch is expected. But in some cases, more than
    one batch might be provided, eg. for semi-supervised training.
    The base model outputs a tuple, but inheriting classes can also output
    a dictionary with a separate key for each mode.

    Args:
      *batches: a batch (or potentially multiple batches) of data.

    Returns:
      A tuple with the processed inputs to be fed to the model. The output tuple
      consists of one or more arrays or a single element which is a dictionary
      if `input_as_dict` is True.
    """
    batch = batches[0]
    if self.input_as_dict:
      return ({k: v for k, v in batch.items() if k in self.modalities},)
    return tuple(batch[m] for m in self.modalities)

  def get_predict_fn(
      self, model: nn.Module, rngs=None, train=False
  ) -> PredictFn:
    """Returns a function that extracts inference outputs from the given model.

    Args:
      model: The model to run inference on.
      rngs: JAX random keys.
      train: whether or not the call is for training. Defaults to False.

    Returns:
      A function that loads the model weights and extracts the outputs to be
      used for evaluations.
    """

    def predict_fn(params, **batch):
      model_inputs = self.model_inputs(batch)
      model_outputs = model.apply(params, *model_inputs, train=train, rngs=rngs)
      # model_outputs is expected to be a tuple
      if not isinstance(model_outputs, (list, tuple)):
        # In single task case, the first element should be the logits.
        if not isinstance(model_outputs, dict):
          model_outputs = (model_outputs, {})
        # In the multitask case, the model can output a dictionary of format
        # {"task_name": (output, aux)}. In this case, this dict is stored in
        # second element of the tuple.
        else:
          model_outputs = (None, model_outputs)
      return model_outputs

    return predict_fn

  @abc.abstractmethod
  def get_loss_and_aux(
      self, model_outputs, *batches: Mapping, train: bool = False
  ) -> tuple[FloatOrArr, dict[str, FloatOrArr]]:
    ...


COMMON_TASKS = {
    "classification": "classification.ClassificationTask",
    "classification_multihead": "classification.MultiHeadClassificationTask",
    "classification_multilabel": "classification.ClassificationTask",
    "masked_autoencoder": "mae.MaeTask",
    "mfp_masked_autoencoder": "mae.MfpMaeTask",
    "classification_and_reconstruction": "mae.ClassificationAndReconstructionTask",  # pylint: disable=line-too-long
    "multitask_classification": "classification.MultitaskClassificationTask",
    "panoptic_segmentation": "segmentation.PanopticSegmentationTask",
    "regression": "regression.RegressionTask",
    "regression_segmentation": "segmentation.RegressionSegmentationTask",
    "segmentation": "segmentation.SegmentationTask",
    "simclr": "contrastive.SimClrTask",
    "ssl_classification": "ssl_tasks.SslClassificationTask",
    "sup_res": "supres.SupResTask",
    "uncertainty_regression": "uncertainty_tasks.UncertaintyRegressionTask",
}


def from_config(config):
  """Creates a list of evaluators based on `config`."""
  task_type = config.get("task_type", "classification")
  task_type = COMMON_TASKS.get(task_type, task_type)
  task_cls = train_utils.import_module(task_type, "tasks")
  if not callable(task_cls):
    task_cls = getattr(task_cls, "Task")

  args = {"loss_name": config.loss} if "loss" in config else {}
  args.update(loss_kw=config.get("loss_kw", {}), **config.get("task_kw", {}))

  logging.info("Seting up task `%s` in class `%s` with args: %s",
               task_type, task_cls, args)
  return task_cls(**args)
