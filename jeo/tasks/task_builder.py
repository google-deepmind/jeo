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

"""Task builder."""
from absl import logging
from jeo import train_utils
from jeo.tools import restore


COMMON_TASKS = {
    "classification": "common_tasks.ClassificationTask",
    "classification_multihead": "common_tasks.MultiHeadClassificationTask",
    "classification_multilabel": "common_tasks.ClassificationTask",
    "masked_autoencoder": "mae.MaeTask",
    "mfp_masked_autoencoder": "mae.MfpMaeTask",
    "classification_and_reconstruction": "mae.ClassificationAndReconstructionTask",  # pylint: disable=line-too-long
    "multitask_classification": "common_tasks.MultitaskClassificationTask",
    "panoptic_segmentation": "segmentation.PanopticSegmentationTask",
    "regression": "common_tasks.RegressionTask",
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


def from_xid(xid, wid=1):
  cfg = restore.get_config(xid, wid)
  return from_config(cfg)
