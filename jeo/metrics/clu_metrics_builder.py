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

"""Metrics (currently heavily based on Hubways CLU metrics)."""
from collections.abc import Callable, Sequence
import functools
import inspect

from absl import logging
from clu import metrics as clu_metrics_builder
import clu.values
import flax
import jax.numpy as jnp
from jeo.metrics import clu_metrics
import numpy as np


DataDict = dict[str, np.ndarray | jnp.ndarray]
SingleFn = Callable[
    [jnp.ndarray, DataDict, DataDict, DataDict], "ExpandedCollection"
]
GatherFn = Callable[
    [jnp.ndarray, DataDict, DataDict, DataDict, str], "ExpandedCollection"
]


@flax.struct.dataclass
class ExpandedCollection(clu_metrics_builder.Collection):
  """An expanded CLU metrics collection that supports nested Collections."""

  def _check_reduction_counter_ndim(
      self, reduction_counter: clu_metrics_builder._ReductionCounter
  ):
    ndim = reduction_counter.value.ndim
    if ndim != 0:
      raise ValueError(
          f"Collection is still replicated (ndim={ndim}). Maybe you forgot to "
          "call a flax.jax_utils.unreplicate() or a Collections.reduce()?"
      )

  @classmethod
  def from_model_output(cls, **kwargs):
    """We need this to be able to get the collection to imitate a metric."""
    return cls._from_model_output(**kwargs)

  def compute(self) -> dict[str, jnp.ndarray]:
    """Returns a dictionary mapping metric field name to `Metric.compute()`."""
    self._check_reduction_counter_ndim(self._reduction_counter)
    results = {}
    for metric_name, metric in vars(self).items():
      if metric_name != "_reduction_counter":
        metric_results = metric.compute()
        if isinstance(metric_results, dict):
          for k, v in metric_results.items():
            results[f"{metric_name}/{k}"] = v
        else:
          results[metric_name] = metric_results
    return results

  def compute_values(self) -> dict[str, clu.values.Value]:
    """Computes metrics and returns them as clu.values.Value."""
    self._check_reduction_counter_ndim(self._reduction_counter)
    results = {}
    for metric_name, metric in vars(self).items():
      if metric_name != "_reduction_counter":
        metric_results = metric.compute_value()
        if isinstance(metric_results, dict):
          for k, v in metric_results.items():
            results[f"{metric_name}/{k}"] = v
        else:
          results[metric_name] = metric_results
    return results


class MetricsCollection:
  """A metrics collection class that manages internal metrics states.

  This class can be subclassed and in dependence of usage, the __init__() and
  the get_gather_fn() methods can be overwritten with customized metrics
  collection class definitions.

  The interface is borrowed from tf.keras.Metric class with similar public
  methods `reset_states`, `updata_state`, and `result`.
  """

  def __init__(self, metrics_list):
    self._collection_cls = get_metrics_cls(metrics_list)
    self._state = None

  def get_gather_fn(self) -> GatherFn:
    """Returns a stateless function for gathering metric update inputs."""
    return functools.partial(get_gathered_update, self._collection_cls)

  def get_single_fn(self) -> SingleFn:
    return functools.partial(get_single_update, self._collection_cls)

  def reset_states(self):
    self._state = None

  def has_state(self):
    return self._state is not None

  def update_state(self, metrics_update: ExpandedCollection):
    if self._state is None:
      self._state = metrics_update
    else:
      self._state = self._state.merge(metrics_update)

  def result(self) -> dict[str, jnp.ndarray]:
    """Returns dict of computed metric values."""
    if self._state is None:
      raise ValueError("No metrics updates received yet.")
    return self._state.compute()


def get_metrics_cls(
    metrics_list: Sequence[str | tuple[str, str | clu_metrics_builder.Metric]],
    class_name: str = "Metrics",
) -> type[ExpandedCollection]:
  """Returns Jax-serializable metrics collection class.

  Based on CLU metrics module ((internal link)).

  Args:
    metrics_list: A list of metrics. One can provide:
      - a single string to use a registered metric.
      - a tuple (name, registered_name) to use a registered metric with a
        different name.
      - a tuple (name, metric_class) to use a custom metric class.
    class_name: A custom name for the class. It will be subclassed from
      clu.metrics.Collection.
  Returns:
    New metrics collection class type.
  """
  metrics_dict = {}
  for metric_config in metrics_list:
    if isinstance(metric_config, str):
      name = metric_config
      reg_name = metric_config
    else:
      name, metric_or_reg_name = metric_config
      if inspect.isclass(metric_or_reg_name) and (
          issubclass(metric_or_reg_name, clu_metrics_builder.Metric)
          or issubclass(metric_or_reg_name, clu_metrics_builder.Collection)
      ):
        metrics_dict[name] = metric_or_reg_name
        continue
      elif isinstance(metric_or_reg_name, str):
        reg_name = metric_or_reg_name
      else:
        raise ValueError("Invalid metric configuration.")

    reg_name = reg_name.lower()
    # Unfortunately, special characters cannot be inside of metrics names.
    name = name.replace(".", "_").replace("@", "_at_")

    # TODO: Change to match..case syntax.
    if reg_name == "loss":
      metrics_dict[name] = clu_metrics_builder.Average.from_output("loss")
    elif reg_name in ["accuracy", "acc"]:
      metrics_dict[name] = clu_metrics.Accuracy
    elif reg_name == "loss_std":
      metrics_dict[name] = clu_metrics_builder.Std.from_output("loss")
    elif reg_name.endswith("_loss"):  # Supporting additional losses.
      metrics_dict[name] = clu_metrics_builder.Average.from_output(name)
    elif reg_name.endswith("_avg"):  # Generic average-based metrics.
      metrics_dict[name] = clu_metrics_builder.Average.from_output(
          name.removesuffix("_avg"))
    elif reg_name in ["learning_rate"]:
      metrics_dict[name] = clu_metrics_builder.LastValue.from_output(name)
    elif reg_name in ["aucpr", "map"]:  # Same as mAP (Mean Average Precision).
      metrics_dict[name] = clu_metrics.AUCPR
    elif reg_name in ["roc_auc", "rocauc", "auc"]:
      metrics_dict[name] = clu_metrics.ROCAUC
    elif reg_name in ["pearson", "pearsonr", "pearson_r", "pearson_corr",
                      "pearsoncorr"]:
      metrics_dict[name] = clu_metrics.ExamplePearsonCorr
    elif reg_name == "mask_count":
      metrics_dict[name] = clu_metrics.MaskCount
    elif reg_name == "expensive_aucpr":
      metrics_dict[name] = clu_metrics.ExpensiveAUCPR
    elif reg_name.startswith("recall_at_precision_"):
      precision_point = float(reg_name[len("recall_at_precision_"):])
      metrics_dict[name] = clu_metrics.recall_at_precision_function(
          precision_point=precision_point
      )
    elif reg_name.startswith("expensive_recall_at_precision_"):
      precision_point = float(reg_name["expensive_recall_at_precision_":])
      metrics_dict[name] = clu_metrics.recall_at_precision_function_expensive(
          precision_point=precision_point
      )
    elif reg_name in ["mae", "umad", "mad"]:
      metrics_dict[name] = clu_metrics.MAE
    elif reg_name in ["r2", "r2_score", "r2score"]:
      metrics_dict[name] = clu_metrics.R2Score
    elif reg_name == "stratified_regression_confusion_matrix":
      metrics_dict[name] = clu_metrics.StratifiedRegressionConfusionMatrix
    elif reg_name.startswith("stratified_"):
      metric_name = reg_name.removeprefix("stratified_")
      metrics_dict[name] = clu_metrics.get_stratified_avg_metric(metric_name)
    elif reg_name == "mse":
      metrics_dict[name] = clu_metrics.MSE
    elif reg_name == "rmse":
      metrics_dict[name] = clu_metrics.RMSE
    elif reg_name == "bias":
      metrics_dict[name] = clu_metrics.Bias
    elif reg_name == "spearman_correlation":
      metrics_dict[name] = clu_metrics.SpearmanCorrelation
    elif reg_name == "pearson_correlation":
      metrics_dict[name] = clu_metrics.PearsonCorrelation
    elif reg_name == "matthews_corrcoef":
      metrics_dict[name] = clu_metrics.MatthewsCorrelation
    elif reg_name == "f1":
      metrics_dict[name] = clu_metrics.F1
    elif reg_name == "f1_macro":
      metrics_dict[name] = clu_metrics.F1Macro
    elif reg_name in ["precision", "prec", "prec@1"]:
      metrics_dict[name] = clu_metrics.Precision
    elif reg_name in ["precision_macro", "prec_macro"]:
      metrics_dict[name] = clu_metrics.PrecisionMacro
    elif reg_name == "recall":
      metrics_dict[name] = clu_metrics.Recall
    elif reg_name == "recall_macro":
      metrics_dict[name] = clu_metrics.RecallMacro
    elif reg_name == "miou":
      metrics_dict[name] = clu_metrics.MIoU
    elif reg_name == "confusion_matrix":  # non-scalar, excluded from XM.
      metrics_dict[name] = clu_metrics.ConfusionMatrix
    elif reg_name == "confusion_matrix_3d":
      metrics_dict[name] = clu_metrics.ConfusionMatrix3D
    elif reg_name == "strata_binary_confusion_matrix":
      # non-scalar, excluded from XM.
      metrics_dict[name] = clu_metrics.PerStrataBinaryConfusionMatrix
    elif reg_name == "strata_confusion_matrix":
      # non-scalar, excluded from XM.
      metrics_dict[name] = clu_metrics.PerStrataConfusionMatrix
    elif reg_name == "min_logvar":
      metrics_dict[name] = clu_metrics.MinLogvar
    elif reg_name == "max_logvar":
      metrics_dict[name] = clu_metrics.MaxLogvar
    elif reg_name == "ece":
      metrics_dict[name] = clu_metrics.ECE
    else:
      logging.warning("Metric `%s` not specified. Assuming Average based "
                      "metric.", name)
      metrics_dict[name] = clu_metrics_builder.Average.from_output(name)
  return create_metrics_collection_cls(class_name, metrics_dict)


def create_metrics_collection_cls(
    class_name: str,
    metrics_dict: dict[
        str, clu_metrics_builder.Metric | type[ExpandedCollection]
    ],
) -> type[ExpandedCollection]:
  """Creates new metrics collection class."""
  cls = type(
      class_name, (ExpandedCollection,), {"__annotations__": metrics_dict}
  )
  cls = flax.struct.dataclass(cls)  # To enable Jax tree serialization.
  return cls


def _get_metrics_inputs(
    *,
    loss: jnp.ndarray,
    outputs: DataDict,
    inputs: DataDict,
    opt_params: DataDict
) -> DataDict:
  """Returns single dictionary that can be used as inputs for metrics."""
  # Loss is always included.
  metrics_inputs = dict(loss=loss, **opt_params)

  # Adding additional model outputs for metrics computation.
  potential_model_outputs = [
      "logits",
      "masked_lm_loss",
      "next_sentence_loss",
      "predictions",
      "log_variances",
      "pred_uncertainty",
      "logit_samples",
  ]
  for key in potential_model_outputs:
    if key in outputs:
      metrics_inputs[key] = outputs[key]

  # Adding labels if available in input data.
  potential_label_keys = ["label_ids", "label", "labels"]
  for key in potential_label_keys:
    if key in inputs:
      assert "labels" not in metrics_inputs
      metrics_inputs["labels"] = inputs[key]

  potential_model_inputs = ["label_weights", "mask", "noisy_pixels"]
  for key in potential_model_inputs:
    if key in inputs:
      metrics_inputs[key] = inputs[key]
  return metrics_inputs


def get_single_update(
    metrics_cls: type[ExpandedCollection],
    loss: jnp.ndarray,
    outputs: DataDict,
    inputs: DataDict,
    opt_params: DataDict,
) -> ExpandedCollection:
  """Constructs metrics update for a single (unreplicated by pmap) output."""
  metrics_inputs = _get_metrics_inputs(
      loss=loss, outputs=outputs, inputs=inputs, opt_params=opt_params
  )
  return metrics_cls.single_from_model_output(**metrics_inputs)


def get_gathered_update(
    metrics_cls: type[ExpandedCollection],
    loss: jnp.ndarray,
    outputs: DataDict,
    inputs: DataDict,
    opt_params: DataDict,
    axis_name: str = "batch",
) -> ExpandedCollection:
  """Constructs metrics update to be used inside pmapped functions."""
  metrics_inputs = _get_metrics_inputs(
      loss=loss, outputs=outputs, inputs=inputs, opt_params=opt_params
  )
  return metrics_cls.gather_from_model_output(
      axis_name=axis_name, **metrics_inputs
  )
