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

"""Metrics (currently heavily based on Hubways CLU metrics)."""
import functools
from typing import Callable, Dict, Optional, Sequence, Tuple, Type, Union

from absl import logging
from clu import metrics as clu_metrics
import flax
import jax
import jax.experimental.sparse
import jax.numpy as jnp
from jeo.metrics import hw_metrics
import numpy as np
import scipy.stats
import sklearn.metrics

Collection = clu_metrics.Collection
DataDict = Dict[str, Union[np.ndarray, jnp.ndarray]]
SingleFn = Callable[[jnp.ndarray, DataDict, DataDict, DataDict],
                    Collection]
GatherFn = Callable[[jnp.ndarray, DataDict, DataDict, DataDict, str],
                    Collection]


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
    return functools.partial(hw_metrics.get_gathered_update,
                             self._collection_cls)

  def get_single_fn(self) -> SingleFn:
    return functools.partial(hw_metrics.get_single_update, self._collection_cls)

  def reset_states(self):
    self._state = None

  def has_state(self):
    return self._state is not None

  def update_state(self, metrics_update: Collection):
    if self._state is None:
      self._state = metrics_update
    else:
      self._state = self._state.merge(metrics_update)

  def result(self) -> Dict[str, jnp.ndarray]:
    """Returns dict of computed metric values."""
    if self._state is None:
      raise ValueError("No metrics updates received yet.")
    return self._state.compute()


def get_metrics_cls(metrics_list: Sequence[Union[str, Tuple[str, str]]],
                    class_name: str = "Metrics") -> Type[Collection]:
  """Returns Jax-serializable metrics collection class.

  Based on CLU metrics module (go/clu-metrics).

  Args:
    metrics_list: A list of metric names. One can provide as well a tuple to
      use not registered names (name, registered_name), so that the unregistered
      name is used for a registered metric.
    class_name: A custom name for the class. It will be subclassed from
      clu.metrics.Collection.
  Returns:
    New metrics collection class type.
  """
  metrics_dict = {}
  for name in metrics_list:
    if isinstance(name, str):
      name = (name, name)
    name, reg_name = name
    reg_name = reg_name.lower()
    # Unfortunately, special characters cannot be inside of metrics names.
    name = name.replace(".", "_").replace("@", "_at_")

    if reg_name == "loss":
      metrics_dict[name] = clu_metrics.Average.from_output("loss")
    elif reg_name in ["accuracy", "acc"]:
      metrics_dict[name] = Accuracy
    elif reg_name == "loss_std":
      metrics_dict[name] = clu_metrics.Std.from_output("loss")
    elif reg_name.endswith("_loss"):  # Supporting additional losses.
      metrics_dict[name] = clu_metrics.Average.from_output(name)
    elif reg_name.endswith("_avg"):  # Generic average-based metrics.
      metrics_dict[name] = clu_metrics.Average.from_output(name)
    elif reg_name in ["learning_rate"]:
      metrics_dict[name] = clu_metrics.LastValue.from_output(name)
    elif reg_name in ["aucpr", "map"]:  # Same as mAP (Mean Average Precision).
      metrics_dict[name] = hw_metrics.AUCPR
    elif reg_name in ["roc_auc", "rocauc", "auc"]:
      metrics_dict[name] = ROCAUC
    elif reg_name in ["pearson", "pearsonr", "pearson_r", "pearson_corr",
                      "pearsoncorr"]:
      metrics_dict[name] = ExamplePearsonCorr
    elif reg_name == "mask_count":
      metrics_dict[name] = hw_metrics.MaskCount
    elif reg_name == "expensive_aucpr":
      metrics_dict[name] = hw_metrics.ExpensiveAUCPR
    elif reg_name.startswith("recall_at_precision_"):
      precision_point = float(reg_name[len("recall_at_precision_"):])
      metrics_dict[name] = hw_metrics.recall_at_precision_function(
          precision_point=precision_point)
    elif reg_name.startswith("expensive_recall_at_precision_"):
      precision_point = float(reg_name["expensive_recall_at_precision_":])
      metrics_dict[name] = hw_metrics.recall_at_precision_function_expensive(
          precision_point=precision_point)
    elif reg_name == "umad" or reg_name == "mad":
      metrics_dict[name] = hw_metrics.UMAD
    elif reg_name == "mse":
      metrics_dict[name] = MSE
    elif reg_name == "rmse":
      metrics_dict[name] = RMSE
    elif reg_name == "bias":
      metrics_dict[name] = Bias
    elif reg_name == "spearman_correlation":
      metrics_dict[name] = hw_metrics.SpearmanCorrelation
    elif reg_name == "pearson_correlation":
      metrics_dict[name] = hw_metrics.PearsonCorrelation
    elif reg_name == "matthews_corrcoef":
      metrics_dict[name] = hw_metrics.MatthewsCorrelation
    elif reg_name == "f1":
      metrics_dict[name] = hw_metrics.F1
    elif reg_name == "f1_macro":
      metrics_dict[name] = F1Macro
    elif reg_name in ["precision", "prec", "prec@1"]:
      metrics_dict[name] = Precision
    elif reg_name in ["precision_macro", "prec_macro"]:
      metrics_dict[name] = PrecisionMacro
    elif reg_name == "recall":
      metrics_dict[name] = Recall
    elif reg_name == "recall_macro":
      metrics_dict[name] = RecallMacro
    elif reg_name == "miou":
      metrics_dict[name] = MIoU
    elif reg_name == "confusion_matrix":  # non-scalar, excluded from XM.
      metrics_dict[name] = ConfusionMatrix
    elif reg_name == "min_logvar":
      metrics_dict[name] = MinLogvar
    elif reg_name == "max_logvar":
      metrics_dict[name] = MaxLogvar
    elif reg_name == "ece":
      metrics_dict[name] = ECE
    else:
      logging.warning("Metric `%s` not specified. Assuming Average based "
                      "metric.", name)
      metrics_dict[name] = clu_metrics.Average.from_output(name)
      # raise ValueError(f"Not supported metric type: {reg_name}")

  return hw_metrics.create_metrics_collection_cls(class_name, metrics_dict)


@flax.struct.dataclass
class ConfusionMatrix(clu_metrics.Metric):
  """Computes confusion matrix for multiclass classification/segmentation tasks.

  Allows to account for per-pixel weights and masking.
  The intention is to use this class to derive additional metrics.
  Logits/predictions shape for classification task is (B,N), and for
  segmentation is (B,H,W,N).

  If exclude_background_class is True, then downstream metrics can select to
  ignore the firest background class in metric computation. However, when for
  a valid class the background class is predicted, it should still count as
  error (false negative).

  Metrics computation with correct axes:
  ```
    true = cm.sum(axis=0)
    pred = cm.sum(axis=1)
    n_total = cm.sum()
    tp = np.diag(cm)
    fp = np.sum(cm, axis=1) - tp
    fn = np.sum(cm, axis=0) - tp
    iou_per_class = np.nan_to_num(tp / (tp + fp + fn))
    prec_per_class = np.nan_to_num(tp / (tp + fp))
    recall_per_class = np.nan_to_num(tp / (tp + fn))
    acc = tp.sum() / n_total  # micro
  ```
  """
  matrix: jnp.ndarray  # (N,N)
  exclude_background_class: bool = False

  @classmethod
  def from_model_output(
      cls,
      labels: jnp.ndarray,  # (B,...) or (B,...,N)
      logits: jnp.ndarray,  # (B,...,N)
      label_weights: Optional[jnp.ndarray] = None,  # (B,...)
      mask: Optional[jnp.ndarray] = None,  # (B,)
      exclude_background_class: bool = False,
      **_,
  ) -> clu_metrics.Metric:
    if labels.ndim == logits.ndim:
      labels = jnp.argmax(labels, -1)  # Reverse one-hot encoding.
    if mask is None:
      mask = jnp.ones(labels.shape[0])
    if label_weights is None:
      label_weights = jnp.ones(labels.shape)
    if label_weights.ndim != labels.ndim:
      raise ValueError(f"Unequal shapes: label_weights({label_weights.shape}), "
                       f"labels({labels.shape})")

    pred = np.argmax(logits, -1).flatten()
    true = labels.flatten()
    num_classes = logits.shape[-1]

    # If samples are masked out (due to batch padding or excluding background
    # labels), set value to `num_classes`, which will be ignored in BCOO for
    # a (num_classes, num_classes) matrix.
    masked_label_weights = (label_weights * jnp.expand_dims(mask, list(range(
        mask.ndim, label_weights.ndim)))).flatten().astype(bool)
    pred = jnp.where(masked_label_weights, pred, num_classes)
    true = jnp.where(masked_label_weights, true, num_classes)
    confusion_matrix = jax.experimental.sparse.BCOO(
        (jnp.ones(len(true), "int32"), jnp.stack((pred, true), 1)),
        shape=(num_classes, num_classes)).todense()

    return cls(matrix=confusion_matrix,
               exclude_background_class=exclude_background_class)

  def merge(self, other: "ConfusionMatrix") -> "ConfusionMatrix":
    assert self.matrix.shape == other.matrix.shape
    return type(self)(matrix=self.matrix + other.matrix,
                      exclude_background_class=jnp.logical_or(  # pytype: disable=wrong-arg-types
                          self.exclude_background_class,
                          other.exclude_background_class))

  def compute(self) -> jnp.ndarray:
    """Returns confusion matrix."""
    return self.matrix


@flax.struct.dataclass
class MIoU(ConfusionMatrix):
  """Computes Mean Intersection Over Union (mIoU) metric."""

  def compute(self) -> jnp.ndarray:
    tp = np.diag(self.matrix)
    fp = np.sum(self.matrix, axis=1) - tp
    fn = np.sum(self.matrix, axis=0) - tp
    iou_per_class = tp / (tp + fp + fn)
    if self.exclude_background_class:
      # Don't count the first excluded background class.
      iou_per_class = iou_per_class[1:]
    # If there are no tp+fp+fn (nan values), then this class should not be
    # counted. Or should it be counted as 100% correct?
    return np.nanmean(iou_per_class)


@flax.struct.dataclass
class F1Macro(ConfusionMatrix):
  """Computes macro averaged F1 score."""

  def compute(self) -> jnp.ndarray:
    tp = np.diag(self.matrix)
    fp = np.sum(self.matrix, axis=1) - tp
    fn = np.sum(self.matrix, axis=0) - tp
    f1 = 2 * tp / (2 * tp + fp + fn)
    if self.exclude_background_class:
      # Don't count the first excluded background class.
      f1 = f1[1:]
    # If tp+fp+fn is zero (no examples with this class, and no false predictions
    # for this class), then this class should be excluded from averaging.
    return np.nanmean(f1)


@flax.struct.dataclass
class PrecisionMacro(ConfusionMatrix):
  """Computes macro averaged precision score."""

  def compute(self) -> jnp.ndarray:
    tp = np.diag(self.matrix)
    fp = np.sum(self.matrix, axis=1) - tp
    # If there are no predicted positives (tp+fp), then it becomes NaN, and will
    # not be counted in the nanmean() below.
    per_class = tp / (tp + fp)
    if self.exclude_background_class:
      # Don't count the first excluded background class.
      per_class = per_class[1:]
    return np.nanmean(per_class)


@flax.struct.dataclass
class RecallMacro(ConfusionMatrix):
  """Computes macro averaged recall score."""

  def compute(self) -> jnp.ndarray:
    tp = np.diag(self.matrix)
    fn = np.sum(self.matrix, axis=0) - tp
    # If there are no positives (tp+fn=0), then it becomes NaN, and will not be
    # counted in the nanmean() below.
    per_class = tp / (tp + fn)
    if self.exclude_background_class:
      # Don't count the first excluded background class.
      per_class = per_class[1:]
    return np.nanmean(per_class)


@flax.struct.dataclass
class Precision(hw_metrics.ConfusionMatrixMultilabel):
  """Computes precision metrics at threshold=0.5 microaveraged."""

  def compute(self) -> jnp.ndarray:
    index = np.where(self.thresholds >= 0.5)[0][0]
    tp = self.true_positives.sum(axis=-1)[index]
    fp = self.false_positives.sum(axis=-1)[index]
    return hw_metrics.divide_no_nan(tp, tp + fp)


@flax.struct.dataclass
class Recall(hw_metrics.ConfusionMatrixMultilabel):
  """Computes recall metrics at threshold=0.5 microaveraged."""

  def compute(self) -> jnp.ndarray:
    index = np.where(self.thresholds >= 0.5)[0][0]
    tp = self.true_positives.sum(axis=-1)[index]
    fn = self.false_negatives.sum(axis=-1)[index]
    return hw_metrics.divide_no_nan(tp, tp + fn)


@flax.struct.dataclass
class Accuracy(clu_metrics.Average):
  """Computes the accuracy from model outputs `logits` and `labels`."""

  @classmethod
  def from_model_output(
      cls,
      *,
      logits: jnp.ndarray,
      labels: jnp.ndarray,
      label_weights: Optional[jnp.ndarray] = None,
      mask: Optional[jnp.ndarray] = None,
      **kwargs,
  ) -> clu_metrics.Metric:
    if logits.ndim == labels.ndim:
      labels = labels.argmax(axis=-1)
    # Per-pixel (label) weights are merged into mask, in order to not count
    # pixels towards the accuracy which have weight==0.
    # To have separate mask and weights inputs is needed for some corner cases
    # where some metrics expect 1d mask, while in other cases we want to mask
    # out sub-example elements.
    # Eg. computing per-example losses, and dense per-pixel accuracy.
    if label_weights is not None:  # Or per-pixel weights.
      if mask is None:
        mask = label_weights
      else:
        mask = label_weights * jnp.expand_dims(mask, list(range(
            mask.ndim, label_weights.ndim)))
    return super().from_model_output(
        values=(logits.argmax(axis=-1) == labels).astype(jnp.float32),
        mask=mask, **kwargs)


@flax.struct.dataclass
class Bias(clu_metrics.Average):
  """Computes the bias from model outputs `logits` and `labels`.

  Can be used for regression tasks. Please check that logits is the right
  comparison variable, or whether normalization or standardization is needed.

  Bias will be negative if the predicted values (logits) are lower than the
  actual values (labels), and positive otherwise.
  """

  @classmethod
  def from_model_output(
      cls,
      *,
      logits: jnp.ndarray,
      labels: jnp.ndarray,
      label_weights: Optional[jnp.ndarray] = None,
      mask: Optional[jnp.ndarray] = None,
      **kwargs,
  ) -> clu_metrics.Metric:
    assert logits.ndim == labels.ndim
    # Per-pixel (label) weights are merged into mask, in order to not count
    # pixels which have weight==0.
    # To have separate mask and weights inputs is needed for some corner cases
    # where some metrics expect 1d mask, while in other cases we want to mask
    # out sub-example elements.
    # Eg. computing per-example losses, and dense per-pixel accuracy.
    if label_weights is not None:  # Or per-pixel weights.
      if mask is None:
        mask = label_weights
      else:
        mask = label_weights * jnp.expand_dims(mask, list(range(
            mask.ndim, label_weights.ndim)))
    return super().from_model_output(
        values=(logits - labels).astype(jnp.float32), mask=mask, **kwargs)


@flax.struct.dataclass
class MSE(clu_metrics.Average):
  """Computes the MSE from model outputs `logits` and `labels`.

  Can be used for regression tasks. Please check that logits is the right
  comparison variable, or whether normalization or standardization is needed.
  """

  @classmethod
  def from_model_output(
      cls,
      *,
      logits: jnp.ndarray,
      labels: jnp.ndarray,
      label_weights: Optional[jnp.ndarray] = None,
      mask: Optional[jnp.ndarray] = None,
      **kwargs,
  ) -> clu_metrics.Metric:
    logits = jnp.squeeze(logits)
    labels = jnp.squeeze(labels)
    assert logits.ndim == labels.ndim
    if label_weights is not None:  # Or per-pixel weights.
      if mask is None:
        mask = label_weights
      else:
        mask = label_weights * jnp.expand_dims(mask, list(range(
            mask.ndim, label_weights.ndim)))
    if mask is not None:
      # Squeeze local batch size of one when then number of tpu chips is equal
      # to batch size.
      mask = jnp.squeeze(mask)
    return super().from_model_output(
        values=((logits - labels)**2), mask=mask, **kwargs)


@flax.struct.dataclass
class RMSE(MSE):
  """Computes the RMSE from model outputs `logits` and `labels`.

  Can be used for regression tasks. Please check that logits is the right
  comparison variable, or whether normalization or standardization is needed.
  """

  def compute(self):
    return jnp.sqrt(super().compute())


@flax.struct.dataclass
class ROCAUC(clu_metrics.CollectingMetric.from_outputs(("targets", "probs"))):
  """Computes Area Under the ROC Curve metric (ROC AUC)."""

  @classmethod
  def from_model_output(
      cls,
      logits: jnp.ndarray,  # (B,...,N)
      labels: jnp.ndarray,  # (B,...)
      label_weights: Optional[jnp.ndarray] = None,  # (B,...)
      mask: Optional[jnp.ndarray] = None,  # (B,)
      # Extra static configuration parameters.
      sample_proportion: float = 1.,
      false_label_ind: int = 0,
      true_label_ind: int = 1,
      **kwargs,
  ) -> "ROCAUC":
    # Setup mask (due to batch padding or excluding some labels).
    if label_weights is None:
      label_weights = jnp.ones(labels.shape)
    label_weights = label_weights * jnp.isin(
        labels, jnp.array([false_label_ind, true_label_ind]))
    if mask is None:
      mask = jnp.ones(labels.shape[0])
    mask = jnp.expand_dims(mask, list(range(mask.ndim, label_weights.ndim)))
    mask = (label_weights * mask).ravel()

    # Targets representing [false, true] labels.
    if labels.ndim == logits.ndim:  # one-hot encoded.
      targets = labels[true_label_ind].ravel()
    else:
      targets = (labels == true_label_ind).astype(jnp.int32).ravel()

    # Probabilities over the [false, true] labels.
    probs = jax.nn.softmax(logits[..., [false_label_ind, true_label_ind]])
    probs = probs[..., 1].ravel()

    # Subsampling.
    idxs = jnp.arange(len(probs))
    num_samples = int(len(probs) * sample_proportion)
    p = mask.astype(jnp.float32) / jnp.sum(mask)
    rng = jax.lax.rng_uniform(0, 0, (2,)).astype(jnp.uint32)
    # Keeping replace=True to avoid the ill-defined case when num_samples is
    # bigger then the number of non-zero probability values.
    subsampled_idxs = jax.random.choice(rng, idxs, shape=(num_samples,), p=p)

    return super().from_model_output(targets=targets[subsampled_idxs],
                                     probs=probs[subsampled_idxs])

  def compute(self):
    values = super().compute()
    y_true = values["targets"]
    y_score = values["probs"]
    return sklearn.metrics.roc_auc_score(y_true, y_score)


@flax.struct.dataclass
class ExamplePearsonCorr(clu_metrics.CollectingMetric.from_outputs((
    "proportion_true", "mean_prob"))):
  """Computes the example-wise Pearson correlation."""

  @classmethod
  def from_model_output(
      cls,
      logits: jnp.ndarray,  # (B,...,N)
      labels: jnp.ndarray,  # (B,...)
      label_weights: Optional[jnp.ndarray] = None,  # (B,...)
      mask: Optional[jnp.ndarray] = None,  # (B,)
      # Extra static configuration parameters.
      false_label_ind: int = 0,
      true_label_ind: int = 1,
      **kwargs,
  ) -> "ExamplePearsonCorr":
    # Setup mask (due to batch padding or excluding some labels).
    if label_weights is None:
      label_weights = jnp.ones(labels.shape)
    # Note that label_weights must be binary for this metric
    # TODO(moverlan): add chex assertion
    label_weights = label_weights * jnp.isin(
        labels, jnp.array([false_label_ind, true_label_ind]))
    if mask is None:
      mask = jnp.ones(labels.shape[0])
    mask = (label_weights.T * mask).T  # (B,...)

    # Targets representing [false, true] labels.
    if labels.ndim == logits.ndim:  # one-hot encoded.
      targets = labels[true_label_ind]
    else:
      targets = (labels == true_label_ind).astype(jnp.int32)
    non_batch_axes = tuple(range(1, targets.ndim))
    proportion_true = jnp.mean(targets, axis=non_batch_axes, where=mask)  # (B,)

    # Probabilities over the [false, true] labels.
    probs = jax.nn.softmax(logits[..., [false_label_ind, true_label_ind]])
    probs = probs[..., 1]
    mean_prob = jnp.mean(probs, axis=non_batch_axes, where=mask)  # (B,)

    return super().from_model_output(
        proportion_true=proportion_true, mean_prob=mean_prob)

  def compute(self):
    values = super().compute()
    x = values["proportion_true"]
    y = values["mean_prob"]
    # mask out NaNs introduced by batch padding
    valid = np.logical_not(np.logical_or(np.isnan(x), np.isnan(y)))
    x = x[valid]
    y = y[valid]
    return scipy.stats.pearsonr(x, y).statistic


@flax.struct.dataclass
class ECE(clu_metrics.Average):
  """Computes the Expected Calibration Error.

  Based on
  https://lars76.github.io/2020/08/07/metrics-for-uncertainty-estimation.html#4
  """

  @classmethod
  def from_model_output(
      cls,
      *,
      logits: jnp.ndarray,
      labels: jnp.ndarray,
      mask: Optional[jnp.ndarray] = None,
      label_weights: Optional[jnp.ndarray] = None,  # (B,...)
      num_bins: int = 15,
      **kwargs,
  ) -> clu_metrics.Metric:
    if logits.ndim == labels.ndim:
      labels = labels.argmax(axis=-1)
    if mask is None:
      mask = jnp.ones_like(labels)
    if label_weights is not None:
      mask = label_weights * jnp.expand_dims(
          mask, list(range(mask.ndim, label_weights.ndim))
      )
    prob = jax.nn.softmax(logits, axis=-1)
    pred = jnp.argmax(prob, axis=-1)
    prob_pred = jnp.max(prob, -1)
    correct = (pred == labels).astype(jnp.float32)
    bins = jnp.digitize(prob_pred, bins=jnp.linspace(0, 1, num_bins))
    ece = 0
    # Apply mask/weights.
    diff = mask * (correct - prob_pred)
    for bin_i in range(num_bins):
      ece += jnp.abs(jnp.where(bins == bin_i, diff, 0).sum())

    return super().from_model_output(values=ece/sum(labels.shape), **kwargs)


@flax.struct.dataclass
class Minimum(clu_metrics.Metric):
  """Computes the min of a given quantity."""

  metric: jnp.ndarray

  def merge(self, other: "Minimum") -> "Minimum":
    return type(self)(
        metric=jnp.min(jnp.array([self.metric, other.metric]))
    )

  def compute(self):
    return self.metric


@flax.struct.dataclass
class Maximum(clu_metrics.Metric):
  """Computes the max of a given quantity."""

  metric: jnp.ndarray

  def merge(self, other: "Maximum") -> "Maximum":
    return type(self)(
        metric=jnp.max(jnp.array([self.metric, other.metric]))
    )

  def compute(self):
    return self.metric


@flax.struct.dataclass
class MinLogvar(Minimum):
  """Computes the min from model outputs log_variances."""

  @classmethod
  def from_model_output(
      cls,
      log_variances: jnp.ndarray,
      label_weights: Optional[jnp.ndarray] = None,
      **kwargs,
  ) -> clu_metrics.Metric:
    if label_weights is None:
      valid_log_vars = log_variances
    else:
      assert label_weights.shape == log_variances.shape
      # Note that we are masking and not weighting the log_vars!
      valid_log_vars = jnp.where(label_weights > 0, log_variances, np.inf)
    return cls(metric=jnp.min(valid_log_vars))


@flax.struct.dataclass
class MaxLogvar(Maximum):
  """Computes the max from model outputs log_variances."""

  @classmethod
  def from_model_output(
      cls,
      log_variances: jnp.ndarray,
      label_weights: Optional[jnp.ndarray] = None,
      **kwargs,
  ) -> clu_metrics.Metric:
    if label_weights is None:
      valid_log_vars = log_variances
    else:
      assert label_weights.shape == log_variances.shape
      # Note that we are masking and not weighting the log_vars!
      valid_log_vars = jnp.where(label_weights > 0, log_variances, -np.inf)
    return cls(metric=jnp.max(valid_log_vars))
