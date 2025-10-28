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
import dataclasses
from typing import Any, Sequence

from clu import metrics as clu_metrics_builder
import flax
import jax
from jax import lax
import jax.experimental.sparse
import jax.numpy as jnp
import numpy as np
import scipy.stats
import sklearn.metrics


def _default_threshold() -> np.ndarray:
  return np.array(
      [0.0 - 1e-7]
      + [(i + 1) * 1.0 / (200 - 1) for i in range(200 - 2)]
      + [1.0 + 1e-7]
  )


@flax.struct.dataclass
class ConfusionMatrixMultilabel(clu_metrics_builder.Metric):
  """Computes confusion matrix for a given set of thresholds.

  Computes true_positives, true_negatives, false_positives and false_negatives
  for a given set of thresholds for multilabel classification task.

  If exclude_background_class is True, then the first background class should
  be ignored in metric computation. However, when for
  a valid class the background class is predicted, it should still count as
  error (false negative) - though, the model should have been trained for this
  not to occur.

  Used dimension annotation abbreviations:
    B: batch size
    M: number of elements (batch size or batch size * number of pixels)
    N: number of classes
    T: number of threshold values
  """

  # shape: (thresholds, num_classes)
  true_positives: jnp.ndarray
  true_negatives: jnp.ndarray
  false_positives: jnp.ndarray
  false_negatives: jnp.ndarray
  thresholds: np.ndarray = dataclasses.field(default_factory=_default_threshold)

  @classmethod
  def from_model_output(
      cls,
      labels: jnp.ndarray,
      predictions: jnp.ndarray,  # Expected to be probabilities (0-1).
      label_weights: jnp.ndarray | None = None,
      mask: jnp.ndarray | None = None,
      exclude_background_class: bool = False,
      **_,
  ) -> clu_metrics_builder.Metric:
    # Handling the non-multilabel case.
    if labels.ndim == predictions.ndim - 1:
      labels = onehot(labels, num_classes=predictions.shape[-1])
    if mask is None:
      mask = jnp.ones(labels.shape[:-1])
    if mask.ndim < labels.ndim - 1:
      mask = jnp.expand_dims(mask, list(range(mask.ndim, labels.ndim - 1)))
      mask = jnp.broadcast_to(mask, labels.shape[:-1])
    if mask.ndim != labels.ndim - 1:
      raise ValueError(f"Expected mask.ndim={mask.ndim} equal to labals.ndim-1")
    if label_weights is None:
      label_weights = jnp.ones(labels.shape)
    if label_weights.ndim == predictions.ndim - 1:
      label_weights = jnp.broadcast_to(label_weights[..., None], labels.shape)
    if labels.ndim != predictions.ndim or labels.ndim != label_weights.ndim:
      raise ValueError(
          f"Expected equal shapes for labels={labels.shape}, "
          f"predictions={predictions.shape}, and "
          f"label_weights={label_weights.shape}")
    if mask.shape != labels.shape[:-1]:
      raise ValueError(f"Shape for mask={mask.shape} != {labels.shape[:-1]}")
    if labels.ndim != 2:  # Adjust for per-pixel and temporal tasks.
      labels = labels.reshape(-1, labels.shape[-1])  # (M, N)
      predictions = predictions.reshape(-1, predictions.shape[-1])  # (M, N)
      label_weights = label_weights.reshape(-1, label_weights.shape[-1])
      mask = mask.reshape(-1)  # (M,)
    if labels.ndim != 2 or predictions.ndim != 2 or label_weights.ndim != 2:
      raise ValueError(
          f"Expected labels.ndim={labels.ndim} equal to 2; "
          f"predictions.ndim={predictions.ndim} equal to 2; "
          f"label_weights.ndim={label_weights.ndim} equal to 2"
      )
    masked_label_weigthts = label_weights * mask[..., None]  # (M, 1)
    masked_label_weigthts = masked_label_weigthts[None, ...]  # (1, M, 1)

    pred_is_pos = jnp.greater(
        predictions, _default_threshold()[..., None, None]
    )  # (T, M, N)
    pred_is_neg = jnp.logical_not(pred_is_pos)
    label_is_pos = jnp.equal(labels, 1)  # (M, N)
    label_is_neg = jnp.equal(labels, 0)  # (M, N)

    tp = pred_is_pos * label_is_pos * masked_label_weigthts  # (T, M, N)
    tn = pred_is_neg * label_is_neg * masked_label_weigthts
    fp = pred_is_pos * label_is_neg * masked_label_weigthts
    fn = pred_is_neg * label_is_pos * masked_label_weigthts

    if exclude_background_class:
      tp = tp[..., 1:]
      tn = tn[..., 1:]
      fp = fp[..., 1:]
      fn = fn[..., 1:]

    return cls(
        true_positives=tp.sum(axis=1),  # (T, N) or (T, N-1)
        true_negatives=tn.sum(axis=1),
        false_positives=fp.sum(axis=1),
        false_negatives=fn.sum(axis=1),
    )

  def merge(
      self, other: "ConfusionMatrixMultilabel"
  ) -> "ConfusionMatrixMultilabel":
    for i in [
        "true_positives",
        "true_negatives",
        "false_positives",
        "false_negatives",
    ]:
      _assert_same_shape(getattr(self, i), getattr(other, i))
    return type(self)(
        true_positives=self.true_positives + other.true_positives,
        true_negatives=self.true_negatives + other.true_negatives,
        false_positives=self.false_positives + other.false_positives,
        false_negatives=self.false_negatives + other.false_negatives,
    )

  def _get_precision_recall(self):
    tp = self.true_positives.sum(axis=-1)
    fp = self.false_positives.sum(axis=-1)
    fn = self.false_negatives.sum(axis=-1)
    precision = divide_no_nan(tp, tp + fp)
    recall = divide_no_nan(tp, tp + fn)
    return (precision, recall)

  def compute(self) -> jnp.ndarray:
    raise NotImplementedError("Must override compute()")


@flax.struct.dataclass
class AUCPR(ConfusionMatrixMultilabel):
  """Approximates the AUCPR curve.

  Simplify implementation of the approach presented here:
  (internal link)/py/keras/metrics.py?l=1825&rcl=358446889.
  """

  def compute(self) -> jnp.ndarray:
    """Interpolation formula inspired by section 4 of Davis & Goadrich 2006.

    https://www.biostat.wisc.edu/~page/rocpr.pdf
    Implementation based on tf.keras.metrics.AUC with
    summation_method='interpolation'.
    (internal link)/tensorflow/python/keras/metrics.py?l=2156-2235&rcl=353688534

    Returns:
      AUCPR value.
    """
    num_thresholds = len(self.thresholds)
    tp = self.true_positives.sum(axis=-1)
    fp = self.false_positives.sum(axis=-1)
    fn = self.false_negatives.sum(axis=-1)
    dtp = tp[: num_thresholds - 1] - tp[1:]

    p = tp + fp
    dp = p[: num_thresholds - 1] - p[1:]
    prec_slope = divide_no_nan(dtp, jnp.maximum(dp, 0))
    intercept = tp[1:] - jnp.multiply(prec_slope, p[1:])

    safe_p_ratio = jnp.where(
        jnp.logical_and(p[: num_thresholds - 1] > 0, p[1:] > 0),
        divide_no_nan(p[: num_thresholds - 1], jnp.maximum(p[1:], 0)),
        jnp.ones_like(p[1:]),
    )

    pr_auc_increment = divide_no_nan(
        prec_slope * (dtp + intercept * jnp.log(safe_p_ratio)),
        jnp.maximum(tp[1:] + fn[1:], 0),
    )
    return pr_auc_increment.sum()


def recall_at_precision_function(precision_point: float) -> Any:
  """Computes recall at a given precision point.

  Args:
    precision_point: A scalar value in range `[0, 1]`.

  Returns:
    clu_metrics_builder.Metric class representing RecallAtPrecision.
  """
  assert 0 <= precision_point <= 1

  @flax.struct.dataclass
  class RecallAtPrecision(ConfusionMatrixMultilabel):
    """Computes recall at a given precision point.

    Note that this metric can be very imprecise when num_thresholds is low.
    """

    def compute(self) -> jnp.ndarray:
      precision, recall = self._get_precision_recall()

      # Finds the first threshold where the precision is at least
      # precision_point.
      return jnp.max((precision >= precision_point) * recall)

  return RecallAtPrecision


def recall_at_precision_function_expensive(precision_point: float):
  """Computes recall at a given precision point in a more accurate way.

  Args:
    precision_point: A scalar value in range `[0, 1]`.

  Returns:
    clu_metrics_builder.Metric class representing RecallAtPrecision.
  """
  assert 0 <= precision_point <= 1

  @flax.struct.dataclass
  class ExpensiveRecallAtPrecision(ExpensiveAUCPR):
    """Computes recall at a given precision point.

    This metric computation doesn't use mask feature.
    """

    def compute(self):
      weights = np.concatenate(self.values["label_weights"])
      precision, recall, _ = sklearn.metrics.precision_recall_curve(
          np.concatenate(self.values["labels"]).flatten(),
          np.concatenate(self.values["predictions"]).flatten(),
          sample_weight=weights.flatten(),
      )

      index = np.where(precision >= precision_point)[0][0]
      recall_at_precision = recall[index]
      return recall_at_precision

  return ExpensiveRecallAtPrecision


@flax.struct.dataclass
class Precision(ConfusionMatrixMultilabel):
  """Computes precision metrics at threshold=0.5 microaveraged."""

  def compute(self) -> jnp.ndarray:
    index = np.where(self.thresholds >= 0.5)[0][0]
    tp = self.true_positives.sum(axis=-1)[index]
    fp = self.false_positives.sum(axis=-1)[index]
    return divide_no_nan(tp, tp + fp)


@flax.struct.dataclass
class Recall(ConfusionMatrixMultilabel):
  """Computes recall metrics at threshold=0.5 microaveraged."""

  def compute(self) -> jnp.ndarray:
    index = np.where(self.thresholds >= 0.5)[0][0]
    tp = self.true_positives.sum(axis=-1)[index]
    fn = self.false_negatives.sum(axis=-1)[index]
    return divide_no_nan(tp, tp + fn)


@flax.struct.dataclass
class MatthewsCorrelation(ConfusionMatrixMultilabel):
  """Computes mathews correlation."""

  def compute(self) -> jnp.ndarray:
    if self.true_positives.shape[1] > 2:
      raise NotImplementedError(
          "MatthewsCorrelation currently supports only binary classification."
      )
    index = np.where(self.thresholds >= 0.5)[0][0]
    tp = self.true_positives[index, 0]
    tn = self.true_negatives[index, 0]
    fp = self.false_positives[index, 0]
    fn = self.false_negatives[index, 0]
    return divide_no_nan(
        tp * tn - fp * fn,
        jnp.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)),
    )


@flax.struct.dataclass
class F1(ConfusionMatrixMultilabel):
  """Computes F1 at 0.5 metric microaveraged."""

  def compute(self) -> jnp.ndarray:
    index = np.where(self.thresholds >= 0.5)[0][0]
    tp = self.true_positives.sum(axis=-1)[index]
    fp = self.false_positives.sum(axis=-1)[index]
    fn = self.false_negatives.sum(axis=-1)[index]
    return divide_no_nan(2 * tp, 2 * tp + fp + fn)


@flax.struct.dataclass
class ConfusionMatrix(clu_metrics_builder.Metric):
  """Computes confusion matrix for multiclass classification/segmentation tasks.

  Allows to account for per-pixel weights and masking.
  The intention is to use this class to derive additional metrics.
  Logits/predictions shape for classification task is (B,N), and for
  segmentation is (B,H,W,N).

  If exclude_background_class is True, then downstream metrics can select to
  ignore the first background class in metric computation. However, when for
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
      label_weights: jnp.ndarray | None = None,  # (B,...)
      mask: jnp.ndarray | None = None,  # (B,)
      exclude_background_class: bool = False,
      **_,
  ) -> "ConfusionMatrix":
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
        (jnp.ones(len(true), "int32"), jnp.stack((pred, true), 1,
                                                 dtype=jnp.int32)),
        shape=(num_classes, num_classes)).todense()

    return cls(matrix=confusion_matrix,
               exclude_background_class=exclude_background_class)

  def merge(self, other: "ConfusionMatrix") -> "ConfusionMatrix":
    assert self.matrix.shape == other.matrix.shape
    # NOTE: For very large datasets the additions below could overflow, we are
    # using only jnp.int32. Unfortunately JAX does not support jnp.int64 out of
    # the box.
    return type(self)(matrix=self.matrix + other.matrix,
                      exclude_background_class=jnp.logical_or(  # pytype: disable=wrong-arg-types
                          self.exclude_background_class,
                          other.exclude_background_class))

  def compute(self) -> jnp.ndarray:
    """Returns confusion matrix."""
    return self.matrix


@flax.struct.dataclass
class ConfusionMatrix3D(ConfusionMatrix):
  """3D confusion matrix for multiclass classification/segmentation tasks.

  The first dimension after the batch dimension of the matrix is the third
  confusion matrix dimension, e.g. year.

  If exclude_background_class is True, then downstream metrics can select to
  ignore the first background class in metric computation.
  """

  @classmethod
  def from_model_output(
      cls,
      labels: jnp.ndarray,  # (B,T,...) or (B,T,...,N)
      logits: jnp.ndarray,  # (B,T,...,N)
      label_weights: jnp.ndarray | None = None,  # (B,T,...)
      mask: jnp.ndarray | None = None,  # (B,)
      exclude_background_class: bool = False,
      **_,
  ) -> clu_metrics_builder.Metric:
    if labels.ndim == logits.ndim:
      labels = jnp.argmax(labels, -1)  # Reverse one-hot encoding.
    if label_weights is None:
      label_weights = jnp.ones(labels.shape)
    if label_weights.ndim != labels.ndim:
      raise ValueError(f"Unequal shapes: label_weights({label_weights.shape}), "
                       f"labels({labels.shape})")
    if mask is None:
      mask = jnp.ones(labels.shape[0])
    b, t, *_, n = logits.shape

    # Flatten spatial dimensions if present (expand 1 dim if classification).
    labels = labels.reshape(b, t, -1)
    logits = logits.reshape(b, t, -1, n)
    label_weights = label_weights.reshape(b, t, -1) * mask.reshape(b, 1, 1)

    time_idx = jnp.arange(t, dtype=jnp.int32).reshape(1, t, 1)
    time_dim = jnp.broadcast_to(time_idx, labels.shape).flatten()

    pred = np.argmax(logits, -1).flatten()
    true = labels.flatten()
    label_weights = label_weights.flatten().astype(bool)

    pred = jnp.where(label_weights, pred, n)
    true = jnp.where(label_weights, true, n)
    time_dim = jnp.where(label_weights, time_dim, t)

    confusion_matrix = jax.experimental.sparse.BCOO((
        jnp.ones(len(true), "int32"),
        jnp.stack((time_dim, pred, true), 1, dtype=jnp.int32),
    ), shape=(t, n, n)).todense()
    return cls(matrix=confusion_matrix,
               exclude_background_class=exclude_background_class)


def isin(x: jnp.ndarray, choices: int | Sequence[int]) -> jnp.ndarray:
  """Returns elementwise True if `x`-element is in `choices`."""
  if isinstance(choices, int):
    return x == choices
  return jnp.isin(x, jnp.array(choices))


@flax.struct.dataclass
class PerStrataBinaryConfusionMatrix(ConfusionMatrix):
  """Computes confusion matrix per strata for binary classification.
  """

  @classmethod
  def from_model_output(
      cls,
      labels: jnp.ndarray,  # (B,...) or (B,...,N)
      logits: jnp.ndarray,  # (B,...,N)
      label_weights: jnp.ndarray | None = None,  # (B,...)
      mask: jnp.ndarray | None = None,  # (B,)
      exclude_background_class: bool = False,
      strata: jnp.ndarray | None = None,  # (B,)
      predictions: jnp.ndarray | None = None,  # (B,...,N)
      strata_size: int = 0,
      true_label_ind: int | Sequence[int] = 1,
      probability_threshold: float = 0.5,
      **_,
  ) -> clu_metrics_builder.Metric:
    assert strata is not None and strata_size > 0
    assert predictions is not None

    if labels.ndim == predictions.ndim:
      labels = jnp.argmax(labels, -1)  # Reverse one-hot encoding.
    if mask is None:
      mask = jnp.ones(labels.shape[0])
    if label_weights is None:
      label_weights = jnp.ones(labels.shape)
    if label_weights.ndim != labels.ndim:
      raise ValueError(f"Unequal shapes: label_weights({label_weights.shape}), "
                       f"labels({labels.shape})")

    pred = predictions[..., true_label_ind]
    if not isinstance(true_label_ind, int):
      pred = pred.sum(axis=-1)
    pred = jnp.where(pred >= probability_threshold, 1, 0).flatten()
    strata = strata.flatten()
    if strata.ndim < labels.ndim:
      strata = jnp.expand_dims(strata,
                               axis=tuple(np.arange(strata.ndim, labels.ndim)))
    strata = jnp.broadcast_to(strata, labels.shape).flatten()
    true = jnp.where(isin(labels, true_label_ind), 1, 0).flatten()
    num_classes = 2

    # If samples are masked out (due to batch padding or excluding background
    # labels), set value to `num_classes`, which will be ignored in BCOO for
    # a (num_classes, num_classes) matrix.
    masked_label_weights = (label_weights * jnp.expand_dims(mask, list(range(
        mask.ndim, label_weights.ndim)))).flatten().astype(bool)
    pred = jnp.where(masked_label_weights, pred, num_classes)
    true = jnp.where(masked_label_weights, true, num_classes)
    strata = jnp.where(masked_label_weights, strata, strata_size)
    confusion_matrix = jax.experimental.sparse.BCOO(
        (jnp.ones(len(true), "int32"), jnp.stack((strata, pred, true), 1)),
        shape=(strata_size, num_classes, num_classes)).todense()

    return cls(matrix=confusion_matrix)


@flax.struct.dataclass
class PerStrataConfusionMatrix(ConfusionMatrix):
  """Computes confusion matrix per strata for classification tasks.
  """

  @classmethod
  def from_model_output(
      cls,
      labels: jnp.ndarray,  # (B,...) or (B,...,N)
      logits: jnp.ndarray,  # (B,...,N)
      label_weights: jnp.ndarray | None = None,  # (B,...)
      mask: jnp.ndarray | None = None,  # (B,)
      exclude_background_class: bool = False,
      strata: jnp.ndarray | None = None,  # (B,)
      predictions: jnp.ndarray | None = None,  # (B,...,N)
      strata_size: int = 0,
      **_,
  ) -> clu_metrics_builder.Metric:
    assert strata is not None and strata_size > 0
    assert predictions is not None

    if labels.ndim == predictions.ndim:
      labels = jnp.argmax(labels, -1)  # Reverse one-hot encoding.
    if mask is None:
      mask = jnp.ones(labels.shape[0])
    if label_weights is None:
      label_weights = jnp.ones(labels.shape)
    if label_weights.ndim != labels.ndim:
      raise ValueError(f"Unequal shapes: label_weights({label_weights.shape}), "
                       f"labels({labels.shape})")

    pred = predictions.argmax(-1).flatten()
    strata = strata.flatten()
    if strata.ndim < labels.ndim:
      strata = jnp.expand_dims(strata,
                               axis=tuple(np.arange(strata.ndim, labels.ndim)))
    strata = jnp.broadcast_to(strata, labels.shape).flatten()
    true = labels.flatten()
    num_classes = logits.shape[-1]

    # If samples are masked out (due to batch padding or excluding background
    # labels), set value to `num_classes`, which will be ignored in BCOO for
    # a (num_classes, num_classes) matrix.
    masked_label_weights = (label_weights * jnp.expand_dims(mask, list(range(
        mask.ndim, label_weights.ndim)))).flatten().astype(bool)
    pred = jnp.where(masked_label_weights, pred, num_classes)
    true = jnp.where(masked_label_weights, true, num_classes)
    strata = jnp.where(masked_label_weights, strata, strata_size)
    confusion_matrix = jax.experimental.sparse.BCOO(
        (jnp.ones(len(true), "int32"), jnp.stack((strata, pred, true), 1,
                                                 dtype=jnp.int32)),
        shape=(strata_size, num_classes, num_classes)).todense()

    return cls(matrix=confusion_matrix)


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
class Accuracy(clu_metrics_builder.Average):
  """Computes the accuracy from model outputs `logits` and `labels`."""

  @classmethod
  def from_model_output(
      cls,
      *,
      logits: jnp.ndarray,
      labels: jnp.ndarray,
      label_weights: jnp.ndarray | None = None,
      mask: jnp.ndarray | None = None,
      **kwargs,
  ) -> clu_metrics_builder.Metric:
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
class Bias(clu_metrics_builder.Average):
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
      label_weights: jnp.ndarray | None = None,
      mask: jnp.ndarray | None = None,
      **kwargs,
  ) -> clu_metrics_builder.Metric:
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
class MSE(clu_metrics_builder.Average):
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
      label_weights: jnp.ndarray | None = None,
      mask: jnp.ndarray | None = None,
      **kwargs,
  ) -> clu_metrics_builder.Metric:
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
class MAE(clu_metrics_builder.Average):
  """Computes the mean absolute error from model outputs `logits` and `labels`.

  Can be used for regression tasks. Please check that logits is the right
  comparison variable, or whether normalization or standardization is needed.
  """

  @classmethod
  def from_model_output(
      cls,
      *,
      logits: jnp.ndarray,
      labels: jnp.ndarray,
      label_weights: jnp.ndarray | None = None,
      mask: jnp.ndarray | None = None,
      **kwargs,
  ) -> clu_metrics_builder.Metric:
    logits = jnp.squeeze(logits)
    labels = jnp.squeeze(labels)
    assert logits.ndim == labels.ndim
    if label_weights is not None:  # Or per-pixel weights.
      if mask is None:
        mask = label_weights
      else:
        mask = label_weights * jnp.expand_dims(
            mask, list(range(mask.ndim, label_weights.ndim))
        )
    if mask is not None:
      # Squeeze local batch size of one when then number of tpu chips is equal
      # to batch size.
      mask = jnp.squeeze(mask)
    return super().from_model_output(
        values=(jnp.abs(logits - labels)), mask=mask, **kwargs
    )


@flax.struct.dataclass
class R2Score(clu_metrics_builder.Metric):
  """Computes the R2 score."""

  n: jnp.ndarray
  sum_y: jnp.ndarray
  sum_yy: jnp.ndarray
  ss_res: jnp.ndarray

  @classmethod
  def from_model_output(
      cls,
      logits: jnp.ndarray,
      labels: jnp.ndarray,
      mask: jnp.ndarray | None = None,
      label_weights: jnp.ndarray | None = None,
      **kwargs,
  ) -> "R2Score":
    if mask is None:
      mask = jnp.ones(labels.shape[0])
    mask = jnp.expand_dims(
        mask, list(range(mask.ndim, labels.ndim))).astype(jnp.float32)
    if label_weights is not None:
      mask = mask * label_weights

    return cls(
        n=jnp.sum(mask),
        sum_y=jnp.sum(mask * labels),
        sum_yy=jnp.sum(mask * labels**2),
        ss_res=jnp.sum(mask * (labels - logits)**2),
        **kwargs,
    )

  def merge(self, other: "R2Score") -> "R2Score":
    return type(self)(
        n=self.n + other.n,
        sum_y=self.sum_y + other.sum_y,
        sum_yy=self.sum_yy + other.sum_yy,
        ss_res=self.ss_res + other.ss_res,
    )

  def compute(self):
    if self.n < 2:
      return jnp.array(np.nan)

    ss_tot = self.sum_yy - (self.sum_y ** 2) / self.n
    return jnp.array(1 - self.ss_res / ss_tot)


@flax.struct.dataclass
class StratifiedRegressionConfusionMatrix(ConfusionMatrix):
  """Computes confusion matrix for binned regression labels and predictions."""

  @classmethod
  def from_model_output(
      cls,
      labels: jnp.ndarray,  # (B,...)
      logits: jnp.ndarray,  # (B,...)
      label_weights: jnp.ndarray | None = None,  # (B,...)
      mask: jnp.ndarray | None = None,  # (B,)
      exclude_background_class: bool = False,
      stratification_bins: jnp.ndarray | None = None,
      **_,
  ):
    label_bins = jax.nn.one_hot(
        jnp.digitize(labels, jnp.array(stratification_bins)) - 1,
        len(stratification_bins),
    )
    logit_bins = jax.nn.one_hot(
        jnp.digitize(logits, jnp.array(stratification_bins)) - 1,
        len(stratification_bins),
    )

    return super().from_model_output(
        labels=label_bins,
        logits=logit_bins,
        label_weights=label_weights,
        mask=mask,
    )


def get_stratified_avg_metric(metric_name: str):
  """Create a stratified metric from a base metric class."""

  try:
    metric_fn, metric_cls = {
        "mae": (lambda logits, labels: jnp.abs(logits - labels), MAE),
        "mse": (lambda logits, labels: (logits - labels) ** 2, MSE),
        "rmse": (lambda logits, labels: (logits - labels) ** 2, RMSE),
        "pearson_correlation": (
            lambda logits, labels: logits,
            PearsonCorrelation,
        ),
    }[metric_name]
  except KeyError as e:
    raise ValueError(f"Unsupported stratified metric: {metric_name}") from e
  stats_names = [f.name for f in dataclasses.fields(metric_cls)]

  @flax.struct.dataclass
  class StratifiedAveragedMetric(clu_metrics_builder.Metric):
    """Computes stratified metric from model outputs `logits` and `labels`."""

    total: jnp.ndarray
    sum_y: jnp.ndarray
    sum_xx: jnp.ndarray
    sum_yy: jnp.ndarray
    sum_xy: jnp.ndarray
    count: jnp.ndarray

    @classmethod
    def from_model_output(
        cls,
        *,
        logits: jnp.ndarray,
        labels: jnp.ndarray,
        stratification_bins: dict[str, list[float]] | list[float],
        label_weights: jnp.ndarray | None = None,
        mask: jnp.ndarray | None = None,
        **kwargs,
    ) -> clu_metrics_builder.Metric:
      logits = jnp.squeeze(logits)
      labels = jnp.squeeze(labels)
      assert logits.ndim == labels.ndim
      if label_weights is not None:  # Or per-pixel weights.
        if mask is None:
          mask = label_weights
        else:
          mask = label_weights * jnp.expand_dims(
              mask, list(range(mask.ndim, label_weights.ndim))
          )
      else:
        label_weights = jnp.ones_like(labels)
      if mask is not None:
        # Squeeze local batch size of one when then number of tpu chips is equal
        # to batch size.
        mask = jnp.squeeze(mask)
      else:
        mask = jnp.ones_like(labels)

      # If stratification_bins is a single array, we assume that we are
      # stratifying by labels.
      if isinstance(stratification_bins, list):
        stratification_bins = dict(labels=stratification_bins)
      stratification_bins = {
          k: jnp.array(v) for k, v in stratification_bins.items()
      }
      kwargs["labels"] = labels

      combined_mask = (label_weights > 0) & (mask > 0)

      segment_ids = []
      bin_shapes = []
      for key, bins in stratification_bins.items():
        if key not in kwargs:
          raise ValueError(
              f"Missing stratification key {key} in kwargs. Available keys:"
              f" {list(kwargs.keys())}"
          )
        data_to_stratify = kwargs[key]
        # Make sure that the data to stratify is of the same shape as the labels
        # and logits so that we can use it to index into the bins.
        assert data_to_stratify.shape == labels.shape == logits.shape

        segment_ids_key = jnp.digitize(data_to_stratify, jnp.array(bins)) - 1
        # Exclude pixels with values outside of the stratification bins.
        bin_mask_key = (segment_ids_key >= 0) & (
            segment_ids_key < len(bins) - 1
        )
        # Also exclude pixels with values outside of the stratification bins.
        combined_mask &= bin_mask_key

        segment_ids.append(segment_ids_key)
        bin_shapes.append(len(bins) - 1)
      segment_ids = tuple(segment_ids)

      # Initialize totals and counts for each stratification key and bins.
      # For example, for key A with 3 bins and key B with 2 bins, the shapes of
      # totals and counts will be (3, 2).
      stats = dict(
          total=jnp.zeros(bin_shapes, dtype=jnp.float32),
          sum_y=jnp.zeros(bin_shapes, dtype=jnp.float32),
          sum_xx=jnp.zeros(bin_shapes, dtype=jnp.float32),
          sum_yy=jnp.zeros(bin_shapes, dtype=jnp.float32),
          sum_xy=jnp.zeros(bin_shapes, dtype=jnp.float32),
          count=jnp.zeros(bin_shapes, dtype=jnp.float32),
      )
      value = metric_fn(logits=logits, labels=labels)
      value = jnp.where(combined_mask, value, 0)
      labels = jnp.where(combined_mask, labels, 0)
      stats["total"] = stats["total"].at[segment_ids].add(value)
      stats["count"] = stats["count"].at[segment_ids].add(combined_mask)
      # Only compute the sums if the metric needs them.
      if "sum_y" in stats_names:
        stats["sum_y"] = stats["sum_y"].at[segment_ids].add(labels)
      if "sum_xx" in stats_names:
        stats["sum_xx"] = stats["sum_xx"].at[segment_ids].add(value**2)
      if "sum_yy" in stats_names:
        stats["sum_yy"] = stats["sum_yy"].at[segment_ids].add(labels**2)
      if "sum_xy" in stats_names:
        stats["sum_xy"] = stats["sum_xy"].at[segment_ids].add(value * labels)
      return cls(**stats)

    def merge(self, other):
      for stat_name in stats_names:
        _assert_same_shape(getattr(self, stat_name), getattr(other, stat_name))
      return type(self)(
          total=self.total + other.total,
          sum_y=self.sum_y + other.sum_y,
          sum_xx=self.sum_xx + other.sum_xx,
          sum_yy=self.sum_yy + other.sum_yy,
          sum_xy=self.sum_xy + other.sum_xy,
          count=self.count + other.count,
      )

    def compute(self) -> jnp.ndarray:
      names = [f.name for f in dataclasses.fields(metric_cls)]  # pytype: disable=wrong-arg-types
      stats = {name: getattr(self, name) for name in names}
      stats["count"] = jnp.where(stats["count"] == 0, 1, stats["count"])
      return metric_cls(**stats).compute()  # pytype: disable=not-callable

  return StratifiedAveragedMetric


@flax.struct.dataclass
class ECE(clu_metrics_builder.Average):
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
      mask: jnp.ndarray | None = None,
      label_weights: jnp.ndarray | None = None,  # (B,...)
      num_bins: int = 15,
      **kwargs,
  ) -> clu_metrics_builder.Metric:
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
class ROCAUC(
    clu_metrics_builder.CollectingMetric.from_outputs(("targets", "probs"))
):
  """Computes Area Under the ROC Curve metric (ROC AUC)."""

  @classmethod
  def from_model_output(
      cls,
      logits: jnp.ndarray,  # (B,...,N)
      labels: jnp.ndarray,  # (B,...)
      label_weights: jnp.ndarray | None = None,  # (B,...)
      mask: jnp.ndarray | None = None,  # (B,)
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
class ExamplePearsonCorr(
    clu_metrics_builder.CollectingMetric.from_outputs(
        ("proportion_true", "mean_prob")
    )
):
  """Computes the example-wise Pearson correlation."""

  @classmethod
  def from_model_output(
      cls,
      logits: jnp.ndarray,  # (B,...,N)
      labels: jnp.ndarray,  # (B,...)
      label_weights: jnp.ndarray | None = None,  # (B,...)
      mask: jnp.ndarray | None = None,  # (B,)
      # Extra static configuration parameters.
      false_label_ind: int = 0,
      true_label_ind: int = 1,
      **kwargs,
  ) -> "ExamplePearsonCorr":
    # Setup mask (due to batch padding or excluding some labels).
    if label_weights is None:
      label_weights = jnp.ones(labels.shape)
    # Note that label_weights must be binary for this metric
    # TODO: add chex assertion
    label_weights = label_weights * jnp.isin(
        labels, jnp.array([false_label_ind, true_label_ind]))
    if mask is None:
      mask = jnp.ones(labels.shape[0])
    mask = (label_weights.T * mask).T.astype(bool)  # (B,...)

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
class ExpensiveAUCPR(
    clu_metrics_builder.CollectingMetric.from_outputs(
        ("labels", "predictions", "label_weights")
    )
):
  """Computes AUC PR curve.

  This metric computation doesn't use mask feature.

  This implementation is using ~800MB for Webit eval.
  3 (predictions, labels, weights) * 5530 (number of classes) *
  13000 (number of test examples) * 4
  """

  @classmethod
  def from_model_output(
      cls,
      predictions: jnp.ndarray,
      labels: jnp.ndarray,
      label_weights: jnp.ndarray,
      mask: jnp.ndarray | None = None,
      **_,
  ) -> "ExpensiveAUCPR":
    if mask is None:
      mask = jnp.ones(labels.shape[0])
    return super().from_model_output(
        predictions=predictions,
        labels=labels,
        label_weights=label_weights * mask[..., None],
    )

  def compute(self):
    values = super().compute()
    weights = values["label_weights"]
    precision, recall, _ = sklearn.metrics.precision_recall_curve(
        y_true=values["labels"].flatten(),
        y_score=values["predictions"].flatten(),
        sample_weight=weights.flatten(),
    )
    return sklearn.metrics.auc(recall, precision)


@flax.struct.dataclass
class SpearmanCorrelation(
    clu_metrics_builder.CollectingMetric.from_outputs(
        ("logits", "labels", "mask")
    )
):
  """Computes spearman correlation."""

  @classmethod
  def from_model_output(
      cls,
      logits: jnp.ndarray,
      labels: jnp.ndarray,
      mask: jnp.ndarray | None = None,
      **kwargs,
  ) -> "SpearmanCorrelation":
    if mask is None:
      mask = jnp.ones(labels.shape[0])
    if "label_weights" in kwargs:
      raise NotImplementedError(
          "No support for label_weights in SpearmanCorrelation."
      )
    return super().from_model_output(
        logits=logits, labels=labels, mask=mask, **kwargs
    )

  def compute(self):
    values = super().compute()
    labels = values["labels"].flatten()[values["mask"] == 1]
    logits = values["logits"].flatten()[values["mask"] == 1]
    return scipy.stats.spearmanr(a=labels, b=logits).correlation


@flax.struct.dataclass
class PearsonCorrelation(clu_metrics_builder.Metric):
  """Computes pearson correlation incrementally."""

  total: jnp.ndarray
  sum_y: jnp.ndarray
  sum_xx: jnp.ndarray
  sum_yy: jnp.ndarray
  sum_xy: jnp.ndarray
  count: jnp.ndarray

  @classmethod
  def from_model_output(
      cls,
      logits: jnp.ndarray,
      labels: jnp.ndarray,
      mask: jnp.ndarray | None = None,
      label_weights: jnp.ndarray | None = None,
      **kwargs,
  ) -> "PearsonCorrelation":
    if mask is None:
      mask = jnp.ones(labels.shape[0])
    if label_weights is not None:
      mask = jnp.expand_dims(mask, list(range(mask.ndim, label_weights.ndim)))
      mask *= label_weights

    mask = mask > 0
    logits = jnp.where(mask, logits, 0)
    labels = jnp.where(mask, labels, 0)
    return cls(
        total=jnp.sum(logits).astype(jnp.float64),
        sum_y=jnp.sum(labels).astype(jnp.float64),
        sum_xx=jnp.sum(logits**2).astype(jnp.float64),
        sum_yy=jnp.sum(labels**2).astype(jnp.float64),
        sum_xy=jnp.sum(logits * labels).astype(jnp.float64),
        count=jnp.sum(mask).astype(jnp.int32),
    )

  def merge(self, other: "PearsonCorrelation") -> "PearsonCorrelation":
    return type(self)(
        total=self.total + other.total,
        sum_y=self.sum_y + other.sum_y,
        sum_xx=self.sum_xx + other.sum_xx,
        sum_yy=self.sum_yy + other.sum_yy,
        sum_xy=self.sum_xy + other.sum_xy,
        count=self.count + other.count,
    )

  def compute(self) -> jnp.ndarray:
    if self.count.sum() <= 2:
      return jnp.array(np.nan)
    sum_x = self.total
    numerator = self.count * self.sum_xy - sum_x * self.sum_y
    denominator_x = self.count * self.sum_xx - sum_x**2
    denominator_y = self.count * self.sum_yy - self.sum_y**2
    return numerator / (jnp.sqrt(denominator_x) * jnp.sqrt(denominator_y))


@flax.struct.dataclass
class Minimum(clu_metrics_builder.Metric):
  """Computes the min of a given quantity."""

  metric: jnp.ndarray

  def merge(self, other: "Minimum") -> "Minimum":
    return type(self)(
        metric=jnp.min(jnp.array([self.metric, other.metric]))
    )

  def compute(self):
    return self.metric


@flax.struct.dataclass
class Maximum(clu_metrics_builder.Metric):
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
      label_weights: jnp.ndarray | None = None,
      **kwargs,
  ) -> clu_metrics_builder.Metric:
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
      label_weights: jnp.ndarray | None = None,
      **kwargs,
  ) -> clu_metrics_builder.Metric:
    if label_weights is None:
      valid_log_vars = log_variances
    else:
      assert label_weights.shape == log_variances.shape
      # Note that we are masking and not weighting the log_vars!
      valid_log_vars = jnp.where(label_weights > 0, log_variances, -np.inf)
    return cls(metric=jnp.max(valid_log_vars))


@flax.struct.dataclass
class MaskCount(clu_metrics_builder.Metric):
  """Calculates how many examples are not masked out.

  Returns:
    clu_metrics_builder.Metric class representing MaskCount.
  """

  count: jnp.ndarray

  @classmethod
  def from_model_output(
      cls, logits: jnp.ndarray, mask: jnp.ndarray | None = None, **_
  ) -> clu_metrics_builder.Metric:
    if mask is None:
      return cls(count=jnp.array(logits.shape[0]))

    return cls(count=mask.sum())

  def merge(self, other: "MaskCount") -> "MaskCount":
    _assert_same_shape(self.count, other.count)
    return type(self)(count=self.count + other.count)

  def compute(self) -> Any:
    return self.count


# Copied from
# (internal link)/py/flax/training/common_utils.py;l=75-93;rcl=550029828
def onehot(labels, num_classes, on_value=1.0, off_value=0.0):
  """Create a dense one-hot version of an indexed array.

  NB: consider using the more standard `jax.nn.one_hot` instead.

  Args:
    labels: an n-dim JAX array whose last dimension contains integer indices.
    num_classes: the maximum possible index.
    on_value: the "on" value for the one-hot array, defaults to 1.0.
    off_value: the "off" value for the one-hot array, defaults to 0.0.

  Returns:
    A (n+1)-dim array whose last dimension contains one-hot vectors of length
    num_classes.
  """
  x = labels[..., None] == jnp.arange(num_classes).reshape(
      (1,) * labels.ndim + (-1,)
  )
  x = lax.select(x, jnp.full(x.shape, on_value), jnp.full(x.shape, off_value))
  return x.astype(jnp.float32)


# Copied from http://github.com/google/CommonLoopUtils/tree/HEAD/clu/metrics.py?l=58-61&rcl=350556413
def _assert_same_shape(a: jnp.ndarray, b: jnp.ndarray):
  """Raises a `ValueError` if shapes of `a` and `b` don't match."""
  if a.shape != b.shape:
    raise ValueError(f"Expected same shape: {a.shape} != {b.shape}")


def divide_no_nan(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
  """Divides x by y. If the result is nan then it's replaced by zero."""
  divided = x / y
  return jnp.where(jnp.isnan(divided), jnp.zeros_like(divided), divided)
