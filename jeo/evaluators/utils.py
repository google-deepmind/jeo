# Copyright 2026 DeepMind Technologies Limited.
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

"""Utils used by evaluators."""

from collections.abc import Iterator, Sequence

import numpy as np


ALL_PER_CLASS_METRICS = frozenset(
    ("f1", "prec", "recall", "iou", "miou", "acc", "fraction", "auprc")
)
DEFAULT_PER_CLASS_METRICS = ("f1", "prec", "recall")
DEFAULT_PER_CLASS_METRICS_MULTILABEL = ("f1", "prec", "recall", "auprc")


def get_per_class_metrics(
    cm,
    label_map: Sequence[str] | None,
    metrics: str | Sequence[str] | bool = True,
    exclude_unknown: bool = False,
) -> Iterator[tuple[str, float]]:
  """Yields per class metrics."""
  metrics = _get_metrics(metrics, DEFAULT_PER_CLASS_METRICS)

  if not label_map:
    label_map = [f"cls{i}" for i in range(cm.shape[0])]

  true = cm.sum(axis=0)
  pred = cm.sum(axis=1)

  tp = np.diag(cm)
  fn = true - tp
  fp = pred - tp
  tn = np.sum(cm) - tp - fn - fp

  metric_values = {
      "f1": np.nan_to_num(2 * tp / (2 * tp + fp + fn)),  # 2*tp / (true+pred)
      "prec": np.nan_to_num(tp / (tp + fp)),  # tp / pred
      "recall": np.nan_to_num(tp / (tp + fn)),  # tp / true
      "iou": np.nan_to_num(tp / (tp + fp + fn)),  # aka Jaccard index.
      "acc": np.nan_to_num((tp + tn) / np.sum(cm)),
      "fraction": true / np.sum(cm),
  }
  yield from _yield_per_class_metrics(
      metrics, label_map, exclude_unknown, metric_values
  )


def get_per_class_multilabel_metrics(
    cm_components,
    label_map: Sequence[str] | None,
    metrics: str | Sequence[str] | bool,
    threshold: float,
    computed_thresholds: np.ndarray | None = None,
    exclude_unknown=False,
):
  """Yields per class metrics for multilabel case."""
  metrics = _get_metrics(metrics, DEFAULT_PER_CLASS_METRICS_MULTILABEL)
  tp, tn, fp, fn = cm_components

  threshold_idx = np.argmin(computed_thresholds > threshold)
  tp_thresh = tp[threshold_idx]
  fp_thresh = fp[threshold_idx]
  fn_thresh = fn[threshold_idx]
  tn_thresh = tn[threshold_idx]

  if not label_map:
    label_map = [f"cls{i}" for i in range(tp_thresh.shape[0])]

  num_thresholds = len(tp)
  dtp = tp[: num_thresholds - 1] - tp[1:]
  p = tp + fp
  dp = p[: num_thresholds - 1] - p[1:]
  prec_slope = np.nan_to_num(dtp / np.maximum(dp, 0))
  intercept = tp[1:] - (prec_slope * p[1:])
  safe_p_ratio = np.where(
      np.logical_and(p[: num_thresholds - 1] > 0, p[1:] > 0),
      np.nan_to_num(p[: num_thresholds - 1] / np.maximum(p[1:], 0)),
      np.ones_like(p[1:]),
  )
  pr_auc_increment = np.nan_to_num(
      (prec_slope * (dtp + intercept * np.log(safe_p_ratio)))
      / np.maximum(tp[1:] + fn[1:], 0)
  )
  auprc = pr_auc_increment.sum(axis=0)

  total_sum = tp_thresh + tn_thresh + fp_thresh + fn_thresh
  metric_values = {
      "f1": np.nan_to_num(
          2 * tp_thresh / (2 * tp_thresh + fp_thresh + fn_thresh)
      ),
      "prec": np.nan_to_num(tp_thresh / (tp_thresh + fp_thresh)),
      "recall": np.nan_to_num(tp_thresh / (tp_thresh + fn_thresh)),
      "iou": np.nan_to_num(tp_thresh / (tp_thresh + fp_thresh + fn_thresh)),
      "acc": np.nan_to_num((tp_thresh + tn_thresh) / total_sum),
      "fraction": (tp_thresh + fn_thresh) / total_sum,
      "auprc": auprc,
  }
  yield from _yield_per_class_metrics(
      metrics, label_map, exclude_unknown, metric_values
  )


def nandiv(x, y):
  """Return nan if denominator is zero or nan (without warnings)."""
  with np.errstate(invalid="ignore"):
    return x / y


def get_stratified_metrics(
    cm, strata_weights, class_reductions=None, exclude_background_class=False
):
  """Yields stratified metrics.

  Args:
    cm: Confusion matrix of shape (num_strata, num_classes, num_classes).
    strata_weights: Weights for each strata.
    class_reductions: Optional mapping of classes to reduce the number of
      classes. If provided, the confusion matrix will be reduced accordingly.
    exclude_background_class: If True, exclude the background class from the
      metrics.

  Yields:
    Tuple of (metric_name, metric_value) for Overall Accuracy (OA),
    User's Accuracy (ua), Producer's Accuracy (pa), and F1-score.
    For stratified and non-stratified (ns_) metrics.
  """
  assert len(cm.shape) == 3
  assert cm.shape[0] == len(strata_weights)
  assert cm.shape[1] == cm.shape[2]
  if class_reductions is not None:
    classes_of_interest = sorted(list(set(class_reductions.values())))
    assert classes_of_interest == list(range(len(classes_of_interest)))
    ncm = np.zeros(
        (cm.shape[0], len(classes_of_interest), len(classes_of_interest))
    )
    for strata in range(cm.shape[0]):
      for orig_pred_class in range(cm.shape[1]):
        pred_class = class_reductions[orig_pred_class]
        for orig_true_class in range(cm.shape[2]):
          true_class = class_reductions[orig_true_class]
          ncm[strata, pred_class, true_class] += cm[
              strata, orig_pred_class, orig_true_class
          ]
    cm = ncm

  overall_correct = 0
  se = 0
  for strata in range(cm.shape[0]):
    if strata_weights[strata] == 0 or np.sum(cm[strata]) == 0:
      continue
    strata_correct = np.diag(cm[strata]).sum()
    strata_total = np.sum(cm[strata])
    yh = strata_correct / strata_total
    overall_correct += yh * strata_weights[strata]

    if strata_total <= 1:
      continue  # Cannot compute variance from 1 or 0 samples.
    sample_variance = (
        ((1 - yh) ** 2) * strata_correct
        + ((0 - yh) ** 2) * (strata_total - strata_correct)
    ) / (strata_total - 1)
    se += (
        (strata_weights[strata] ** 2)
        * (1 - strata_total / strata_weights[strata])
        * sample_variance
        / strata_total
    )

  se = (se**0.5) / np.sum(strata_weights)

  nostrata_cm = cm.sum(axis=0)
  yield "stratified/oa", overall_correct / np.sum(strata_weights)
  yield "stratified/ns_oa", np.diag(nostrata_cm).sum() / np.sum(nostrata_cm)
  yield "stratified/oa_se", se

  # Area proportions of each class based on reference data.
  area_proportions = (
      cm.sum(axis=1).T
      / np.maximum(cm.sum(axis=(1, 2)), 1e-10)
      @ np.array(strata_weights)
      / sum(strata_weights)
  )
  f1s = []
  for class_of_interest in range(cm.shape[1]):
    if exclude_background_class and class_of_interest == 0:
      continue
    overall_y_u = 0
    overall_ua_den = 0
    overall_pa_den = 0
    ua_se = 0
    pa_se = 0
    area_se = 0
    r_nom = 0
    r_ua_denom = 0
    r_pa_denom = 0
    for strata in range(cm.shape[0]):
      if strata_weights[strata] == 0 or np.sum(cm[strata]) == 0:
        continue
      weight = strata_weights[strata] / np.sum(cm[strata])
      r_nom += cm[strata, class_of_interest, class_of_interest] * weight
      r_ua_denom += cm[strata, class_of_interest, :].sum() * weight
      r_pa_denom += cm[strata, :, class_of_interest].sum() * weight
    r_ua = nandiv(r_nom, r_ua_denom)
    r_pa = nandiv(r_nom, r_pa_denom)
    for strata in range(cm.shape[0]):
      if strata_weights[strata] == 0 or np.sum(cm[strata]) == 0:
        continue
      strata_correct = cm[strata, class_of_interest, class_of_interest]
      num_reference = np.sum(cm[strata, :, class_of_interest])
      strata_total = np.sum(cm[strata])
      yh_hat = strata_correct / strata_total
      ua_xh_hat = np.sum(cm[strata, class_of_interest, :]) / strata_total
      pa_xh_hat = num_reference / strata_total
      weight = strata_weights[strata] / strata_total
      strata_y_u = strata_correct * weight
      strata_ua_den = np.sum(cm[strata, class_of_interest, :]) * weight
      strata_pa_den = num_reference * weight
      overall_y_u += strata_y_u
      overall_ua_den += strata_ua_den
      overall_pa_den += strata_pa_den
      ua_s_xyh = 0
      pa_s_xyh = 0
      for pred_class in range(cm.shape[1]):
        for true_class in range(cm.shape[2]):
          y = (
              1
              if (pred_class == true_class and pred_class == class_of_interest)
              else 0
          )
          ua_x = 1 if pred_class == class_of_interest else 0
          pa_x = 1 if true_class == class_of_interest else 0
          n = cm[strata, pred_class, true_class]
          ua_s_xyh += (y - yh_hat) * (ua_x - ua_xh_hat) * n
          pa_s_xyh += (y - yh_hat) * (pa_x - pa_xh_hat) * n
      if strata_total <= 1:
        continue  # Cannot compute variance from 1 or 0 samples.
      ua_s_xyh /= strata_total - 1
      pa_s_xyh /= strata_total - 1

      sample_variance_y = (
          ((1 - yh_hat) ** 2) * strata_correct
          + ((0 - yh_hat) ** 2) * (strata_total - strata_correct)
      ) / (strata_total - 1)
      ua_sample_variance_x = (
          ((1 - ua_xh_hat) ** 2) * np.sum(cm[strata, class_of_interest, :])
          + ((0 - ua_xh_hat) ** 2)
          * (strata_total - np.sum(cm[strata, class_of_interest, :]))
      ) / (strata_total - 1)
      ua_se += (
          (strata_weights[strata] ** 2)
          * (1 - strata_total / strata_weights[strata])
          * (
              sample_variance_y
              + r_ua * r_ua * ua_sample_variance_x
              - 2 * r_ua * ua_s_xyh
          )
          / strata_total
      )

      pa_sample_variance_x = (
          ((1 - pa_xh_hat) ** 2) * num_reference
          + ((0 - pa_xh_hat) ** 2) * (strata_total - num_reference)
      ) / (strata_total - 1)
      pa_se += (
          (strata_weights[strata] ** 2)
          * (1 - strata_total / strata_weights[strata])
          * (
              sample_variance_y
              + r_pa * r_pa * pa_sample_variance_x
              - 2 * r_pa * pa_s_xyh
          )
          / strata_total
      )

      area_se += (
          (strata_weights[strata] ** 2)
          * (1 - strata_total / strata_weights[strata])
          * pa_sample_variance_x
          / strata_total
      )

    ua_se = (ua_se**0.5) / overall_ua_den
    pa_se = (pa_se**0.5) / overall_pa_den
    area_se = (area_se**0.5) / np.sum(strata_weights)
    ua = nandiv(overall_y_u, overall_ua_den)
    pa = nandiv(overall_y_u, overall_pa_den)
    f1 = 2 * ua * pa / (ua + pa)
    ns_ua = nandiv(nostrata_cm[class_of_interest, class_of_interest],
                   nostrata_cm[class_of_interest, :].sum())
    ns_pa = nandiv(nostrata_cm[class_of_interest, class_of_interest],
                   nostrata_cm[:, class_of_interest].sum())
    ns_f1 = 2 * ns_ua * ns_pa / (ns_ua + ns_pa)
    yield f"stratified/class_{class_of_interest}_ua", ua
    yield f"stratified/class_{class_of_interest}_ua_se", ua_se
    yield f"stratified/class_{class_of_interest}_pa", pa
    yield f"stratified/class_{class_of_interest}_pa_se", pa_se
    yield f"stratified/class_{class_of_interest}_f1", f1
    yield f"stratified/ns_class_{class_of_interest}_ua", ns_ua
    yield f"stratified/ns_class_{class_of_interest}_pa", ns_pa
    yield f"stratified/ns_class_{class_of_interest}_f1", ns_f1
    yield (
        f"stratified/class_{class_of_interest}_area_proportion",
        area_proportions[class_of_interest],
    )
    yield f"stratified/class_{class_of_interest}_area_proportion_se", area_se
    if not (exclude_background_class and class_of_interest == 0):
      f1s.append(f1)
  yield "stratified/f1_macro", np.mean(f1s)


def _get_metrics(
    metrics: str | Sequence[str] | bool, default_metrics: Sequence[str]
) -> Sequence[str]:
  """Returns the metrics to use."""
  if isinstance(metrics, bool):
    metrics = default_metrics
  if isinstance(metrics, str):
    metrics = (metrics,)
  if unknown_metrics := set(metrics) - ALL_PER_CLASS_METRICS:
    raise ValueError(
        f"Unsupported metrics found: {unknown_metrics}. Only"
        f" {ALL_PER_CLASS_METRICS} are supported."
    )
  return metrics


def _yield_per_class_metrics(
    metrics, label_map, exclude_unknown, metric_values
):
  """Yields per class metrics based on the provided metric values."""
  for i, class_name in enumerate(label_map):
    if exclude_unknown and (
        class_name == "unknown" or class_name.startswith("_")
    ):
      continue
    class_name = class_name.lower().replace("/", "_")
    if "f1" in metrics:
      yield f"per_class/{i:02}_{class_name}_f1", metric_values["f1"][i]
    if "iou" in metrics or "miou" in metrics:
      yield f"per_class/{i:02}_{class_name}_iou", metric_values["iou"][i]
    if "prec" in metrics:
      yield f"per_class/{i:02}_{class_name}_prec", metric_values["prec"][i]
    if "recall" in metrics:
      yield f"per_class/{i:02}_{class_name}_recall", metric_values["recall"][i]
    if "acc" in metrics:
      yield f"per_class/{i:02}_{class_name}_acc", metric_values["acc"][i]
    if "fraction" in metrics:
      yield (
          f"per_class/{i:02}_{class_name}_fraction",
          metric_values["fraction"][i],
      )
    if "auprc" in metrics:
      yield f"per_class/{i:02}_{class_name}_auprc", metric_values["auprc"][i]
