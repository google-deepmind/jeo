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

"""Utils used by evaluators."""
import numpy as np


DEFAULT_PER_CLASS_METRICS = ("f1", "prec", "recall")


def get_per_class_metrics(cm, label_map, metrics=DEFAULT_PER_CLASS_METRICS,
                          exclude_unknown=False):
  """Yields per class metrics."""
  if isinstance(metrics, bool):
    metrics = DEFAULT_PER_CLASS_METRICS
  assert not (set(metrics) - {"f1", "prec", "recall", "iou", "miou", "acc"})
  if not label_map:
    label_map = [f"cls{i}" for i in range(cm.shape[0])]

  true = cm.sum(axis=0)
  pred = cm.sum(axis=1)

  tp = np.diag(cm)
  fn = true - tp
  fp = pred - tp
  tn = np.sum(cm) - tp - fn - fp

  f1 = np.nan_to_num(2 * tp / (2 * tp + fp + fn))  # 2*tp / (true+pred)
  prec = np.nan_to_num(tp / (tp + fp))  # tp / pred
  recall = np.nan_to_num(tp / (tp + fn))  # tp / true
  iou_per_class = np.nan_to_num(tp / (tp + fp + fn))  # aka Jaccard index.
  acc_per_class = np.nan_to_num((tp + tn) / np.sum(cm))

  for i, class_name in enumerate(label_map):
    if exclude_unknown and (class_name == "unknown" or class_name[0] == "_"):
      continue
    class_name = class_name.lower().replace("/", "_")
    if "f1" in metrics:
      yield f"per_class/{i:02}_{class_name}_f1", f1[i]
    if "iou" in metrics or "miou" in metrics:
      yield f"per_class/{i:02}_{class_name}_iou", iou_per_class[i]
    if "prec" in metrics:
      yield f"per_class/{i:02}_{class_name}_prec", prec[i]
    if "recall" in metrics:
      yield f"per_class/{i:02}_{class_name}_recall", recall[i]
    if "acc" in metrics:
      yield f"per_class/{i:02}_{class_name}_acc", acc_per_class[i]


def get_stratified_metrics(cm, strata_weights, class_reductions=None):
  """Yields stratified metrics.

  Args:
    cm: Confusion matrix of shape (num_strata, num_classes, num_classes).
    strata_weights: Weights for each strata.
    class_reductions: Optional mapping of classes to reduce the number of
      classes. If provided, the confusion matrix will be reduced accordingly.

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
    ncm = np.zeros((cm.shape[0], len(classes_of_interest),
                    len(classes_of_interest)))
    for strata in range(cm.shape[0]):
      for orig_pred_class in range(cm.shape[1]):
        pred_class = class_reductions[orig_pred_class]
        for orig_true_class in range(cm.shape[2]):
          true_class = class_reductions[orig_true_class]
          ncm[strata, pred_class, true_class] += cm[strata, orig_pred_class,
                                                    orig_true_class]
    cm = ncm

  overall_correct = 0
  se = 0
  for strata in range(cm.shape[0]):
    strata_correct = np.diag(cm[strata]).sum()
    strata_total = np.sum(cm[strata])
    yh = strata_correct / strata_total
    overall_correct += yh * strata_weights[strata]

    sample_variance = (
        ((1 - yh) ** 2) * strata_correct +
        ((0 - yh) ** 2) * (strata_total - strata_correct)) / (strata_total - 1)
    se += ((strata_weights[strata] ** 2) *
           (1 - strata_total / strata_weights[strata]) *
           sample_variance / strata_total)

  se = (se ** 0.5) / np.sum(strata_weights)

  nostrata_cm = cm.sum(axis=0)
  yield "stratified/oa", overall_correct / np.sum(strata_weights)
  yield "stratified/ns_oa", np.diag(nostrata_cm).sum() / np.sum(nostrata_cm)
  yield "stratified/oa_se", se

  # Area proportions of each class based on reference data.
  area_proportions = (cm.sum(axis=1).T / cm.sum(axis=(1, 2)) @
                      np.array(strata_weights) / sum(strata_weights))
  for class_of_interest in range(cm.shape[1]):
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
      weight = strata_weights[strata] / np.sum(cm[strata])
      r_nom += cm[strata, class_of_interest, class_of_interest] * weight
      r_ua_denom += cm[strata, class_of_interest, :].sum() * weight
      r_pa_denom += cm[strata, :, class_of_interest].sum() * weight
    r_ua = r_nom / r_ua_denom
    r_pa = r_nom / r_pa_denom
    for strata in range(cm.shape[0]):
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
          y = 1 if (pred_class == true_class and
                    pred_class == class_of_interest) else 0
          ua_x = (1 if pred_class == class_of_interest else 0)
          pa_x = (1 if true_class == class_of_interest else 0)
          n = cm[strata, pred_class, true_class]
          ua_s_xyh += (y - yh_hat) * (ua_x - ua_xh_hat) * n
          pa_s_xyh += (y - yh_hat) * (pa_x - pa_xh_hat) * n
      ua_s_xyh /= (strata_total - 1)
      pa_s_xyh /= (strata_total - 1)

      sample_variance_y = (
          ((1 - yh_hat) ** 2) * strata_correct +
          ((0 - yh_hat) ** 2) * (strata_total - strata_correct)) / (
              strata_total - 1)
      ua_sample_variance_x = (
          ((1 - ua_xh_hat) ** 2) * np.sum(cm[strata, class_of_interest, :]) +
          ((0 - ua_xh_hat) ** 2) * (
              strata_total - np.sum(cm[strata, class_of_interest, :]))) / (
                  strata_total - 1)
      ua_se += ((strata_weights[strata] ** 2) * (
          1 - strata_total / strata_weights[strata]) * (
              sample_variance_y + r_ua * r_ua * ua_sample_variance_x -
              2 * r_ua * ua_s_xyh) / strata_total)

      pa_sample_variance_x = (
          ((1 - pa_xh_hat) ** 2) * num_reference +
          ((0 - pa_xh_hat) ** 2) * (strata_total - num_reference)
          ) / (strata_total - 1)
      pa_se += ((strata_weights[strata] ** 2) * (
          1 - strata_total / strata_weights[strata]) * (
              sample_variance_y + r_pa * r_pa * pa_sample_variance_x -
              2 * r_pa * pa_s_xyh) / strata_total)

      area_se += ((strata_weights[strata] ** 2) *
                  (1 - strata_total / strata_weights[strata]) *
                  pa_sample_variance_x / strata_total)

    ua_se = (ua_se ** 0.5) / overall_ua_den
    pa_se = (pa_se ** 0.5) / overall_pa_den
    area_se = (area_se ** 0.5) / np.sum(strata_weights)
    ua = overall_y_u / overall_ua_den
    pa = overall_y_u / overall_pa_den
    f1 = 2 * ua * pa / (ua + pa)
    ns_ua = (nostrata_cm[class_of_interest, class_of_interest] /
             nostrata_cm[class_of_interest, :].sum())
    ns_pa = (nostrata_cm[class_of_interest, class_of_interest] /
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
    yield (f"stratified/class_{class_of_interest}_area_proportion",
           area_proportions[class_of_interest])
    yield f"stratified/class_{class_of_interest}_area_proportion_se", area_se
