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

"""Tests for clu metrics."""

from absl.testing import absltest
from absl.testing import parameterized
from flax import linen as nn
import jax
import jax.numpy as jnp
from jeo.metrics import clu_metrics
import numpy as np
import scipy.stats
import sklearn.metrics
import tensorflow as tf


def concat_outputs(model_outputs_dict, name):
  return jnp.concatenate(
      [model_output[name] for model_output in model_outputs_dict]
  )


class LegacyMetricsTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(20210305)
    self.model_outputs = (
        dict(
            labels=np.random.randint(2, size=[8, 30], dtype=np.int32),
            predictions=np.random.rand(8, 30).astype(np.float32),
            label_weights=np.random.randint(2, size=[8, 30], dtype=np.int32),
        ),
        dict(
            labels=np.concatenate(
                [np.ones((7, 30)), np.zeros((1, 30))], axis=0
            ).astype(np.int32),
            predictions=np.ones((8, 30)).astype(np.float32),
            label_weights=np.random.randint(2, size=[8, 30], dtype=np.int32),
        ),
    )
    self.masks = (
        np.array([False, False, False, False, False, False, False, True]),
        np.array([True, True, False, True, True, True, True, False]),
    )
    self.model_outputs_masked = tuple(
        dict(mask=mask, **model_output)
        for mask, model_output in zip(self.masks, self.model_outputs)
    )

    self.model_outputs_regression = tuple(
        dict(
            logits=model_output["predictions"][:, 0],
            **{k: v[:, 0].astype(np.float32) for k, v in model_output.items()}
        )
        for model_output in self.model_outputs
    )
    self.model_outputs_regression_masked = tuple(
        dict(mask=mask, **model_output)
        for mask, model_output in zip(self.masks, self.model_outputs_regression)
    )

    self.model_outputs_single_classification = (
        dict(
            labels=np.random.randint(2, size=[8], dtype=np.int32),
            predictions=nn.softmax(
                np.random.rand(8, 2).astype(np.float32), axis=-1
            ),
            label_weights=np.random.rand(8).astype(np.float32),
        ),
        dict(
            labels=np.random.randint(2, size=[8], dtype=np.int32),
            predictions=nn.softmax(
                np.random.rand(8, 2).astype(np.float32), axis=-1
            ),
            label_weights=np.random.rand(8).astype(np.float32),
        ),
    )
    self.model_outputs_single_classification_masked = tuple(
        dict(mask=mask, **model_output)
        for mask, model_output in zip(
            self.masks, self.model_outputs_single_classification
        )
    )

  # Based on
  # http://github.com/google/CommonLoopUtils/tree/HEAD/clu/metrics_test.py?l=91-120&rcl=341411410
  def make_compute_metric(self, metric_class, reduce, to_jit=True):
    """Returns a jitted function to compute metrics.

    Args:
      metric_class: Metric class to instantiate.
      reduce: If set to `True`
      to_jit: Whether to compile the function.

    Returns:
      A function that takes `model_outputs` (list of dictionaries of values) as
      an input and returns the value from `metric.compute()`.
    """

    def compute_metric(model_outputs):
      if reduce:
        metric_list = [
            metric_class.from_model_output(**model_output)
            for model_output in model_outputs
        ]
        metric_stacked = jax.tree.map(
            lambda *args: jnp.stack(args), *metric_list
        )
        metric = metric_stacked.reduce()
      else:
        metric = None
        for model_output in model_outputs:
          update = metric_class.from_model_output(**model_output)
          metric = update if metric is None else metric.merge(update)
      return metric.compute()  # pytype: disable=attribute-error

    if to_jit:
      return jax.jit(compute_metric)
    else:
      return compute_metric

  def test_aucpr_expensive(self):
    aucpr = self.make_compute_metric(
        clu_metrics.ExpensiveAUCPR, reduce=False, to_jit=False
    )(self.model_outputs)
    self.assertAllClose(aucpr, 0.852988)
    aucpr_masked = self.make_compute_metric(
        clu_metrics.ExpensiveAUCPR, reduce=False, to_jit=False
    )(self.model_outputs_masked)
    self.assertNotAllClose(aucpr, aucpr_masked)

  def test_recall_at_precision_expensive(self):
    hubways_r_at_p_metric = clu_metrics.recall_at_precision_function_expensive(
        precision_point=0.7
    )
    hubways_r_at_p = self.make_compute_metric(
        hubways_r_at_p_metric, reduce=False, to_jit=False
    )(self.model_outputs)
    self.assertAllClose(hubways_r_at_p, 0.895425)

    hubways_r_at_p_masked = self.make_compute_metric(
        hubways_r_at_p_metric, reduce=False, to_jit=False
    )(self.model_outputs_masked)
    self.assertNotAllClose(hubways_r_at_p, hubways_r_at_p_masked)

  def test_spearman_correlation(self):
    model_outputs_regression = tuple(
        {k: v for k, v in model_output.items() if k != "label_weights"}
        for model_output in self.model_outputs_regression
    )
    model_outputs_regression_masked = tuple(
        {k: v for k, v in model_output.items() if k != "label_weights"}
        for model_output in self.model_outputs_regression_masked
    )

    spearman_correlation = self.make_compute_metric(
        clu_metrics.SpearmanCorrelation, reduce=False, to_jit=False
    )(model_outputs_regression)
    self.assertAllClose(spearman_correlation, 0.217393)
    spearman_correlation_masked = self.make_compute_metric(
        clu_metrics.SpearmanCorrelation, reduce=False, to_jit=False
    )(model_outputs_regression_masked)
    self.assertNotAllClose(spearman_correlation, spearman_correlation_masked)

  def test_pearson_correlation(self):
    model_outputs_regression = tuple(
        {k: v for k, v in model_output.items() if k != "label_weights"}
        for model_output in self.model_outputs_regression
    )
    model_outputs_regression_masked = tuple(
        {k: v for k, v in model_output.items() if k != "label_weights"}
        for model_output in self.model_outputs_regression_masked
    )

    pearson_correlation = self.make_compute_metric(
        clu_metrics.PearsonCorrelation, reduce=False, to_jit=False
    )(model_outputs_regression)
    self.assertAllClose(pearson_correlation, 0.183312)
    pearson_correlation_masked = self.make_compute_metric(
        clu_metrics.PearsonCorrelation, reduce=False, to_jit=False
    )(model_outputs_regression_masked)
    self.assertNotAllClose(pearson_correlation, pearson_correlation_masked)

  def test_pearson_correlation_with_mask(self):
    logits = np.arange(4**3).reshape((4, 4, 4))
    labels = 3 * logits
    label_weights = np.random.randint(2, size=(4, 4, 4))
    logits[label_weights == 0] = 0

    # Masked arrays are perfectly correlated.
    pearson_correlation = self.make_compute_metric(
        clu_metrics.PearsonCorrelation, reduce=False, to_jit=False
    )((dict(logits=logits, labels=labels, label_weights=label_weights),))
    self.assertAllClose(pearson_correlation, 1.0)

    # Non masked arrays are not perfectly correlated.
    pearson_correlation = self.make_compute_metric(
        clu_metrics.PearsonCorrelation, reduce=False, to_jit=False
    )((dict(logits=logits, labels=labels),))
    self.assertNotAllClose(pearson_correlation, 1.0)

  def test_matthews_correlation(self):
    # Labels and predictions for binary classification
    labels = clu_metrics.onehot(
        concat_outputs(self.model_outputs_single_classification, "labels"),
        num_classes=2,
    )[:, 1]
    predictions = (
        concat_outputs(self.model_outputs_single_classification, "predictions")
        >= 0.5
    )[:, 1]
    label_weights = concat_outputs(
        self.model_outputs_single_classification, "label_weights"
    )
    matthews_correlation_expected = sklearn.metrics.matthews_corrcoef(
        y_true=labels, y_pred=predictions, sample_weight=label_weights
    )
    matthews_correlation = self.make_compute_metric(
        clu_metrics.MatthewsCorrelation, reduce=False, to_jit=False
    )(self.model_outputs_single_classification)

    self.assertAllClose(matthews_correlation, matthews_correlation_expected)
    matthews_correlation_masked = self.make_compute_metric(
        clu_metrics.MatthewsCorrelation, reduce=False, to_jit=False
    )(self.model_outputs_single_classification_masked)
    self.assertNotAllClose(matthews_correlation, matthews_correlation_masked)


class MetricsTest(parameterized.TestCase):

  @parameterized.named_parameters(("jit", True), ("no-jit", False))
  def test_confusion_matrix(self, jitted):
    labels = jnp.array([0, 1, 2, 2, 0])
    probs = jnp.array([[0.9, 0.0, 0.1],  # Correct.
                       [0.8, 0.1, 0.1],  # Wrong.
                       [0.1, 0.2, 0.7],  # Correct.
                       [0.1, 0.2, 0.7],  # Correct.
                       [0.3, 0.3, 0.4]])  # Wrong.
    expected = jnp.array([[1, 1, 0], [0, 0, 0], [1, 0, 2]])

    metrics_fn = clu_metrics.ConfusionMatrix.from_model_output
    if jitted:
      metrics_fn = jax.jit(metrics_fn)

    update = metrics_fn(labels, probs)
    result = update.compute()
    self.assertEqual(result.shape, expected.shape)
    np.testing.assert_array_equal(result, expected)

    # Test that an update with same values results in doubled conf matrix.
    new_update = metrics_fn(labels, probs)
    merged = update.merge(new_update)
    merged_result = merged.compute()
    np.testing.assert_array_equal(merged_result, expected * 2)

  @parameterized.named_parameters(("jit", True), ("no-jit", False))
  def test_confusion_matrix_with_label_weights(self, jitted):
    labels = jnp.array([0, 1, 2, 0])
    probs = jnp.array([[0.9, 0.0, 0.1],  # Correct.
                       [0.8, 0.1, 0.1],  # Wrong.
                       [0.1, 0.2, 0.7],  # Correct.
                       [0.3, 0.3, 0.4]])  # Wrong.
    weights = (labels > 0).astype("float32")
    expected = jnp.array([[0, 1, 0], [0, 0, 0], [0, 0, 1]])

    metrics_fn = clu_metrics.ConfusionMatrix.from_model_output
    if jitted:
      metrics_fn = jax.jit(metrics_fn)
    update = metrics_fn(labels, probs, weights, exclude_background_class=True)
    result = update.compute()
    self.assertEqual(result.shape, expected.shape)
    np.testing.assert_array_equal(result, expected)

    # Test that an update with same values results in doubled conf matrix.
    new_update = metrics_fn(labels, probs, weights,
                            exclude_background_class=True)
    merged = update.merge(new_update)
    merged_result = merged.compute()
    np.testing.assert_array_equal(merged_result, expected * 2)

  @parameterized.named_parameters(("jit", True), ("no-jit", False))
  def test_confusion_matrix_3d(self, jitted):
    # (B, T)
    labels = jnp.array([[0, 1], [2, 2], [0, 1]])
    # (B, T, N)
    probs = jnp.array([
        [[0.9, 0.0, 0.1], [0.1, 0.8, 0.1]],  # B0: (T0: 0->0, T1: 1->1)
        [[0.1, 0.2, 0.7], [0.1, 0.2, 0.7]],  # B1: (T0: 2->2, T1: 2->2)
        [[0.3, 0.3, 0.4], [0.1, 0.9, 0.0]],  # B2: (T0: 0->2, T1: 1->1)
    ])
    # Expected: (T, N, N)
    expected = jnp.array([
        [[1, 0, 0], [0, 0, 0], [1, 0, 1]],  # T=0: (0,0), (2,2), (0,2)
        [[0, 0, 0], [0, 2, 0], [0, 0, 1]],  # T=1: (1,1), (2,2), (1,1)
    ])

    metrics_fn = clu_metrics.ConfusionMatrix3D.from_model_output
    if jitted:
      metrics_fn = jax.jit(metrics_fn)

    update = metrics_fn(labels=labels, logits=probs)
    result = update.compute()
    self.assertEqual(result.shape, expected.shape)
    np.testing.assert_array_equal(result, expected)

    # Test that an update with same values results in doubled conf matrix.
    new_update = metrics_fn(labels=labels, logits=probs)
    merged = update.merge(new_update)
    merged_result = merged.compute()
    np.testing.assert_array_equal(merged_result, expected * 2)

    # Test with mask and weights.
    mask = jnp.array([True, True, False])  # Mask out last batch element
    label_weights = jnp.ones_like(labels, dtype=jnp.float32)
    expected_masked = jnp.array([
        [[1, 0, 0], [0, 0, 0], [0, 0, 1]],  # T=0: (0,0), (2,2)
        [[0, 0, 0], [0, 1, 0], [0, 0, 1]],  # T=1: (1,1), (2,2)
    ])
    update_masked = metrics_fn(
        labels=labels, logits=probs, mask=mask, label_weights=label_weights
    )
    result_masked = update_masked.compute()
    np.testing.assert_array_equal(result_masked, expected_masked)

  @parameterized.named_parameters(("jit", True), ("no-jit", False))
  def test_miou(self, jitted):
    labels = jnp.array([0, 1, 2, 0])
    probs = jnp.array([[0.9, 0.0, 0.1],  # Correct.
                       [0.8, 0.1, 0.1],  # Wrong.
                       [0.1, 0.2, 0.7],  # Correct.
                       [0.3, 0.3, 0.4]])  # Wrong.
    weights = (labels > 0).astype("float32")
    expected = 0.5  # Second wrong, third correct, others ignored.

    metrics_fn = clu_metrics.MIoU.from_model_output
    if jitted:
      metrics_fn = jax.jit(metrics_fn)

    update = metrics_fn(labels, probs, weights, exclude_background_class=True)
    result = update.compute()
    np.testing.assert_array_equal(result, expected)

    # Test that an update with same values results in the same mIoU.
    new_update = metrics_fn(labels, probs, weights,
                            exclude_background_class=True)
    merged = update.merge(new_update)
    merged_result = merged.compute()
    np.testing.assert_array_equal(merged_result, expected)

  @parameterized.named_parameters(
      ("f1", clu_metrics.F1Macro, 0.611111),
      ("miou", clu_metrics.MIoU, 0.444444),
      ("prec", clu_metrics.PrecisionMacro, 0.66666667),
      ("recall", clu_metrics.RecallMacro, 0.66666667),
  )
  def test_macro_metrics(self, metric, expected):
    labels = jnp.array([0, 1, 2, 0, 1])
    probs = jnp.array([[0.9, 0.0, 0.1],  # Correct.
                       [0.8, 0.1, 0.1],  # Wrong.
                       [0.1, 0.2, 0.7],  # Correct.
                       [0.3, 0.3, 0.4],  # Wrong.
                       [0.3, 0.4, 0.3],  # Correct.
                       ])
    weights = np.ones_like(labels).astype("float32")

    metrics_fn = jax.jit(metric.from_model_output)
    update = metrics_fn(labels, probs, weights, exclude_background_class=False)
    result = update.compute()
    np.testing.assert_array_almost_equal(result, expected)

  @parameterized.named_parameters(
      ("f1", clu_metrics.F1Macro, 0.611111),
      ("miou", clu_metrics.MIoU, 0.444444),
      ("prec", clu_metrics.PrecisionMacro, 0.66666667),
      ("recall", clu_metrics.RecallMacro, 0.66666667),
  )
  def test_macro_metrics_no_examples(self, metric, expected):
    # Adding a class, which has no examples, and no predictions (tp+fp+fn=0).
    labels = jnp.array([0, 1, 2, 0, 1])
    probs = jnp.array([[0.9, 0.0, 0.1, 0],  # Correct.
                       [0.8, 0.1, 0.1, 0],  # Wrong.
                       [0.1, 0.2, 0.7, 0],  # Correct.
                       [0.3, 0.3, 0.4, 0],  # Wrong.
                       [0.3, 0.4, 0.3, 0],  # Correct.
                       ])
    weights = np.ones_like(labels).astype("float32")

    metrics_fn = jax.jit(metric.from_model_output)
    update = metrics_fn(labels, probs, weights, exclude_background_class=False)
    result = update.compute()
    np.testing.assert_array_almost_equal(result, expected)

  @parameterized.named_parameters(
      ("f1", clu_metrics.F1Macro, 0.5),
      ("miou", clu_metrics.MIoU, 0.375),
      ("prec", clu_metrics.PrecisionMacro, 0.625),
      ("recall", clu_metrics.RecallMacro, 0.66666667),
  )
  def test_macro_metrics_no_positives(self, metric, expected):
    # Adding a class, which has no positives (tp+fn=0).
    labels = jnp.array([0, 1, 2, 0, 1])
    probs = jnp.array([[0.9, 0.0, 0.1, 0],  # Correct.
                       [0.0, 0.1, 0.1, 0.8],  # Wrong.
                       [0.1, 0.2, 0.7, 0],  # Correct.
                       [0.3, 0.3, 0.4, 0],  # Wrong.
                       [0.3, 0.4, 0.3, 0],  # Correct.
                       ])
    weights = np.ones_like(labels).astype("float32")

    metrics_fn = jax.jit(metric.from_model_output)
    update = metrics_fn(labels, probs, weights, exclude_background_class=False)
    result = update.compute()
    np.testing.assert_array_almost_equal(result, expected)

  @parameterized.named_parameters(
      ("f1", clu_metrics.F1Macro, 0.5),
      ("miou", clu_metrics.MIoU, 0.375),
      ("prec", clu_metrics.PrecisionMacro, 0.66666667),
      ("recall", clu_metrics.RecallMacro, 0.625),
  )
  def test_macro_metrics_no_predicted_positives(self, metric, expected):
    # Adding a class, which has no predicted positives (tp+fp=0).
    labels = jnp.array([0, 1, 2, 3, 1])
    probs = jnp.array([[0.9, 0.0, 0.1, 0],  # Correct.
                       [0.8, 0.1, 0.1, 0],  # Wrong.
                       [0.1, 0.2, 0.7, 0],  # Correct.
                       [0.3, 0.3, 0.4, 0],  # Wrong.
                       [0.3, 0.4, 0.3, 0],  # Correct.
                       ])
    weights = np.ones_like(labels).astype("float32")

    metrics_fn = jax.jit(metric.from_model_output)
    update = metrics_fn(labels, probs, weights, exclude_background_class=False)
    result = update.compute()
    np.testing.assert_array_almost_equal(result, expected)

  @parameterized.named_parameters(("jit", True), ("no-jit", False))
  def test_roc_auc(self, jitted):
    labels = jnp.array([0, 1, 2, 2, 0])
    logits = jnp.array([[0.9, 0.0, 0.1],  # Correct.
                        [0.8, 0.1, 0.1],  # Wrong.
                        [0.1, 0.2, 0.7],  # Correct.
                        [0.1, 0.2, 0.7],  # Correct.
                        [0.3, 0.3, 0.4]])  # Wrong.
    # Expected auc, using manually computed softmax of filtered examples, which
    # corresponds to = softmax([[0.1, 0.1],
    #                           [0.2, 0.7],
    #                           [0.2, 0.7]])[:, 1]
    expected = clu_metrics.sklearn.metrics.roc_auc_score(
        [0, 1, 1], [0.5, 0.6226, 0.6226]
    )

    metrics_fn = clu_metrics.ROCAUC.from_model_output
    if jitted:
      metrics_fn = jax.jit(metrics_fn, static_argnums=(2, 3, 4))
    update = metrics_fn(labels=labels, logits=logits, sample_proportion=1,
                        false_label_ind=1, true_label_ind=2)
    result = update.compute()
    np.testing.assert_array_equal(result, expected)

    # Test that an update with same values results in the same result.
    new_update = metrics_fn(labels=labels, logits=logits, sample_proportion=1,
                            false_label_ind=1, true_label_ind=2)
    merged = update.merge(new_update)
    merged_result = merged.compute()
    np.testing.assert_array_equal(merged_result, expected)

  @parameterized.named_parameters(("jit", True), ("no-jit", False))
  def test_pearson_r(self, jitted):
    ignore_label, false_label_ind, true_label_ind = (0, 1, 2)
    labels = np.array([[1, 1, 1], [0, 1, 2], [2, 0, 2]])
    logits = np.array([
        [[1, 0, -1],
         [0, 2, 1],
         [-1, -2, 1]],
        [[1, 1, 2],
         [0, 2, 1],
         [-1, -1, 1]],
        [[2, 0, -1],
         [2, 2, 1],
         [0, 0, 1]],
    ])
    mask = jnp.not_equal(labels, ignore_label)
    proportion_true = jnp.mean(labels, axis=1, where=mask)
    probs = jax.nn.softmax(logits[..., [false_label_ind, true_label_ind]])
    probs = probs[..., 1]
    mean_prob = jnp.mean(probs, axis=1, where=mask)
    expected = scipy.stats.pearsonr(proportion_true, mean_prob).statistic

    metrics_fn = clu_metrics.ExamplePearsonCorr.from_model_output
    if jitted:
      metrics_fn = jax.jit(metrics_fn, static_argnames=(
          "false_label_ind", "true_label_ind"))
    update = metrics_fn(labels=labels, logits=logits, false_label_ind=1,
                        true_label_ind=2)
    actual = update.compute()

    self.assertAlmostEqual(actual, expected, places=4)

  @parameterized.named_parameters(("jit", True), ("no-jit", False))
  def test_bias(self, jitted):
    labels = jnp.array([15., 25., 35.])
    logits = jnp.array([10., 20., 30.])
    expected = jnp.array(-5.)

    metrics_fn = clu_metrics.Bias.from_model_output
    if jitted:
      metrics_fn = jax.jit(metrics_fn)

    update = metrics_fn(labels=labels, logits=logits)
    result = update.compute()
    self.assertEqual(result.shape, expected.shape)
    np.testing.assert_array_equal(result, expected)

  @parameterized.named_parameters(("jit", True), ("no-jit", False))
  def test_ece(self, jitted):
    labels = jnp.array([0, 1, 2, 0, 1])
    logits = jnp.array([
        [0.9, 0.0, 0.1, 0],  # Correct.
        [0.0, 0.1, 0.1, 0.8],  # Wrong.
        [0.1, 0.2, 0.7, 0],  # Correct.
        [0.3, 0.3, 0.4, 0],  # Wrong.
        [0.3, 0.4, 0.3, 0],  # Correct.
    ])
    mask = np.ones_like(labels).astype("float32")
    expected = jnp.array(0.24, dtype=jnp.float32)

    metrics_fn = clu_metrics.ECE.from_model_output
    if jitted:
      metrics_fn = jax.jit(metrics_fn)

    update = metrics_fn(logits=logits, labels=labels, mask=mask)
    result = update.compute()
    np.testing.assert_array_equal(round(result, 2), expected)

  @parameterized.named_parameters(("jit", True), ("no-jit", False))
  def test_minlogvar(self, jitted):
    weights = jnp.array([0.0, 0.2, 0.8])
    log_variances = jnp.array([-10., 20., 30.])
    expected = jnp.array(20.)

    metrics_fn = clu_metrics.MinLogvar.from_model_output
    if jitted:
      metrics_fn = jax.jit(metrics_fn)

    update = metrics_fn(log_variances=log_variances, label_weights=weights)
    result = update.compute()
    self.assertEqual(result.shape, expected.shape)
    np.testing.assert_array_equal(result, expected)

  @parameterized.named_parameters(("jit", True), ("no-jit", False))
  def test_maxlogvar(self, jitted):
    weights = jnp.array([[0.0, 0.2, 0.1], [0.1, 0.2, 0.4]])
    log_variances = jnp.array([[100., -120., 10], [3.6, 0.0, 10.4]])
    expected = jnp.array(10.4)

    metrics_fn = clu_metrics.MaxLogvar.from_model_output
    if jitted:
      metrics_fn = jax.jit(metrics_fn)

    update = metrics_fn(log_variances=log_variances, label_weights=weights)
    result = update.compute()
    self.assertEqual(result.shape, expected.shape)
    np.testing.assert_array_equal(result, expected)

  @parameterized.named_parameters(("jit", True, False), ("no-jit", False, True))
  def test_mse_rmse(self, jitted, mse=False):
    logits = jnp.array([15., 25., 35.])
    labels = jnp.array([10., 20., 30.])
    if mse:
      metrics_fn = clu_metrics.MSE.from_model_output
      expected = jnp.array(25.)
    else:
      metrics_fn = clu_metrics.RMSE.from_model_output
      expected = jnp.array(5.)

    if jitted:
      metrics_fn = jax.jit(metrics_fn)

    update = metrics_fn(labels=labels, logits=logits)
    result = update.compute()
    self.assertEqual(result.shape, expected.shape)
    np.testing.assert_array_equal(result, expected)

  @parameterized.named_parameters(("jit", True), ("no-jit", False))
  def test_mae(self, jitted):
    logits = jnp.array([15.0, 25.0, 35.0])
    labels = jnp.array([10.0, 20.0, 30.0])
    metrics_fn = clu_metrics.MAE.from_model_output
    expected = jnp.array(5.0)

    if jitted:
      metrics_fn = jax.jit(metrics_fn)

    update = metrics_fn(labels=labels, logits=logits)
    result = update.compute()
    self.assertEqual(result.shape, expected.shape)
    np.testing.assert_array_equal(result, expected)

  @parameterized.named_parameters(("jit", True), ("no-jit", False))
  def test_r2score(self, jitted):
    logits = jnp.array([15.0, 25.0, 35.0])
    labels = jnp.array([10.0, 20.0, 30.0])
    mask = jnp.array([1, 1, 1])
    weights = jnp.array([1, 1, 1])
    metrics_fn = clu_metrics.R2Score.from_model_output
    expected = jnp.array(0.625)

    if jitted:
      metrics_fn = jax.jit(metrics_fn)

    update = metrics_fn(labels=labels, logits=logits, mask=mask,
                        label_weights=weights)
    result = update.compute()
    self.assertEqual(result.shape, expected.shape)
    np.testing.assert_array_equal(result, expected)

  @parameterized.named_parameters(("jit", True), ("no-jit", False))
  def test_r2score_weighted(self, jitted):
    logits = jnp.array([15.0, 25.0, 35.0])
    labels = jnp.array([10.0, 20.0, 30.0])
    mask = jnp.array([1, 1, 1])
    weights = jnp.array([0.5, 1, 0.5])
    metrics_fn = clu_metrics.R2Score.from_model_output
    expected = jnp.array(0.5)

    if jitted:
      metrics_fn = jax.jit(metrics_fn)

    update = metrics_fn(labels=labels, logits=logits, mask=mask,
                        label_weights=weights)
    result = update.compute()
    self.assertEqual(result.shape, expected.shape)
    np.testing.assert_array_equal(result, expected)

  def test_stratified_avg_metric(self):
    # Simulate inputs
    logits = jnp.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])
    labels = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    # Stratification bins
    stratification_bins = {
        "key1": [0, 2.5, 5.0, 7.0],  # 3 bins
        "key2": [0, 3.5, 7.0],  # 2 bins
    }

    # Extra data for stratification
    extra = {
        "key1": jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        # note that the last value is out of bounds.
        "key2": jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 9.0]),
    }

    metric_fn = clu_metrics.get_stratified_avg_metric("mae")
    metric = metric_fn.from_model_output(
        logits=logits,
        labels=labels,
        stratification_bins=stratification_bins,
        **extra
    )

    # Expected totals and counts
    expected_totals = np.array([[0.3, 0.0], [0.3, 0.4], [0.0, 0.5]])
    expected_counts = np.array([[2.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    expected_mae = np.array([[0.15, 0.0], [0.3, 0.4], [0.0, 0.5]])

    self.assertTrue(jnp.allclose(metric.total, expected_totals))
    self.assertTrue(jnp.allclose(metric.count, expected_counts))
    self.assertTrue(np.allclose(metric.compute(), expected_mae))

  def test_stratified_pearson_correlation(self):
    # Simulate inputs
    # Bin 1 (0-2.5): Perfectly correlated
    # Bin 2 (2.5-5.0): Negatively correlated
    # Bin 3 (5.0-7.0): No correlation
    logits = jnp.array([1.0, 2.0, 3.0, 4.0, 5.5, 6.0, 6.5])
    labels = jnp.array([1.1, 2.1, 4.0, 3.0, 5.0, 6.5, 6.0])
    # Stratification bins
    stratification_bins = [0, 2.5, 5.0, 7.0]

    metric_fn = clu_metrics.get_stratified_avg_metric("pearson_correlation")
    metric = metric_fn.from_model_output(
        logits=logits,
        labels=labels,
        stratification_bins=stratification_bins,
    )

    # Expected Pearson correlation for each bin
    # Bin 1: [1.0, 2.0], [1.1, 2.1] -> Perfect positive correlation
    # Bin 2: [3.0, 4.0], [4.0, 3.0] -> Perfect negative correlation
    # Bin 3: [5.5, 6.0, 6.5], [5.0, 6.5, 6.0] -> Close to zero correlation
    expected_pearson = np.array([1.0, -1.0, 0.65465367])
    self.assertTrue(np.allclose(metric.compute(), expected_pearson, atol=1e-6))

  def test_stratified_pearson_correlation_with_mask(self):
    # Simulate inputs where a mask changes the correlation.
    logits = jnp.array([1.0, 2.0, 10.0, 5.5, 7.5])
    labels = jnp.array([1.0, 2.0, -10.0, 5.5, 7.5])
    # Without the mask, the correlation is not 1. With the mask, it is.
    label_weights = jnp.array([1, 1, 0, 1, 1])
    stratification_bins = [0, 5.0, 11.0]

    metric_fn = clu_metrics.get_stratified_avg_metric("pearson_correlation")
    metric = metric_fn.from_model_output(
        logits=logits,
        labels=labels,
        label_weights=label_weights,
        stratification_bins=stratification_bins,
    )

    # Bin 1 has perfect correlation. Bin 2 is empty due to the mask, so its
    # correlation is 0.
    expected_pearson = np.array([1.0, 1.0])
    self.assertTrue(np.allclose(metric.compute(), expected_pearson, atol=1e-6))

  def test_per_strata_confusion_matrix(self):
    # Simulate inputs where a mask changes the correlation.
    logits = jnp.array([[1.0, 2.0, 10.0, 5.5, 7.5], [10, 4, 2, 1, 7]])
    predictions = jax.nn.softmax(logits, axis=-1)
    labels = jnp.array([4, 2])
    strata = jnp.array([2, 1])
    strata_size = 3

    metric = clu_metrics.PerStrataConfusionMatrix.from_model_output(
        logits=logits,
        labels=labels,
        predictions=predictions,
        strata=strata,
        strata_size=strata_size,
    )

    confusion_matrix = metric.compute()
    self.assertEqual(confusion_matrix.shape, (3, 5, 5))
    expected_confusion_matrix = np.array([
        [[0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         ],
        [[0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         ],
        [[0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         ]
    ])
    self.assertTrue(np.allclose(confusion_matrix, expected_confusion_matrix))


if __name__ == "__main__":
  jax.config.update("jax_threefry_partitionable", False)
  absltest.main()
