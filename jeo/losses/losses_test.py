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

import functools
import itertools

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from jeo import losses
import numpy as np
import optax  # For testing against.

jax.config.update("jax_threefry_partitionable", False)


# Loss functions that expect labels to be one-hot encoded. All other loss
# functions expect labels to be one dim less than logits.
EXPECT_ONE_HOT_LABELS = ("sigmoid_xent", "softmax_xent",
                         "softmax_focal_loss")


def random_binary_mask(rng, shape, probability):
  """Creates a random binary mask."""
  tensor = jax.random.uniform(rng, shape, dtype=jnp.float32)
  mask = tensor >= 1 - probability
  return mask.astype(jnp.int32)


class LossesTest(parameterized.TestCase):

  @parameterized.parameters(*(itertools.product(losses.CLASSIFICATION_LOSS_FNS,
                                                [True, False])))
  def test_classification_loss_fn(self, fn_name, reduction):
    batch_size, num_classes = 4, 10
    expects_onehot = fn_name in EXPECT_ONE_HOT_LABELS
    logits = jnp.ones((batch_size, num_classes), dtype=jnp.float32)
    labels = jnp.ones((batch_size, num_classes) if expects_onehot
                      else (batch_size,), dtype=jnp.int32)
    loss_fn = losses.get_loss_fn(fn_name)
    loss = loss_fn(logits=logits, labels=labels, reduction=reduction)
    if reduction:
      self.assertEqual(loss.shape, ())
    else:
      self.assertEqual(loss.shape, (batch_size,))

  @parameterized.parameters([[True], [False]])
  def test_softmax_focal_loss(self, reduction):
    batch_size, num_classes = 4, 10
    logits = jnp.ones((batch_size, num_classes), dtype=jnp.float32)
    for expects_onehot in [True, False]:
      labels = jnp.ones(
          (batch_size, num_classes) if expects_onehot else (batch_size,),
          dtype=jnp.int32,
      )
      loss = losses.softmax_focal_loss(logits=logits, labels=labels,
                                       reduction=reduction)
      if reduction:
        self.assertEqual(loss.shape, ())
      else:
        self.assertEqual(loss.shape, (batch_size,))

  @parameterized.parameters([[1.0], [np.ones(10)]])
  def test_softmax_focal_loss_alpha(self, alpha):
    batch_size, num_classes = 4, 10
    logits = jnp.ones((batch_size, num_classes), dtype=jnp.float32)
    labels = jnp.ones((batch_size,), dtype=jnp.int32)
    loss = losses.softmax_focal_loss(logits=logits, labels=labels, alpha=alpha)
    self.assertEqual(loss.shape, ())

  @parameterized.parameters(*(itertools.product(losses.REGRESSION_LOSS_FNS,
                                                [True, False])))
  def test_regression_loss_fn(self, fn_name, reduction):
    batch_size = 4
    rng = jax.random.PRNGKey(42)
    rngs = jax.random.split(rng, 2)
    logits = jax.random.uniform(rngs[0], (batch_size, 1))
    labels = jax.random.uniform(rngs[1], (batch_size,))
    loss_fn = losses.get_loss_fn(fn_name)
    loss = loss_fn(logits=logits, labels=labels, reduction=reduction)
    if reduction:
      self.assertEqual(loss.shape, ())
    else:
      self.assertEqual(loss.shape, (batch_size,))

  @parameterized.parameters(*(itertools.product(
      losses.REGRESSION_SEGMENTATION_FNS, [True, False])))
  def test_segmentation_regression_loss_fn(self, fn_name, reduction):
    batch_size = 4
    rng = jax.random.PRNGKey(42)
    rngs = jax.random.split(rng, 2)
    logits = jax.random.uniform(rngs[0], (batch_size, 32, 32, 1))
    labels = jax.random.uniform(rngs[1], (batch_size, 32, 32, 1))
    loss_fn = losses.get_loss_fn(fn_name)
    loss = loss_fn(logits=logits, labels=labels, reduction=reduction)
    if reduction:
      self.assertEqual(loss.shape, ())
    else:
      self.assertEqual(loss.shape, (batch_size,))

  def test_get_loss_fn_from_module(self):
    fn_path = "losses.nt_xent"
    loss_kw = {"temperature": 1., "axis_name": ""}
    loss_fn = losses.get_loss_fn(fn_path, **loss_kw)
    self.assertTrue(callable(loss_fn))

  @parameterized.parameters(
      (False, 1, 0.32542178), (True, 1, 0.71335447), (False, 2, 0.3209438))
  def test_supres_losses(self, with_sigma, c, expected_loss):
    b, h, w = 4, 32, 32
    rng = jax.random.PRNGKey(42)
    rngs = jax.random.split(rng, 3)
    sr = jax.random.uniform(rngs[0], (b, h, w, c))
    hr = jax.random.uniform(rngs[1], (b, h, w, c))
    sr_sigma = jax.random.uniform(rngs[2], (b, h, w, c))
    loss, aux = losses.supres_losses(sr, sr_sigma, hr, border=1,
                                     with_sigma=with_sigma)
    self.assertCountEqual(aux, ("cpsnr", "l1", "sr_mse"))
    self.assertAlmostEqual(loss, expected_loss, places=5)

  def test_l2(self):
    b, h, w, c = 2, 12, 12, 3
    rngs = jax.random.split(jax.random.PRNGKey(42), 2)
    true = jax.random.uniform(rngs[0], (b, h, w, c))
    pred = true + 0.5 * jax.random.uniform(rngs[1], (b, h, w, c))
    expected = ((pred - true)**2).mean()
    out = losses.l2_loss(logits=pred, labels=true)
    self.assertAlmostEqual(out, expected)

  def test_l1(self):
    b, h, w, c = 2, 12, 12, 3
    rngs = jax.random.split(jax.random.PRNGKey(42), 2)
    true = jax.random.uniform(rngs[0], (b, h, w, c))
    pred = true + 0.5 * jax.random.uniform(rngs[1], (b, h, w, c))
    expected = ((pred - true)).mean()
    out = losses.l1_loss(logits=pred, labels=true)
    self.assertAlmostEqual(out, expected, places=5)

  @parameterized.parameters((False,), (True,))
  def test_l2_with_weights(self, reduction):
    shape = (4, 120, 120)
    rngs = jax.random.split(jax.random.PRNGKey(0), 3)
    x = jax.random.uniform(rngs[0], shape)
    y = jax.random.uniform(rngs[1], shape)
    w = jax.random.uniform(rngs[2], shape) < 0.5  # masking weights
    se = (y - x) ** 2
    out = losses.l2_loss(logits=x, labels=y, weights=w, reduction=reduction)
    if reduction:
      expected = (se * w).sum() / w.sum()
      self.assertAlmostEqual(out, expected, places=5)
    else:
      expected = (se * w).sum(axis=(1, 2)) / w.sum(axis=(1, 2))
      np.testing.assert_array_almost_equal(out, expected)

  def test_sigmoid_xent(self):
    b, h, w = 2, 12, 12
    rngs = jax.random.split(jax.random.PRNGKey(42), 2)
    labels = jax.random.uniform(rngs[0], (b, h, w))
    logits = labels + jax.random.uniform(rngs[1], (b, h, w))
    expected = optax.sigmoid_binary_cross_entropy(logits, labels).mean()
    out = losses.sigmoid_xent(logits=logits, labels=labels)
    self.assertAlmostEqual(out, expected, places=6)

  def test_generalized_softmax_xent(self):
    b, h, w, n = 2, 12, 12, 10
    rngs = jax.random.split(jax.random.PRNGKey(42), 2)
    labels = jax.random.randint(rngs[0], (b, h, w), 0, n)
    labels_onehot = losses.onehot(labels, n)
    logits = labels_onehot + 5 * jax.random.uniform(rngs[1], (b, h, w, n))

    # Test with int labels.
    expected = optax.softmax_cross_entropy(logits, labels_onehot).mean()
    out = losses.generalized_softmax_xent(logits=logits, labels=labels)
    self.assertAlmostEqual(out, expected, 5)

    # Test with one-hot labels.
    expected = optax.softmax_cross_entropy(logits, labels_onehot).mean()
    out = losses.generalized_softmax_xent(logits=logits, labels=labels_onehot)
    self.assertAlmostEqual(out, expected, 5)

  def test_lq_loss(self):
    b, h, w, n = 2, 12, 12, 10
    rngs = jax.random.split(jax.random.PRNGKey(42), 2)
    labels = jax.random.randint(rngs[0], (b, h, w), 0, n)
    labels_onehot = losses.onehot(labels, n)
    logits = labels_onehot + 5 * jax.random.uniform(rngs[1], (b, h, w, n))

    # Test q=0 with int labels.
    expected = optax.softmax_cross_entropy(logits, labels_onehot).mean()
    out = losses.lq_loss(logits=logits, labels=labels, q=0.0)
    self.assertAlmostEqual(out, expected, 5)

    # Test q=0 with one-hot labels.
    out = losses.lq_loss(logits=logits, labels=labels_onehot, q=0.0)
    self.assertAlmostEqual(out, expected, 5)

    # Test q=1 with int labels.
    probs = jax.nn.softmax(logits, axis=-1)
    expected_q1 = (1.0 - jnp.sum(labels_onehot * probs**1.0, axis=-1)).mean()
    out = losses.lq_loss(logits=logits, labels=labels, q=1.0)
    self.assertAlmostEqual(out, expected_q1, 5)

    # Test q=0.5 with int labels.
    probs = jax.nn.softmax(logits, axis=-1)
    expected_q05 = jnp.mean(
        (1.0 - jnp.sum(labels_onehot * probs**0.5, axis=-1)) / 0.5
    )
    out = losses.lq_loss(logits=logits, labels=labels, q=0.5)
    self.assertAlmostEqual(out, expected_q05, 5)

    # Test reduction=False
    out_no_reduction = losses.lq_loss(
        logits=logits, labels=labels, q=1.0, reduction=False)
    self.assertEqual(out_no_reduction.shape, (b,))

  def test_softmax_xent(self):
    b, h, w, n = 2, 12, 12, 10
    rngs = jax.random.split(jax.random.PRNGKey(42), 2)
    labels = jax.random.randint(rngs[0], (b, h, w, n), 0, 2)
    logits = labels + 5 * jax.random.uniform(rngs[1], (b, h, w, n))
    expected = optax.softmax_cross_entropy(logits, labels).mean()
    out = losses.softmax_xent(logits=logits, labels=labels)
    self.assertAlmostEqual(out, expected)

  def test_weighted_softmax_xent(self):
    b, h, w, n = 2, 12, 12, 10
    rngs = jax.random.split(jax.random.PRNGKey(42), 2)
    labels = jax.random.randint(rngs[0], (b, h, w, n), 0, 2)
    logits = labels + 5 * jax.random.uniform(rngs[1], (b, h, w, n))
    weights = jnp.argmax(labels, axis=-1) > 0
    num_zero_weighted = jnp.sum(weights == 0).item()
    loss_non_zero_weighted = optax.softmax_cross_entropy(
        logits[weights], labels[weights]
    )
    expected = jnp.concat(
        [jnp.zeros(num_zero_weighted), loss_non_zero_weighted]
    ).mean()
    out = losses.softmax_xent(logits=logits, labels=labels, weights=weights)
    self.assertAlmostEqual(out, expected, places=5)

  def test_kld_loss_softmax(self):
    b, n = 8, 10
    rngs = jax.random.split(jax.random.PRNGKey(42), 2)

    true_logits = jax.random.normal(rngs[0], (b, n))
    pred_logits = jax.random.normal(rngs[1], (b, n))
    true_probs = jax.nn.softmax(true_logits)
    pred_probs = jax.nn.softmax(pred_logits)
    expected = optax.kl_divergence(jnp.log(pred_probs), true_probs).mean()

    # Model inputs and labels are both probabilities.
    out = losses.kld_loss(
        logits=pred_probs, labels=true_probs, activation="none"
    )
    self.assertAlmostEqual(out, expected, places=5)

    # Model inputs are probabilities, labels are logits.
    out = losses.kld_loss(
        logits=pred_probs,
        labels=true_logits,
        activation="none",
        labels_activation="softmax",
    )
    self.assertAlmostEqual(out, expected, places=5)

    # # Model inputs are logits, labels are probabilities.
    out = losses.kld_loss(
        logits=pred_logits, labels=true_probs, activation="softmax"
    )
    self.assertAlmostEqual(out, expected, places=5)

    # Model inputs are logits, labels are logits.
    out = losses.kld_loss(
        logits=pred_logits,
        labels=true_logits,
        activation="softmax",
        labels_activation="softmax",
    )
    self.assertAlmostEqual(out, expected, places=5)

  def test_kld_loss_sigmoid(self):
    b = 8
    rngs = jax.random.split(jax.random.PRNGKey(42), 2)

    true_logits = jax.random.normal(rngs[0], (b,))
    pred_logits = jax.random.normal(rngs[1], (b,))
    true_probs = jax.nn.sigmoid(true_logits)
    pred_probs = jax.nn.sigmoid(pred_logits)
    multiclass_true_probs = jnp.stack((1 - true_probs, true_probs), axis=-1)
    multiclass_pred_probs = jnp.stack((1 - pred_probs, pred_probs), axis=-1)

    expected = optax.kl_divergence(
        jnp.log(multiclass_pred_probs), multiclass_true_probs
    ).mean()

    # Model inputs and labels are both probabilities.
    out = losses.kld_loss(
        logits=multiclass_pred_probs,
        labels=multiclass_true_probs,
        activation="none",
    )
    self.assertAlmostEqual(out, expected, places=5)

    # Model inputs are probabilities, labels are logits.
    out = losses.kld_loss(
        logits=multiclass_pred_probs,
        labels=true_logits,
        activation="none",
        labels_activation="sigmoid",
    )
    self.assertAlmostEqual(out, expected, places=5)

    # # # Model inputs are logits, labels are probabilities.
    out = losses.kld_loss(
        logits=pred_logits, labels=multiclass_true_probs, activation="sigmoid"
    )
    self.assertAlmostEqual(out, expected, places=5)

    # # Model inputs are logits, labels are logits.
    out = losses.kld_loss(
        logits=pred_logits,
        labels=true_logits,
        activation="sigmoid",
        labels_activation="sigmoid",
    )
    self.assertAlmostEqual(out, expected, places=5)

  @parameterized.parameters((True,), (False,))
  def test_nt_xent(self, jit):
    a = jnp.array([[1.0, 5.0, 2.3], [2.0, -1.0, 0.0],
                   [-5.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
    b = jnp.array([[1.0, -4.0, 2.3], [2.0, 1.0, 2.0],
                   [-5.5, 1.0, 0.0], [0.5, 1.9, 3.1]])
    loss_fn = functools.partial(losses.nt_xent, temperature=0.1, reduction=True,
                                axis_name=None)
    if jit:
      loss_fn = jax.jit(loss_fn)
    logits = jnp.stack((a, b), 1)
    logits /= jnp.linalg.norm(logits, axis=-1, keepdims=True)
    loss = loss_fn(logits=logits)
    self.assertAlmostEqual(8.149471, loss, places=5)
    logits = jnp.stack((a, -b), 1)
    logits /= jnp.linalg.norm(logits, axis=-1, keepdims=True)
    loss = loss_fn(logits=logits)
    self.assertAlmostEqual(25.766865, loss, places=5)
    logits = jnp.stack((a, a), 1)
    logits /= jnp.linalg.norm(logits, axis=-1, keepdims=True)
    loss = loss_fn(logits=logits)
    self.assertAlmostEqual(0.3855347, loss, places=5)
    logits = jnp.stack((b, b), 1)
    logits /= jnp.linalg.norm(logits, axis=-1, keepdims=True)
    loss = loss_fn(logits=logits)
    self.assertAlmostEqual(0.30277473, loss, places=5)

  def test_nt_xent_pmap(self):
    b, v, d = 4, 2, 32
    rng = jax.random.PRNGKey(0)
    logits = jax.random.normal(rng, (jax.local_device_count(), b, v, d))
    logits /= jnp.linalg.norm(logits, axis=-1, keepdims=True)
    non_pmap_loss = losses.nt_xent(
        jnp.concatenate(logits, axis=0), temperature=0.1, reduction=True,
        axis_name=None)

    @functools.partial(jax.pmap, axis_name="batch")
    def compute_loss(logits):
      return losses.nt_xent(logits, temperature=0.1, reduction=True,
                            axis_name="batch")

    losses_per_device = compute_loss(logits)
    pmap_loss = jnp.mean(losses_per_device)
    self.assertAlmostEqual(non_pmap_loss, pmap_loss, places=5)

  def test_unused_loss(self):
    out = losses.get_loss_fn("unused")(logits=[1, 2, 3], labels=[0, 1, 0])
    self.assertIsNone(out)

  def test_generalized_dice(self):
    labels = jnp.array([[[1, 2, 0], [4, 3, 4], [1, 4, 1]]])
    logits = jnp.array([[
        [
            [-0.267, -0.994, -0.398, -0.31, 0.133],
            [-0.806, -0.102, 0.668, 0.609, -1.406],
            [1.404, -1.006, 0.925, 0.806, -1.623],
        ],
        [
            [0.602, 0.266, 0.111, 2.081, -1.014],
            [0.662, -2.159, 0.118, -1.758, -1.142],
            [2.555, -0.566, -0.716, 0.65, -1.76],
        ],
        [
            [0.923, 1.576, 0.301, 0.663, 0.595],
            [0.959, -3.683, -0.247, 1.325, 1.481],
            [-0.127, -2.327, 2.09, 0.164, -0.563],
        ],
    ]])
    weights = jnp.array([[[1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [1.0, 0.0, 1.0]]])

    kwargs = dict(logits=logits, labels=labels, weights=weights)
    loss_fn = functools.partial(losses.generalized_dice, **kwargs)

    st = loss_fn(norm_type="standard").item()
    st_dual = loss_fn(norm_type="standard", dual=True, **kwargs).item()
    sq = loss_fn(norm_type="squared").item()
    sq_dual = loss_fn(norm_type="squared", dual=True, **kwargs).item()
    tan = loss_fn(norm_type="tanimoto").item()
    tan_dual = loss_fn(norm_type="tanimoto", dual=True, **kwargs).item()
    tvs = loss_fn(norm_type="tversky").item()
    tvs_dual = loss_fn(norm_type="tversky", dual=True, **kwargs).item()
    tvs_1 = loss_fn(norm_type="tversky", beta=0.1, **kwargs).item()
    tvs_1_dual = loss_fn(norm_type="tversky", dual=True, beta=0.1).item()
    tvs_9 = loss_fn(norm_type="tversky", beta=0.9, **kwargs).item()
    tvs_9_dual = loss_fn(norm_type="tversky", dual=True, beta=0.9).item()

    self.assertSequenceAlmostEqual(
        (st, st_dual, sq, sq_dual, tan, tan_dual, tvs, tvs_dual, tvs_1,
         tvs_1_dual, tvs_9, tvs_9_dual),
        (0.8815, 0.5488, 0.8392, 0.4960, 0.9040, 0.5827, 0.8555, 0.5166, 0.6691,
         0.3688, 0.8628, 0.5283),
        places=3,
    )

  @parameterized.parameters([[True], [False]])
  def test_weighted_losses_sum(self, reduction):
    batch_size, num_classes, size = 4, 10, 32
    logits = jnp.ones((batch_size, size, size, num_classes))
    labels = jnp.ones((batch_size, size, size), "int32")

    losses_config = dict(
        focal=dict(loss="softmax_focal_loss", weight=2),
        tversky=dict(
            loss="generalized_dice", loss_kw=dict(norm_type="tversky")
        ),
        xent=dict(loss="generalized_softmax_xent"),
    )

    combined_loss, losses_values = losses.weighted_losses_sum(
        logits, labels, losses_config=losses_config, reduction=reduction
    )

    # Check that the individual losses are computed correctly.
    for name, loss_kwg in losses_config.items():
      loss_fn = losses.get_loss_fn(
          loss_kwg["loss"], **loss_kwg.get("loss_kw", {})
      )
      loss = loss_fn(logits=logits, labels=labels)
      self.assertAlmostEqual(np.mean(loss), np.mean(losses_values[name]))

    self.assertEqual(combined_loss.shape, () if reduction else (batch_size,))

  def test_huber(self):
    delta = 1.0
    b, h, w, c = 2, 12, 12, 3
    rngs = jax.random.split(jax.random.PRNGKey(42), 2)
    true = jax.random.uniform(rngs[0], (b, h, w, c))
    pred = true + 0.5 * jax.random.uniform(rngs[1], (b, h, w, c))
    abs_errors = jnp.abs(pred - true)
    quadratic = jnp.minimum(abs_errors, delta)
    linear = abs_errors - quadratic
    expected = (0.5 * quadratic**2 + delta * linear).mean()

    out = losses.huber_loss(logits=pred, labels=true)
    self.assertAlmostEqual(out, expected)

  @parameterized.parameters(
      itertools.product((True, False), ("l2_loss", "huber_loss"))
  )
  def test_generalized_supres_loss(self, brightness_bias, loss_name):
    batch_size, height, width, channels = 2, 32, 32, 3
    rng = jax.random.PRNGKey(42)
    logits = jnp.zeros((batch_size, height, width, channels)) + 0.5
    labels = jnp.zeros((batch_size, height, width, channels)) + 0.4
    weights = random_binary_mask(rng, (batch_size, height, width), 0.8)
    border = 3

    loss = losses.generalized_supres_loss(
        logits=logits,
        labels=labels,
        loss=loss_name,
        reduction=False,
        weights=weights,
        border=border,
        brightness_bias=brightness_bias,
    )
    self.assertEqual(loss.shape, (batch_size,))

    loss_reduced = losses.generalized_supres_loss(
        logits=logits, labels=labels, loss=loss_name, weights=weights
    )
    self.assertEqual(loss_reduced.shape, ())
    if loss_name == "l2_loss":
      expected_loss = losses.l2_loss(logits, labels, weights=weights).mean()
      self.assertAlmostEqual(loss_reduced.item(), expected_loss, places=5)
    elif loss_name == "huber_loss":
      expected_loss = losses.huber_loss(logits, labels, weights=weights).mean()
      self.assertAlmostEqual(loss_reduced.item(), expected_loss, places=5)


if __name__ == "__main__":
  absltest.main()
