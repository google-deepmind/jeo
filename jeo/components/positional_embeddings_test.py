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

"""Tests for positional embeddings."""
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import flax.linen as nn
import jax
import jax.numpy as jnp
from jeo.components import positional_embeddings
import numpy as np


class ComponentsTest(parameterized.TestCase):

  def test_sinusoidal_init(self):
    max_len, emb_dim = 3, 4
    pos_emb = positional_embeddings.sinusoidal_init(max_len)(None, (emb_dim,))
    expected_shape = (1, max_len, emb_dim)
    self.assertEqual(pos_emb.shape, expected_shape)
    expected_values = np.array([
        [0, 0, 1, 1],
        [0.84147, 0.0001, 0.540302, 1],
        [0.909297, 0.0002, -0.4161468, 1]])
    np.testing.assert_allclose(pos_emb[0], expected_values, atol=1e-5)

  def test_sincos_1d(self):
    d, emb_dim = 3, 4
    pos_emb = positional_embeddings.posemb_sincos_1d(d, emb_dim)
    expected_shape = (1, d, emb_dim)
    self.assertEqual(pos_emb.shape, expected_shape)
    expected_values = np.array([
        [0, 0, 1, 1],
        [0.84147, 0.0001, 0.540302, 1],
        [0.909297, 0.0002, -0.4161468, 1]])
    np.testing.assert_allclose(pos_emb[0], expected_values, atol=1e-5)

  def test_sine_2d_init(self):
    max_len, emb_dim = 4, 8
    pos_emb = positional_embeddings.sine2d_init(max_len)(None, (emb_dim,))
    expected_shape = (1, max_len, emb_dim)
    self.assertEqual(pos_emb.shape, expected_shape)
    expected_values = np.array([
        [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
        [0.841, 1e-4, 0.54, 1.0, 0.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, 1.0, 1.0, 0.841, 1e-4, 0.54, 1.0],
        [0.841, 1e-4, 0.54, 1.0, 0.841, 1e-4, 0.54, 1.0]])
    np.testing.assert_allclose(pos_emb[0], expected_values, atol=1e-2)

  def test_sincos_2d(self):
    h, w, emb_dim = 2, 2, 8
    pos_emb = positional_embeddings.posemb_sincos_2d(h, w, emb_dim)
    expected_shape = (1, h*w, emb_dim)
    self.assertEqual(pos_emb.shape, expected_shape)
    expected_values = np.array([
        [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
        [0.841, 1e-4, 0.54, 1.0, 0.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, 1.0, 1.0, 0.841, 1e-4, 0.54, 1.0],
        [0.841, 1e-4, 0.54, 1.0, 0.841, 1e-4, 0.54, 1.0]])
    np.testing.assert_allclose(pos_emb[0], expected_values, atol=1e-2)

  def test_sincos_3d(self):
    t, h, w, emb_dim = 3, 2, 2, 12
    pos_emb = positional_embeddings.posemb_sincos_3d(t, h, w, emb_dim)
    expected_shape = (1, t*h*w, emb_dim)
    self.assertEqual(pos_emb.shape, expected_shape)
    expected_row0 = np.array(
        [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0])
    np.testing.assert_allclose(pos_emb[0, 0], expected_row0, atol=1e-5)
    expected_diag = np.array([
        0., 0.0001, 1., 1., 0., 0., 0.54030228, 1., 0.90929741, 0.0002,
        -0.41614684, 1.])
    np.testing.assert_allclose(np.diag(pos_emb[0]), expected_diag, atol=1e-5)

  @parameterized.parameters(("learn", 1), ("learned", 2), ("sine", 1),
                            ("sine", 2), ("fixed", 3))
  def test_get_posemb(self, kind, ndim):
    emb_dim = 12
    dims = [i+1 for i in range(ndim)]
    expected_shape = (1, np.prod(dims), emb_dim)
    m = mock.MagicMock()
    m.param.return_value = jnp.zeros(expected_shape)

    pos_emb = positional_embeddings.get_posemb(m, kind, dims, emb_dim)
    self.assertEqual(pos_emb.shape, expected_shape)

  @parameterized.parameters("fixed", "learned", "none", "learned-at_attn",
                            "fixed-2d")
  def test_add_pos_emb(self, posemb):
    batch_size, seq_len, emb_dim = 2, 4, 8
    m = MockModel(posemb)
    targets = jnp.ones((batch_size, seq_len, emb_dim), dtype=jnp.int64)
    params = m.init(jax.random.PRNGKey(0), targets)
    _, y, pe = m.apply(params, targets)
    if posemb != "none":
      self.assertEqual((1, seq_len, emb_dim), pe.shape)
    self.assertEqual((batch_size, seq_len, emb_dim), y.shape)
    np.testing.assert_allclose(y-pe, targets)

  def test_add_pos_emb_decode(self):
    batch_size, seq_len, emb_dim = 2, 10, 8
    rng = jax.random.PRNGKey(0)
    m = MockModel("fixed")
    targets = jnp.ones((batch_size, seq_len, emb_dim), dtype=jnp.int64)
    params = m.init(rng, targets)
    def tokens_to_logits(tokens, cache):
      return m.apply({"params": params["params"], "cache": cache}, tokens,
                     decode=True, mutable=True, rngs={"params": rng})
    cache = m.apply(params, targets, decode=True, mutable=True,
                    rngs={"params": rng})[1]["cache"]
    # Cache is initialized with index 0.
    self.assertEqual(cache["add_pos_emb"]["cache_index"], 0)
    for i in range(3):
      cache = tokens_to_logits(targets[:, i:i+1, :], cache)[1]["cache"]
    # After 3 decode steps, cache index is at position 3.
    self.assertEqual(cache["add_pos_emb"]["cache_index"], 3)


class MockModel(nn.Module):
  posemb: str = "fixed"

  @nn.compact
  def __call__(self, targets, decode=False, train=False):
    y, pe = positional_embeddings.AddPositionEmbs(
        decode=decode, type=self.posemb, name="add_pos_emb")(
            targets, deterministic=not train)
    z = nn.Dense(4, use_bias=False)(y)
    return z, y, pe


if __name__ == "__main__":
  absltest.main()
