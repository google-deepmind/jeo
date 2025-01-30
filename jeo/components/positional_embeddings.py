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

"""Positional embeddings."""
from collections.abc import Sequence
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np


def get_posemb(
    module: nn.Module,
    kind: str,
    pos_dims: Sequence[int],
    width: int,
    name: str = "posemb",
    dtype: Any = jnp.float32,
) -> jnp.ndarray:
  """Returns positional embeddings based on specified kind."""
  # Rename `kind` for backwards compatibility with old-style naming.
  # Remove dim marker - it is automatically extracted from pos_dims.
  kind = kind.replace("-1d", "").replace("-2d", "").replace("-3d", "")
  rename = {"learned": "learn", "fixed": "sine", "sincos2d": "sine"}
  kind = rename.get(kind, kind)

  if kind == "learn":
    return module.param(name, nn.initializers.normal(stddev=1/np.sqrt(width)),
                        (1, np.prod(pos_dims), width), dtype)
  elif kind == "sine" and len(pos_dims) == 1:
    return posemb_sincos_1d(*pos_dims, width, dtype=dtype)
  elif kind == "sine" and len(pos_dims) == 2:
    return posemb_sincos_2d(*pos_dims, width, dtype=dtype)
  elif kind == "sine" and len(pos_dims) == 3:
    return posemb_sincos_3d(*pos_dims, width, dtype=dtype)
  else:
    raise ValueError(f"Unknown posemb type: {kind}")


def sinusoidal_init(
    max_len: int = 512, min_scale: float = 1.0, max_scale: float = 10000.0
):
  """1D Sinusoidal Position Embedding Initializer.

  Args:
      max_len: maximum possible length for the input.
      min_scale: float: minimum frequency-scale in sine grating.
      max_scale: float: maximum frequency-scale in sine grating.

  Returns:
      output: init function returning `(1, max_len, emb_dim)`
  """

  def init(key, shape, dtype=np.float32):
    """Sinusoidal init."""
    del key  # Needed args to be usable as initializer.
    emb_dim = shape[-1]
    pe = np.zeros((max_len, emb_dim), dtype=np.float32)
    position = np.arange(0, max_len)[:, np.newaxis]
    scale_factor = -np.log(max_scale / min_scale) / (emb_dim // 2 - 1)
    div_term = min_scale * np.exp(np.arange(0, emb_dim // 2) * scale_factor)
    pe[:, :emb_dim // 2] = np.sin(position * div_term)
    pe[:, emb_dim // 2: 2 * (emb_dim // 2)] = np.cos(position * div_term)
    return jnp.asarray(pe, dtype)[None, :, :]  # [1, max_len, emb_dim]

  return init


def posemb_sincos_1d(
    d: int, width: int, temperature: float = 10_000.0, dtype: Any = jnp.float32
) -> jnp.ndarray:
  """Returns 1-dim sincos positional embeddings."""
  x = jnp.mgrid[:d]
  assert width % 2 == 0, "Width must be mult of 2 for sincos posemb"
  omega = jnp.arange(width // 2) / (width // 2 - 1)
  omega = 1. / (temperature**omega)
  x = jnp.einsum("m,d->md", x.flatten(), omega)
  pe = jnp.concatenate([jnp.sin(x), jnp.cos(x)], axis=1)
  return jnp.asarray(pe, dtype)[None, :, :]


def posemb_sincos_2d(
    h: int,
    w: int,
    width: int,
    temperature: float = 10_000.0,
    dtype: Any = jnp.float32,
) -> jnp.ndarray:
  """Follows the MoCo v3 logic (from BV ViT)."""
  y, x = jnp.mgrid[:h, :w]
  assert width % 4 == 0, "Width must be mult of 4 for sincos posemb"
  omega = jnp.arange(width // 4) / (width // 4 - 1)
  omega = 1. / (temperature**omega)
  y = jnp.einsum("m,d->md", y.flatten(), omega)
  x = jnp.einsum("m,d->md", x.flatten(), omega)
  pe = jnp.concatenate([jnp.sin(x), jnp.cos(x), jnp.sin(y), jnp.cos(y)], axis=1)
  return jnp.asarray(pe, dtype)[None, :, :]


def posemb_sincos_3d(
    t: int,
    h: int,
    w: int,
    width: int,
    temperature: float = 10_000.0,
    dtype: Any = jnp.float32,
) -> jnp.ndarray:
  """Follows the MoCo v3 logic."""
  assert width % 6 == 0, "Width must be mult of 6 for sincos posemb"
  z, y, x = jnp.mgrid[:t, :h, :w]
  omega = jnp.arange(width // 6) / (width // 6 - 1)
  omega = 1. / (temperature**omega)
  z = jnp.einsum("m,d->md", z.flatten(), omega)
  y = jnp.einsum("m,d->md", y.flatten(), omega)
  x = jnp.einsum("m,d->md", x.flatten(), omega)
  pe = jnp.concatenate([jnp.sin(x), jnp.cos(x), jnp.sin(y), jnp.cos(y),
                        jnp.sin(z), jnp.cos(z)], axis=1)
  return jnp.asarray(pe, dtype)[None, :, :]  # (1, len, emb_dim)


def sine2d_init(max_len: int = 196, max_scale: float = 10000.0):
  """2D Sinusoidal Position Embedding Initializer.

  Args:
      max_len: maximum possible length for the input. Should be a square number.
      max_scale: float: maximum frequency-scale in sine grating.

  Returns:
      output: init function returning `(1, max_len, emb_dim)`
  """
  h = int(np.sqrt(max_len))

  def init(key, shape, dtype=jnp.float32):
    """Sine 2d pos embeddings based on ViT & MoCo v3."""
    del key  # Needed args to be usable as initializer.
    width = shape[-1]
    return posemb_sincos_2d(h, h, width, temperature=max_scale, dtype=dtype)

  return init


class AddPositionEmbs(nn.Module):
  """Adds (optionally learned) positional embeddings to the inputs.

  Attributes:
    posemb_init: positional embedding initializer.
    decode: whether to run in single-position autoregressive mode.
    max_len: maximal sequence length. Optional, uses targets sequence length if
      not specified.
  """
  max_len: int | None = None
  decode: bool = False
  # Higher level default was None, alternatively, here it has been
  # nn.initializers.normal(stddev=0.02)  # from BERT.
  posemb_init: Any = None
  type: str = "fixed"  # {fixed, learned, ...}

  @nn.compact
  def __call__(
      self, inputs: jnp.ndarray, deterministic: bool = True
  ) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Applies AddPositionEmbs module.

    By default this layer uses a fixed sinusoidal embedding table. If a
    learned position embedding is desired, pass an initializer to
    posemb_init in the configuration.

    Args:
      inputs: input data [batch_size, seq_len, emb_dim].
      deterministic: bool, pseudo for whether in inference or not.

    Returns:
      output: inputs modulated by pos-embeddings [batch_size, seq_len, emb_dim].
      pe: positional embeddings [batch_size, seq_len, emb_dim].
    """
    assert inputs.ndim == 3, f"Unexpected inputs shape: {inputs.shape}"
    choices = self.type.split("-")
    for k in choices:
      assert k in ["fixed", "learned", "1d", "2d", "at_attn", "rand_roll",
                   "rand_roll5", "init_sine", "none"], self.type
    _, seq_len, emb_dim = inputs.shape
    max_len = self.max_len or seq_len
    if "fixed" in choices:
      assert self.posemb_init is None, "posemb_init not used if fixed"
      if "2d" in choices:
        pos_embedding = sine2d_init(max_len=max_len)(None, inputs.shape)
      else:  # 1d is implied by default for backwards compatibility.
        pos_embedding = sinusoidal_init(max_len=max_len)(None, inputs.shape)
    elif "learned" in choices:
      pos_emb_shape = (1, max_len, emb_dim)
      if "init_sine" in choices:
        if "2d" in choices:
          posemb_init = sine2d_init(max_len=max_len)
        else:  # 1d is implied by default for backwards compatibility.
          posemb_init = sinusoidal_init(max_len=max_len)
      else:
        posemb_init = self.posemb_init or nn.initializers.glorot_normal()
      pos_embedding = self.param("pos_embedding", posemb_init, pos_emb_shape)
    elif self.type == "none":
      pos_embedding = jnp.zeros((1, 1, 1), dtype=jnp.float32)
    else:
      raise ValueError(f"Unrecognized positional embedding type: {self.type}")
    if "rand_roll" in self.type.split("-") and not deterministic:
      shift = (jax.random.randint(self.make_rng("dropout"), (), 0, 2) *
               jax.random.randint(self.make_rng("dropout"), (), 0, max_len))
      pos_embedding = jnp.roll(pos_embedding, shift, axis=1)
    if "rand_roll5" in self.type.split("-") and not deterministic:
      shift = (jax.random.randint(self.make_rng("dropout"), (), 0, 2) * 5 *
               jax.random.randint(self.make_rng("dropout"), (), 0, max_len//5))
      pos_embedding = jnp.roll(pos_embedding, shift, axis=1)

    pe = pos_embedding[:, :seq_len, :]  # If seq_len < max_len.

    if self.decode:
      is_initialized = self.has_variable("cache", "cache_index")
      # We use a cache position index for tracking decoding position.
      cache_index = self.variable("cache", "cache_index",
                                  lambda: jnp.array(0, dtype=jnp.int32))
      if is_initialized:
        i = cache_index.value
        cache_index.value = i + 1
        # Returns pos_embedding[0, i, :], the positional embedding for the
        # current decoding position.
        pe = jax.lax.dynamic_slice(
            pos_embedding, start_indices=jnp.array((0, i, 0)),
            slice_sizes=(1, 1, emb_dim))
    return inputs + pe, pe
