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

"""Common transformer blocks."""
from collections.abc import Callable
import functools
from typing import Any

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp

Initializer = flax.typing.Initializer


class MlpBlock(nn.Module):
  """Transformer MLP / feed-forward block."""
  # Based on third_party/py/scenic/projects/baselines/detr/model.py.

  mlp_dim: int
  out_dim: int | None = None
  dropout_rate: float = 0.1
  kernel_init: Callable[..., Any] = nn.initializers.xavier_uniform()
  bias_init: Callable[..., Any] = nn.initializers.normal(stddev=1e-6)
  activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.gelu
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self,
               inputs: jnp.ndarray,
               *,
               deterministic: bool = True) -> jnp.ndarray:
    """Applies Transformer MlpBlock model."""
    actual_out_dim = inputs.shape[-1] if self.out_dim is None else self.out_dim
    x = nn.Dense(
        self.mlp_dim,
        dtype=self.dtype,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init)(
            inputs)
    x = self.activation_fn(x)
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
    output = nn.Dense(
        actual_out_dim,
        dtype=self.dtype,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init)(
            x)
    output = nn.Dropout(rate=self.dropout_rate)(
        output, deterministic=deterministic)
    return output


class PatchMerger3D(nn.Module):
  """Merges adjacent patches together as described in SwinTransformer.

  Patches are merged together in a local, 4-connected neighbourhood.
  The input shape is [b, t*h*w, c].
  The output shape is [b, t * h/2 * w/2, output_channels].

  Attributes:
    input_resolution: The input spatial resolution of the flattened tokens that
      are passed as input. Defined as [height, width].
    output_channels: Number of channels at the output. If None, doubles the
      channels of the input.
    use_bias: Whether to use a bias in the final linear projection.
    kernel_init: Initializer function to use for the final linear projection.
    bias_init: Initializer function to use for the bias term of the final linear
      projection. Ignored if use_bias is False.
    dtype: The data type of the module.
    precision: The precision to use in the final linear projection.
  """

  input_resolution: tuple[int, int, int]
  output_channels: int | None = None
  use_bias: bool = False
  kernel_init: Initializer = nn.initializers.truncated_normal()
  bias_init: Initializer = nn.initializers.zeros
  dtype: jnp.dtype = jnp.float32
  precision: jax.lax.Precision | None = None

  @nn.compact
  def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
    time, height, width = self.input_resolution
    batch, n_tokens, channels = inputs.shape

    if n_tokens != time * height * width:
      raise ValueError("Inputs have wrong size. num_tokens should equal t*h*w.")
    if height % 2 != 0 or width % 2 != 0:
      raise ValueError(
          f"Width and height should be even. Got w={width}, h={height}"
      )
    output_dim = self.output_channels or channels * 2

    x = jnp.reshape(inputs, [batch, time, height, width, channels])
    x0 = x[:, :, 0::2, 0::2, :]  # [b, t, h/2, w/2, c]
    x1 = x[:, :, 1::2, 0::2, :]  # [b, t, h/2, w/2, c]
    x2 = x[:, :, 0::2, 1::2, :]  # [b, t, h/2, w/2, c]
    x3 = x[:, :, 1::2, 1::2, :]  # [b, t, h/2, w/2, c]
    x_concat = jnp.concatenate([x0, x1, x2, x3], axis=-1)
    # [b, t*h/2*w/2, 4c]
    x_subsampled = jnp.reshape(x_concat, [batch, n_tokens // 4, 4 * channels])

    y = nn.LayerNorm(dtype=self.dtype)(x_subsampled)
    return nn.Dense(
        output_dim,
        dtype=self.dtype,
        use_bias=self.use_bias,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        precision=self.precision,
    )(y)


class DetrMultiHeadDotProductAttention(nn.Module):
  """DETR Customized Multi-head dot-product attention.

  Based on third_party/py/scenic/projects/baselines/detr/model.py.

  Attributes:
    num_heads: Number of attention heads. Features (i.e. inputs_q.shape[-1])
      should be divisible by the number of heads.
    pos_emb_q: Positional embedding to be added to the query.
    pos_emb_k: Positional embedding to be added to the key.
    pos_emb_v: Positional embedding to be added to the value.
    qkv_features: dimension of the key, query, and value.
    out_features: dimension of the last projection
    dropout_rate: dropout rate
    broadcast_dropout: use a broadcasted dropout along batch dims.
    kernel_init: initializer for the kernel of the Dense layers.
    bias_init: initializer for the bias of the Dense layers.
    use_bias: bool: whether pointwise QKV dense transforms use bias. In DETR
      they always have a bias on the output.
    dtype: the dtype of the computation (default: float32)
  """

  num_heads: int
  qkv_features: int | None = None
  out_features: int | None = None
  dropout_rate: float = 0.
  broadcast_dropout: bool = False
  kernel_init: Callable[..., Any] = nn.initializers.xavier_uniform()
  bias_init: Callable[..., Any] = nn.initializers.zeros
  use_bias: bool = True
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self,
               inputs_q: jnp.ndarray,
               inputs_kv: jnp.ndarray | None = None,
               *,
               pos_emb_q: jnp.ndarray | None = None,
               pos_emb_k: jnp.ndarray | None = None,
               pos_emb_v: jnp.ndarray | None = None,
               key_padding_mask: jnp.ndarray | None = None,
               train: bool = False) -> jnp.ndarray:
    """Applies multi-head dot product attention on the input data.

    Projects the inputs into multi-headed query, key, and value vectors,
    applies dot-product attention and project the results to an output vector.

    This can be used for encoder-decoder attention by specifying both `inputs_q`
    and `inputs_kv` or for self-attention by only specifying `inputs_q` and
    setting `inputs_kv` to None.

    Args:
      inputs_q: Input queries of shape  `[bs, len, features]`.
      inputs_kv: Key/values of shape `[bs, len, features]` or None for
        self-attention, in which case key/values will be derived from inputs_q.
      pos_emb_q: Positional embedding to be added to the query.
      pos_emb_k: Positional embedding to be added to the key.
      pos_emb_v: Positional embedding to be added to the value.
      key_padding_mask: Binary array. Key-value tokens that are padded are 0,
        and 1 otherwise.
      train: Train or not (to apply dropout).

    Returns:
      output of shape `[bs, len, features]`.
    """
    if inputs_kv is None:
      inputs_kv = inputs_q

    assert inputs_kv.ndim == inputs_q.ndim == 3
    features = self.out_features or inputs_q.shape[-1]
    qkv_features = self.qkv_features or inputs_q.shape[-1]

    assert qkv_features % self.num_heads == 0, (
        "Memory dimension must be divisible by number of heads.")
    head_dim = qkv_features // self.num_heads

    def add_positional_emb(x, pos_emb_x):
      return x if pos_emb_x is None else x + pos_emb_x

    query, key, value = (add_positional_emb(inputs_q, pos_emb_q),
                         add_positional_emb(inputs_kv, pos_emb_k),
                         add_positional_emb(inputs_kv, pos_emb_v))

    dense = functools.partial(
        nn.DenseGeneral,
        axis=-1,
        features=(self.num_heads, head_dim),
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        use_bias=self.use_bias,
        dtype=self.dtype)
    # project inputs_q to multi-headed q/k/v
    # dimensions are then [bs, l, n_heads, n_features_per_head]
    query, key, value = (dense(name="query")(query), dense(name="key")(key),
                         dense(name="value")(value))

    # create attention masks
    if key_padding_mask is not None:
      attention_bias = (1 - key_padding_mask) * -1e10
      # add head and query dimension.
      attention_bias = jnp.expand_dims(attention_bias, -2)
      attention_bias = jnp.expand_dims(attention_bias, -2)
    else:
      attention_bias = None

    # apply attention
    x = dot_product_attention(
        query,
        key,
        value,
        dtype=self.dtype,
        bias=attention_bias,
        dropout_rate=self.dropout_rate,
        broadcast_dropout=self.broadcast_dropout,
        dropout_rng=self.make_rng("dropout") if train else None,
        deterministic=not train,
        capture_attention_weights=True)

    # back to the original inputs dimensions
    out = nn.DenseGeneral(
        features=features,
        axis=(-2, -1),
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        use_bias=True,
        dtype=self.dtype,
        name="out")(
            x)

    return out


class DetrDecoderBlock(nn.Module):
  """DETR Transformer decoder block.

  From third_party/py/scenic/projects/baselines/detr/model.py

  Attributes:
    num_heads: Number of heads.
    qkv_dim: Dimension of the query/key/value.
    mlp_dim: Dimension of the mlp on top of attention block.
    pre_norm: If use LayerNorm before attention/mlp blocks.
    dropout_rate:Dropout rate.
    attention_dropout_rate:Dropout rate for attention weights.
    dtype: Data type of the computation (default: float32).
  """

  num_heads: int
  qkv_dim: int
  mlp_dim: int
  pre_norm: bool = False
  dropout_rate: float = 0.0
  attention_dropout_rate: float = 0.0
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self,
               obj_queries: jnp.ndarray,
               encoder_output: jnp.ndarray,
               *,
               pos_embedding: jnp.ndarray | None = None,
               query_pos_emb: jnp.ndarray | None = None,
               key_padding_mask: jnp.ndarray | None = None,
               train: bool = False):
    """Applies DecoderBlock module.

    Args:
      obj_queries: Input data for decoder.
      encoder_output: Output of encoder, which are encoded inputs.
      pos_embedding: Positional Embedding to be added to the keys in
        cross-attention.
      query_pos_emb: Positional Embedding to be added to the queries.
      key_padding_mask: Binary mask containing 0 for pad tokens in key.
      train: Train or not (to apply dropout)

    Returns:
      Output after transformer decoder block.
    """

    assert query_pos_emb is not None, ("Given that object_queries are zeros "
                                       "and not learnable, we should add "
                                       "learnable query_pos_emb to them.")
    # Seems in DETR the self-attention in the first layer basically does
    # nothing, as the  value vector is a zero vector and we add no learnable
    # positional embedding to it!
    self_attn = DetrMultiHeadDotProductAttention(
        num_heads=self.num_heads,
        qkv_features=self.qkv_dim,
        broadcast_dropout=False,
        dropout_rate=self.attention_dropout_rate,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.zeros,
        use_bias=True,
        dtype=self.dtype)

    cross_attn = DetrMultiHeadDotProductAttention(
        num_heads=self.num_heads,
        qkv_features=self.qkv_dim,
        broadcast_dropout=False,
        dropout_rate=self.attention_dropout_rate,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.zeros,
        use_bias=True,
        dtype=self.dtype)

    mlp = MlpBlock(
        mlp_dim=self.mlp_dim,
        activation_fn=nn.relu,
        dtype=self.dtype,
        dropout_rate=self.dropout_rate)

    assert obj_queries.ndim == 3
    if self.pre_norm:
      # self attention block
      x = nn.LayerNorm(dtype=self.dtype)(obj_queries)
      x = self_attn(
          inputs_q=x,
          pos_emb_q=query_pos_emb,
          pos_emb_k=query_pos_emb,
          pos_emb_v=None,
          train=train)
      x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
      x = x + obj_queries
      # cross attention block
      y = nn.LayerNorm(dtype=self.dtype)(x)
      y = cross_attn(
          inputs_q=y,
          inputs_kv=encoder_output,
          pos_emb_q=query_pos_emb,
          pos_emb_k=pos_embedding,
          pos_emb_v=None,
          key_padding_mask=key_padding_mask,
          train=train)
      y = nn.Dropout(rate=self.dropout_rate)(y, deterministic=not train)
      y = y + x
      # mlp block
      z = nn.LayerNorm(dtype=self.dtype)(y)
      z = mlp(z, deterministic=not train)
      out = y + z

    else:
      # self attention block
      x = self_attn(
          inputs_q=obj_queries,
          pos_emb_q=query_pos_emb,
          pos_emb_k=query_pos_emb,
          pos_emb_v=None,
          train=train)
      x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
      x = x + obj_queries
      x = nn.LayerNorm(dtype=self.dtype)(x)
      # cross attention block
      y = cross_attn(
          inputs_q=x,
          inputs_kv=encoder_output,
          pos_emb_q=query_pos_emb,
          pos_emb_k=pos_embedding,
          pos_emb_v=None,
          key_padding_mask=key_padding_mask,
          train=train)
      y = nn.Dropout(rate=self.dropout_rate)(y, deterministic=not train)
      y = y + x
      y = nn.LayerNorm(dtype=self.dtype)(y)
      # mlp block
      z = mlp(y, deterministic=not train)
      z = y + z
      out = nn.LayerNorm(dtype=self.dtype)(z)

    return out


class DetrQueryPosEmbedding(nn.Module):
  """Creates learned positional embeddings for object queries.

  From third_party/py/scenic/projects/baselines/detr/model.py

  Attributes:
    hidden_dim: Hidden dimension for the pos embeddings.
    num_queries: Number of object queries.
    posemb_init: Positional embeddings initializer.
    dtype: Jax dtype; The dtype of the computation (default: float32).
  """
  hidden_dim: int
  num_queries: int
  posemb_init: Callable[..., Any] = nn.initializers.normal(stddev=1.0)
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self) -> jnp.ndarray:
    """Creates the positional embeddings for queries.

    Returns:
      Positional embedding for object queries.
    """
    query_pos = self.param("query_emb", self.posemb_init,
                           (self.num_queries, self.hidden_dim))
    query_pos = jnp.expand_dims(query_pos, 0)
    return jnp.asarray(query_pos, self.dtype)


class IdentityLayer(nn.Module):
  """Identity layer, convenient for giving a name to an array."""

  @nn.compact
  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    return x


def dot_product_attention(
    query: jnp.ndarray,
    key: jnp.ndarray,
    value: jnp.ndarray,
    *,
    bias: jnp.ndarray | None = None,
    bias_kv: jnp.ndarray | None = None,
    broadcast_dropout: bool = True,
    dropout_rate: float = 0.1,
    dtype: jnp.dtype = jnp.float32,
    precision: jax.lax.Precision | None = None,
    deterministic: bool,
    dropout_rng: jnp.ndarray | None = None,
    capture_attention_weights: bool = True) -> jnp.ndarray:
  """Computes the dot-product attention given query, key and value.

  This is the core function for applying attention based on
  https://arxiv.org/abs/1706.03762. It calculates the attention weights given
  query and key and combines the values using the attention weights.

  Note: query, key, value needn't have any batch dimensions.

  Args:
    query: Queries for calculating attention with shape of `[batch..., q_length,
      num_heads, qk_depth_per_head]`.
    key: Keys for calculating attention with shape of `[batch..., kv_length,
      num_heads, qk_depth_per_head]`.
    value: Values to be used in attention with shape of `[batch..., kv_length,
      num_heads, v_depth_per_head]`.
    bias: Bias for the attention weights. This should be
      broadcastable to the shape: `[batch..., num_heads, q_length, kv_length]`
        This can be used for incorporating causal masks, padding masks,
        proximity bias, etc.
    bias_kv: Attention bias defined for keys only which has shape
      `[batch..., kv_length]`. Can be used for masking elements in k/v.
    broadcast_dropout: Use a broadcasted dropout along batch dims.
    dropout_rate: Dropout rate.
    dtype: The dtype of the computation (default: float32).
    precision: Numerical precision of the computation see `jax.lax.Precision`
      for details.
    deterministic: Deterministic or not (to apply dropout).
    dropout_rng: Optional JAX PRNGKey to be used for dropout.
    capture_attention_weights: Whether to add an identity layer to tag the
      attention weights to be used for capturing them using Flax
      capture_intermediate, e.g. for visualization. Note that if this is set to
      True, this function can be only called within a Flax module.

  Returns:
    Output of shape `[batch..., length, num_heads, v_depth_per_head]`.
  """
  assert key.ndim == query.ndim == value.ndim, "q, k, v must have same rank."
  assert query.shape[:-3] == key.shape[:-3] == value.shape[:-3], (
      "q, k, v batch dims must match.")
  assert query.shape[-2] == key.shape[-2] == value.shape[-2], (
      "q, k, v num_heads must match.")
  assert key.shape[-3] == value.shape[-3], "k, v lengths must match."
  assert query.shape[-1] == key.shape[-1], "q, k depths must match."

  # Calculate attention matrix.
  depth = query.shape[-1]
  query = query / jnp.sqrt(depth).astype(dtype)
  # attn weight shape is (batch..., num_heads, q_length, kv_length)
  attn_weights = jnp.einsum(
      "...qhd,...khd->...hqk", query, key, precision=precision)

  # Apply attention bias: masking, dropout, proximity bias, etc.
  if bias is not None:
    attn_weights = attn_weights + bias
  if bias_kv is not None:
    bias_kv = bias_kv[..., jnp.newaxis, jnp.newaxis, :]
    attn_weights += bias_kv

  # Normalize the attention weights.
  attn_weights = jax.nn.softmax(attn_weights).astype(dtype)

  if capture_attention_weights:
    # Tag the intermediate weights for logging/visualization.
    attn_weights = IdentityLayer(name="attn_weights")(attn_weights)

  # Apply attention dropout.
  if not deterministic and dropout_rate > 0.:
    if dropout_rng is None:
      raise ValueError("Did not provide `rng` to dot_product_attention().")
    attn_weights = _attention_dropout(
        attn_weights,
        rate=dropout_rate,
        broadcast=broadcast_dropout,
        dropout_rng=dropout_rng)

  # Return weighted sum over values for each query position.
  return jnp.einsum(
      "...hqk,...khd->...qhd", attn_weights, value, precision=precision)


def _attention_dropout(attn_weights: jnp.ndarray,
                       *,
                       rate: float,
                       broadcast: bool = True,
                       dropout_rng: jnp.ndarray) -> jnp.ndarray:
  """Applies dropout on attention weights.

  This always applies the dropout. There is no `deterministic` parameter.

  Arguments:
    attn_weights: Attention weights.
    rate: The dropout rate. (_not_ the keep rate!)
    broadcast: Whether to broadcast on first and second last axis.
    dropout_rng: RNG.

  Returns:
    Weights after dropout.
  """
  keep_prob = 1.0 - rate
  if broadcast:
    # Dropout is broadcast across the batch+head+non-attention dimension.
    dropout_shape = list(attn_weights.shape)
    dropout_shape[0] = 1  # Broadcast batch.
    dropout_shape[-2] = 1  # Broadcast heads.
    keep = jax.random.bernoulli(dropout_rng, keep_prob, dropout_shape)
  else:
    keep = jax.random.bernoulli(dropout_rng, keep_prob, attn_weights.shape)
  multiplier = (
      keep.astype(attn_weights.dtype) /
      jnp.asarray(keep_prob, dtype=attn_weights.dtype))
  return attn_weights * multiplier
