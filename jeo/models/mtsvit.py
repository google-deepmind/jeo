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

"""MTSViT (Multi-Modality Temporal-Spatial Vision Transformer).

Extend the model from:
Paper: https://arxiv.org/pdf/2301.04944.pdf
Github: https://github.com/michaeltrs/DeepSatModels

The following abbreviations / notation are used throughout this file:
b - the batch dimension
t - temporal dimension
h - height
w - width
d - channels
N - number of classes
m = number of modalities
"""

from collections.abc import Callable, Sequence
from typing import Any

from absl import logging
import einops
import flax.linen as nn
import jax.numpy as jnp
from jeo.components import positional_embeddings
from jeo.components import transformers
from jeo.models import vit


class Model(nn.Module):
  """MTSViT model.

  Expectations:
  - First modality is the reference modality, determining the output shape.
  - Spatial extent of each modality (after patchification) either matches the
    reference modality or is 1 (nonspatial). Otherwise needed to add padding.
  - Height and width expected equal per modality (for simplicity).
  - Temporal extent can be variable (but if it's very long for one modality, and
    others will be padded to it, this will be computationally expensive).
  """

  mods: Sequence[str]  # Modality names (should be in inputs and patch sizes).
  patch_sizes: dict[str, Sequence[int]]  # Dict of (t,h,w) patch sizes.
  num_classes: int
  emb_dim: int = 192
  temporal_depth: int = 2
  spatial_depth: int = 2
  decoder_depth: int = 2
  mlp_dim: int | None = None  # Defaults to 4x emb_dim.
  num_heads: int = 6
  dropout: float = 0.0
  head: str = "segmentation"  # {"classification", "segmentation"}
  use_multiscale_decoder: bool = False
  use_upscale_pool: bool = False
  spatial_first: bool = True

  @nn.compact
  def __call__(self, inputs, *, train=False):
    for k in self.mods:
      if k not in inputs or k not in self.patch_sizes:
        raise ValueError(f"{k} not in inputs or patch_sizes.")
    # Output the same size as the 1st "reference" modality.
    b, _, output_h, _, _ = inputs[self.mods[0]].shape

    # Tokenize. patched_shapes[k] = (t, h, w) of modality k.
    _, tokens, patched_shapes = self.tokenize(inputs, separate=True)
    _, patched_h, _ = patched_shapes[self.mods[0]]

    # Separate spatial and nonspatial modalities.
    x_s, x_ns, mods = [], [], []  # x_s = spatial, x_ns = nonspatial
    for k in self.mods:
      _, ph, pw = patched_shapes[k]
      assert (ph == patched_h or ph == 1) and ph == pw, "Unequal patch sizes."
      if ph > 1:
        x_s.append(tokens[k])
        mods.append(k)
      else:
        x_ns.append(tokens[k])
    mods += [m for m in self.mods if patched_shapes[m][1] == 1]

    if self.spatial_first:
      # Prepare for spatial encoder: target shape (bt, hw, d).
      # Neglects nonspatial modalities, and moves temporal dim into batch dim.
      x, ps = einops.pack(x_s, "* s d")
      x = x + positional_embeddings.get_posemb(
          self, "learn", (x.shape[1],), self.emb_dim, "pe_spatial", x.dtype)

      # Spatial encoder (each modality only interacts with itself).
      spatial_encoder = vit.Encoder(
          depth=self.spatial_depth,
          mlp_dim=self.mlp_dim,
          num_heads=self.num_heads,
          dropout=self.dropout,
          name="transformer_spatial")
      x, _ = spatial_encoder(x, deterministic=not train)
      x_s = einops.unpack(x, ps, "* s d")

      # Prepare for temporal encoder: target shape (bhw, t, d).
      # Include non-temporal modalities, and move spatial dims into batch dim.
      # Pad all temp patches and concat along batch-spatial dim.
      x = x_s + x_ns
      x = [einops.rearrange(y, "b t s d -> (b s) t d") for y in x]
      x, valid_mask, ps = pad_and_pack_sequences(x)
      x += positional_embeddings.get_posemb(
          self, "learn", (x.shape[1],), self.emb_dim, "pe_temp", x.dtype)

      # Add class token embeddings.
      cls_temp = self.param("cls_temp", nn.initializers.zeros,
                            (1, self.num_classes, self.emb_dim), x.dtype)
      cls_temp = einops.repeat(cls_temp, "1 n d -> b n d", b=x.shape[0])
      x = jnp.concatenate([cls_temp, x], axis=1)  # (mBhw,N+t,d)
      # Update valid_mask with class tokens for attention mask construction.
      valid_mask = jnp.concatenate([
          jnp.ones(cls_temp.shape[:2]), valid_mask], axis=1)
      attn_mask = get_attention_masks(valid_mask, self.num_heads)

      temporal_encoder = vit.Encoder(
          depth=self.temporal_depth,
          mlp_dim=self.mlp_dim,
          num_heads=self.num_heads,
          dropout=self.dropout,
          name="transformer_temporal")
      x, _ = temporal_encoder(x, deterministic=not train, mask=attn_mask)
      x = x[:, : self.num_classes, :]  # Only class tokens  (mBhw,N,d)
      x = einops.unpack(x, ps, "* n d")
      x = [einops.rearrange(y, "(b s) n d -> (b n) s d", b=b) for y in x]
    else:  # Temporal first.
      x = [einops.rearrange(y, "b t s d -> (b s) t d") for y in x_s + x_ns]
      x, valid_mask, ps = pad_and_pack_sequences(x)
      x += positional_embeddings.get_posemb(
          self, "learn", (x.shape[1],), self.emb_dim, "pe_temp", x.dtype)
      # Add class token embeddings.
      cls_temp = self.param("cls_temp", nn.initializers.zeros,
                            (1, self.num_classes, self.emb_dim), x.dtype)
      cls_temp = einops.repeat(cls_temp, "1 n d -> b n d", b=x.shape[0])
      x = jnp.concatenate([cls_temp, x], axis=1)  # (mBhw,N+t,d)
      # Update valid_mask with class tokens for attention mask construction.
      valid_mask = jnp.concatenate([
          jnp.ones(cls_temp.shape[:2]), valid_mask], axis=1)
      attn_mask = get_attention_masks(valid_mask, self.num_heads)
      temporal_encoder = vit.Encoder(
          depth=self.temporal_depth,
          mlp_dim=self.mlp_dim,
          num_heads=self.num_heads,
          dropout=self.dropout,
          name="transformer_temporal")
      x, _ = temporal_encoder(x, deterministic=not train, mask=attn_mask)
      x = x[:, : self.num_classes, :]  # Only class tokens  (mBhw,N,d)
      x = einops.unpack(x, ps, "* n d")  # Seq of ((b s) n d)
      x = [einops.rearrange(y, "(b s) n d -> (b n) s d", b=b) for y in x]
      x_ns = [y for y in x if y.shape[1] == 1]
      x = [y for y in x if y.shape[1] > 1]
      x, ps = einops.pack(x, "* s d")
      x = x + positional_embeddings.get_posemb(
          self, "learn", (x.shape[1],), self.emb_dim, "pe_spatial", x.dtype)
      spatial_encoder = vit.Encoder(
          depth=self.spatial_depth,
          mlp_dim=self.mlp_dim,
          num_heads=self.num_heads,
          dropout=self.dropout,
          name="transformer_spatial")
      x, _ = spatial_encoder(x, deterministic=not train)
      x = einops.unpack(x, ps, "* s d")  # Seq of ((b n) s d)
      x += x_ns  # Add nonspatial modalities back.

    # Decoder for multi-modality interaction.
    # Use patches from 1st modality as queries.
    queries = x[0]
    values = queries if len(mods) == 1 else jnp.concatenate(x[1:], axis=1)

    if self.head == "classification":  # Add classification token.
      cls_tok = self.param("cls_tok", nn.initializers.zeros,
                           (1, 1, queries.shape[-1]), queries.dtype)
      queries = jnp.concatenate([jnp.tile(cls_tok, [queries.shape[0], 1, 1]),
                                 queries], axis=1)

    decoder = ChunkDecoder(
        num_heads=self.num_heads,
        num_layers=self.decoder_depth,
        qkv_dim=self.emb_dim,
        mlp_dim=self.emb_dim * 4,
        num_classes=self.num_classes,
        patched_shapes=patched_shapes,
        mods=mods[1:] if len(mods) > 1 else mods,
        multiscale=self.use_multiscale_decoder,
        name="spatial_decoder",
    )
    x = decoder(obj_queries=queries, encoder_output=values, train=train)

    out = {}
    patch_size = self.patch_sizes[mods[0]]
    if self.head == "classification":  # Default TSViT classification head.
      x = x[:, 0, :]  # (BN,d)
      x = nn.LayerNorm()(x)
      x = nn.Dense(1, name="head_cls")(x)
      x = out["logits"] = x.reshape((b, self.num_classes))
    elif self.head == "segmentation":  # Default TSViT segmentation head.
      x = nn.LayerNorm()(x)
      if self.use_upscale_pool:
        # Project segmentation results to a higher resolution and then pool it.
        upscale = 2
        x = nn.Dense(patch_size[-2] * patch_size[-1] * upscale * upscale,
                     name="head_cls")(x)
        x = einops.rearrange(
            x, "(b n) (h w) (p q) -> b (h p) (w q) n",
            b=b, h=output_h // patch_size[-2], p=patch_size[-2]*upscale)
        # Downsize to output resolution.
        x = nn.avg_pool(x, (3, 3), strides=(2, 2), padding="SAME")
      else:
        x = nn.Dense(patch_size[-2] * patch_size[-1], name="head_cls")(x)
        x = einops.rearrange(
            x, "(b n) (h w) (p q) -> b (h p) (w q) n",
            b=b, h=output_h // patch_size[-2], p=patch_size[-2])

      out["logits_2d"] = x
    return x, out

  def tokenize(self, inputs, separate=False):
    """Tokanizes each modality separately.

    Args:
      inputs: Inputs dict with shapes (b, t_k, h_k, w_k, c_k).
        t_k, h_k, w_k, c_k can be different for each modality.
      separate: If True, tokens are separated by temporal and spatial dims.

    Returns:
      patches: Patchifyed inputs before projecting them to embedding space with
        shapes (b, num_patches_k, c_k * total_patch_size_k). Where
        total_patch_size_k = dt * dh * dw from patch_sizes dict.
        Note: t_k * h_k * w_k * c_k == num_patches_k * total_patch_size_k.
        num_patches_k == T_k *  H_k * W_k.
        If separate=True, then num_patches_k == (T_k, H_k * W_k).
      tokens: Tokens dict with shapes (b, num_patches_k, emb_dim).
        If separate=True, then num_patches_k == (T_k, H_k * W_k).
      patched_shapes: Dict denoting shapes after patchification that are then
        reshaped into num_patches_k: (T_k=t_k/dt_k, H_k=h_k/dh_k, W_k=w_k/dw_k).
    """
    patches, tokens, patched_shapes = {}, {}, {}
    for k in self.mods:
      x, patch_size = maybe_make_5d(inputs[k], self.patch_sizes[k])

      patched_shape = [x.shape[1 + i] // patch_size[i] for i in range(3)]
      patched_shapes[k] = patched_shape  # (t, h, w)
      dt, dh, dw = patch_size
      if separate:
        x = einops.rearrange(
            x, "b (t p) (h q) (w r) c -> b t (h w) (p q r c)", p=dt, q=dh, r=dw)
      else:
        x = einops.rearrange(
            x, "b (t p) (h q) (w r) c -> b (t h w) (p q r c)", p=dt, q=dh, r=dw)
      patches[k] = x
      x = nn.Dense(self.emb_dim, name=f"proj_inputs_{k}")(x)  # (b,p,d).
      tokens[k] = x
      logging.info(
          "Tokenized %s (%s) into %s (patches: %s, patched_shape: %s).",
          k, inputs[k].shape, tokens[k].shape, patches[k].shape, patched_shape)
    return patches, tokens, patched_shapes  # _, (n, p, d), (t, h, w)


class ChunkDecoder(nn.Module):
  """DETR Transformer Decoder.

  Attributes:
    num_heads: Number of heads.
    num_layers: Number of layers.
    qkv_dim: Dimension of the query/key/value.
    mlp_dim: Dimension of the mlp on top of attention block.
    normalize_before: If use LayerNorm before attention/mlp blocks.
    return_intermediate: If return the outputs from intermediate layers.
    padding_mask: Binary mask containing 0 for padding tokens.
    dropout_rate:Dropout rate.
    attention_dropout_rate:Dropout rate for attention weights.
    dtype: Data type of the computation (default: float32).
  """

  num_heads: int
  num_layers: int
  qkv_dim: int
  mlp_dim: int
  num_classes: int
  patched_shapes: dict[str, Sequence[int]]
  mods: Sequence[str]
  normalize_before: bool = False
  norm: Callable[..., Any] | None = None
  return_intermediate: bool = False
  dropout_rate: float = 0.0
  attention_dropout_rate: float = 0.0
  dtype: jnp.dtype = jnp.float32
  multiscale: bool = False

  @nn.compact
  def __call__(self,
               obj_queries: jnp.ndarray,
               encoder_output: jnp.ndarray,
               *,
               key_padding_mask: jnp.ndarray | None = None,
               pos_embedding: jnp.ndarray | None = None,
               train: bool = False) -> jnp.ndarray:
    """Applies Decoder on the inputs.

    Args:
      obj_queries: Input data for decoder.
      encoder_output: Output of encoder, which are encoded inputs.
      key_padding_mask: Binary mask containing 0 for padding tokens in the keys.
      pos_embedding: Positional Embedding to be added to the keys.
      train: Whether it is training.

    Returns:
      Output of a transformer decoder.
    """
    assert encoder_output.ndim == 3  # `[batch, len, features]`
    assert obj_queries.ndim == 3  # `[batch, num queries, embedding size]`
    y = obj_queries
    patched_shapes = {k: v for k, v in self.patched_shapes.items()}
    outputs = []

    query_pos_emb = transformers.DetrQueryPosEmbedding(
        y.shape[-1], y.shape[1])()

    for lyr in range(self.num_layers):
      y = transformers.DetrDecoderBlock(
          qkv_dim=self.qkv_dim,
          mlp_dim=self.mlp_dim,
          num_heads=self.num_heads,
          pre_norm=self.normalize_before,
          dropout_rate=self.dropout_rate,
          attention_dropout_rate=self.attention_dropout_rate,
          dtype=self.dtype,
          name=f"decoderblock_{lyr}")(
              y,
              encoder_output,
              pos_embedding=pos_embedding,
              query_pos_emb=query_pos_emb,
              key_padding_mask=key_padding_mask,
              train=train)

      if self.multiscale:
        start_idx = 0
        encoder_output_merged = []
        for k in self.mods:
          _, patched_h_k, patched_w_k = patched_shapes[k]
          end_idx = start_idx + patched_h_k * patched_w_k
          encoder_output_k = encoder_output[:, start_idx:end_idx, :]
          if patched_h_k > 1:
            encoder_output_k = transformers.PatchMerger3D(
                input_resolution=(1, patched_h_k, patched_w_k),
                output_channels=encoder_output_k.shape[-1],
                name=f"patch_merger_{lyr}_{k}",
            )(
                encoder_output_k
            )  # (b, h*w, d) --> (b, h//2*w//2, d)
            patched_shapes[k] = (1, patched_h_k // 2, patched_w_k // 2)
          encoder_output_merged.append(encoder_output_k)
          start_idx = end_idx
        encoder_output = jnp.concatenate(encoder_output_merged, axis=1)

      if self.return_intermediate:
        outputs.append(y)

    if self.return_intermediate:
      y = jnp.stack(outputs, axis=0)
    y = y if self.norm is None else self.norm(y)
    return y


def get_attention_masks(m, num_heads):
  """Constructs attention mask from valid mask."""
  # From (B, P) to (B, num_heads, P, P).
  m = einops.einsum(m, m, "b p, b q -> b p q")
  m = einops.repeat(m, "b p q -> b num_heads p q", num_heads=num_heads)
  return m


def pad_and_concat_sequences(sequences, padding_value=0.0):
  """Pad and concatenate sequences.

  Args:
    sequences: List of sequences of shape [B_k, L_k, D]. B_k and L_k can be
      different for each sequence.
    padding_value: Value to use for padding.

  Returns:
    Padded and concatenated sequence of shape (sum(B_k), max(L_k), D).
    Valid mask of concatenated sequence (sum(B_k), max(L_k)).
  """
  max_len = max(s.shape[1] for s in sequences)
  padded_sequences = []
  valid_masks = []  # 1 = valid pixels, 0 = padded values.
  for s in sequences:
    padded_sequences.append(jnp.pad(
        s, [(0, 0), (0, max_len - s.shape[1]), (0, 0)],
        constant_values=padding_value))
    valid_masks.append(jnp.pad(
        jnp.ones(s.shape[:2]), [(0, 0), (0, max_len - s.shape[1])],
        constant_values=0))
  padded_sequence = jnp.concatenate(padded_sequences, axis=0)
  valid_mask = jnp.concatenate(valid_masks, axis=0)
  return padded_sequence, valid_mask


def pad_and_pack_sequences(sequences, padding_value=0.0):
  """Pad and concatenate sequences.

  Args:
    sequences: List of sequences of shape [B_k, S_k, D]. B_k and S_k can be
      different for each sequence.
    padding_value: Value to use for padding.

  Returns:
    Padded and concatenated sequence of shape (sum(B_k), max(S_k), D).
    Valid mask of concatenated sequence (sum(B_k), max(S_k)).
    Packed shapes for the packed sequences.
  """
  max_len = max(s.shape[1] for s in sequences)
  padded_sequences = []
  valid_masks = []  # 1 = valid pixels, 0 = padded values.
  for s in sequences:
    padded_sequences.append(jnp.pad(
        s, [(0, 0), (0, max_len - s.shape[1]), (0, 0)],
        constant_values=padding_value))
    valid_masks.append(jnp.pad(
        jnp.ones(s.shape[:2]), [(0, 0), (0, max_len - s.shape[1])],
        constant_values=0))
  padded_sequence, ps = einops.pack(padded_sequences, "* s d")
  valid_mask, _ = einops.pack(valid_masks, "* s")
  return padded_sequence, valid_mask, ps


def maybe_make_5d(x, patch_size):
  # Ensures patch_size is [t, h, w] and x is [b, T, H, W, c].
  if len(patch_size) == 2:
    patch_size = [1] + list(patch_size)
  if x is not None:
    x = expand_to_ndim(x, 5)  # Add temporal dimension if needed.
    assert len(patch_size) == x.ndim - 2
  return x, patch_size


def expand_to_ndim(x, ndim, start_at=1):
  """If x is not ndim, appends dimensions staring at `start_at` dimension."""
  if x.ndim > ndim:
    raise ValueError(f"Input x {x.shape} has more than {ndim} dimenstions.")
  if x.ndim < ndim:
    x = jnp.expand_dims(x, tuple(range(start_at, start_at + (ndim - x.ndim))))
  return x
