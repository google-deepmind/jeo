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

"""Swin Transformer V2 implementation.

Based on https://github.com/microsoft/Swin-Transformer.
"""

from collections.abc import Sequence
import enum
import functools
from typing import Any, Callable

import einops
import flax.linen as nn
import jax
import jax.numpy as jnp
from jeo.tools import restore
from jeo.tools import tree_utils
import numpy as np
import optax

Array = jax.Array
LayerNorm = functools.partial(
    nn.LayerNorm,
    epsilon=1e-4,
    # We add a small bias to the LayerNorm to improve the stability of training
    # for dataset with low magnitude inputs. LN gradients would otherwise
    # explode (see (internal link) for details).
    bias_init=nn.initializers.truncated_normal(stddev=1e-2),
)
Dense = functools.partial(nn.Dense, bias_init=nn.initializers.zeros)
Conv = functools.partial(
    nn.Conv, kernel_init=nn.initializers.truncated_normal()
)


class ReductionType(enum.Enum):
  MEAN = 'mean'
  LAST = 'last'
  FIRST = 'first'
  CONV = 'conv'
  NONE = 'none'


class TemporalReduction(nn.Module):
  """Supported temporal reduction types for factorized attention models."""

  reduction_type: ReductionType

  @nn.compact
  def __call__(self, x: Array) -> Array:
    if self.reduction_type == ReductionType.LAST:
      return x[:, -1:]
    elif self.reduction_type == ReductionType.FIRST:
      return x[:, :1]
    elif self.reduction_type == ReductionType.MEAN:
      return jnp.mean(x, axis=1, keepdims=True)
    elif self.reduction_type == ReductionType.CONV:
      t = x.shape[1]
      return nn.Conv(
          features=x.shape[-1], kernel_size=(t, 1, 1), padding='VALID'
      )(x)
    elif self.reduction_type == ReductionType.NONE:
      return x
    else:
      raise ValueError(f'Unsupported reduction type: {self.reduction_type}')


def apply_norm(x: Array, norm_type: str, train: bool):
  if norm_type == 'batch':
    return nn.BatchNorm(use_running_average=not train, momentum=0.9)(x)
  if norm_type == 'layer':
    return LayerNorm()(x)
  return x


def pad_to_be_divisible(x: Array, divisor: int, axis: int) -> Array:
  """Pads a single axis of an array to be divisible by a divisor."""
  axis_len = x.shape[axis]
  if axis_len % divisor == 0:
    return x
  pad_amount = (divisor - axis_len % divisor) % divisor
  pad_width = [(0, 0)] * x.ndim
  pad_width[axis] = (0, pad_amount)
  return jnp.pad(x, pad_width)


def pad_len_to_be_divisible(tensor_len: int, divisor: int) -> int:
  """Calculates the padded length needed for divisibility."""
  return tensor_len + (divisor - tensor_len % divisor) % divisor


def _window_partition(x: Array, window_size: tuple[int, int, int]) -> Array:
  """Partitions a 3D input into 3D windows."""
  wd, wh, ww = window_size
  if not (wd > 0 and wh > 0 and ww > 0):
    raise ValueError('window_size dimensions must be positive')

  # Pad the input if the dimension is not exactly divisible by the window size.
  for i, size in enumerate(window_size):
    if size > 1:
      x = pad_to_be_divisible(x, size, i + 1)
  rearrange_str = 'b (nwd wd) (nwh wh) (nww ww) c -> (b nwd nwh nww) wd wh ww c'
  return einops.rearrange(x, rearrange_str, wd=wd, wh=wh, ww=ww)


def _window_reverse(
    windows: Array,
    window_size: tuple[int, int, int],
    orig_size: tuple[int, int, int],
) -> Array:
  """Reverses the window partitioning."""
  wd, wh, ww = window_size
  od, oh, ow = orig_size

  pd = pad_len_to_be_divisible(od, wd)
  ph = pad_len_to_be_divisible(oh, wh)
  pw = pad_len_to_be_divisible(ow, ww)

  num_windows_d = pd // wd
  num_windows_h = ph // wh
  num_windows_w = pw // ww
  b = windows.shape[0] // (num_windows_d * num_windows_h * num_windows_w)

  x = windows.reshape(
      (b, num_windows_d, num_windows_h, num_windows_w, wd, wh, ww, -1)
  )
  x = x.transpose((0, 1, 4, 2, 5, 3, 6, 7)).reshape((b, pd, ph, pw, -1))
  return x[:, :od, :oh, :ow, :]


def _create_window_attention_mask(
    shift_size: tuple[int, int, int],
    input_resolution: tuple[int, int, int],
    window_size: tuple[int, int, int],
) -> Array | None:
  """Creates attention mask for shifted windows."""
  sd, sh, sw = shift_size
  if sd == 0 and sh == 0 and sw == 0:
    return None
  wd, wh, ww = window_size

  img_mask = jnp.zeros((1, *input_resolution, 1))  # B, D, H, W, C
  # Define slices for each dimension based on shift
  # If shift is 0 for a dim, only one slice covers the whole dim. Otherwise, 3
  # slices: (0, -win_dim), (-win_dim, -shift_dim), (-shift_dim, None)
  slices_d = (
      (slice(0, -wd), slice(-wd, -sd), slice(-sd, None))
      if sd != 0 and wd > sd
      else (slice(0, input_resolution[0]),)
  )
  slices_h = (
      (slice(0, -wh), slice(-wh, -sh), slice(-sh, None))
      if sh != 0 and wh > sh
      else (slice(0, input_resolution[1]),)
  )
  slices_w = (
      (slice(0, -ww), slice(-ww, -sw), slice(-sw, None))
      if sw != 0 and ww > sw
      else (slice(0, input_resolution[2]),)
  )

  cnt = 0
  for ds in slices_d:
    for hs in slices_h:
      for ws in slices_w:
        img_mask = img_mask.at[:, ds, hs, ws, :].set(cnt)
        cnt += 1

  # (num_total_win * 1, WD, WH, WW, 1)
  mask_windows = _window_partition(img_mask, window_size)
  # (num_total_win, WD*WH*WW)
  mask_windows = mask_windows.reshape((-1, wd * wh * ww))
  # (num_total_win, N_elems, N_elems)
  attn_mask = mask_windows[:, None, :] - mask_windows[:, :, None]
  # True where attention is allowed (same segment ID)
  attn_mask = jnp.array(attn_mask == 0)
  return attn_mask.astype(jnp.float32)


def get_relative_position_index(window_size: tuple[int, int, int]) -> Array:
  """Computes the relative position index for each position in the window.

  Args:
    window_size: A tuple of three integers representing the window size in
      dimensions (D, H, W).

  Returns:
    A 1D array of size (Wd*Wh*Ww)**2, where each entry corresponds to a unique
    relative position index.
  """
  # [3, Wd, Wh, Ww]
  coords_grid = jnp.meshgrid(*map(jnp.arange, window_size), indexing='ij')
  # [3, Wd*Wh*Ww]
  coords = jnp.stack(coords_grid, axis=0).reshape(3, -1)
  # [Wd*Wh*Ww, Wd*Wh*Ww, 3] with values in range
  # `-window_size[i] + 1` to `window_size[i] - 1`
  coords = (coords[:, :, None] - coords[:, None, :]).transpose(1, 2, 0)

  # Mapping to range `[0, 2 * (window_size[i] - 1)]`
  coords = coords.at[:, :, 0].set(coords[:, :, 0] + window_size[0] - 1)
  coords = coords.at[:, :, 1].set(coords[:, :, 1] + window_size[1] - 1)
  coords = coords.at[:, :, 2].set(coords[:, :, 2] + window_size[2] - 1)

  # Multiplication factors to create unique indices.
  idx1 = 2 * window_size[1] - 1
  idx2 = 2 * window_size[2] - 1
  coords = coords.at[:, :, 0].set(coords[:, :, 0] * idx1 * idx2)
  coords = coords.at[:, :, 1].set(coords[:, :, 1] * idx2)

  return coords.sum(-1).astype(jnp.int32).reshape(-1)  # (Wd*Wh*Ww)**2


def get_relative_coords_table_const(
    window_size: tuple[int, int, int],
    pretrained_window_size: tuple[int, int, int] | None,
) -> Array:
  """Computes the relative coordinates table for the window."""
  relative_coords_d = jnp.arange(-(window_size[0] - 1), window_size[0])
  relative_coords_h = jnp.arange(-(window_size[1] - 1), window_size[1])
  relative_coords_w = jnp.arange(-(window_size[2] - 1), window_size[2])
  grid = jnp.meshgrid(
      relative_coords_d, relative_coords_h, relative_coords_w, indexing='ij'
  )
  # 1, (2Wd-1), (2Wh-1), (2Ww-1), 3
  coords = jnp.expand_dims(jnp.stack(grid, axis=-1), axis=0)

  # Normalize by pretrained_window_size or current window_size
  norms = pretrained_window_size or window_size
  for i in range(3):
    coords = coords.at[:, ..., i].set(coords[:, ..., i] / norms[i])

  # Logarithmic scaling factor
  coords = jnp.sign(coords) * jnp.log2(jnp.abs(8 * coords) + 1.0) / np.log2(8)
  return coords.astype(jnp.float32)  # Shape (1, 2Wd-1, 2Wh-1, 2Ww-1, 3)


class MLP(nn.Module):
  """Simple MLP block."""

  hidden_features: int
  out_features: int
  dropout_rate: float = 0.0

  @nn.compact
  def __call__(self, x: Array, deterministic: bool) -> Array:
    x = Dense(features=self.hidden_features)(x)
    x = nn.gelu(x)
    if self.dropout_rate > 0.0:
      x = nn.Dropout(rate=self.dropout_rate, deterministic=deterministic)(x)
    x = Dense(features=self.out_features)(x)
    if self.dropout_rate > 0.0:
      x = nn.Dropout(rate=self.dropout_rate, deterministic=deterministic)(x)
    return x


class WindowAttention(nn.Module):
  """Window multi-head self attention."""

  num_heads: int
  window_size: tuple[int, int, int]
  cpb_mlp: bool
  pretrained_window_size: tuple[int, int, int] | None = None
  qkv_bias: bool = True
  attn_drop_rate: float = 0.0
  proj_drop_rate: float = 0.0

  @nn.compact
  def __call__(
      self, x: Array, *, attn_mask: Array | None, deterministic: bool
  ) -> Array:
    """Forward pass with Scaled Cosine Attention and Log-CPB."""
    b, n, c = x.shape  # (num_windows*B), (WD*WH*WW), C
    head_dim = c // self.num_heads

    ### Scaled Cosine Attention
    logit_scale = self.param(
        'logit_scale',
        nn.initializers.constant(jnp.log(10.0)),
        (self.num_heads, 1, 1),
        jnp.float32,
    )
    qkv = Dense(features=c * 3, use_bias=False, name='qkv')(x)
    if self.qkv_bias:
      qkv_parts = jnp.split(qkv, 3, axis=-1)
      q = qkv_parts[0] + self.param('q_bias', nn.initializers.zeros, (1, 1, c))
      k = qkv_parts[1]  # No bias for K as suggested in the paper appendix.
      v = qkv_parts[2] + self.param('v_bias', nn.initializers.zeros, (1, 1, c))
      qkv = jnp.concatenate([q, k, v], axis=-1)
    # each (num_windows*B), num_heads, N, head_dim
    q, k, v = qkv.reshape(b, n, 3, self.num_heads, head_dim).transpose(
        2, 0, 3, 1, 4
    )
    # Safe norm to avoid zero denominator.
    q_norm = optax.safe_norm(q, 1e-5, axis=-1, keepdims=True)
    k_norm = optax.safe_norm(k, 1e-5, axis=-1, keepdims=True)
    # Normalise and rescale.
    attn = (q / q_norm) @ (k / k_norm).transpose((0, 1, 3, 2))
    current_tau = jnp.exp(jnp.clip(logit_scale, max=jnp.log(100)))
    attn = attn * current_tau[None, ...]  # Apply learnable scale

    ### Log-spaced Continuous Position Bias
    if self.cpb_mlp:
      # [2Wd-1, 2Wh-1, 2Ww-1, 3]
      cpb = get_relative_coords_table_const(
          self.window_size, self.pretrained_window_size
      )
      # [(2Wd-1)*(2Wh-1)*(2Ww-1), num_heads]
      cpb = Dense(features=512, name='cpb_1')(cpb)
      cpb = nn.relu(cpb)
      cpb = Dense(features=self.num_heads, use_bias=False, name='cpb_2')(cpb)
      cpb = cpb.reshape(-1, self.num_heads)
    else:
      cpb = self.param(
          'cpb',
          nn.initializers.zeros,
          (
              (2 * self.window_size[0] - 1)
              * (2 * self.window_size[1] - 1)
              * (2 * self.window_size[2] - 1),
              self.num_heads,
          ),
          jnp.float32,
      )
    # [num_heads, Wd*Wh*Ww, Wd*Wh*Ww]
    rel_pos_idx_flat = get_relative_position_index(self.window_size)
    cpb = cpb[rel_pos_idx_flat]
    n = self.window_size[0] * self.window_size[1] * self.window_size[2]
    cpb = cpb.reshape(n, n, self.num_heads).transpose(2, 0, 1)
    # Add continuous position bias.
    attn = attn + (16 * nn.sigmoid(cpb))

    # Apply window attention mask (if provided)
    if attn_mask is not None:
      # Reshape mask for broadcasting (N = window_size*window_size)
      # Assumes mask is [num_mask_windows, N, N] or tiled version [B, N, N]
      if attn_mask.shape[0] != b:
        num_windows_per_mask = b // attn_mask.shape[0]
        attn_mask = jnp.tile(attn_mask, (num_windows_per_mask, 1, 1))
      # Add head dim
      attn_mask = attn_mask[:, None, :, :]  # [B, 1, N, N]
      # Apply mask where False. -inf is converted to zero in softmax.
      attn = jnp.where(attn_mask, attn, -jnp.inf)

    ## Window Attention
    attn = nn.softmax(attn, axis=-1)
    attn = nn.Dropout(
        rate=self.attn_drop_rate,
        deterministic=deterministic,
        name='attn_dropout',
    )(attn)
    x = (attn @ v).transpose((0, 2, 1, 3)).reshape((b, n, c))
    x = Dense(features=c, name='proj')(x)
    x = nn.Dropout(
        rate=self.proj_drop_rate,
        deterministic=deterministic,
        name='proj_dropout',
    )(x)

    return x


class PatchMerging(nn.Module):
  """Patch Merging Layer."""

  resolution: tuple[int, int, int] = (1, 2, 2)
  norm_type: str | None = 'layer'

  @nn.compact
  def __call__(self, x: Array, *, train: bool) -> Array:
    *_, c = x.shape
    pd, ph, pw = self.resolution
    # [B, D/pd, H/pd, W/pw, 4*C]
    rearrange_str = 'b (d pd) (h ph) (w pw) c -> b d h w (pd ph pw c)'
    x = einops.rearrange(x, rearrange_str, pd=pd, ph=ph, pw=pw)
    x = Dense(features=2 * c, use_bias=False, name='proj')(x)
    x = apply_norm(x, self.norm_type, train)
    return x


class PatchExpand(nn.Module):
  """Patch Expand Layer."""

  resolution: tuple[int, int, int] = (1, 2, 2)
  norm_type: str | None = 'layer'

  @nn.compact
  def __call__(self, x: Array, *, train: bool) -> Array:
    *_, c = x.shape
    x = Dense(features=2 * c, use_bias=False, name='proj')(x)
    x = apply_norm(x, self.norm_type, train)
    pd, ph, pw = self.resolution
    # [B, pd*D, ph*H, pw*W, C/2]
    rearrange_str = 'b d h w (pd ph pw c) -> b (d pd) (h ph) (w pw) c'
    x = einops.rearrange(x, rearrange_str, pd=pd, ph=ph, pw=pw)
    return x


class StochasticDepth(nn.Module):
  """Stochastic Depth / Drop Path."""

  drop_rate: float

  @nn.compact
  def __call__(self, x: Array, deterministic: bool) -> Array:
    if deterministic or self.drop_rate == 0.0:
      return x
    keep_prob = 1.0 - self.drop_rate
    rng = self.make_rng('dropout')
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = jax.random.uniform(rng, shape, dtype=x.dtype)
    binary_tensor = jnp.floor(keep_prob + random_tensor)
    return (x / keep_prob) * binary_tensor


class SwinTransformerBlock(nn.Module):
  """A SWIN transformer V2 block."""

  num_heads: int
  cpb_mlp: bool
  norm_type: str | None
  window_size: tuple[int, int, int]
  shift_size: tuple[int, int, int]
  mlp_ratio: float = 4.0
  qkv_bias: bool = True
  drop_path_rate: float = 0.0
  attn_drop_rate: float = 0.0
  mlp_drop_rate: float = 0.0

  @nn.compact
  def __call__(self, x: Array, deterministic: bool) -> Array:
    _, d, h, w, c = x.shape
    input_resolution = (d, h, w)
    shortcut = x

    # Determine effective window_size and shift_size for each dimension
    eff_ws = []
    eff_ss = []
    any_dim_shifted = False
    for i in range(3):  # D, H, W
      if input_resolution[i] <= self.window_size[i]:
        eff_ws.append(input_resolution[i])
        eff_ss.append(0)  # No shift if window is as large as input dim
      else:
        eff_ws.append(self.window_size[i])
        eff_ss.append(self.shift_size[i])
        if self.shift_size[i] > 0:
          any_dim_shifted = True
        # Basic validation for non-degenerate cases
        if not (0 <= self.shift_size[i] < self.window_size[i]):
          raise ValueError(
              f'shift_size[{i}] must be < window_size[{i}] if input_res[{i}] >'
              f' window_size[{i}]'
          )
    effective_window_size = tuple(eff_ws)
    effective_shift_size = tuple(eff_ss)

    attn_mask = None
    # Create mask only if there's a non-zero shift in any dimension
    if any_dim_shifted:
      attn_mask = _create_window_attention_mask(
          shift_size=effective_shift_size,
          input_resolution=input_resolution,
          window_size=effective_window_size,
      )

    ### Attention Block
    if any_dim_shifted:
      roll_shifts = tuple(-s for s in effective_shift_size)
      x = jnp.roll(x, shift=roll_shifts, axis=(1, 2, 3))
    # Partition windows
    x = _window_partition(x, effective_window_size)
    x = x.reshape((-1, np.prod(effective_window_size), c))
    # W-MSA / SW-MSA
    x = WindowAttention(
        num_heads=self.num_heads,
        window_size=effective_window_size,
        qkv_bias=self.qkv_bias,
        attn_drop_rate=self.attn_drop_rate,
        proj_drop_rate=self.mlp_drop_rate,
        cpb_mlp=self.cpb_mlp,
    )(x, attn_mask=attn_mask, deterministic=deterministic)
    # Merge windows back
    x = x.reshape((-1, *effective_window_size, c))
    x = _window_reverse(x, effective_window_size, input_resolution)
    # Reverse cyclic shift
    if any_dim_shifted:
      x = jnp.roll(x, shift=effective_shift_size, axis=(1, 2, 3))
    x = apply_norm(x, self.norm_type, not deterministic)
    x = StochasticDepth(self.drop_path_rate)(x, deterministic=deterministic)
    x = shortcut + x

    ### FFN Block
    shortcut = x
    x = MLP(int(c * self.mlp_ratio), c, self.mlp_drop_rate)(
        x, deterministic=deterministic
    )
    x = apply_norm(x, self.norm_type, not deterministic)
    x = StochasticDepth(self.drop_path_rate)(x, deterministic=deterministic)
    x = shortcut + x
    return x


def _check_shapes(x: Array, patch_size: tuple[int, int, int], mod: str):
  input_spatial_shape = x.shape[1:4]
  for i in range(3):
    if input_spatial_shape[i] % patch_size[i] != 0:
      raise ValueError(
          f'input_spatial_shape[{i}] ({input_spatial_shape[i]}) must be'
          f' divisible by patch_size[{i}] ({patch_size[i]})'
          f' input shape: {x.shape} for modality: {mod}'
      )


class Embed(nn.Module):
  """Image to Patch Embedding."""

  patch_size: tuple[int, int, int] | dict[str, tuple[int, int, int]]
  embed_dim: int
  norm_type: str | None
  compute_patches: bool

  @nn.compact
  def __call__(
      self, x: Array | dict[str, Array], *, train: bool
  ) -> Array | dict[str, Array]:
    embed_module = lambda ps: nn.Sequential([
        nn.Conv(
            # No padding needed due to divisibility check
            features=self.embed_dim,
            kernel_size=ps,
            strides=ps if self.compute_patches else (1, 1, 1),
            padding='VALID' if self.compute_patches else 'SAME',
        ),
        functools.partial(apply_norm, norm_type=self.norm_type, train=train),
    ])

    if isinstance(x, dict):
      for k, v in x.items():
        if self.compute_patches:
          _check_shapes(v, self.patch_size[k], k)
        x[k] = embed_module(self.patch_size[k])(v)
      # TODO: we are currently assuming that all the modalities
      # produce the same embedding dimension. This may not be the case.
      x = jnp.concatenate(list(x.values()), axis=-1)
    else:
      if self.compute_patches:
        _check_shapes(x, self.patch_size, 'image')
      x = embed_module(self.patch_size)(x)

    # Output shape: B, D/P, H/P, W/P, C
    return x


class SwinBlock(nn.Module):
  """One stage of the Swin Transformer V2."""

  depth: int
  num_heads: int
  cpb_mlp: bool
  norm_type: str | None
  window_size: tuple[int, int, int]
  mlp_ratio: float = 4.0
  qkv_bias: bool = True
  drop_path_rates: Sequence[float] = ()
  attn_drop_rate: float = 0.0
  mlp_drop_rate: float = 0.0
  add_final_conv: bool = False

  @nn.compact
  def __call__(self, x: Array, deterministic: bool) -> Array:
    """Applies Swin blocks and optional downsampling."""
    if len(self.drop_path_rates) != self.depth:
      raise ValueError(
          f'drop_path_rates length mismatch: {len(self.drop_path_rates)} vs'
          f' {self.depth}'
      )
    shortcut = x  # only needed when adding a final conv block

    for i in range(self.depth):
      shift_size = tuple(
          ws // 2 if i % 2 != 0 else 0 for ws in self.window_size
      )
      x = SwinTransformerBlock(
          num_heads=self.num_heads,
          window_size=self.window_size,
          shift_size=shift_size,
          mlp_ratio=self.mlp_ratio,
          qkv_bias=self.qkv_bias,
          drop_path_rate=self.drop_path_rates[i],
          attn_drop_rate=self.attn_drop_rate,
          mlp_drop_rate=self.mlp_drop_rate,
          cpb_mlp=self.cpb_mlp,
          norm_type=self.norm_type,
      )(x, deterministic=deterministic)
    if self.add_final_conv:
      x = Conv(x.shape[-1], (1, 3, 3))(x)
      x = shortcut + x
    return x


class SwinTransformerV2(nn.Module):
  """Swin Transformer V2."""

  depths: Sequence[int]
  num_heads: Sequence[int]
  window_size: int | tuple[int, int, int] | Sequence[tuple[int, int, int]]
  mlp_ratio: float
  qkv_bias: bool
  attn_drop_rate: float
  mlp_drop_rate: float
  drop_path_rate: float
  cpb_mlp: bool
  add_final_conv: bool
  patch_module: Callable[[], PatchMerging | PatchExpand] | None
  norm_type: str | None
  temporal_reduction_after_stage: int = -1
  reduction_type: ReductionType | None = None

  @nn.compact
  def __call__(
      self,
      x: Array,
      *,
      skips: Sequence[Array] | None = None,
      deterministic: bool,
  ) -> tuple[Array, list[Array]]:

    dpr_offset = 0
    num_layers = len(self.depths)
    dpr = list(
        np.linspace(0, self.drop_path_rate, sum(self.depths))
    )  # Stochastic depth decay rule

    # Build normalized per-layer window sizes.
    if isinstance(self.window_size, int):
      window_sizes = [(self.window_size,) * 3] * num_layers
    elif not isinstance(self.window_size, Sequence):
      raise ValueError(f'Invalid window_size type: {type(self.window_size)}')
    elif len(self.window_size) == 0:
      raise ValueError('window_size cannot be empty')
    elif isinstance(self.window_size[0], int):
      if len(self.window_size) != 3:
        raise ValueError(
            'Single window_size must have length 3, got'
            f' {len(self.window_size)}'
        )
      window_sizes = [self.window_size] * num_layers
    elif isinstance(self.window_size[0], Sequence):
      if len(self.window_size) != num_layers:
        raise ValueError(
            f'window_size length mismatch: {len(self.window_size)} vs'
            f' {num_layers}'
        )
      for i, ws in enumerate(self.window_size):
        if len(ws) != 3:
          raise ValueError(
              f'window_size at index {i} must have length 3, got {len(ws)}'
          )
      window_sizes = self.window_size
    else:
      raise ValueError(
          f'Invalid window_size element type: {type(self.window_size[0])}'
      )

    ### Swin Layers
    layer_feature_outputs = []
    for i_layer in range(num_layers):
      layer_depth = self.depths[i_layer]
      layer_dpr = dpr[dpr_offset : dpr_offset + layer_depth]
      dpr_offset += layer_depth

      if skips and i_layer > 1:
        x = jnp.concatenate([x, skips[i_layer - 1]], axis=-1)
        x = Dense(skips[i_layer - 1].shape[-1])(x)

      x = SwinBlock(
          depth=layer_depth,
          num_heads=self.num_heads[i_layer],
          window_size=window_sizes[i_layer],
          mlp_ratio=self.mlp_ratio,
          qkv_bias=self.qkv_bias,
          drop_path_rates=layer_dpr,
          attn_drop_rate=self.attn_drop_rate,
          mlp_drop_rate=self.mlp_drop_rate,
          cpb_mlp=self.cpb_mlp,
          add_final_conv=self.add_final_conv,
          norm_type=self.norm_type,
      )(x, deterministic=deterministic)

      if self.reduction_type is not None:
        if self.temporal_reduction_after_stage == i_layer or (
            self.temporal_reduction_after_stage == -1
            and i_layer == num_layers - 1
        ):
          x = TemporalReduction(reduction_type=self.reduction_type)(x)

      if i_layer < num_layers - 1:
        layer_feature_outputs.insert(0, x)
        if self.patch_module:
          x = self.patch_module()(x, train=not deterministic)

    return x, layer_feature_outputs


class Logits(nn.Module):
  """Logits."""

  num_classes: int | dict[str, int]
  prelogits: Callable[[Array], Array] | None = None

  @nn.compact
  def __call__(
      self, x: Array, aux: dict[str, Array]
  ) -> tuple[Array | None, dict[str, Array]]:
    if isinstance(self.num_classes, int):
      x = aux['prelogits'] = self.prelogits(x) if self.prelogits else x
      x = Dense(self.num_classes, bias_init=nn.initializers.zeros)(x)
      return x, aux
    for name, num_classes in self.num_classes.items():
      y = aux[f'{name}_prelogits'] = self.prelogits(x) if self.prelogits else x
      aux[name] = Dense(num_classes, bias_init=nn.initializers.zeros)(y)
    return None, aux


class ConvSegmentationHead(nn.Module):
  """Conv+Bilinear decoder."""

  num_classes: int | dict[str, int] | None
  patch_size: tuple[int, int, int]
  norm_type: str | None  # ['batch', 'layer', None]
  temporal_reduction: str = 'mean'  # ['mean', 'last']

  @nn.compact
  def __call__(
      self,
      x: Array,
      aux: dict[str, Array],
      skips: Sequence[Array],
      *,
      train: bool,
  ) -> tuple[Array | None, dict[str, Array]]:

    # Reduce temporal dimension.
    # skips: List[(B, T, H, W, C)] -> List[(B, H, W, C)]
    if self.temporal_reduction == 'last':
      # Select last time slice: x: (B, T, H', W', C') -> (B, H', W', C')
      x = x[:, -1]
      skips = [skip[:, -1] for skip in skips]
    elif self.temporal_reduction == 'mean':
      # Average over time axis: x: (B, T, H', W', C') -> (B, H', W', C')
      x = jnp.mean(x, axis=1)
      skips = [jnp.mean(skip, axis=1) for skip in skips]
    elif self.temporal_reduction == 'conv':
      # Conv over time axis: x: (B, T, H', W', C') -> (B, H', W', C')
      t = x.shape[1]
      x = Conv(x.shape[-1], (t, 1, 1), padding='VALID', name='temporal_conv')(x)
      x = jnp.squeeze(x, axis=1)

      skips_reduced = []
      for i, skip in enumerate(skips):
        skip_reduced = Conv(
            skip.shape[-1],
            (t, 1, 1),
            padding='VALID',
            name=f'skip_temporal_conv_{i}',
        )(skip)
        skips_reduced.append(skip_reduced[:, 0])
      skips = skips_reduced
    else:
      raise ValueError(
          f'Unsupported temporal_reduction: {self.temporal_reduction}'
      )

    ### Decoder Blocks with skip connections
    for i, skip_features in enumerate(skips):
      skip_c = skip_features.shape[-1]

      # Upsample and incorporate skip features.
      out_shape = x.shape[0], 2 * x.shape[1], 2 * x.shape[2], skip_c
      x = jax.image.resize(x, out_shape, method='bilinear')
      x = jnp.concatenate([x, skip_features], axis=-1)
      x = Dense(skip_c)(x)

      # CNN block.
      x = Conv(skip_c, (3, 3))(x)
      x = apply_norm(x, self.norm_type, train)
      x = aux[f'skip_up_{i}'] = nn.gelu(x)

    # Final upsampling.
    b, h, w, c = x.shape
    _, ph, pw = self.patch_size
    shape = b, ph * h, pw * w, c

    def _final_upsample(x):
      x = jax.image.resize(x, shape=shape, method='bilinear')
      x = Conv(c, (3, 3))(x)
      x = apply_norm(x, self.norm_type, train)
      x = nn.gelu(x)
      return x

    if self.num_classes is None:
      return _final_upsample(x), aux
    return Logits(self.num_classes, _final_upsample)(x, aux)


class SwinSegmentationHead(nn.Module):
  """Swin-Unet Decoder."""

  num_classes: int | dict[str, int] | None
  patch_size: tuple[int, int, int]
  swin_kwargs: dict[str, Any]
  conv_upsample: bool
  conv_norm: str | None  # ['batch', 'layer', None]

  @nn.compact
  def __call__(
      self,
      x: Array,
      aux: dict[str, Array],
      skips: Sequence[Array],
      *,
      train: bool,
  ) -> tuple[Array | None, dict[str, Array]]:

    swin_kwargs = {k: v for k, v in self.swin_kwargs.items()}
    swin_kwargs['depths'] = list(reversed(swin_kwargs['depths']))
    swin_kwargs['num_heads'] = list(reversed(swin_kwargs['num_heads']))
    norm_type = swin_kwargs['norm_type']
    patch_module = functools.partial(PatchExpand, norm_type=norm_type)
    # Swin Up with reversed depths and heads.
    x, upskip = SwinTransformerV2(
        patch_module=patch_module, **swin_kwargs, name='swin_up'
    )(x, skips=skips, deterministic=not train)
    aux['up_out'] = x
    for i, upskip in enumerate(upskip):
      aux[f'skip_up_{len(upskip) - i - 1}'] = upskip
    x = jnp.mean(x, axis=1, keepdims=not self.conv_upsample)

    def _final_upsample(y: Array):
      if self.conv_upsample:
        # Upsample with bilinear interpolation.
        b, h, w, c = y.shape
        _, ph, pw = self.patch_size
        shape = b, ph * h, pw * w, c
        y = jax.image.resize(y, shape=shape, method='bilinear')
      else:
        # Upsample with pixel shuffle.
        patch_size = (1, *self.patch_size[1:])  # no need of temporal dimension
        y = PatchExpand(patch_size, self.conv_norm)(y, train=train)
        y = jnp.squeeze(y, axis=1)
      y = Conv(y.shape[-1], (3, 3))(y)
      y = apply_norm(y, self.conv_norm, train)
      y = nn.gelu(y)
      return y

    if self.num_classes is None:
      return _final_upsample(x), aux
    return Logits(self.num_classes, _final_upsample, name='segmentation_head')(
        x, aux
    )


class ClassificationHead(nn.Module):
  """Classification head."""

  num_classes: int | dict[str, int]

  @nn.compact
  def __call__(
      self,
      x: Array,
      aux: dict[str, Array],
      skips: Sequence[Array],
      *,
      train: bool,
  ) -> tuple[Array | None, dict[str, Array]]:
    # Scale all feature maps to match the largest one and concatenate them.
    skips = [
        jax.image.resize(skip, shape=x.shape, method='linear') for skip in skips
    ]
    x = jnp.concatenate(skips + [x], axis=-1)
    x = aux['prelogits'] = x.mean(axis=(1, 2, 3))  # avg pool

    return Logits(self.num_classes, name='classification_head')(x, aux)


class SuperResHead(nn.Module):
  """SwinIR with V2 updated features (https://arxiv.org/abs/2209.11345)."""

  num_classes: int | dict[str, int]
  upscale_factor: int
  conv_norm: str | None  # ['batch', 'layer', None]

  @nn.compact
  def __call__(
      self, x: Array, aux: dict[str, Array], *, train: bool
  ) -> tuple[Array | None, dict[str, Array]]:
    # Reduce temporal dimension.
    shallow_features = jnp.mean(aux['embedded'], axis=1)
    image_size = x.shape[-2]
    x = jnp.mean(x, axis=1)

    # Add residual connection with shallow features.
    x = Conv(x.shape[-1], (1, 3, 3))(x) + shallow_features
    x = apply_norm(x, self.conv_norm, train)
    x = jax.nn.leaky_relu(x, negative_slope=0.2)

    # Conv layer before upsampling.
    x = Conv(64, kernel_size=(3, 3), padding='SAME')(x)
    x = apply_norm(x, self.conv_norm, train)
    x = jax.nn.leaky_relu(x, negative_slope=0.2)

    def _final_upsample(y: Array):
      # ConvNet upsampling is more common in superesolution sota models.
      while y.shape[1] < image_size * self.upscale_factor:
        out_shape = y.shape[0], 2 * y.shape[1], 2 * y.shape[2], y.shape[3]
        y = jax.image.resize(y, out_shape, method='bilinear')
        y = Conv(64, kernel_size=(3, 3), padding='SAME')(y)
        y = apply_norm(y, self.conv_norm, train)
        y = jax.nn.leaky_relu(y, negative_slope=0.2)
      return y

    return Logits(self.num_classes, _final_upsample, name='super_res_head')(
        x, aux
    )


def _maybe_make_5d(x: Array):
  if x.ndim == 4:
    x = jnp.expand_dims(x, axis=1)
  if x.ndim != 5:
    raise ValueError(f'Input x must be 5D (B,D,H,W,C), got {x.shape}')
  return x


class SwinV2Model(nn.Module):
  """SwinV2 model."""

  # Encoder Config
  patch_size: tuple[int, int, int] | dict[str, tuple[int, int, int]] = (1, 4, 4)
  embed_dim: int = 128
  depths: Sequence[int] = (2, 2, 18, 2)
  num_heads: Sequence[int] = (4, 8, 16, 32)
  window_size: tuple[int, int, int] | int = 7
  mlp_ratio: float = 4.0
  qkv_bias: bool = True
  attn_drop_rate: float = 0.0
  mlp_drop_rate: float = 0.0
  drop_path_rate: float = 0.1
  cpb_mlp: bool = True
  add_final_conv: bool = False
  encoder_norm: str | None = 'layer'  # ['batch', 'layer', None]
  mods: Sequence[str] = ()
  embed_module: Callable[..., nn.Module] | None = None

  # Head Config
  head_type: str = ''
  num_classes: int = 1
  decoder_channels: Sequence[int] = ()  # or (512, 256, 128)
  conv_upsample: bool = False
  conv_norm: str | None = 'batch'  # ['batch', 'layer', None]
  upscale_factor: int = 0
  # temporal_reduction configures the temporal reduction strategy used in the
  # ConvSegmentationHead decoder.
  temporal_reduction: str = 'mean'  # ['mean', 'last', 'conv']
  # encoder_temporal_reduction configures the temporal reduction strategy used
  # in the Swin Backbone encoder.
  encoder_temporal_reduction: str = 'none'
  # temporal_reduction_after_stage specifies after which stage in the Swin
  # Backbone encoder the temporal reduction is applied. Defaults to -1, which
  # applies temporal reduction after the last stage.
  temporal_reduction_after_stage: int = -1

  @nn.compact
  def __call__(
      self, x: Array | dict[str, Array], *, train: bool
  ) -> tuple[Array | Sequence[Array] | None, dict[str, Array]]:
    """Forward pass through encoder and decoder."""
    if isinstance(x, dict):
      mods = self.mods or list(x.keys())  # Take input keys as mods by default.

      # Convert patch_size to a dict if it's not already when input is a dict.
      if not hasattr(self.patch_size, 'keys'):
        patch_size = {m: self.patch_size for m in mods}
      else:
        patch_size = self.patch_size

      if not all(k in x and k in patch_size for k in mods):
        raise ValueError(
            f'One of {mods} is not in inputs {x.keys()} '
            f'or patch sizes {patch_size.keys()}'
        )

      x = {k: _maybe_make_5d(x[k]) for k in mods}
      out_patch_size = patch_size[mods[0]]
    else:
      x = _maybe_make_5d(x)
      patch_size = out_patch_size = self.patch_size

    if isinstance(self.window_size, int):
      window_size = (self.window_size,) * 3
    else:
      window_size = self.window_size

    aux = {}
    if self.head_type == 'super_res':
      assert self.upscale_factor > 0, 'upscale_factor must be specified.'
      assert self.add_final_conv, 'add_final_conv must be True.'
      # For the SwinIR we don't downsample at each stage.
      down_module = None
      # For the SwinIR we compute shallow features instead of creating patch
      # embeddings.
      compute_patches = False
    else:
      compute_patches = True
      down_module = functools.partial(PatchMerging, norm_type=self.encoder_norm)
      assert self.patch_size, 'patch_size must be specified.'

    embed_cls = self.embed_module or Embed
    x = aux['embedded'] = embed_cls(
        patch_size=patch_size,
        embed_dim=self.embed_dim,
        norm_type=self.encoder_norm,
        compute_patches=compute_patches,
    )(x, train=train)

    swin_kwargs = dict(
        depths=self.depths,
        num_heads=self.num_heads,
        window_size=window_size,
        mlp_ratio=self.mlp_ratio,
        qkv_bias=self.qkv_bias,
        attn_drop_rate=self.attn_drop_rate,
        mlp_drop_rate=self.mlp_drop_rate,
        drop_path_rate=self.drop_path_rate,
        cpb_mlp=self.cpb_mlp,
        add_final_conv=self.add_final_conv,
        norm_type=self.encoder_norm,
        temporal_reduction_after_stage=self.temporal_reduction_after_stage,
        reduction_type=ReductionType(self.encoder_temporal_reduction)
        if self.encoder_temporal_reduction
        else None,
    )

    # Swin Down
    x, skips = SwinTransformerV2(
        patch_module=down_module, **swin_kwargs, name='swin_down'
    )(x, deterministic=not train)

    aux['bottleneck'] = x
    for i, sk in enumerate(skips):
      aux[f'skip_down_{len(skips) - i - 1}'] = sk

    if self.head_type == 'classification':
      return ClassificationHead(self.num_classes)(x, aux, skips, train=train)

    if self.head_type == 'segmentation':
      return SwinSegmentationHead(
          self.num_classes,
          out_patch_size,
          swin_kwargs,
          self.conv_upsample,
          self.conv_norm,
      )(x, aux, skips, train=train)

    if self.head_type == 'conv_segmentation':
      return ConvSegmentationHead(
          self.num_classes,
          out_patch_size,
          self.conv_norm,
          temporal_reduction=self.temporal_reduction,
      )(x, aux, skips, train=train)

    if self.head_type == 'super_res':
      return SuperResHead(
          self.num_classes, self.upscale_factor, self.conv_norm
      )(x, aux, train=train)

    return [jnp.mean(y, axis=1) for y in [x] + skips], aux


def Model(num_classes=None, *, variant=None, **kw):  # pylint: disable=invalid-name
  """Factory function, because linen really don't like what I'm doing!"""
  return SwinV2Model(
      **(get_variant(variant) | kw | dict(num_classes=num_classes))
  )


def get_variant(variant):
  """Converts a string like "B" or "B/32" into a params dict."""
  if variant is None:
    return {}

  v, patch = variant, {}
  if '/' in variant:
    v, patch = variant.split('/')
    patch = int(patch)
    patch = {'patch_size': (1, patch, patch)}

  return {
      'Ti': {  # Tiny
          'embed_dim': 96,
          'depths': [2, 2, 6, 2],
          'num_heads': [3, 6, 12, 24],
          'drop_path_rate': 0.2,
      },
      'S': {  # Small
          'embed_dim': 96,
          'depths': [2, 2, 18, 2],
          'num_heads': [3, 6, 12, 24],
          'drop_path_rate': 0.3,
      },
      'B': {  # Base
          'embed_dim': 128,
          'depths': [2, 2, 18, 2],
          'num_heads': [4, 8, 16, 32],
          'drop_path_rate': 0.5,
      },
      'L': {  # Large
          'embed_dim': 192,
          'depths': [2, 2, 18, 2],
          'num_heads': [6, 12, 24, 48],
          'drop_path_rate': 0.2,
      },
      'G': {  # Giant
          'embed_dim': 512,
          'depths': [2, 2, 42, 2],
          'num_heads': [16, 32, 64, 128],
          'drop_path_rate': 0.2,
      },
  }[v] | patch


def load(
    init_params,
    ckpt_path: str,
    model_cfg,
    init_state,
    dont_load_params=(),
    dont_load_states=(),
):
  """Loads init from checkpoint, both old model and this one."""
  del model_cfg  # Unused

  params, states = restore.load_params_and_states(ckpt_path)
  params = tree_utils.merge_params(params, init_params, dont_load_params)
  states = tree_utils.merge_params(states, init_state, dont_load_states)
  return params, states
