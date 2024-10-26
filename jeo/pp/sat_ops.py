# Copyright 2024 The jeo Authors.
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

"""PP ops for multi-modal multi-channel multi-temporal satellite data."""
import math
from big_vision.pp.registry import Registry  # pylint: disable=g-importing-member
import einops
from jeo.pp import image_ops
from jeo.pp import pp_utils
import tensorflow as tf
import tensorflow_probability as tfp


# Values are from:
# http://google3/learning/multipod/pax/climate/normalization.py;l=11;rcl=658449988
# Mean and STD per channel after `NormalizeWithScaledTanh` with tanh_scale=3000
# S2 bands are: B1, B2, B3, B4, B5, B6, B7, B8, B8A, B9, B11, B12
SENTINEL2_SCALED_TANH_MEANS = [
    0.2778, 0.3010, 0.3578, 0.3853, 0.4723, 0.5964, 0.6323, 0.6424, 0.6584,
    0.6690, 0.5849, 0.4718]  # pyformat: disable
SENTINEL2_SCALED_TANH_STDS = [
    0.2567, 0.2505, 0.2377, 0.2611, 0.2336, 0.2205, 0.2242, 0.2268, 0.2260,
    0.2228, 0.2416, 0.2381]  # pyformat: disable
# Normalization values for Top of Atmosphere (TOA) L1C product.
# L1C product bands are: B1, B2, B3, B4, B5, B6, B7, B8, B8A, B9, B10, B11, B12.
SENTINEL2_TOA_SCALED_TANH_MEANS = [
    0.4976, 0.4379, 0.4119, 0.4071, 0.4610, 0.5871, 0.6338, 0.6221, 0.6648,
    0.2933, 0.0166, 0.5521, 0.4095]  # pyformat: disable
SENTINEL2_TOA_SCALED_TANH_STDS = [
    0.1668, 0.1875, 0.1960, 0.2326, 0.2133, 0.1966, 0.2018, 0.2048, 0.2063,
    0.2038, 0.0398, 0.2352, 0.2259]  # pyformat: disable
# Normalization values for Top of Atmosphere (TOA) L1C product without B10.
# L1C product bands are: B1, B2, B3, B4, B5, B6, B7, B8, B8A, B9, B10, B11, B12.
SENTINEL2_TOA_NO_B10_SCALED_TANH_MEANS = [
    0.4976, 0.4379, 0.4119, 0.4071, 0.4610, 0.5871, 0.6338, 0.6221, 0.6648,
    0.2933, 0.5521, 0.4095]  # pyformat: disable
SENTINEL2_TOA_NO_B10_SCALED_TANH_STDS = [
    0.1668, 0.1875, 0.1960, 0.2326, 0.2133, 0.1966, 0.2018, 0.2048, 0.2063,
    0.2038, 0.2352, 0.2259]  # pyformat: disable


@Registry.register("preprocess_ops.pad_seq")
def get_pad_seq(keys, length, allow_crop=True, **kwargs):
  """Pads (and crops) sequences in first dimension."""
  if isinstance(keys, str):
    keys = [keys]

  length_of_dims_to_pad = {0: length}
  def _pp(data):
    for k in keys:
      if k in data:
        data[k] = _crop_or_pad_tensor(
            data[k], length_of_dims_to_pad, allow_crop, **kwargs
        )
    return data
  return _pp


@Registry.register("preprocess_ops.pad_tensors")
def get_pad_tensors_to_match(keys, dims_to_pad, **kwargs):
  """Pads tensors along the provided dims to match the max dim's length."""
  if isinstance(keys, str):
    keys = [keys]

  def _pp(data):
    ranks = [data[key].get_shape().rank for key in keys]
    rank = ranks[0]
    if not all(rank == elem_rank for elem_rank in ranks):
      raise ValueError(
          "the rank of all tensors must be identical, at least one"
          f" rank is not matching others {ranks}"
      )
    # Compute max length of each dimension of interest.
    length_of_dims_to_pad = dict()
    for dim in dims_to_pad:
      length_of_dims_to_pad[dim] = max([data[k].get_shape()[dim] for k in keys])
    # Pad each key.
    for k in keys:
      data[k] = _crop_or_pad_tensor(data[k], length_of_dims_to_pad, **kwargs)
    return data

  return _pp


def _crop_or_pad_tensor(
    tensor, length_of_dims_to_pad, allow_crop=True, float_pad_value=0.0
):
  """Crops or pads a tensor along given dimensions."""
  rank = tf.shape(tensor).shape[0]
  target_shape = [
      length_of_dims_to_pad.get(dim, tensor.get_shape()[dim])
      for dim in range(rank)
  ]
  paddings = [(0, target_shape[x] - tf.shape(tensor)[x]) for x in range(rank)]

  if allow_crop:
    paddings = tf.maximum(paddings, 0)
  if tensor.dtype == tf.string:
    padded = tf.pad(tensor, paddings, constant_values="")
  elif tensor.dtype == tf.bool:
    padded = tf.pad(tensor, paddings, constant_values=False)
  elif tensor.dtype == tf.float32:
    padded = tf.pad(tensor, paddings, constant_values=float_pad_value)
  else:
    padded = tf.pad(tensor, paddings, constant_values=-1)
  if allow_crop:
    padded = tf.slice(padded, begin=[0] * rank, size=target_shape)
  return padded


@Registry.register("preprocess_ops.rearrange")
def get_rearrange(pattern, keys=("image",), **kwargs):
  """Arbitary rearranges dimensions using einops."""
  def _pp(data):
    for k in keys:
      data[k] = einops.rearrange(data[k], pattern, **kwargs)
    return data
  return _pp


@Registry.register("preprocess_ops.merge_spatial_dimensions")
def get_merge_spatial_dimensions(keys=("image",)):
  """Merges the spatial dimensions into a single dimension."""
  return get_rearrange("t h w c -> t (h w) c", keys)


@Registry.register("preprocess_ops.merge_all_dimensions")
def get_merge_all_dimensions(keys=("image",)):
  """Merges all dimensions into a single dimension."""
  return get_rearrange("t h w c -> (t h w c)", keys)


@Registry.register("preprocess_ops.reduce_temporal")
def get_reduce_temporal(
    keys=("image",), kind="first", timestamp_key=None, repeat=None
):
  """Reduces temporal dimension to a single element."""
  if isinstance(keys, str):
    keys = [keys]

  def _pp(data):

    temp_dim = tf.shape(data[keys[0]])[0]
    if kind == "random":
      idx = tf.random.uniform(shape=(), maxval=temp_dim, dtype=tf.int32)
    elif kind == "latest":
      idx = tf.argmax(data[timestamp_key])
    else:
      idx = None

    for key in keys:
      img = data[key]  # (t,h,w,c).

      if kind == "median":  # Get the median image across all dates (is slow).
        img = tfp.stats.percentile(img, 50, axis=0, interpolation="nearest")
      elif kind == "first":  # Select first frame.
        img = img[0]
      elif kind == "last":  # Select last frame.
        img = img[-1]
      elif kind in [
          "random",  # Select random frame.
          "latest",  # Select most recent frame based on `timestamp_key`.
      ]:
        img = img[idx]

      if repeat:
        img = tf.expand_dims(img, axis=0)
        if repeat > 1:
          img = tf.repeat(img, repeat, axis=0)

      data[key] = img

    return data

  return _pp


@Registry.register("preprocess_ops.add_lat_lon_channels", replace=True)
def get_add_lat_lon_channels(key):
  """Appends unit-sphere cartesian coordinates as channels."""
  def _pp(data):
    norm = math.pi / 180
    z = tf.broadcast_to(tf.sin(data["lat"] * norm), tf.shape(data[key])[:-1])
    y = tf.broadcast_to(tf.sin(data["lon"] * norm) * tf.cos(data["lat"] * norm),
                        tf.shape(data[key])[:-1])
    x = tf.broadcast_to(tf.cos(data["lon"] * norm) * tf.cos(data["lat"] * norm),
                        tf.shape(data[key])[:-1])
    z = tf.expand_dims(z, -1)
    y = tf.expand_dims(y, -1)
    x = tf.expand_dims(x, -1)
    z = tf.cast(z, data[key].dtype)
    y = tf.cast(y, data[key].dtype)
    x = tf.cast(x, data[key].dtype)
    data[key] = tf.concat([data[key], z, y, x], axis=-1)
    return data
  return _pp


@Registry.register("preprocess_ops.concat_satellites", replace=True)
def get_concat_satellites(sats, key=None):
  """Restricts the mask to forested areas."""
  key = key or sats[0]

  def _pp(data):
    first_axis = data[key].shape[0]
    h_axis = data[key].shape[1]
    w_axis = data[key].shape[2]
    for sat in sats[1:]:
      x = data[sat]
      if len(x.shape) == 3:
        x = tf.expand_dims(x, 0)
      x = tf.repeat(x, first_axis // x.shape[0], axis=0)
      x = tf.repeat(x, h_axis // x.shape[1], axis=1)
      x = tf.repeat(x, w_axis // x.shape[2], axis=2)
      data[key] = tf.concat([data[key], x], axis=-1)
    return data

  return _pp


@Registry.register("preprocess_ops.scale_spatial_dims_like", replace=True)
def get_scale_spatial_dims_like(
    keys_to_resize: tuple[str, ...], reference_key: str, height_axis: int = -2,
    width_axis: int = -3,
):
  """Scales spatial dims of given keys to match the dims of the reference key.

  This op scales images under the given keys to match the spatial dimensions of
  the image at the reference key. The height and width dimensions of the keys to
  resize must be divisible by the height and width dimensions of the reference
  key. The op can only scale images to larger dimensions, i.e., reference height
  and width dimensions must be greater or equal to the height and width
  dimensions of the keys to resize.

  Args:
    keys_to_resize: Keys of the data to resize the spatial dimensions of.
    reference_key: Key of the data to use as reference for the target spatial
      resolution.
    height_axis: Axis determining the height dimension.
    width_axis: Axis determining the width dimension.

  Returns:
    Function to resize the spatial dimensions of the given keys.
  """

  def _pp(data):
    n_dims = len(data[reference_key].shape)
    if height_axis % n_dims == width_axis % n_dims:
      raise ValueError(
          "height and width axes must be different, got"
          f" height_axis: {height_axis}, width_axis: {width_axis}"
      )
    ref_height_dim = data[reference_key].shape[height_axis]
    ref_width_dim = data[reference_key].shape[width_axis]
    for key in keys_to_resize:
      current_height_dim = data[key].shape[height_axis]
      current_width_dim = data[key].shape[width_axis]
      resized_data = data[key]
      if current_height_dim != ref_height_dim:
        if ref_height_dim < current_height_dim:
          raise ValueError(
              "Reference height dimension must be greater or equal, got"
              f" current height dimension: {current_height_dim}, reference"
              f" height dimension: {ref_height_dim}"
          )
        if ref_height_dim % current_height_dim != 0:
          raise ValueError(
              f"Reference height dimension {ref_height_dim} must be divisible"
              f' by height dimension {current_height_dim} of key "{key}"'
          )
        dx = ref_height_dim // current_height_dim
        resized_data = tf.repeat(resized_data, dx, axis=height_axis)
      if current_width_dim != ref_width_dim:
        if ref_width_dim < current_width_dim:
          raise ValueError(
              "Reference width dimension must be greater or equal, got current"
              f" width dimension: {current_width_dim}, reference width"
              f" dimension: {ref_width_dim}"
          )
        if ref_width_dim % current_width_dim != 0:
          raise ValueError(
              f"Reference width dimension {ref_width_dim} must be divisible by"
              f' width dimension {current_width_dim} of key "{key}"'
          )
        dx = ref_width_dim // current_width_dim
        resized_data = tf.repeat(resized_data, dx, axis=width_axis)
      data[key] = resized_data
    return data

  return _pp


@Registry.register("preprocess_ops.scale_sentinel2_by_tanh", replace=True)
@pp_utils.InKeyOutKey()
def get_scale_sentinel2_by_tanh(imagery_type: str):
  """Rescale sentinel2 imagery with tanh normalization.

  Args:
    imagery_type: The type of sentilnel2 data to scale. Available types: toa
      (Top of Atmosphere), toa_no_b10, sr (Top of Atmosphere with b10 band
      dropped), and sr (Surfance Reflectance).

  Returns:
    A function to scale the sentinel2 imagery.
  """

  if imagery_type == "toa":
    means = SENTINEL2_TOA_SCALED_TANH_MEANS
    stds = SENTINEL2_TOA_SCALED_TANH_STDS
  elif imagery_type == "toa_no_b10":
    means = SENTINEL2_TOA_NO_B10_SCALED_TANH_MEANS
    stds = SENTINEL2_TOA_NO_B10_SCALED_TANH_STDS
  elif imagery_type == "sr":
    means = SENTINEL2_SCALED_TANH_MEANS
    stds = SENTINEL2_SCALED_TANH_STDS
  else:
    raise ValueError(
        f"Unsupported imagery type: {imagery_type}. Available types: toa,"
        " toa_no_b10, sr"
    )

  return image_ops.get_tanh_value_range(means, stds, 3_000.0)


@Registry.register("preprocess_ops.s2_to_rgb", replace=True)
@pp_utils.InKeyOutKey(indefault="s2", outdefault="rgb")
def get_s2_to_rgb(bands=(2, 1, 0), temporal="first", bias=0, scale=3000):
  """Gather the required s2 bands and convert them to rgb."""
  def _pp(s2):
    if s2.get_shape().rank == 4:
      if temporal == "first":
        s2 = s2[0]
      else:
        raise ValueError(f"Unknown temporal reduce: {temporal}")
    elif s2.get_shape().rank != 3:
      raise ValueError(f"Rank must be 3, not {s2.get_shape().rank}")

    s2 = tf.unstack(s2, axis=-1)
    image = tf.stack([s2[bands[0]], s2[bands[1]], s2[bands[2]]], axis=2)
    rgb_image = (image - bias)/ scale
    return tf.clip_by_value(rgb_image, 0, 1)

  return _pp
