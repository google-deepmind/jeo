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

"""Random deterministic preprocessing ops.

These functions were inspired by preprocesing in the big_vision UViM project.
"""

import einops
from jeo.pp import pp_utils
from jeo.pp.pp_builder import Registry  # pylint: disable=g-importing-member
import keras.ops.image as keras_image
import numpy as np
import tensorflow as tf


@Registry.register("preprocess_ops.randu", replace=True)
def get_randu(key: str):
  """Creates a random uniform float [0, 1) in `key`."""

  def _randu(data):
    data[key] = tf.random.uniform([])
    return data

  return _randu


@Registry.register("preprocess_ops.det_roll", replace=True)
@pp_utils.InKeyOutKey(with_data=True)
def get_det_roll(*, randkey1: str = "rollx", randkey2: str = "rolly"):
  def _det_roll(image, data):
    axis1, axis2 = (-2, -1) if image.shape.rank == 2 else (-3, -2)
    h = tf.cast(tf.floor(data[randkey1] * image.shape[axis1]), dtype=tf.int32)
    w = tf.cast(tf.floor(data[randkey2] * image.shape[axis2]), dtype=tf.int32)
    image = tf.roll(image, [h, w], axis=[axis1, axis2])
    return image

  return _det_roll


def _create_rotation_matrix(rads: float, width: int, height: int) -> tf.Tensor:
  """Creates a rotation matrix for keras.ops.image.affine_transform."""
  sin, cos = tf.math.sin(rads), tf.math.cos(rads)
  x_offset = (width - 1) - (cos * (width - 1) - sin * (height - 1))
  y_offset = (height - 1) - (sin * (width - 1) + cos * (height - 1))
  return tf.stack([cos, -sin, x_offset / 2.0, sin, cos, y_offset / 2.0, 0, 0])


@Registry.register("preprocess_ops.det_rotate", replace=True)
@pp_utils.InKeyOutKey(with_data=True)
def get_det_rotate(*, randkey: str = "angle",
                   interpolation="bilinear", fill_value=0):
  """Rotate by an angle between -/+ `angle` using `randkey` in range (0, 1]."""

  def _det_rotate(image, data):
    orig_shape = image.shape
    orig_ndims = len(orig_shape)
    if orig_ndims == 5:
      image = einops.rearrange(image, "b t h w c -> (b t) h w c")
    *_, w, h, _ = image.shape
    image = keras_image.affine_transform(
        image,
        _create_rotation_matrix(data[randkey] * 2.0 * np.pi, w, h),
        interpolation=interpolation,
        fill_mode="constant",
        fill_value=fill_value,
        data_format="channels_last",
    )
    if orig_ndims == 5:
      image = einops.rearrange(
          image, "(b t) h w c -> b t h w c", b=orig_shape[0], t=orig_shape[1]
      )
    return image

  return _det_rotate


@Registry.register("preprocess_ops.det_rotate90", replace=True)
@pp_utils.InKeyOutKey(with_data=True)
def get_det_rotate90(*, randkey: str = "angle"):
  """Rotate by an orthogonal angle using `randkey` in range (0, 1]."""

  def _det_rotate90(image, data):
    orig_shape = image.shape
    orig_ndims = image.ndim
    if orig_ndims == 5:
      image = einops.rearrange(
          image,
          "b t h w c -> (b t) h w c",
      )
    elif orig_ndims == 2:
      image = image[..., None]
    maximum_num_rotations = 4.0
    num_rotations = tf.cast(
        data[randkey] * maximum_num_rotations, dtype=tf.int32
    )
    image = tf.image.rot90(image, k=num_rotations)
    if orig_ndims == 5:
      image = einops.rearrange(
          image,
          "(b t) h w c -> b t h w c",
          b=orig_shape[0],
          t=orig_shape[1],
      )
    elif orig_ndims == 2:
      image = image[..., 0]
    return image

  return _det_rotate90


@Registry.register("preprocess_ops.det_fliplr", replace=True)
@pp_utils.InKeyOutKey(with_data=True)
def get_det_fliplr(*, randkey: str = "fliplr"):
  """Flips an image horizontally based on `randkey`."""

  # From third_party/py/big_vision/pp/proj/uvim/pp_ops.py.
  # NOTE: we could unify this with regular flip when randkey=None.
  def _det_fliplr(orig_image, data):
    flip_image = tf.image.flip_left_right(orig_image)
    flip = tf.cast(data[randkey] > 0.5, orig_image.dtype)
    return flip_image * flip + orig_image * (1 - flip)

  return _det_fliplr


@Registry.register("preprocess_ops.det_flip_rot", replace=True)
@pp_utils.InKeyOutKey(with_data=True)
def get_det_flip_rot(*, randkey: str = "flip_rot"):
  """Deterministically flip and rotate by an orthogonal angle."""

  def _pp(image, data):
    orig_shape = image.shape
    orig_ndims = image.ndim
    orig_dtype = image.dtype
    if orig_dtype == tf.bool:
      image = tf.cast(image, tf.uint8)
    if orig_ndims == 5:
      image = einops.rearrange(image, "b t h w c -> (b t) h w c")
    elif orig_ndims == 2:
      image = image[..., None]
    max_num_transformations = 8.0
    num_rotations = (
        tf.cast(data[randkey] * max_num_transformations, dtype=tf.int32) % 4
    )
    image = tf.image.rot90(image, k=num_rotations)
    flip_image = tf.image.flip_left_right(image)
    to_flip = tf.cast(data[randkey] > 0.5, image.dtype)
    image = flip_image * to_flip + image * (1 - to_flip)
    if orig_ndims == 5:
      image = einops.rearrange(
          image, "(b t) h w c -> b t h w c", b=orig_shape[0], t=orig_shape[1]
      )
    elif orig_ndims == 2:
      image = image[..., 0]
    image = tf.cast(image, orig_dtype)
    return image

  return _pp


@Registry.register("preprocess_ops.det_crop", replace=True)
@pp_utils.InKeyOutKey(with_data=True)
def get_det_crop(crop_size: int, num_crops: int = 1, randkey: str = "crop"):
  """Deterministically crops an image `num_crops` times."""
  crop_size = pp_utils.maybe_repeat(crop_size, 2)
  crop_fn = tf.image.stateless_random_crop

  def _det_crop(image, data):
    seed = tf.cast(2**30 * data[randkey], "int32")
    shape = crop_size
    if image.ndim == 3:
      shape = (*crop_size, image.shape[-1])
    elif image.ndim == 4:
      shape = (image.shape[0], *crop_size, image.shape[-1])
    if num_crops > 1:
      return tf.vectorized_map(
          lambda _: crop_fn(image, shape, [seed, seed + 1]), tf.range(num_crops)
      )
    return crop_fn(image, shape, [seed, seed + 1])

  return _det_crop


@Registry.register("preprocess_ops.det_resize", replace=True)
@pp_utils.InKeyOutKey(with_data=True)
def get_det_resize(
    min_ratio: float = 0.8, max_ratio: float = 1.3, randkey="resize"
):
  """Deterministically resizes an image with a ratio from the given range."""

  def _det_crop(image, data):
    # [0, 1] -> [min_ratio, max_ratio]
    ratio = (max_ratio - min_ratio) * data[randkey] + min_ratio

    # Computes new shape.
    *_, h, w, _ = image.shape
    h, w = int(h * ratio), int(w * ratio)

    # Bilinear is more suited to floating points.
    method = "bilinear" if image.dtype.is_floating else "nearest"

    # `tf.image.resize` does not accept bool.
    image = tf.cast(image, "float32")
    return tf.cast(tf.image.resize(image, (h, w), method), image.dtype)

  return _det_crop
