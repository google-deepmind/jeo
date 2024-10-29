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

"""Preprocessing ops for image-like data.

Many ops are based on third_party/py/big_vision/pp/ops_image.py.
"""
from collections.abc import Sequence

from jeo.pp import pp_utils
from jeo.pp.pp_builder import Registry  # pylint: disable=g-importing-member
import tensorflow as tf


@Registry.register("preprocess_ops.decode")
@pp_utils.InKeyOutKey(indefault="image", outdefault="image")
def get_decode(channels=3, precise=False):
  """Decode an encoded image string, see tf.io.decode_image.

  Args:
    channels: see tf.io.decode_image.
    precise: if False, use default TF image decoding algorithm. If True, change
      DCT method for JPEG decoding to match PIL/cv2/PyTorch.

  Returns:
    The decoded image.
  """

  def _decode(image):
    if precise:
      return tf.image.decode_jpeg(  # Also supports png btw.
          image, channels=channels, dct_method="INTEGER_ACCURATE"
      )
    else:
      return tf.io.decode_image(
          image, channels=channels, expand_animations=False
      )

  return _decode


@Registry.register("preprocess_ops.resize", replace=True)
@pp_utils.InKeyOutKey(indefault="image", outdefault="image")
def get_resize(size, method="bilinear", antialias=False):
  """Resizes image to a given size.

  Args:
    size: either an integer H, where H is both the new height and width
      of the resized image, or a list or tuple [H, W] of integers, where H and W
      are new image"s height and width respectively.
    method: resize method, see tf.image.resize docs for options.
    antialias: see tf.image.resize. Ideally set to True for all new configs.

  Returns:
    A function for resizing an image.

  """
  size = pp_utils.maybe_repeat(size, 2)

  def _resize(image):
    """Resizes image to a given size."""
    # Note: use TF-2 version of tf.image.resize as the version in TF-1 is
    # buggy: https://github.com/tensorflow/tensorflow/issues/6720.
    # In particular it was not equivariant with rotation and lead to the network
    # to learn a shortcut in self-supervised rotation task, if rotation was
    # applied after resize.
    dtype = image.dtype
    tf_dtype = tf.type_spec_from_value(image).dtype
    image = tf.image.resize(image, size, method=method, antialias=antialias)
    return tf.cast(tf.clip_by_value(image, tf_dtype.min, tf_dtype.max), dtype)

  return _resize


# This functionality is used by resize_small and resize_long. But we're not
# registering it as a pp op yet, as there is no need for it. However, it can
# probably be slightly generalized into "scale augmentation" eventually.
def _resize_factor(image, factor, method="area", antialias=True):
  """Resizes the image by a (float) `factor`, keeping the aspect ratio fixed."""
  h, w = tf.shape(image)[0], tf.shape(image)[1]

  h = tf.cast(tf.round(tf.cast(h, tf.float32) * factor), tf.int32)
  w = tf.cast(tf.round(tf.cast(w, tf.float32) * factor), tf.int32)

  dtype = image.dtype
  tf_dtype = tf.type_spec_from_value(image).dtype
  image = tf.image.resize(image, (h, w), method=method, antialias=antialias)
  return tf.cast(tf.clip_by_value(image, tf_dtype.min, tf_dtype.max), dtype)


@Registry.register("preprocess_ops.resize_small", replace=True)
@pp_utils.InKeyOutKey(indefault="image", outdefault="image")
def get_resize_small(smaller_size, method="area", antialias=False):
  """Resizes the smaller side to `smaller_size` keeping aspect ratio.

  Args:
    smaller_size: an integer, that represents a new size of the smaller side of
      an input image.
    method: the resize method. `area` is a meaningful, bwd-compat default.
    antialias: see tf.image.resize. Ideally set to True for all new configs.

  Returns:
    A function, that resizes an image and preserves its aspect ratio.

  Note:
    backwards-compat for "area"+antialias tested here:
    (internal link)?usp=sharing
  """

  def _resize_small(image):  # pylint: disable=missing-docstring
    h, w = tf.shape(image)[0], tf.shape(image)[1]
    factor = (
        tf.cast(smaller_size, tf.float32) /
        tf.cast(tf.minimum(h, w), tf.float32))
    return _resize_factor(image, factor, method=method, antialias=antialias)
  return _resize_small


@Registry.register("preprocess_ops.resize_long", replace=True)
@pp_utils.InKeyOutKey(indefault="image", outdefault="image")
def get_resize_long(longer_size, method="area", antialias=True):
  """Resizes the longer side to `longer_size` keeping aspect ratio.

  Args:
    longer_size: an integer, that represents a new size of the longer side of
      an input image.
    method: the resize method. `area` is a meaningful, bwd-compat default.
    antialias: see tf.image.resize. Ideally set to True for all new configs.

  Returns:
    A function, that resizes an image and preserves its aspect ratio.
  """

  def _resize_long(image):  # pylint: disable=missing-docstring
    h, w = tf.shape(image)[0], tf.shape(image)[1]
    factor = (
        tf.cast(longer_size, tf.float32) /
        tf.cast(tf.maximum(h, w), tf.float32))
    return _resize_factor(image, factor, method=method, antialias=antialias)
  return _resize_long


@Registry.register("preprocess_ops.inception_crop", replace=True)
@pp_utils.InKeyOutKey(indefault="image", outdefault="image")
def get_inception_crop(size=None, area_min=5, area_max=100,
                       method="bilinear", antialias=False):
  """Makes inception-style image crop.

  Inception-style crop is a random image crop (its size and aspect ratio are
  random) that was used for training Inception models, see
  https://www.cs.unc.edu/~wliu/papers/GoogLeNet.pdf.

  Args:
    size: Resize image to [size, size] after crop.
    area_min: minimal crop area.
    area_max: maximal crop area.
    method: rezied method, see tf.image.resize docs for options.
    antialias: see tf.image.resize. Ideally set to True for all new configs.

  Returns:
    A function, that applies inception crop.
  """

  def _inception_crop(image):  # pylint: disable=missing-docstring
    begin, crop_size, _ = tf.image.sample_distorted_bounding_box(
        tf.shape(image),
        tf.zeros([0, 0, 4], tf.float32),
        area_range=(area_min / 100, area_max / 100),
        min_object_covered=0,  # Don't enforce a minimum area.
        use_image_if_no_bounding_boxes=True)
    crop = tf.slice(image, begin, crop_size)
    # Unfortunately, the above operation loses the depth-dimension. So we need
    # to restore it the manual way.
    crop.set_shape([None, None, image.shape[-1]])
    if size:
      crop = get_resize(size, method, antialias)({"image": crop})["image"]
    return crop

  return _inception_crop


@Registry.register("preprocess_ops.random_crop", replace=True)
@pp_utils.InKeyOutKey(indefault="image", outdefault="image")
def get_random_crop(crop_size: int, num_crops: int = 1):
  """Makes a random crop of a given size.

  Args:
    crop_size: either an integer H, where H is both the height and width of the
      random crop, or a list or tuple [H, W] of integers, where H and W are
      height and width of the random crop respectively.
    num_crops: the number of different random crops to make.

  Returns:
    A function, that applies random crop.
  """
  crop_size = pp_utils.maybe_repeat(crop_size, 2)

  def crop_fn(image):
    shape = (*crop_size, image.shape[-1])
    if num_crops > 1:
      return tf.vectorized_map(
          lambda _: tf.image.random_crop(image, shape), tf.range(num_crops)
      )
    return tf.image.random_crop(image, shape)

  return crop_fn


@Registry.register("preprocess_ops.central_crop", replace=True)
@pp_utils.InKeyOutKey(indefault="image", outdefault="image")
def get_central_crop(crop_size=None):
  """Makes central crop of a given size.

  Args:
    crop_size: either an integer H, where H is both the height and width of the
      central crop, or a list or tuple [H, W] of integers, where H and W are
      height and width of the central crop respectively. If `crop_size` is not
      specified, then the largest possible center crop will be taken.

  Returns:
    A function, that applies central crop.
  """
  if crop_size:
    crop_size = pp_utils.maybe_repeat(crop_size, 2)

  def _crop(image):
    if crop_size:
      h, w = crop_size[0], crop_size[1]
    else:
      h = w = tf.minimum(tf.shape(image)[0], tf.shape(image)[1])
    is_2d = tf.shape(image).shape[0] == 2
    if is_2d:  # tf.image.crop_to_bounding_box expects channels dimension.
      image = tf.expand_dims(image, axis=-1)
    dy = (tf.shape(image)[-3] - h) // 2
    dx = (tf.shape(image)[-2] - w) // 2
    out = tf.image.crop_to_bounding_box(image, dy, dx, h, w)
    if is_2d:
      out = out[..., 0]
    return out

  return _crop


@Registry.register("preprocess_ops.extract_patches", replace=True)
@pp_utils.InKeyOutKey(indefault="image", outdefault="image")
def get_extract_patches(crop_size: int, padding: str, flatten: bool = False):
  """Extract overlapping patches from an image.

  Args:
    crop_size: the height and width of the patches to extract.
    padding: If VALID, only patches which are fully contained in the input image
      are included. If SAME, all patches whose starting point is inside the
      input are included, and areas outside the input default to zero.
    flatten: If True, the patches are flattened into a 1D vector.

  Returns:
    A function, that extracts patches.
  """
  assert padding in [
      "VALID",
      "SAME",
  ], f"Unsupported padding: {padding}. Available options are: VALID, SAME."

  def _extract(image):
    # The extracted patches function only takes 4D tensors.
    assert len(image.shape) == 4, "Image must be [B, H, W, C]."

    # Determine the output shape based on args.
    out_shape = (crop_size**2,) if flatten else (crop_size, crop_size)

    return tf.reshape(
        tf.image.extract_patches(
            image,
            sizes=[1, crop_size, crop_size, 1],
            strides=[1, 1, 1, 1],
            rates=[1, 1, 1, 1],
            padding=padding,
        ),
        (-1, *out_shape, image.shape[-1]),
    )

  return _extract


@Registry.register("preprocess_ops.flip_lr", replace=True)
@pp_utils.InKeyOutKey(indefault="image", outdefault="image")
def get_random_flip_lr():
  """Flips an image horizontally with probability 50%."""

  def _random_flip_lr_pp(image):
    return tf.image.random_flip_left_right(image)

  return _random_flip_lr_pp


@Registry.register("preprocess_ops.decode_jpeg_and_inception_crop",
                   replace=True)
@pp_utils.InKeyOutKey(indefault="image", outdefault="image")
def get_decode_jpeg_and_inception_crop(size=None, area_min=5, area_max=100,
                                       ratio_min=0.75, ratio_max=1.33,
                                       method="bilinear", antialias=False):
  """Decode jpeg string and make inception-style image crop.

  Inception-style crop is a random image crop (its size and aspect ratio are
  random) that was used for training Inception models, see
  https://www.cs.unc.edu/~wliu/papers/GoogLeNet.pdf.

  From: third_party/py/big_vision/pp/ops_image.py

  Args:
    size: Resize image to [size, size] after crop.
    area_min: minimal crop area.
    area_max: maximal crop area.
    ratio_min: minimal aspect ratio.
    ratio_max: maximal aspect ratio.
    method: rezied method, see tf.image.resize docs for options.
    antialias: see tf.image.resize. Ideally set to True for all new configs.

  Returns:
    A function, that applies inception crop.
  """

  def _inception_crop(image_data):  # pylint: disable=missing-docstring
    shape = tf.image.extract_jpeg_shape(image_data)
    begin, crop_size, _ = tf.image.sample_distorted_bounding_box(
        shape,
        tf.zeros([0, 0, 4], tf.float32),
        area_range=(area_min / 100, area_max / 100),
        aspect_ratio_range=(ratio_min, ratio_max),
        min_object_covered=0,  # Don't enforce a minimum area.
        use_image_if_no_bounding_boxes=True)

    # Crop the image to the specified bounding box.
    offset_y, offset_x, _ = tf.unstack(begin)
    target_height, target_width, _ = tf.unstack(crop_size)
    crop_window = tf.stack([offset_y, offset_x, target_height, target_width])
    image = tf.image.decode_and_crop_jpeg(image_data, crop_window, channels=3)

    if size:
      image = get_resize(size, method, antialias)({"image": image})["image"]

    return image

  return _inception_crop


@Registry.register("preprocess_ops.cutout_mask", replace=True)
def cutout_mask(
    min_size_ratio=0.01,
    max_size_ratio=0.5,
    inkey="image",
    outkey="cutout_mask",
):
  """Creates a mask which can be used to cutout a random region of the image."""

  def _pp(data):
    sp = tf.shape(data[inkey])
    h, w = sp[0], sp[1]
    fh = tf.cast(h, tf.float32)
    fw = tf.cast(w, tf.float32)

    pad_h = tf.random.uniform(
        [],
        minval=tf.cast(fh * min_size_ratio / 2.0, tf.int32),
        maxval=tf.cast(fh * max_size_ratio / 2.0 + 1.0, tf.int32),
        dtype=tf.int32,
    )
    pad_w = tf.random.uniform(
        [],
        minval=tf.cast(fw * min_size_ratio / 2.0, tf.int32),
        maxval=tf.cast(fw * max_size_ratio / 2.0 + 1.0, tf.int32),
        dtype=tf.int32,
    )
    pad_size = (pad_h, pad_w)

    # Center location in the image where the zero mask will be applied.
    cutout_center_height = tf.random.uniform(
        shape=[], minval=0, maxval=h, dtype=tf.int32
    )
    cutout_center_width = tf.random.uniform(
        shape=[], minval=0, maxval=w, dtype=tf.int32
    )

    lower_pad = tf.maximum(0, cutout_center_height - pad_size[0])
    upper_pad = tf.maximum(0, h - (cutout_center_height + pad_size[0]))
    left_pad = tf.maximum(0, cutout_center_width - pad_size[1])
    right_pad = tf.maximum(0, w - (cutout_center_width + pad_size[1]))

    cutout_shape = [h - (lower_pad + upper_pad), w - (left_pad + right_pad)]
    padding_dims = [[lower_pad, upper_pad], [left_pad, right_pad]]
    mask = tf.pad(
        tf.zeros(cutout_shape, dtype=tf.int32), padding_dims, constant_values=1
    )
    data[outkey] = mask
    return data

  return _pp


@Registry.register("preprocess_ops.cutout_from_mask", replace=True)
@pp_utils.InKeyOutKey(with_data=True)
def get_cutout_from_mask(mask_key="cutout_mask", replace=0):
  """Cuts out a region of an image according to mask in `cutout_mask`."""

  def _cutout_from_mask(image, data):
    shape = tf.shape(image)
    c = shape[-1]
    mask = data[mask_key]
    mask = tf.expand_dims(mask, -1)
    mask = tf.tile(mask, [1, 1, c])
    return tf.where(
        tf.equal(mask, 0),
        tf.ones_like(image, dtype=image.dtype) * replace,
        image,
    )

  return _cutout_from_mask


@Registry.register("preprocess_ops.random_fill_along_dim", replace=True)
@pp_utils.InKeyOutKey(with_data=False)
def get_random_fill_along_dim(*, axis=1, probability=0.25, fill_value=0):
  """Random masks some channels from a given dimension."""

  def _random_fill_along_dim(image):
    orig_shape = image.shape
    orig_ndims = image.ndim

    mask_axis = axis if axis >= 0 else axis + orig_ndims
    dim_len = orig_shape[mask_axis]
    mask = tf.random.uniform(shape=[dim_len]) < probability

    # expand mask for broadcasting
    for d in range(mask_axis):
      mask = tf.expand_dims(mask, axis=d)
    for d in range(mask_axis + 1, orig_ndims):
      mask = tf.expand_dims(mask, axis=d)

    replace = tf.ones_like(image, dtype=image.dtype) * fill_value

    image = tf.where(mask, replace, image)
    return image

  return _random_fill_along_dim


@Registry.register("preprocess_ops.rot90", replace=True)
@pp_utils.InKeyOutKey(indefault="image", outdefault="image")
def get_random_rotation90():
  """Randomly rotate an image by multiples of 90 degrees."""

  def _pp(image):
    """Rotation function."""
    num_rotations = tf.random.uniform(shape=(), maxval=4, dtype=tf.int32)
    if image.shape.rank == 2:  # 2-dim images, eg. segmentation mask.
      return tf.image.rot90(image[..., None], k=num_rotations)[..., 0]
    return tf.image.rot90(image, k=num_rotations)

  return _pp


@Registry.register("preprocess_ops.rand_shift", replace=True)
@pp_utils.InKeyOutKey(indefault="image", outdefault="image")
def get_random_shift_reflect(max_shift=4):
  """Randomly shifts given image by specified amount."""
  # By up to w pixels, with reflection at boundary.

  def _pp(image):
    pad_extend = [[max_shift] * 2, [max_shift] * 2, [0] * 2]
    padded_x = tf.pad(image, pad_extend, mode="REFLECT")
    return tf.image.random_crop(padded_x, tf.shape(image))

  return _pp


@Registry.register("preprocess_ops.flip_ud_with_label", replace=True)
def get_flip_ud_with_label(outkey="flipped", key="image"):
  """Flips image up or down and saves if it was flipped."""

  def _pp(features):
    features[outkey] = tf.random.uniform([], maxval=2, dtype=tf.int32)
    features[key] = tf.cond(
        features[outkey] > 0,
        lambda: tf.image.flip_up_down(features[key]),
        lambda: features[key],
    )
    return features

  return _pp


@Registry.register("preprocess_ops.random_resize", replace=True)
def get_random_resize(
    ref_size,
    min_ratio,
    max_ratio,
    key="image",
    method="bicubic",
    mask_key="padding_mask",
):
  """Randomly resizes image preserving aspect ratio."""

  def _pp(features):
    scale = tf.random.uniform([], min_ratio, max_ratio)
    scaled_size = tf.cast(scale * ref_size, tf.int32)
    scaled_size = tf.stack([scaled_size, scaled_size])
    features[key] = tf.image.resize(
        features[key], scaled_size, method=method, preserve_aspect_ratio=True
    )
    if mask_key in features:
      features[mask_key] = tf.image.resize(
          features[mask_key][:, :, None],
          scaled_size,
          method="nearest",
          preserve_aspect_ratio=True,
      )[:, :, 0]
    return features

  return _pp


@Registry.register("preprocess_ops.flip_ud", replace=True)
@pp_utils.InKeyOutKey(indefault="image", outdefault="image")
def get_random_flip_ud():
  """Flips an image vertically with probability 50%."""

  # Note, instead of using this op, for satellite data prefer to get the 8
  # random combinations with "|flip_lr|rot90".
  # TODO: Support for such random operations flipping labels & masks.
  def _pp(image):
    return tf.image.random_flip_up_down(image)

  return _pp


@Registry.register("preprocess_ops.clear_boundary", replace=True)
@pp_utils.InKeyOutKey(indefault="image", outdefault="image")
def get_clear_boundary(margin, inverse=False):
  """Sets a zero value on a boundary of a tensor (of width "margin").

  Args:
    margin: width of the boundary.
    inverse: if True, sets zero in the middle of the tensor instead.
  Returns:
    a transform fn
  """

  def _pp(x):
    n = x.shape[0]
    assert len(x.shape) == 2
    assert n > margin * 2
    y = tf.concat([
        tf.zeros((margin, n), dtype=x.dtype),
        tf.concat([tf.zeros((n - margin * 2, margin), dtype=x.dtype),
                   x[margin:-margin, margin:-margin],
                   tf.zeros((n - margin * 2, margin), dtype=x.dtype)],
                  axis=1),
        tf.zeros((margin, n), dtype=x.dtype),
        ], axis=0)
    if inverse:
      return x - y
    else:
      return y

  return _pp


@Registry.register("preprocess_ops.vgg_value_range", replace=True)
@pp_utils.InKeyOutKey(indefault="image", outdefault="image")
def get_vgg_value_range(
    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), scale=255.0
):
  """VGG-style preprocessing, subtracts mean and divides by stddev.

  This preprocessing is very common for ImageNet pre-trained models since VGG,
  and to this day the standard for models coming from most PyTorch codes.

  From: third_party/py/big_vision/pp/ops_image.py

  Args:
    mean: Tuple of values to be subtracted. Default to widespread VGG values.
    std: Tuple of values to be divided by. Default to widespread VGG values.
    scale: The scale of the image values. Default to 255.

  Returns:
    A function to rescale the values.
  """
  mean = tf.constant(mean, tf.float32) * scale
  std = tf.constant(std, tf.float32) * scale

  def _vgg_value_range(image):
    return (tf.cast(image, tf.float32) - mean) / std

  return _vgg_value_range


@Registry.register("preprocess_ops.tanh_value_range", replace=True)
@pp_utils.InKeyOutKey(indefault="image", outdefault="image")
def get_tanh_value_range(
    mean: Sequence[float], std: Sequence[float], tanh_scale=1.0
):
  """Normalizing image pixels by a scaled tanh function.

  Args:
    mean: Tuple of values to be subtracted.
    std: Tuple of values to be divided by.
    tanh_scale: The scale of the image values.

  Returns:
    A function to rescale the values.
  """
  mean = tf.constant(mean, tf.float32)
  std = tf.constant(std, tf.float32)
  scale = tf.constant(tanh_scale, tf.float32)

  def _tanh_value_range(image):
    tanh_image = tf.math.tanh(image / scale)
    return (tf.cast(tanh_image, tf.float32) - mean) / std

  return _tanh_value_range


@Registry.register("preprocess_ops.pansharpen", replace=True)
def get_pansharpen(
    keys: str | Sequence[str], out_key: str = "image"
):
  """Pansharpen rgb image.

  Args:
    keys: It can either be a single key corresponding to a [H, W, 4] array with
      rgbp channels on the last dimensions, or two keys with rgb and p channels.
      or a list of four keys for r,g,b,p channels. The value must be in [0, 1].
    out_key: The name of the key for the pansharpend image.

  Returns:
    A function that pansharpen an image.
  """

  if isinstance(keys, str):
    keys = [keys]

  def _format(arr):
    if arr.shape[-1] not in (1, 3, 4):
      arr = arr[..., None]
    return arr

  def _pansharpen(data):

    # Ensure correct format of the channels.
    rgbp = [_format(data[k]) for k in keys]

    # Extract rgb and p channels.
    if len(keys) == 4:
      r, g, b, p = rgbp
      rgb = tf.concat([r, g, b], axis=-1)
    elif len(keys) == 2:
      rgb, p = rgbp
    elif len(keys) == 1:
      rgb, p = rgbp[0][..., :3], rgbp[0][..., -1:]
    else:
      raise ValueError(f"Invalid number of keys: {len(keys)}")

    # Check that channels are formatted correctly.
    if not (rgb.shape[-1] == 3 and p.shape[-1] == 1) or (
        rgb.shape[:-1] != p.shape[:-1]
    ):
      raise ValueError(f"Invalid shapes: {rgb.shape} and {p.shape}")

    # Pansharpen.
    hs = tf.image.rgb_to_hsv(rgb)[..., :2]
    data[out_key] = tf.image.hsv_to_rgb(tf.concat([hs, p], axis=-1))

    return data

  return _pansharpen
