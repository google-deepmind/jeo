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

"""Datasets decoding preprocessing ops."""

from big_vision.pp.registry import Registry  # pylint: disable=g-importing-member
from jeo.datasets.const import pastis
from jeo.datasets.const import planted
from jeo.pp import pp_utils
import tensorflow as tf
import tensorflow_probability as tfp


PASTIS_S2_AVG = [1180.17966309, 1387.73000488, 1436.66230469, 1773.64633789,
                 2735.82133789, 3080.08549805, 3223.57607422, 3338.31606445,
                 2418.06196289, 1630.16706543]
PASTIS_S2_STD = [1976.67634277, 1916.79592285, 1996.20566406, 1903.12397461,
                 1784.92321777, 1796.31408691, 1811.76420898, 1793.33869629,
                 1474.41660156, 1309.8329834]
PASTIS_S1A_AVG = [-10.91893864, -17.34335365, 6.42436066]
PASTIS_S1A_STD = [3.26800823, 3.19879241, 3.33648825]
PASTIS_S1D_AVG = [-11.07381592, -17.45235901, 6.38642101]
PASTIS_S1D_STD = [3.33763394, 3.15572557, 3.33574982]
PLANTED_SATS = ("s1", "s2", "l7", "alos", "modis")


@Registry.register("preprocess_ops.decode")
@pp_utils.InKeyOutKey(indefault="image", outdefault="image")
def get_decode(channels=3, precise=False):
  """Decode an encoded image string, see tf.io.decode_image.

  From: third_party/py/big_vision/pp/ops_image.py

  Args:
    channels: see tf.io.decode_image.
    precise: if False, use default TF image decoding algorithm.
        If True, change DCT method for JPEG decoding to match PIL/cv2/PyTorch.
        See also go/tf-decode-quirks for a concrete example.

  Returns:
    The decoded image.
  """

  def _decode(image):
    if precise:
      return tf.image.decode_jpeg(  # Also supports png btw.
          image, channels=channels, dct_method="INTEGER_ACCURATE")
    else:
      return tf.io.decode_image(
          image, channels=channels, expand_animations=False)

  return _decode


@Registry.register("preprocess_ops.pastis_jeo_decode", replace=True)
def get_pastis_jeo_decode():
  """Decodes pastis example in pastis-jeo-recreation."""
  label_map = pastis.CODE_INDEX_TO_CLASS_INDEX_MAP
  # Unused classes are void label, class_index = 19.
  culture_table = pp_utils.get_lookup_table(label_map, default_value=19)

  def _pp(data):
    # Recontract segmentation_mask based on parcel_culture(CODE_CULTU).
    segmentation = culture_table.lookup(
        tf.cast(data["parcel_culture"], tf.int32)
    )
    # Background=0: no data area.
    segmentation = tf.where(data["parcel_culture_mask"] > 0, segmentation, 0)
    # Void=19: unused classes.
    segmentation = tf.where(data["parcel_culture"] < 0, 19, segmentation)

    # Recontract segmentation_mask based on parcel_code(CODE_GROUP).
    # CODE_GROUP=22 or 25: 12
    segmentation = tf.where(data["parcel_code"] == 22, 12, segmentation)
    segmentation = tf.where(data["parcel_code"] == 25, 12, segmentation)
    # CODE_GROUP=20: 16
    segmentation = tf.where(data["parcel_code"] == 20, 16, segmentation)

    # According to the paper, void(19) class is ignored, while background(0)
    # class is predicted. Move void class to 0 in order to disregard it.
    # Note that this shifts all classes by 1.
    data["segmentation_mask"] = (segmentation + 1) % 20

    return data

  return _pp


@Registry.register("preprocess_ops.pastis_decode", replace=True)
def get_pastis_decode(norm=True, target="semantic", include_s1=False):
  """Decodes pastis example."""
  assert target == "semantic", "Panoptic not supported yet."
  size = 128

  def _pp(data):
    if target == "semantic":
      segmentation = tf.cast(data.pop("semantics"), tf.int32)
      segmentation = tf.ensure_shape(segmentation, (size, size, 1))
      # According to the paper, void(19) class is ignored, while background(0)
      # class is predicted. Move void class to 0 in order to disregard it.
      # Note that this shifts all classes by 1.
      data["segmentation_mask"] = (segmentation + 1) % 20

    s2 = data.pop("s2")
    if norm:
      s2 = (s2 - tf.constant(PASTIS_S2_AVG)) / tf.constant(PASTIS_S2_STD)
    data["s2"] = tf.ensure_shape(s2, (None, size, size, 10))

    if include_s1:
      s1a = data.pop("s1a")
      s1d = data.pop("s1d")
      if norm:
        s1a = (s1a - tf.constant(PASTIS_S1A_AVG)) / tf.constant(PASTIS_S1A_STD)
        s1d = (s1d - tf.constant(PASTIS_S1D_AVG)) / tf.constant(PASTIS_S1D_STD)
      data["s1a"] = tf.ensure_shape(s1a, (None, size, size, 3))
      data["s1d"] = tf.ensure_shape(s1d, (None, size, size, 3))

    return data

  return _pp


@Registry.register("preprocess_ops.pastis_decode_semantic_simple", replace=True)
def get_pastis_decode_semantic_simple(clip=False, temp_select="median"):
  """Decodes semantic example from pastis TFDS."""

  def _pp(data):
    size = 128
    segmentation = tf.cast(data.pop("semantics"), tf.int32)
    segmentation = tf.ensure_shape(segmentation, (size, size, 1))
    # According to the paper, void(19) class is ignored, while background(0)
    # class is predicted. Move void class to 0 in order to disregard it.
    # Note that this shifts all classes by 1.
    data["segmentation_mask"] = (segmentation + 1) % 20
    # For quick testing only.
    img = tf.gather(data.pop("s2"), (2, 1, 0), axis=-1)  # RGB channels.
    if temp_select == "median":
      # Get the median image across all dates.
      img = tfp.stats.percentile(img, 50, axis=0, interpolation="nearest")
    elif temp_select == "random":
      ind = tf.random.uniform(shape=(), maxval=tf.shape(img)[0], dtype=tf.int32)
      img = img[ind]
    elif temp_select == "first":
      img = img[0]
    elif temp_select == "aug":
      # Select only August images. dates_s2 has the form YYYYMMDD.
      is_aug = tf.where(
          tf.equal(tf.strings.substr(data["dates_s2"], 5, 1), b"8"))[..., 0]
      img = tf.gather(img, is_aug)
      # Select the one with smallest mean (usually lowest cloud coverage).
      means = tf.reduce_mean(img, axis=(1, 2, 3))
      img = img[tf.argmin(means)]
    # Rescale to [-1-.., 1+..].
    mean = tf.gather(tf.constant(PASTIS_S2_AVG), (2, 1, 0))
    std = tf.gather(tf.constant(PASTIS_S2_STD), (2, 1, 0))
    img = (img - mean) / std
    if clip:
      img = tf.clip_by_value(img, -1., 1.)
    data["image"] = tf.ensure_shape(img, (size, size, 3))
    return data

  return _pp


@Registry.register("preprocess_ops.pastis_decode_panoptic_simple", replace=True)
def get_pastis_decode_panoptic_simple(clip=False):
  """Decodes semantic example from pastis TFDS."""

  def _pp(data):
    size = 128
    segmentation = tf.cast(data.pop("semantics"), tf.int32)
    data["segmentation_mask"] = tf.ensure_shape(segmentation, (size, size, 1))
    instances = tf.cast(data.pop("instances"), tf.int32)
    # Original instances start at 1. 0 is background/"no instance".
    # Make it start at 0, and "no instance" be at -1 in order to get an empty
    # one-hot mask for background pixels.
    instances = instances - 1
    data["instances"] = tf.ensure_shape(instances, (size, size, 1))
    # For quick testing only.
    img = tf.gather(data.pop("s2"), (2, 1, 0), axis=-1)  # RGB channels.
    # Get the median image across all dates.
    img = tfp.stats.percentile(img, 50, axis=0, interpolation="nearest")
    # Rescale to [-1-.., 1+..].
    mean = tf.gather(tf.constant(PASTIS_S2_AVG), (2, 1, 0))
    std = tf.gather(tf.constant(PASTIS_S2_STD), (2, 1, 0))
    img = (img - mean) / std
    if clip:
      img = tf.clip_by_value(img, -1., 1.)
    data["image"] = tf.ensure_shape(img, (size, size, 3))
    return data

  return _pp


@Registry.register("preprocess_ops.decode_fmow_s2", replace=True)
def get_decode_fmow_s2(rgb=False, temp_select=None, clip=False):
  """Decodes example from FMoW-S2 dataset."""
  center, scale = pp_utils.load_normalization_ranges(
      "fmow_s2/1.0.0", "train", "s2")
  if rgb:
    center = tf.gather(center, (3, 2, 1), axis=-1)
    scale = tf.gather(scale, (3, 2, 1), axis=-1)
  num_channels = 3 if rgb else 13

  def _pp(data):
    img = data.pop("s2")
    if rgb:
      img = tf.gather(img, (3, 2, 1), axis=-1)  # RGB channels.
    if temp_select is None:
      pass
    elif temp_select == "median":
      # Get the median image across all dates.
      img = tfp.stats.percentile(img, 50, axis=0, interpolation="nearest")
    elif temp_select == "random":
      # Pick a single random image from the time series.
      ind = tf.random.uniform(shape=(), maxval=tf.shape(img)[0], dtype=tf.int32)
      img = img[ind]
    elif temp_select == "first":
      # For debugging/testing, just take the first image.
      img = img[0]
    else:
      raise ValueError(f"Not supported temporal selection: {temp_select}")
    img = (img - center) / scale
    if clip:
      img = tf.clip_by_value(img, -1., 1.)
    # Fix shape for channels dim.
    if temp_select is None:
      data["image"] = tf.ensure_shape(img, [None, None, None, num_channels])
    else:
      data["image"] = tf.ensure_shape(img, [None, None, num_channels])
    return data

  return _pp


@Registry.register("preprocess_ops.dw_decode_semantic_simple", replace=True)
def get_dw_decode_semantic_simple(clip=False, rgb=True):
  """Decodes semantic example from dynamicworld TFDS."""
  assert rgb, "Non-rgb processing is not supported yet."
  s2_mean_std = tf.constant(
      [[1146.0770543, 1067.73446121, 1006.22073859, 1234.06404657,
        1866.40691978, 2157.97865832, 2114.55712782, 1831.00579254,
        1160.10163639],
       [591.94262794, 618.30252184, 786.53500061, 736.00089648,
        825.44881527, 946.01503245, 948.50804702, 1002.58732544,
        825.0339669]])
  # s2_bands = [f"B{b}" for b in [2, 3, 4, 5, 6, 7, 8, 11, 12]]

  def _pp(data):
    size = 512
    segmentation = tf.cast(data.pop("semantics"), tf.int32)
    segmentation = tf.ensure_shape(segmentation, (size, size, 1))
    # Labels 1 to 9 are the target classes.
    # Label=0 is void/not labelled, and label=10 is clouds. None should be used
    # for evaluation or training loss computation.
    data["segmentation_mask"] = segmentation % 10
    # For quick testing only.
    img = data["s2"]
    # Rescale to ~[-1-.., 1+..].
    img = (img - s2_mean_std[0]) / s2_mean_std[1]
    if rgb:
      img = tf.gather(img, (2, 1, 0), axis=-1)  # RGB channels.
    if clip:
      img = tf.clip_by_value(img, -1., 1.)
    data["image"] = tf.ensure_shape(img, (size, size, 3))
    return data

  return _pp


# TODO(mnn): This op is deprecated and will be removed soon.
@Registry.register("preprocess_ops.decode_planted_centroid", replace=True)
def get_decode_planted_centroid(label_key="common_name", mask_value=60000,
                                zero_pad=True, clip=False,
                                version="sse_asia:1.0.0", stats_split="train",
                                sat_keys=PLANTED_SATS,
                                add_frequency=True):
  """Decodes planted centroid example."""
  variant = version.split(":")[0]
  if label_key == "common_name":
    label_map = planted.LABEL_MAPS[variant]
  else:
    raise ValueError(f"Label key `{label_key}` not supported yet.")
  table = pp_utils.get_lookup_table(label_map)
  add_frequency = add_frequency and variant in planted.FREQUENCY_SPLITS
  if add_frequency:
    subsplit_table = pp_utils.get_lookup_table(
        {({"frequent": 2, "common": 1, "rare": 0})[k]: v
         for k, v in planted.FREQUENCY_SPLITS[variant].items()})

  center_and_scale_per_satellite = {
      sat: pp_utils.load_normalization_ranges(
          f"planted/{version}", stats_split, sat)
      for sat in sat_keys}

  def _pp(data):
    def _normalize_and_add_pading_mask_per_satellite(sat):
      # Add padding mask.
      pad_mask_key = f"{sat}_mask"
      if pad_mask_key in data:
        # If available, mask denotes valid pixels, and we need to reverse it.
        data[pad_mask_key] = ~tf.cast(data[pad_mask_key], tf.bool)
      else:
        # TODO(mnn): remove mask_value flows when switching completely.
        data[pad_mask_key] = (data[sat] == mask_value)
      # Normalize.
      center, scale = center_and_scale_per_satellite[sat]
      data[sat] = (tf.cast(data[sat], tf.float32) - center) / scale
      if zero_pad:  # Set masked values to 0.
        data[sat] = tf.where(data[pad_mask_key], 0., data[sat])
      if clip:
        data[sat] = tf.clip_by_value(data[sat], -1., 1.)

    for sat in sat_keys:
      _normalize_and_add_pading_mask_per_satellite(sat)
      # Re-shape modis into 4D.
      if sat == "modis" and data["modis"].ndim == 2:
        data["modis"] = data["modis"][:, None, None, :]
      if sat == "modis_sr" and data["modis_sr"].ndim == 2:
        data["modis_sr"] = data["modis_sr"][:, None, None, :]

    # Get label.
    if add_frequency:
      data["subsplit"] = subsplit_table.lookup(data["common_name"])
    data["label"] = table.lookup(data.pop(label_key))
    return data

  return _pp


@Registry.register("preprocess_ops.decode_coffee", replace=True)
def get_decode_coffee(binary=True, mask_value=60000, zero_pad=True):
  """Decodes planted centroid example."""
  label_map = [
      "Coffee", "Primary", "Acacia/Wattle", "Banana", "Cashew", "Coconut Palm",
      "Eucalyptus", "Monterey Pine", "Oil Palm", "Rubber", "Shining Gum",
      "Tasmanian Bluegum"]  # 12 classes.
  table = pp_utils.get_lookup_table(label_map)

  # Exclude high-res "hr" data for now (not present in many locations).
  sat_keys = ["s1", "s2", "l7", "l8", "nicfi", "nicfi_monthly", "modis", "alos"]
  stats_per_satellite = {
      sat: pp_utils.load_normalization_ranges(
          "coffee/recognition_240m/0.2.0", "train", sat) for sat in sat_keys}

  def _pp(data):
    def _normalize_and_add_pading_mask_per_satellite(sat):
      # Add padding mask.
      pad_mask_key = f"{sat}_mask"
      if pad_mask_key in data:
        # If available, mask denotes valid pixels, and we need to reverse it.
        data[pad_mask_key] = ~tf.cast(data[pad_mask_key], tf.bool)
      else:
        # TODO(mnn): remove mask_value flows when switching completely.
        data[pad_mask_key] = (data[sat] == mask_value)
      # Normalize.
      center, scale = stats_per_satellite[sat]
      data[sat] = (tf.cast(data[sat], tf.float32) - center) / scale
      if zero_pad:  # Set masked values to 0.
        data[sat] = tf.where(data[pad_mask_key], 0., data[sat])

    for sat in sat_keys:
      _normalize_and_add_pading_mask_per_satellite(sat)
    # Re-shape modis into 4D.
    data["modis"] = data["modis"][:, None, None, :]
    # Get labels.
    data["labels"] = table.lookup(data.pop("common_name"))
    if binary:  # {0: non-coffee, 1: coffee}.
      data["labels"] = tf.cast(data["labels"] == 0, tf.int32)
    return data

  return _pp


@Registry.register("preprocess_ops.decode_planted", replace=True)
def get_decode_planted(label_key="genus", zero_pad=True, clip=False,
                       data_dir="/placer/prod/home/dune/datasets",
                       dataset="planted_global:4.6.0", variant="global",
                       stats_split="train", sat_keys=PLANTED_SATS,
                       add_frequency=True):
  """Decodes planted centroid example."""
  if label_key == "common_name":
    label_map = planted.LABEL_MAPS[variant]
  elif label_key == "genus":
    label_map = planted.GENUS_LABEL_MAP
    genus_mapping = pp_utils.get_lookup_table(planted.GENUS_MAPPING, "")
  elif label_key == "species":
    label_map = planted.SPECIES_LABEL_MAP
  else:
    raise ValueError(f"Label key `{label_key}` not supported yet.")
  table = pp_utils.get_lookup_table(label_map)
  if add_frequency:
    frequency_splits = {"common_name": planted.FREQUENCY_SPLITS,
                        "genus": planted.get_genus_frequency_splits(),
                        "species": planted.get_species_frequency_splits()}
    frequency_splits = frequency_splits[label_key][variant]
    subsplit_table = pp_utils.get_lookup_table(
        {({"frequent": 2, "common": 1, "rare": 0})[k]: v
         for k, v in frequency_splits.items()})

  center_and_scale_per_satellite = {
      sat: pp_utils.load_normalization_ranges(dataset, stats_split, sat,
                                              data_dir=data_dir)
      for sat in sat_keys}

  def _pp(data):
    def _normalize_and_add_pading_mask_per_satellite(sat):
      # Add padding mask.
      pad_mask_key = f"{sat}_mask"
      if pad_mask_key in data:
        # If available, mask denotes valid pixels, and we need to reverse it.
        data[pad_mask_key] = ~tf.cast(data[pad_mask_key], tf.bool)
      # Normalize.
      center, scale = center_and_scale_per_satellite[sat]
      data[sat] = (tf.cast(data[sat], tf.float32) - center) / scale
      if zero_pad:  # Set masked values to 0.
        data[sat] = tf.where(data[pad_mask_key], 0., data[sat])
      if clip:
        data[sat] = tf.clip_by_value(data[sat], -1., 1.)

    for sat in sat_keys:
      _normalize_and_add_pading_mask_per_satellite(sat)
      # Re-shape modis into 4D.
      if sat == "modis" and data["modis"].ndim == 2:
        data["modis"] = data["modis"][:, None, None, :]

    # Get label.
    if label_key == "genus" and label_key not in data:
      data[label_key] = genus_mapping.lookup(data.pop("common_name"))
    if add_frequency:
      data["subsplit"] = subsplit_table.lookup(data[label_key])
    data["label"] = table.lookup(data.pop(label_key))
    return data

  return _pp


@Registry.register("preprocess_ops.decode_jeo_satellites", replace=True)
def get_decode_jeo_satellites(
    sat_keys, data_dir, dataset, zero_pad=True, clip=False, normalize=True,
    norm_split="train", vmin=-1., vmax=1.):
  """Decodes standardized drivers example."""
  if normalize:
    center_and_scale_per_satellite = {
        sat: pp_utils.load_normalization_ranges(
            dataset, split_name=norm_split, postfix=sat, data_dir=data_dir)
        for sat in sat_keys}

  def _pp(data):
    def _normalize_and_add_pading_mask_per_satellite(sat):
      # Add padding mask.
      pad_mask_key = f"{sat}_mask"
      if pad_mask_key in data:
        # If available, mask denotes valid pixels, and we need to reverse it.
        data[pad_mask_key] = ~tf.cast(data[pad_mask_key], tf.bool)
      # Normalize.
      center, scale = center_and_scale_per_satellite[sat]
      data[sat] = (tf.cast(data[sat], tf.float32) - center) / scale
      if zero_pad:  # Set masked values to 0.
        data[sat] = tf.where(data[pad_mask_key], 0., data[sat])
      if clip:
        data[sat] = tf.clip_by_value(data[sat], vmin, vmax)

    if normalize:
      for sat in sat_keys:
        _normalize_and_add_pading_mask_per_satellite(sat)
    return data

  return _pp
