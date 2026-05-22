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

"""Datasets decoding preprocessing ops."""

from absl import logging
from jeo.pp import pp_utils
from jeo.pp.pp_builder import Registry  # pylint: disable=g-importing-member
import tensorflow as tf


@Registry.register("preprocess_ops.decode_jeo_satellites", replace=True)
def get_decode_jeo_satellites(
    sat_keys, data_dir, dataset, zero_pad=True, clip=False, normalize=True,
    norm_split="train", vmin=-1., vmax=1., centers=None, scales=None,
    norm_overrides=None):
  """Decodes standardized drivers example."""
  if normalize:
    centers = centers or {}
    scales = scales or {}
    norm_overrides = norm_overrides or {}
    center_and_scale_per_satellite = {}
    for sat in sat_keys:
      sat_for_stats = norm_overrides.get(sat, sat)
      is_small_valued_data = (sat_for_stats.startswith("efm") or
                              sat_for_stats.startswith("ccdc"))
      center_default = "mean" if is_small_valued_data else "bins_median"
      scale_default = "std" if is_small_valued_data else "bins_mad_std"
      center_and_scale_per_satellite[sat] = pp_utils.load_normalization_ranges(
          dataset, split_name=norm_split, postfix=sat_for_stats,
          data_dir=data_dir,
          center=centers.get(sat_for_stats, center_default),
          scale=scales.get(sat_for_stats, scale_default))

  def _pp(data):
    for sat in sat_keys:
      if normalize:
        if sat.endswith("_mask"):  # Keep original mask (1: valid, 0: invalid).
          continue
        center, scale = center_and_scale_per_satellite[sat]  # pytype: disable=name-error
        if center is not None and scale is not None:
          data[sat] = (tf.cast(data[sat], tf.float32) - center) / scale
        if clip:  # Clipping values only if normalizing data.
          data[sat] = tf.clip_by_value(data[sat], vmin, vmax)
      # Add padding mask.
      pad_mask_key = f"{sat}_mask"
      if pad_mask_key in data:
        # If available, mask denotes valid pixels, and we need to reverse it.
        data[pad_mask_key] = ~tf.cast(data[pad_mask_key], tf.bool)
        if zero_pad:  # Set masked values to 0.
          data[sat] = tf.where(
              data[pad_mask_key], tf.cast(0.0, data[sat].dtype), data[sat]
          )
      elif zero_pad:
        logging.warning("No mask available for satellite %s.", sat)
    return data

  return _pp
