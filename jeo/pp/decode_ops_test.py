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

"""Tests for decode_ops."""
import copy
from unittest import mock

from absl.testing import parameterized
from jeo.pp import decode_ops
from jeo.pp import pp_utils
import numpy as np
import tensorflow as tf


class DecodeOpsTest(tf.test.TestCase, parameterized.TestCase):

  def tfrun(self, fn, features):
    # Run once as standalone, as could happen eg in colab.
    yield {k: np.array(v) for k, v in fn(copy.deepcopy(features)).items()}

    # And then once again as part of tf.data.Dataset pipeline.
    tf_features = tf.data.Dataset.from_tensors(copy.deepcopy(features))
    for ex in tf_features.map(fn).as_numpy_iterator():
      yield ex

  def test_get_pastis_decode_semantic_simple(self):
    features = {
        "s2": tf.zeros((3, 128, 128, 10), tf.float32),
        "semantics": tf.zeros((128, 128, 1), tf.uint8),
    }
    for out in self.tfrun(decode_ops.get_pastis_decode_semantic_simple(),
                          features):
      self.assertAllEqual(out["image"].shape, [128, 128, 3])
      self.assertAllEqual(out["segmentation_mask"].shape, [128, 128, 1])

  def test_get_decode_pastis_jeo(self):
    features = {
        "s1": tf.ones((3, 12, 12, 2), tf.float32),
        "s2": tf.ones((3, 12, 12, 2), tf.float32),
        "parcel_culture": tf.ones((12, 12, 1), tf.float32),
        "parcel_culture_mask": tf.ones((12, 12, 2), tf.int32),
        "parcel_code": tf.ones((12, 12, 1), tf.float32),
    }
    for out in self.tfrun(decode_ops.get_pastis_jeo_decode(), features):
      self.assertEqual(out["segmentation_mask"].shape, (12, 12, 2))

  @parameterized.parameters(("random", True), ("first", False),
                            ("median", True), (None, False))
  @mock.patch.object(decode_ops, "pp_utils")
  def test_get_decode_fmow_s2(self, temp_select, rgb, mock_pp_utils):
    center = tf.constant(
        [1475.5, 1218.5, 1112.5, 1051.5, 1264.5, 1928.5, 2189.5, 2103.5,
         2374.5, 606.5, 13.5, 1932.5, 1274.5])
    scale = tf.constant(
        [251.3007, 325.4307, 388.4412, 664.9461, 604.9008, 761.3151,
         930.3315, 943.6749, 1040.044, 393.6303, 5.1891, 1016.3223, 942.9336])
    mock_pp_utils.load_normalization_ranges.return_value = (center, scale)
    features = {
        "s2": tf.ones((3, 128, 128, 13), tf.float32),
        "label": tf.zeros((1,), tf.int32),
    }
    expected_shape = [128, 128, 3 if rgb else 13]
    if temp_select is None:
      expected_shape = [3] + expected_shape
    for out in self.tfrun(decode_ops.get_decode_fmow_s2(
        temp_select=temp_select, rgb=rgb), features):
      # Check RGB construction and temporal selection.
      self.assertAllEqual(out["image"].shape, expected_shape)
      # Check normalization.
      if rgb:  # Normalization of the "Blue" channel.
        expected_norm = np.full(expected_shape[:-1], (1-1218.5) / 325.4307)
        self.assertAllClose(out["image"][..., 2], expected_norm)
      else:  # Normalization of the 0-th "Ultra-Blue" channel.
        expected_norm = np.full(expected_shape[:-1], (1-1475.5) / 251.3007)
        self.assertAllClose(out["image"][..., 0], expected_norm)

  @mock.patch.object(pp_utils, "load_normalization_ranges")
  def test_get_decode_planted(self, mock_fn):
    center = tf.constant([1.0, 0.5])
    scale = tf.constant([1.0, 10.0])
    mock_fn.return_value = (center, scale)
    features = {
        "s1": tf.ones((3, 12, 12, 2), tf.float32),
        "s2": tf.ones((3, 12, 12, 2), tf.int64),
        "l7": tf.ones((3, 4, 4, 2), tf.int64),
        "alos": tf.ones((4, 6, 6, 2), tf.float32),
        "modis": tf.ones((3, 1, 1, 2), tf.float32),
        "common_name": tf.constant("Cacao", dtype=tf.string),
    }
    expected_keys = [k for k in features.keys() if k != "common_name"]
    expected_keys += [f"{k}_mask" for k in expected_keys]
    expected_keys += ["label"]
    for out in self.tfrun(decode_ops.get_decode_planted_centroid(), features):
      # Check elements content.
      self.assertCountEqual(list(out.keys()), expected_keys)
      # Check conversion of string to label index (Coffee is at 19).
      self.assertEqual(out["label"], 19)
      # Check normalization.
      self.assertAllClose(out["modis"][0, 0, 0, 0], 0.)
      self.assertAllClose(out["modis"][0, 0, 0, 1], 0.05)
      # Check mask.
      self.assertAllClose(out["modis_mask"][0, 0, 0, 0], False)

  @mock.patch.object(pp_utils, "load_normalization_ranges")
  def test_get_decode_planted_with_frequency(self, _):
    features = {"common_name": tf.constant("Coffee", dtype=tf.string)}
    expected_keys = ["label", "subsplit"]
    for out in self.tfrun(decode_ops.get_decode_planted_centroid(
        version="global:1.0.0", add_frequency=True, sat_keys=[]), features):
      # Check elements content.
      self.assertCountEqual(list(out.keys()), expected_keys)
      # Check conversion of string to label index (Coffee is at 39 in global).
      self.assertEqual(out["label"], 39)
      # Check frequency subsplit (Coffee is "common": 1).
      self.assertEqual(out["subsplit"], 1)

  @parameterized.parameters((True,), (False,))
  @mock.patch.object(pp_utils, "load_normalization_ranges")
  def test_get_decode_coffee(self, binary, mock_fn):
    center = tf.constant([1.0, 0.5])
    scale = tf.constant([1.0, 10.0])
    mock_fn.return_value = (center, scale)
    features = {
        "s1": tf.ones((3, 12, 12, 2), tf.float32),
        "s2": tf.ones((3, 12, 12, 2), tf.float32),
        "l7": tf.ones((3, 4, 4, 2), tf.float32),
        "l8": tf.ones((4, 4, 4, 2), tf.float32),
        "alos": tf.ones((6, 3, 3, 2), tf.float32),
        "nicfi": tf.ones((2, 24, 24, 2), tf.float32),
        "nicfi_monthly": tf.ones((12, 24, 24, 2), tf.float32),
        "modis": tf.ones((3, 2), tf.float32),
        "common_name": tf.constant("Coffee", dtype=tf.string),
    }
    expected_keys = [k for k in features.keys() if k != "common_name"]
    expected_keys += [f"{k}_mask" for k in expected_keys]
    expected_keys += ["labels"]
    for out in self.tfrun(decode_ops.get_decode_coffee(binary), features):
      # Check elements content.
      self.assertCountEqual(list(out.keys()), expected_keys)
      # Check conversion of string to label index.
      self.assertEqual(out["labels"], 1 if binary else 0)
      # Check normalization.
      self.assertAllClose(out["modis"][0, 0, 0, 0], 0.)
      self.assertAllClose(out["modis"][0, 0, 0, 1], 0.05)
      # Check mask.
      self.assertAllClose(out["modis_mask"][0, 0], False)

  @mock.patch.object(pp_utils, "load_normalization_ranges")
  def test_decode_planted(self, mock_fn):
    center = tf.constant([1.0, 0.5])
    scale = tf.constant([1.0, 10.0])
    mock_fn.return_value = (center, scale)
    features = {
        "s1": tf.ones((3, 12, 12, 2), tf.float32),
        "s1_mask": tf.ones((3, 12, 12, 2), tf.uint8),
        "modis": tf.ones((3, 1, 1, 2), tf.float32),
        "modis_mask": tf.ones((3, 1, 1, 2), tf.uint8),
        "common_name": tf.constant("Cacao", dtype=tf.string),
    }
    expected_keys = [k for k in features.keys() if k != "common_name"]
    expected_keys += ["label", "subsplit"]
    for out in self.tfrun(decode_ops.get_decode_planted(
        sat_keys=["s1", "modis"], label_key="common_name"), features):
      # Check elements content.
      self.assertCountEqual(list(out.keys()), expected_keys)
      # Check conversion of string to label index (Cacao is at 62).
      self.assertEqual(out["label"], 62)
      # Check normalization.
      self.assertAllClose(out["modis"][0, 0, 0, 0], 0.)
      self.assertAllClose(out["modis"][0, 0, 0, 1], 0.05)
      # Check mask.
      self.assertAllClose(out["modis_mask"][0, 0, 0, 0], False)

  @mock.patch.object(pp_utils, "load_normalization_ranges")
  def test_decode_planted_genus(self, mock_fn):
    center = tf.constant([1.0, 0.5])
    scale = tf.constant([1.0, 10.0])
    mock_fn.return_value = (center, scale)
    features = {
        "s1": tf.ones((3, 12, 12, 2), tf.float32),
        "s1_mask": tf.ones((3, 12, 12, 2), tf.uint8),
        "modis": tf.ones((3, 1, 1, 2), tf.float32),
        "modis_mask": tf.ones((3, 1, 1, 2), tf.uint8),
        "common_name": tf.constant("Cacao", dtype=tf.string),
    }
    expected_keys = [k for k in features.keys() if k != "common_name"]
    expected_keys += ["label", "subsplit"]
    for out in self.tfrun(decode_ops.get_decode_planted(
        sat_keys=["s1", "modis"], label_key="genus"), features):
      # Check elements content.
      self.assertCountEqual(list(out.keys()), expected_keys)
      # Check conversion to label index (Cacao genus `Theobroma` is at 45).
      self.assertEqual(out["label"], 45)
      # Check normalization.
      self.assertAllClose(out["modis"][0, 0, 0, 0], 0.)
      self.assertAllClose(out["modis"][0, 0, 0, 1], 0.05)
      # Check mask.
      self.assertAllClose(out["modis_mask"][0, 0, 0, 0], False)

  @mock.patch.object(pp_utils, "load_normalization_ranges")
  def test_get_decode_jeo_satellites(self, mock_fn):
    center = tf.constant([1.0, 2.0])
    scale = tf.constant([3.0, 4.0])
    mock_fn.return_value = (center, scale)
    features = {
        "l7": tf.ones((20, 64, 64, 2), tf.float32),
        "l7_mask": tf.ones((20, 64, 64, 2), tf.int32),}

    for out in self.tfrun(decode_ops.get_decode_jeo_satellites(
        ["l7"], "", "some_data_generated_by_jeo_geo_datagen:0.0.1"), features):
      # Check normalization.
      self.assertAllClose(out["l7"][0, 0, 0, 0], (1 - 1) / 3)
      self.assertAllClose(out["l7"][0, 0, 0, 1], (1 - 2) / 4)

if __name__ == "__main__":
  tf.test.main()
