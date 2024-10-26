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

"""Tests for sat_ops."""

import copy

from absl.testing import parameterized
from jeo.pp import sat_ops
import numpy as np
import tensorflow as tf


class SatOpsTest(tf.test.TestCase, parameterized.TestCase):

  def tfrun(self, fn, features):
    # Run once as standalone, as could happen eg in colab.
    yield {k: np.array(v) for k, v in fn(copy.deepcopy(features)).items()}

    # And then once again as part of tf.data.Dataset pipeline.
    tf_features = tf.data.Dataset.from_tensors(copy.deepcopy(features))
    for ex in tf_features.map(fn).as_numpy_iterator():
      yield ex

  def test_pad_seq(self):
    features = {
        "s2": tf.zeros((3, 128, 128, 10), tf.float32),
        "semantics": tf.zeros((128, 128, 1), tf.uint8),
        "s2_days": tf.zeros((3,), tf.int32),
    }
    for out in self.tfrun(sat_ops.get_pad_seq(["s2", "s2_days"], 5), features):
      self.assertAllEqual(out["s2"].shape, [5, 128, 128, 10])
      self.assertAllEqual(out["semantics"].shape, [128, 128, 1])
      self.assertAllEqual(out["s2_days"].shape, [5])

  def test_pad_merged_tensors(self):
    features = {
        "s2": tf.zeros((4, 12, 12, 9), tf.float32),
        "s1": tf.zeros((4, 12, 12, 3), tf.float32),
        "l7": tf.zeros((20, 4, 4, 6), tf.float32),
        "modis": tf.zeros((36, 1, 1, 11), tf.float32),
    }
    keys = tuple(features.keys())

    def op_sequence(x):
      x = sat_ops.get_merge_spatial_dimensions(keys=keys)(x)
      x = sat_ops.get_pad_tensors_to_match(keys=keys, dims_to_pad=(0, 2))(x)
      return x

    for out in self.tfrun(op_sequence, features):
      self.assertAllEqual(out["s2"].shape, (36, 144, 11))
      self.assertAllEqual(out["s1"].shape, (36, 144, 11))
      self.assertAllEqual(out["l7"].shape, (36, 16, 11))
      self.assertAllEqual(out["modis"].shape, (36, 1, 11))

  def test_pad_tensors(self):
    features = {
        "s2": tf.zeros((4, 12, 12, 9), tf.float32),
        "s1": tf.zeros((4, 12, 12, 3), tf.float32),
        "l7": tf.zeros((20, 4, 4, 6), tf.float32),
        "modis": tf.zeros((36, 1, 1, 11), tf.float32),
    }
    keys = tuple(features.keys())

    for out in self.tfrun(
        sat_ops.get_pad_tensors_to_match(keys=keys, dims_to_pad=(0, 3)),
        features,
    ):
      self.assertAllEqual(out["s2"].shape, (36, 12, 12, 11))
      self.assertAllEqual(out["s1"].shape, (36, 12, 12, 11))
      self.assertAllEqual(out["l7"].shape, (36, 4, 4, 11))
      self.assertAllEqual(out["modis"].shape, (36, 1, 1, 11))

  def test_get_merge_spatial_dimensions(self):
    t, h, w, c = 3, 128, 128, 10
    features = {"image": tf.zeros((t, h, w, c), tf.float32)}
    for out in self.tfrun(sat_ops.get_merge_spatial_dimensions(), features):
      self.assertEqual(out["image"].shape, (t, h * w, c))

  def test_get_merge_spatial_dimensions_multiple(self):
    t, h, w, c = 3, 128, 128, 10
    t2, h2, w2, c2 = 7, 32, 32, 3
    features = {"s1": tf.zeros((t, h, w, c), tf.float32),
                "s2": tf.zeros((t2, h2, w2, c2), tf.float32)}
    for out in self.tfrun(sat_ops.get_merge_spatial_dimensions(["s1", "s2"]),
                          features):
      self.assertEqual(out["s1"].shape, (t, h * w, c))
      self.assertEqual(out["s2"].shape, (t2, h2 * w2, c2))

  @parameterized.parameters(
      ("b h w c -> b h w c", (32, 30, 40, 3)),
      ("b h w c -> (b h) w c", (960, 40, 3)),
      ("b h w c -> b c h w", (32, 3, 30, 40)),
      ("b h w c -> b (c h w)", (32, 3600)),
  )
  def test_get_rearrange(self, pattern, expected_shape):
    t, h, w, c = 32, 30, 40, 3
    features = {"image": tf.zeros((t, h, w, c), tf.float32)}
    for out in self.tfrun(sat_ops.get_rearrange(pattern), features):
      self.assertEqual(out["image"].shape, expected_shape)

  @parameterized.parameters(
      ("b (h1 h) (w1 w) c -> (b h1 w1) h w c", 2, 2, (128, 15, 20, 3)),
      ("b (h h1) (w w1) c -> b h w (c h1 w1)", 2, 2, (32, 15, 20, 12)))
  def test_get_rearrange_with_kwargs(self, pattern, h1, w1, expected_shape):
    t, h, w, c = 32, 30, 40, 3
    features = {"image": tf.zeros((t, h, w, c), tf.float32)}
    for out in self.tfrun(sat_ops.get_rearrange(pattern, h1=h1, w1=w1),
                          features):
      self.assertEqual(out["image"].shape, expected_shape)

  @parameterized.parameters(("median",), ("random",), ("first",), ("last",))
  def test_get_reduce_temporal(self, kind):
    t, h, w, c = 4, 32, 32, 9
    features = {"image": tf.ones((t, h, w, c), tf.float32)}
    for out in self.tfrun(sat_ops.get_reduce_temporal(kind=kind), features):
      self.assertEqual(out["image"].shape, (h, w, c))

  def test_get_reduce_temporal_latest(self):
    t, h, w, c = 4, 32, 32, 9
    features = {
        "image": tf.ones((t, h, w, c), tf.float32),
        "timestamp": tf.random.uniform((t,), dtype=tf.float32),
    }
    for out in self.tfrun(
        sat_ops.get_reduce_temporal(kind="latest", timestamp_key="timestamp"),
        features,
    ):
      self.assertEqual(out["image"].shape, (h, w, c))

  def test_get_add_lat_lon_channels(self):
    features = {
        "l7": tf.ones((20, 64, 64, 2), tf.float32),
        "lat": 30.0,
        "lon": 60.0
    }

    for out in self.tfrun(sat_ops.get_add_lat_lon_channels("l7"), features):
      self.assertEqual(out["l7"].shape, (20, 64, 64, 5))

  @parameterized.named_parameters(
      dict(testcase_name="resize_height", width=64, height=32),
      dict(
          testcase_name="resize_height_custom_axis",
          width=32,
          height=96,
          height_axis=1,
          width_axis=2,
      ),
      dict(testcase_name="resize_width", width=32, height=96),
      dict(
          testcase_name="resize_width_custom_axis",
          width=64,
          height=32,
          height_axis=2,
          width_axis=1,
      ),
      dict(testcase_name="resize_height_and_width", height=32, width=8),
  )
  def test_resize_spatial_dims(
      self, width: int, height: int, height_axis: int = -2, width_axis: int = -3
  ):
    keys_to_resize = ["s1", "s2"]
    current_dims = (3, width, height, 5)
    reference_key = "reference"
    ref_dims = (32, 64, 96, 10)
    data = {
        reference_key: tf.zeros(ref_dims, tf.float32),
        "s1": tf.zeros(current_dims, tf.float32),
        "s2": tf.zeros(current_dims, tf.float32),
    }

    expected_dims = [3, width, height, 5]
    expected_dims[height_axis] = ref_dims[height_axis]
    expected_dims[width_axis] = ref_dims[width_axis]
    expected_dims = tuple(expected_dims)
    for out in self.tfrun(
        sat_ops.get_scale_spatial_dims_like(
            keys_to_resize, reference_key, height_axis, width_axis
        ),
        data,
    ):
      self.assertCountEqual(out.keys(), data.keys())
      self.assertEqual(out["s1"].shape, expected_dims)
      self.assertEqual(out["s2"].shape, expected_dims)
      self.assertEqual(out[reference_key].shape, ref_dims)

  def test_resize_spatial_dims_raises_errors(self):
    with self.subTest("same_height_and_width_axes"):
      data = {
          "reference": tf.zeros((32, 64, 96, 10), tf.float32),
          "s1": tf.zeros((3, 8, 32, 5), tf.float32),
      }
      with self.assertRaisesRegex(ValueError, "height and width axes"):
        for _ in self.tfrun(
            sat_ops.get_scale_spatial_dims_like(["s1"], "reference", 1, -3),
            data,
        ):
          pass

    with self.subTest("ref_height_smaller"):
      data = {
          "reference": tf.zeros((32, 64, 96, 10), tf.float32),
          "s1": tf.zeros((3, 64, 192, 5), tf.float32),
      }
      with self.assertRaisesRegex(
          ValueError, "Reference height dimension must be greater"
      ):
        for _ in self.tfrun(
            sat_ops.get_scale_spatial_dims_like(["s1"], "reference"),
            data,
        ):
          pass

    with self.subTest("ref_height_divisible"):
      data = {
          "reference": tf.zeros((32, 64, 96, 10), tf.float32),
          "s1": tf.zeros((3, 64, 88, 5), tf.float32),
      }
      with self.assertRaisesRegex(
          ValueError, "Reference height dimension 96 must be divisible"
      ):
        for _ in self.tfrun(
            sat_ops.get_scale_spatial_dims_like(["s1"], "reference"),
            data,
        ):
          pass

    with self.subTest("ref_width_smaller"):
      data = {
          "reference": tf.zeros((32, 64, 96, 10), tf.float32),
          "s1": tf.zeros((3, 65, 96, 5), tf.float32),
      }
      with self.assertRaisesRegex(
          ValueError, "Reference width dimension must be greater"
      ):
        for _ in self.tfrun(
            sat_ops.get_scale_spatial_dims_like(["s1"], "reference"),
            data,
        ):
          pass

    with self.subTest("ref_width_divisible"):
      data = {
          "reference": tf.zeros((32, 64, 96, 10), tf.float32),
          "s1": tf.zeros((3, 63, 96, 5), tf.float32),
      }
      with self.assertRaisesRegex(
          ValueError, "Reference width dimension 64 must be divisible"
      ):
        for _ in self.tfrun(
            sat_ops.get_scale_spatial_dims_like(["s1"], "reference"),
            data,
        ):
          pass

  def test_s2_to_rgb_shape(self):
    features = {
        "s2": tf.zeros((4, 128, 128, 10), tf.int32),
    }
    for out in self.tfrun(sat_ops.get_s2_to_rgb(), features):
      self.assertAllEqual(out["rgb"].shape, [128, 128, 3])
      self.assertEqual(out["rgb"].dtype, tf.float64)

  def test_s2_to_rgb_values(self):
    l = tf.zeros(shape=[7, 128, 128], dtype=float)
    p = tf.ones(shape=[128, 128], dtype=float) * 3000.0
    q = tf.ones(shape=[128, 128], dtype=float) * 1500.0
    r = tf.ones(shape=[128, 128], dtype=float) * 750.0
    s2 = tf.stack([p, q, r, *tf.unstack(l)], axis=-1)
    features = {"s2": s2}
    for out in self.tfrun(sat_ops.get_s2_to_rgb(temporal=""), features):
      self.assertAllEqual(out["rgb"][0][0], [0.25, 0.5, 1.0])

  @parameterized.parameters(
      dict(
          features={"s2": tf.zeros((1, 4, 128, 128, 10), tf.int32)},
          temporal="first",
          error_string="Rank must be 3, not 5"
      ),
      dict(
          features={"s2": tf.zeros((4, 128, 128, 10), tf.int32)},
          temporal="second",
          error_string="Unknown temporal reduce: second"
      ),
  )
  def test_s2_to_rgb_incorrect_input(self, features, temporal, error_string):
    with self.assertRaisesRegex(ValueError, error_string):
      for _ in self.tfrun(sat_ops.get_s2_to_rgb(temporal=temporal), features):
        pass


if __name__ == "__main__":
  tf.test.main()
