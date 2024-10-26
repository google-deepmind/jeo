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

"""Tests for rand_det_ops."""
import copy

from absl.testing import parameterized
from jeo.pp import rand_det_ops
import numpy as np
import tensorflow as tf


def get_image_data(dtype=tf.uint8):
  img = tf.random.uniform((640, 480, 3), 0, 255, tf.int32)  # Can't ask uint8!?
  return {"image": tf.cast(img, dtype)}


class PpOpsTest(tf.test.TestCase, parameterized.TestCase):

  def tfrun(self, fn, features):
    # Run once as standalone, as could happen eg in colab.
    yield {k: np.array(v) for k, v in fn(copy.deepcopy(features)).items()}

    # And then once again as part of tf.data.Dataset pipeline.
    tf_features = tf.data.Dataset.from_tensors(copy.deepcopy(features))
    for ex in tf_features.map(fn).as_numpy_iterator():
      yield ex

  def test_randu(self):
    for out in self.tfrun(rand_det_ops.get_randu("flip"), {"dummy": 0.}):
      self.assertEqual(out["flip"].shape, ())
      self.assertAllGreaterEqual(out["flip"], 0.0)
      self.assertAllLessEqual(out["flip"], 1.0)

  @parameterized.parameters((2,), (3,), (4,), (5,))
  def test_det_roll(self, n_dim):
    features = {"image": tf.zeros((5,) * n_dim), "rollx": 0.3, "rolly": 0.9}
    for out in self.tfrun(rand_det_ops.get_det_roll(key="image"), features):
      self.assertAllEqual(out["image"].shape, features["image"].shape)

  def test_det_rotate(self):
    features = {
        "3D_modality": tf.ones((20, 20, 1)),
        "4D_modality": tf.ones((2, 20, 20, 9)),
        "5D_modality": tf.ones((2, 2, 20, 20, 3)),
        "angle": 0.5,
    }
    for key in features:
      if key == "angle":
        continue
      fn = rand_det_ops.get_det_rotate(key=key)
      for out in self.tfrun(fn, features):
        expected_shape = features[key].shape.as_list()  # pytype: disable=attribute-error
        self.assertAllEqual(out[key].shape, expected_shape)

  @parameterized.parameters((2), (3), (4), (5))
  def test_random_rot90(self, n_dim):
    channels_offset = 1 if n_dim == 2 else 0
    features = {f"{n_dim}d_modality": tf.ones(range(2, n_dim+2)),}
    for angle, axis in zip((0.0, 0.25, 0.5, 0.75, 1.0),
                           ((-3, -2), (-2, -3), (-3, -2), (-2, -3), (-3, -2),)):
      features["angle"] = angle
      for key in features:
        if key == "angle":
          continue
        for out in self.tfrun(rand_det_ops.get_det_rotate90(key=key), features):
          expected_shape = features[key].shape.as_list()  # pytype: disable=attribute-error
          # adjust shape (may flip h <->w or not):
          h = expected_shape[axis[0] + channels_offset]
          w = expected_shape[axis[1] + channels_offset]
          expected_shape[-3 + channels_offset] = h
          expected_shape[-2 + channels_offset] = w
          self.assertAllEqual(expected_shape, out[key].shape)

  def test_det_flip_lr(self):
    # Test both dtypes to make sure it can be applied correctly to both.
    for dtype in [tf.uint8, tf.float32]:
      image_data = get_image_data(dtype=dtype)
      for out in self.tfrun(rand_det_ops.get_det_fliplr(
          randkey="rand", key="image"), {"rand": 0.1, **image_data}):
        self.assertTrue(np.all(image_data["image"] == out["image"]))
        self.assertEqual(out["image"].dtype, dtype)
      for out in self.tfrun(rand_det_ops.get_det_fliplr(
          randkey="rand", key="image"), {"rand": 0.6, **image_data}):
        self.assertTrue(np.all(image_data["image"][:, ::-1, :] == out["image"]))
        self.assertEqual(out["image"].dtype, dtype)

  @parameterized.parameters(
      ("2D_modality", tf.ones((99, 20)), (20, 99)),
      ("3D_modality", tf.ones((99, 20, 1)), (20, 99, 1)),
      ("4D_modality", tf.ones((2, 99, 20, 9)), (2, 20, 99, 9)),
      ("5D_modality", tf.ones((2, 2, 99, 20, 3)), (2, 2, 20, 99, 3)))
  def test_det_flip_rot(self, key, data, expected_shape):
    flip_rot = 0.2  # Corresponds to no flip and a single rotation.
    fn = rand_det_ops.get_det_flip_rot(key=key)
    # Test both dtypes to make sure it can be applied correctly to both.
    for dtype in [tf.uint8, tf.float32, tf.bool]:
      data = tf.cast(data, dtype)
      for out in self.tfrun(fn, {key: data, "flip_rot": flip_rot}):
        self.assertAllEqual(out[key].shape, expected_shape)

  def test_det_crop(self):
    image = np.arange(36).reshape(6, 6, 1)

    # Check that the crops are consistent across keys for 10 different seeds.
    for _ in range(10):
      data = {"a": image, "b": image, "crop": np.random.uniform(0, 1)}

      crop_a = rand_det_ops.get_det_crop(key="a", crop_size=2)(data)["a"]  # pylint: disable=no-value-for-parameter
      crop_b = rand_det_ops.get_det_crop(key="b", crop_size=2)(data)["b"]  # pylint: disable=no-value-for-parameter

      self.assertEqual(crop_a.shape, (2, 2, 1))
      self.assertEqual(crop_b.shape, (2, 2, 1))
      self.assertTrue(np.all(crop_a == crop_b))

  def test_det_crop_with_four_crops(self):
    crops, crop_size = 4, 12
    data = {"image": tf.ones((64, 64, 3)), "crop": np.random.uniform(0, 1)}
    for out in self.tfrun(rand_det_ops.get_det_crop(
        crop_size, crops, key="image"), data):
      self.assertAllEqual(out["image"].shape, (crops, crop_size, crop_size, 3))

  def test_det_crop_2d(self):
    crops, crop_size = 4, 12
    data = {"image": tf.ones((64, 64)), "crop": np.random.uniform(0, 1)}
    for out in self.tfrun(rand_det_ops.get_det_crop(
        crop_size, crops, key="image"), data):
      self.assertAllEqual(out["image"].shape, (crops, crop_size, crop_size))

  def test_det_crop_4d(self):
    crops, crop_size = 4, 12
    data = {"image": tf.ones((7, 64, 64, 3)), "crop": np.random.uniform(0, 1)}
    for out in self.tfrun(rand_det_ops.get_det_crop(
        crop_size, crops, key="image"), data):
      self.assertAllEqual(out["image"].shape,
                          (crops, 7, crop_size, crop_size, 3))

  @parameterized.parameters(
      ("3D", tf.ones((512, 512, 3))), ("4D", tf.ones((2, 224, 224, 9)))
  )
  def test_det_resize(self, key, data):
    min_ratio = 0.8
    max_ratio = 1.3
    size = data.shape[-2]

    fn = rand_det_ops.get_det_resize(key=key)
    # `resize` chooses the resize value in the given interval.
    for resize in [0.1, 0.3, 0.5, 0.7, 1.0]:
      expected_size = int(size * ((max_ratio - min_ratio) * resize + min_ratio))
      # floating types using bilinear interpolation whereas other uses nearest.
      for dtype in [tf.uint8, tf.float32, tf.bool]:
        data = tf.cast(data, dtype)
        for out in self.tfrun(fn, {key: data, "resize": resize}):
          self.assertAllEqual(out[key].shape[-2], expected_size)
          # self.assertEqual(out[key].dtype, dtype)


if __name__ == "__main__":
  tf.test.main()
