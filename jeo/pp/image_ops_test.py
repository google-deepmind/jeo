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

"""Tests for image preprocessing operations."""

import copy

from absl.testing import parameterized
from jeo.pp import image_ops
import numpy as np
import tensorflow as tf


def get_image_data(dtype=tf.uint8):
  img = tf.random.uniform((640, 480, 3), 0, 255, tf.int32)  # Can't ask uint8!?
  return {"image": tf.cast(img, dtype)}


class PpImageOpsTest(tf.test.TestCase, parameterized.TestCase):

  def tfrun(self, fn, features):
    # Run once as standalone, as could happen eg in colab.
    yield {k: np.array(v) for k, v in fn(copy.deepcopy(features)).items()}

    # And then once again as part of tf.data.Dataset pipeline.
    tf_features = tf.data.Dataset.from_tensors(copy.deepcopy(features))
    for ex in tf_features.map(fn).as_numpy_iterator():
      yield ex

  def test_resize(self):
    for data in self.tfrun(image_ops.get_resize([120, 80]),
                           get_image_data()):
      self.assertEqual(data["image"].shape, (120, 80, 3))

  def test_resize_small(self):
    for data in self.tfrun(image_ops.get_resize_small(240),
                           get_image_data()):
      self.assertEqual(data["image"].shape, (320, 240, 3))

  def test_resize_long(self):
    for data in self.tfrun(image_ops.get_resize_long(320),
                           get_image_data()):
      self.assertEqual(data["image"].shape, (320, 240, 3))

  def test_inception_crop(self):
    for data in self.tfrun(image_ops.get_inception_crop(),
                           get_image_data()):
      self.assertEqual(data["image"].shape[-1], 3)

  @parameterized.parameters(10, [[10, 8]])
  def test_random_crop(self, crop_size):
    for data in self.tfrun(
        image_ops.get_random_crop(crop_size), get_image_data()
    ):
      if isinstance(crop_size, int):
        crop_size = (crop_size, crop_size)
      self.assertEqual(data["image"].shape, (*crop_size, 3))

  @parameterized.parameters(10, [[10, 8]])
  def test_multiple_random_crop(self, crop_size):
    for data in self.tfrun(
        image_ops.get_random_crop(crop_size, 10), get_image_data()
    ):
      if isinstance(crop_size, int):
        crop_size = (crop_size, crop_size)
      self.assertEqual(data["image"].shape, (10, *crop_size, 3))

  def test_central_crop(self):
    for data in self.tfrun(image_ops.get_central_crop([20, 80]),
                           get_image_data()):
      self.assertEqual(data["image"].shape, (20, 80, 3))

  def test_central_crop_on_2d(self):
    features = {"mask": tf.random.uniform((640, 480), 0, 2, tf.int32)}
    for data in self.tfrun(image_ops.get_central_crop([20, 80], key="mask"),
                           features):
      self.assertEqual(data["mask"].shape, (20, 80))

  @parameterized.parameters(
      [5, 3, "VALID", True],
      [10, 5, "SAME", False],
      [7, 3, "VALID", False],
      [12, 5, "SAME", True],
  )
  def test_extract_patches(self, n, crop_size, padding, flatten):
    # NxN image with values 1, 2, 3, ..., N**2
    image_data = dict(
        image=tf.constant(
            [[[[x * n + y + 1] for y in range(n)] for x in range(n)]]
        )
    )
    for data in self.tfrun(
        image_ops.get_extract_patches(crop_size, padding, flatten), image_data
    ):
      # The number of patches extracted.
      num_out = (n - crop_size + 1) ** 2 if padding == "VALID" else n**2

      # Check that the values in the first patch are as expected.
      expected_values = np.array([[
          [[x * n + y + 1] for y in range(crop_size)] for x in range(crop_size)
      ]])
      if padding == "SAME":
        # Account for the presence of the zeros on left and top.
        k = int(crop_size / 2)
        expected_values = expected_values.squeeze()[:-k, :-k]
      self.assertEqual(data["image"][0].squeeze().sum(), expected_values.sum())

      shape = (crop_size**2,) if flatten else (crop_size, crop_size)
      self.assertEqual(data["image"].shape, (num_out, *shape, 1))

  def test_random_flip_lr(self):
    data_orig = get_image_data()
    for data in self.tfrun(image_ops.get_random_flip_lr(), data_orig):
      self.assertTrue(
          np.all(data_orig["image"].numpy() == data["image"]) or
          np.all(data_orig["image"].numpy() == data["image"][:, ::-1]))

  @parameterized.parameters((10), (10, 8))
  def test_simclr_crop_and_resize(self, height, width=None):
    features = {
        "s1": tf.ones((20, 20, 1)),
        "s2": tf.ones((7, 20, 20, 3)),
        "l8": tf.ones((20, 20, 3)),
    }
    for key in features:
      fn = image_ops.get_simclr_crop_and_resize(
          height=height, width=width, key=key
      )
      for out in self.tfrun(fn, features):
        expected_shape = features[key].shape.as_list()
        expected_shape[-2] = width if width else height
        expected_shape[-3] = height
        self.assertAllEqual(out[key].shape, expected_shape)

  def test_simclr_random_color_jitter(self):
    # TODO(crisnv): get_simclr_random_color_jitter support > 3 dimentions
    features = {
        "s1": tf.ones((20, 20, 1)),
        # "s2": tf.ones((7, 20, 20, 3)),
        "l8": tf.ones((20, 20, 3)),
    }
    for key in features:
      fn = image_ops.get_simclr_random_color_jitter(
          p=1.0, strength=0.5, key=key
      )
      for out in self.tfrun(fn, features):
        expected_shape = features[key].shape.as_list()
        self.assertAllEqual(out[key].shape, expected_shape)

  @parameterized.parameters((3), (4), (5))
  def test_random_drop_all_channel(self, n_dim):
    features = {
        f"{n_dim}d_modality": tf.ones(range(2, n_dim + 2)),
    }
    for key in features:
      for a in range(features[key].ndim):
        # test all channels erased
        fn = image_ops.get_random_fill_along_dim(
            key=key, axis=a, probability=1.0
        )
        for out in self.tfrun(fn, features):
          expected_shape = features[key].shape.as_list()
          self.assertAllEqual(out[key].shape, expected_shape)
          self.assertAllEqual(
              out[key], tf.zeros_like(out[key], dtype=out[key].dtype)
          )

  @parameterized.parameters((3), (4), (5))
  def test_random_drop_zero_channels(self, n_dim):
    # test no channels are erased when prob is 0, and shape is preserved
    features = {
        f"{n_dim}d_modality": tf.ones(range(2, n_dim + 2)),
    }
    for key in features:
      for a in range(features[key].ndim):
        fn = image_ops.get_random_fill_along_dim(
            key=key, axis=a, probability=0.0
        )
        for out in self.tfrun(fn, features):
          expected_shape = features[key].shape.as_list()
          self.assertAllEqual(out[key].shape, expected_shape)
          self.assertAllEqual(
              out[key], tf.ones_like(out[key], dtype=out[key].dtype)
          )

        fn = image_ops.get_random_fill_along_dim(
            key=key, axis=a, probability=0.5
        )
        for out in self.tfrun(fn, features):
          expected_shape = features[key].shape.as_list()
          self.assertAllEqual(out[key].shape, expected_shape)

  @parameterized.parameters((3), (4), (5))
  def test_random_drop_negative_indexes(self, n_dim):
    # test that shape is preserved when using of negative indexes
    features = {
        f"{n_dim}d_modality": tf.ones(range(2, n_dim + 2)),
    }
    for key in features:
      for a in range(-1, -1 - features[key].ndim, -1):
        fn = image_ops.get_random_fill_along_dim(
            key=key, axis=a, probability=0.5
        )
        for out in self.tfrun(fn, features):
          expected_shape = features[key].shape.as_list()
          self.assertAllEqual(out[key].shape, expected_shape)

  def test_get_flip_ud_with_label(self):
    img = np.tri(3)[:, :, None]  # (3, 3, 1) diag image.
    flipped = img[::-1, :, :]  # Flipped vertically.
    data = {"image": tf.constant(img)}
    for out in self.tfrun(image_ops.get_flip_ud_with_label(), data):
      self.assertCountEqual(list(out.keys()), ["image", "flipped"])
      if out["flipped"] == 1:
        np.testing.assert_array_equal(out["image"], flipped)
      else:
        np.testing.assert_array_equal(out["image"], img)

  def test_get_randaug_geo(self):
    data = {"image": tf.constant(np.zeros((224, 224, 6)), tf.uint8)}
    for out in self.tfrun(image_ops.get_randaug_geo(), data):
      self.assertEqual(out["image"].shape, data["image"].shape)

  def test_get_cutout(self):
    data = {"image": tf.constant(np.zeros((224, 224, 6)), tf.uint8)}
    for out in self.tfrun(image_ops.get_cutout(20, 128), data):
      self.assertEqual(out["image"].shape, data["image"].shape)

  def test_get_cutout_randsize(self):
    data = {"image": tf.constant(np.zeros((224, 224, 6)), tf.uint8)}
    for out in self.tfrun(image_ops.get_cutout_randsize(0.1, 0.5), data):
      self.assertEqual(out["image"].shape, data["image"].shape)

  def test_get_cutout_randsize_deterministic(self):
    data = {"image": tf.constant(np.ones((224, 224, 6)), tf.uint8)}
    for out in self.tfrun(image_ops.get_cutout_randsize(2., 2., 99), data):
      self.assertEqual(out["image"].shape, data["image"].shape)
      self.assertAllEqual(out["image"], data["image"] * 99)

  def test_get_random_resize(self):
    data = {"image": tf.constant(np.ones((224, 224, 6)), tf.float32),
            "padding_mask": tf.constant(np.ones((224, 224)), tf.int32),}
    for out in self.tfrun(image_ops.get_random_resize(10, .5, .5), data):
      self.assertEqual(out["image"].shape, (5, 5, 6))
      self.assertEqual(out["padding_mask"].shape, (5, 5))

  def test_random_flip_ud(self):
    data = {"image": tf.random.normal((224, 224, 6), 0, 1, tf.float32)}
    for out in self.tfrun(image_ops.get_random_flip_ud(), data):
      self.assertTrue(
          np.all(data["image"].numpy() == out["image"]) or
          np.all(data["image"].numpy() == out["image"][::-1, :]))

  def test_random_rot90(self):
    data = {"image": tf.random.normal((224, 224, 6), 0, 1, tf.float32)}
    for out in self.tfrun(image_ops.get_random_rotation90(), data):
      self.assertTrue(
          np.all(data["image"].shape == out["image"].shape) or
          np.all(data["image"].shape == out["image"].shape[1, 0, 2]))

  def test_random_rot90_2d(self):
    data = {"image": tf.random.normal((224, 128), 0, 1, tf.float32)}
    for out in self.tfrun(image_ops.get_random_rotation90(), data):
      self.assertTrue(
          np.all(out["image"].shape == (224, 128)) or
          np.all(out["image"].shape == (128, 224)))

  def test_clear_boundary(self):
    size = 128
    data = {"label": tf.ones((size, size))}
    margin = 32
    for inv in [False, True]:
      for out in self.tfrun(
          image_ops.get_clear_boundary(inkey="label", outkey="test",
                                       margin=margin, inverse=inv), data):
        self.assertEqual(out["test"].shape, (size, size))
        print(out["test"])
        self.assertTrue(
            np.all(out["test"][margin: -margin, margin: -margin] == (
                0 if inv else 1)))
        self.assertTrue(np.all(out["test"][:margin] == (1 if inv else 0)))
        self.assertTrue(np.all(out["test"][-margin:] == (1 if inv else 0)))
        self.assertTrue(np.all(out["test"][..., :margin] == (1 if inv else 0)))
        self.assertTrue(np.all(out["test"][..., -margin:] == (1 if inv else 0)))

  def test_get_tanh_value_range(self):
    data = {"image": tf.ones((224, 224, 2), tf.float32)}
    th_1 = tf.tanh(1.0).numpy()
    for out in self.tfrun(
        image_ops.get_tanh_value_range(mean=(th_1, 0), std=(1, th_1)), data
    ):
      self.assertEqual(out["image"].shape, (224, 224, 2))
      # Check correct mean subtraction.
      self.assertTrue((out["image"][..., 0] == 0.0).all())
      # Check correct std scaling.
      self.assertTrue((out["image"][..., 1] == 1.0).all())

  def test_get_pansharpen(self):
    """Test that different keys leads to same pansharpening."""

    rgbp = np.random.rand(64, 64, 4).astype("float32")

    res = []
    for keys, data in (
        (["rgbp"], dict(rgbp=rgbp)),
        (["rgb", "p"], dict(rgb=rgbp[..., :3], p=rgbp[..., -1])),
        (["r", "g", "b", "p"], dict(r=rgbp[..., 0], g=rgbp[..., 1],
                                    b=rgbp[..., 2], p=rgbp[..., 3]))
    ):
      for out in self.tfrun(
          image_ops.get_pansharpen(keys=keys), data
      ):
        res.append(out["image"])

    a = res.pop(0)
    for x in res:
      self.assertTrue(np.isclose(a, x).all())


if __name__ == "__main__":
  tf.test.main()
