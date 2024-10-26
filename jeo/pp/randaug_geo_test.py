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

"""Tests for randaug_geo."""
import copy
import functools

from absl.testing import parameterized
from jeo.pp import randaug_geo
import numpy as np
import tensorflow as tf

ALL_AVAILABLE_OPS = [
    "AutoContrast",
    "Equalize",
    "Invert",
    "Rotate",
    "Posterize",
    "Solarize",
    "Color",
    "Contrast",
    "Brightness",
    "Sharpness",
    "ShearX", "ShearY",
    "TranslateX", "TranslateY",
    "Cutout",
    "CutoutRandsize",
    "SolarizeAdd",
]


class RandaugGeoTest(tf.test.TestCase, parameterized.TestCase):

  def tfrun(self, fn, img):
    # Run once as standalone, as could happen eg in colab.
    yield np.array(fn(copy.deepcopy(img)))

    # # And then once again as part of tf.data.Dataset pipeline.
    tf_features = tf.data.Dataset.from_tensors(copy.deepcopy(img))
    for ex in tf_features.map(fn).as_numpy_iterator():
      yield ex

  def test_distort_image_with_randaugment(self):
    img = tf.ones((16, 16, 6), dtype=tf.uint8)
    fn = functools.partial(randaug_geo.distort_image_with_randaugment,
                           num_layers=4, magnitude=10)
    for out in self.tfrun(fn, img):
      self.assertAllEqual(out.shape, img.shape)

  @parameterized.parameters(*ALL_AVAILABLE_OPS)
  def test_all_available_ops(self, op_name):
    augmentation_hparams = randaug_geo.HParams(
        cutout_const=10, translate_const=10)
    def run_op(image):
      replace_value = 128 + tf.zeros((tf.shape(image)[2]), dtype=image.dtype)
      func, _, args = randaug_geo._parse_policy_info(
          op_name, 1., 10., replace_value, augmentation_hparams)
      image = func(image, *args)
      return image

    img = tf.ones((16, 16, 6), dtype=tf.uint8)
    for out in self.tfrun(run_op, img):
      self.assertAllEqual(out.shape, img.shape)


if __name__ == "__main__":
  tf.test.main()
