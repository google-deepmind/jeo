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

import copy

from absl.testing import parameterized
from jeo.pp import pp_ops
import numpy as np
import tensorflow as tf


class PpOpsTest(tf.test.TestCase, parameterized.TestCase):

  def tfrun(self, fn, features):
    # Run once as standalone, as could happen eg in colab.
    yield {k: np.array(v) for k, v in fn(copy.deepcopy(features)).items()}

    # And then once again as part of tf.data.Dataset pipeline.
    tf_features = tf.data.Dataset.from_tensors(copy.deepcopy(features))
    for ex in tf_features.map(fn).as_numpy_iterator():
      yield ex

  def test_squeeze_dim(self):
    features = {"image": tf.constant(np.zeros((1, 224, 224, 3)))}
    for out in self.tfrun(pp_ops.get_squeeze_dim("image", axis=0), features):
      self.assertAllEqual(out["image"].shape, [224, 224, 3])

  def test_expand_dim(self):
    features = {"arr": tf.constant(np.zeros((3, 3))),
                "seq": tf.constant(np.zeros((7,)))}
    for out in self.tfrun(pp_ops.get_expand_dim("arr", "seq"), features):
      self.assertAllEqual(out["arr"].shape, [1, 3, 3])
      self.assertAllEqual(out["seq"].shape, [1, 7])

  def test_rename(self):
    features = {"a": tf.constant([1]),
                "b": tf.constant([2]), "c": tf.constant([3]),}
    for out in self.tfrun(pp_ops.get_rename(a="X", b="Y"), features):
      self.assertEqual(list(out.keys()), ["X", "Y", "c"])

  def test_concat(self):
    features = {
        "s1": tf.ones((20, 20, 1)),
        "s2": tf.ones((20, 20, 3)),
        "l8": tf.ones((20, 20, 3)),
    }
    for out in self.tfrun(pp_ops.get_concat(["s1", "s2"], outkey="img"),
                          features):
      self.assertCountEqual(list(out.keys()), ["img", "l8"])
      self.assertAllEqual(out["img"].shape, [20, 20, 4])

  def test_concat_broadcast(self):
    features = {
        "a": tf.ones((4, 3, 3, 2)),
        "b": tf.ones((1, 3, 3, 1)),
        "c": tf.ones((1, 3, 3, 2)),
    }
    concat_op = pp_ops.get_concat(
        ["a", "b", "c"], axis=-1, outkey="img", broadcast=True)
    for out in self.tfrun(concat_op, features):
      self.assertCountEqual(list(out.keys()), ["img"])
      self.assertAllEqual(out["img"].shape, [4, 3, 3, 5])

  def test_get_keep(self):
    data = {"s1": 1, "s2": 2, "l8": 3}
    for data_keep in self.tfrun(pp_ops.get_keep(["s1", "l8"]), data):
      self.assertAllEqual(set(data_keep.keys()), {"s1", "l8"})

  def test_get_drop(self):
    data = {"s1": 1, "s2": 2, "l8": 3}
    for data_drop in self.tfrun(pp_ops.get_drop(["s1", "l8"]), data):
      self.assertAllEqual(set(data_drop.keys()), {"s2"})

  def test_ensure_4d(self):
    data = {"2d": tf.ones((1, 2), tf.float32),
            "3d": tf.ones((1, 2, 3), tf.float32),
            "4d": tf.ones((1, 2, 3, 4), tf.float32),}
    for out in self.tfrun(pp_ops.get_ensure_4d(["3d", "4d"]), data):
      self.assertAllEqual(out["2d"].shape, [1, 2])
      self.assertAllEqual(out["3d"].shape, [1, 1, 2, 3])
      self.assertAllEqual(out["4d"].shape, [1, 2, 3, 4])

  @parameterized.parameters(((24, 24, 6),), ((None, None, 6),))
  def test_ensure_shape(self, shape):
    data = {"image": tf.ones((24, 24, 6), tf.float32)}
    for out in self.tfrun(pp_ops.get_ensure_shape(shape, key="image"), data):
      self.assertCountEqual(out.keys(), ["image"])

  @parameterized.named_parameters(
      dict(
          testcase_name="multiple_keys",
          data_inputs={
              "k_int": tf.constant((1, 0, 2)),
              "k_bool": tf.constant((True, False)),
          },
          expected_outputs={
              "k_int": tf.constant((0, 1, 0)),
              "k_bool": tf.constant((False, True)),
          },
      ),
      dict(
          testcase_name="single_key",
          data_inputs={"k_bool": tf.constant((True, False))},
          expected_outputs={"k_bool": tf.constant((False, True))},
      ),
  )
  def test_invert_bool(self, data_inputs, expected_outputs):
    for out in self.tfrun(
        pp_ops.get_invert_bool(data_inputs.keys()), data_inputs
    ):
      self.assertDictEqual(out, expected_outputs)

  @parameterized.parameters(((24, 34, 6),), ((None, None, 8),))
  def test_ensure_shape_fails(self, shape):
    data = {"image": tf.ones((24, 24, 6), tf.float32)}
    with self.assertRaises(tf.errors.InvalidArgumentError):
      for _ in self.tfrun(pp_ops.get_ensure_shape(shape, key="image"), data):
        pass

  @parameterized.parameters(([0], [1]), (0, []), ([3, 2, 1], [3]))
  def test_select_channels(self, s1_ind, expected):
    features = {
        "s1": tf.zeros((6, 20, 20, 9)),
        "s2": tf.zeros((20, 20, 13))}
    key_channels = {"s1": s1_ind, "s2": [3, 2, 1]}
    for out in self.tfrun(pp_ops.get_select_channels(key_channels), features):
      self.assertAllEqual(out["s1"].shape, [6, 20, 20] + expected)
      self.assertAllEqual(out["s2"].shape, [20, 20, 3])

  @parameterized.parameters((["B1"], [1]), (["B3", "B2", "B1"], [3]))
  def test_select_channels_by_name(self, s2_ind, expected):
    features = {
        "s1": tf.zeros((6, 20, 20, 3)),
        "s2": tf.zeros((5, 20, 20, 10))}
    band_names = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B10"]
    for out in self.tfrun(pp_ops.get_select_channels_by_name(
        "s2", s2_ind, band_names), features):
      self.assertAllEqual(out["s1"].shape, [6, 20, 20, 3])
      self.assertAllEqual(out["s2"].shape, [5, 20, 20] + expected)

  def test_extract_channels(self):
    features = {"s2": tf.zeros((20, 20, 13))}
    key_channels = {"rgb": [0, 1, 2], "p": [3], "n": [4]}
    for out in self.tfrun(
        pp_ops.get_extract_channels("s2", key_channels), features
    ):
      self.assertAllEqual(out["s2"].shape, [20, 20, 13])
      self.assertAllEqual(out["rgb"].shape, [20, 20, 3])
      self.assertAllEqual(out["p"].shape, [20, 20, 1])
      self.assertAllEqual(out["n"].shape, [20, 20, 1])

  @parameterized.parameters(([7.],), ([7],), (7,))
  def test_meta_as_channel(self, value):
    features = {"lat": tf.constant(value), "img": tf.zeros((2, 2, 3))}
    for out in self.tfrun(pp_ops.get_add_meta_as_channel("lat", "img"),
                          features):
      self.assertAllEqual(out["img"].shape, [2, 2, 4])
      expected = np.zeros((2, 2, 4))
      expected[:, :, -1] = 7
      self.assertAllEqual(out["img"], expected)

  def test_remap_ints(self):
    features = {"image": tf.constant((0, 1, 2, 2, 3))}
    mapping = [0, 2, 1, 1]  # 0->0, 1->2, 2->1, 3->1
    for out in self.tfrun(pp_ops.get_remap_ints(
        mapping, key="image"), features):
      self.assertAllEqual(out["image"].shape, features["image"].shape)
      expected = np.array([0, 2, 1, 1, 1])
      self.assertAllEqual(out["image"], expected)

  def test_ensure_remap_ints_fails(self):
    # Number of unique labels is higher than lenght of mapping.
    features = {"image": tf.constant((0, 1, 2, 2, 3, 4))}
    mapping = [0, 2, 1, 1]  # 0->0, 1->2, 2->1, 3->1, 4->MISSING
    with self.assertRaises(tf.errors.InvalidArgumentError):
      for _ in self.tfrun(pp_ops.get_remap_ints(
          mapping, key="image"), features):
        pass

  def test_remap_with_str_list(self):
    features = {"label": tf.constant(["A", "A", "B", "C", "Unknown"])}
    mapping = ["A", "B", "C"]  # Index is the new mapped value.
    expected = np.array([0, 0, 1, 2, -1])  # Unknown is mapped to default "-1".
    for out in self.tfrun(
        pp_ops.get_remap(mapping, key="label", default_value=-1), features
    ):
      self.assertAllEqual(out["label"].shape, features["label"].shape)
      self.assertAllEqual(out["label"], expected)

  def test_remap_with_dict(self):
    features = {"image": tf.constant([0, 1, 2, 2, 3, 4])}
    mapping = {0: 0, 1: 2, 2: 1, 3: 1}  # 0->0, 1->2, 2->1, 3->1, others->99
    expected = np.array([0, 2, 1, 1, 1, 99])
    for out in self.tfrun(pp_ops.get_remap(mapping, 99, key="image"), features):
      self.assertAllEqual(out["image"].shape, features["image"].shape)
      self.assertAllEqual(out["image"], expected)

  def test_remap_with_dict_and_default(self):
    features = {"image": tf.constant([0, 1, 2, 2, 3, 4])}
    mapping = {0: 0, 1: 2, 2: 1, 3: 1}  # 0->0, 1->2, 2->1, 3->1, others->99
    expected = np.array([0, 2, 1, 1, 1, 99])
    for out in self.tfrun(pp_ops.get_remap(mapping, 99, key="image"), features):
      self.assertAllEqual(out["image"].shape, features["image"].shape)
      self.assertAllEqual(out["image"], expected)

  def test_reduce(self):
    features = {"a": tf.constant([0, 1, 2]), "b": tf.constant([1, 2, 3]),
                "c": tf.constant([2, 3, 4])}
    kw = dict(keys=["a", "b", "c"], outkey="reduced")
    for r, expected in [("sum", [3, 6, 9]), ("max", [2, 3, 4]),
                        ("min", [0, 1, 2]), ("mean", [1, 2, 3]),
                        ("prod", [0, 6, 24])]:
      for out in self.tfrun(pp_ops.get_reduce(reduction=r, **kw), features):
        self.assertAllEqual(out["reduced"], expected)


if __name__ == "__main__":
  tf.test.main()
