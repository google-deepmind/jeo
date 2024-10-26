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

"""Tests for pp_utils."""
from unittest import mock

from jeo.pp import pp_utils
import tensorflow as tf


class PpUtilsTest(tf.test.TestCase):

  def test_lookup_table_from_list(self):
    label_map = ["label_0", "label_1"]
    table = pp_utils.get_lookup_table(label_map)
    self.assertAllEqual(table.size(), 2)
    self.assertAllEqual(table.lookup(tf.constant("label_0")), 0)
    self.assertAllEqual(table.lookup(tf.constant("label_1")), 1)
    self.assertAllEqual(table.lookup(tf.constant("label_X")), -1)

  def test_lookup_table_from_int_list(self):
    label_map = [0, 1, 2]
    table = pp_utils.get_lookup_table(label_map)
    self.assertAllEqual(table.size(), 3)
    self.assertAllEqual(table.lookup(tf.constant(0)), 0)
    self.assertAllEqual(table.lookup(tf.constant(1)), 1)
    self.assertAllEqual(table.lookup(tf.constant(5)), -1)

  def test_lookup_table_from_dict(self):
    label_map = {"label_0": 100, "label_1": 200}
    table = pp_utils.get_lookup_table(label_map)
    self.assertAllEqual(table.size(), 2)
    self.assertAllEqual(table.lookup(tf.constant("label_0")), 100)
    self.assertAllEqual(table.lookup(tf.constant("label_1")), 200)
    self.assertAllEqual(table.lookup(tf.constant("label_X")), -1)

  def test_lookup_table_from_reversed_map(self):
    label_map = {"rare": ["L0", "L1", "L2"], "frequent": ["L10", "L20"]}
    table = pp_utils.get_lookup_table(label_map, default_value="unknown")
    self.assertAllEqual(table.size(), 5)
    self.assertAllEqual(table.lookup(tf.constant("L0")), "rare")
    self.assertAllEqual(table.lookup(tf.constant("L1")), "rare")
    self.assertAllEqual(table.lookup(tf.constant("L2")), "rare")
    self.assertAllEqual(table.lookup(tf.constant("L10")), "frequent")
    self.assertAllEqual(table.lookup(tf.constant("L20")), "frequent")
    self.assertAllEqual(table.lookup(tf.constant("L-1")), "unknown")

  @mock.patch.object(pp_utils, "stats_util")
  def test_load_normalization_ranges(self, mock_stats_util):
    mock_stats_util.load_json.return_value = {
        0: {"bins_median": 0., "bins_mad_std": 1.},
        1: {"bins_median": -1., "bins_mad_std": 2.}}
    centers, scales = pp_utils.load_normalization_ranges("dummy_path")
    self.assertAllEqual(centers, [0., -1.])
    self.assertAllEqual(scales, [1., 2.])

  def test_inkeyoutkey(self):
    @pp_utils.InKeyOutKey()
    def get_pp_fn(shift, scale=0):
      def _pp_fn(x):
        return scale * x + shift
      return _pp_fn

    data = {"k_in": 2, "other": 3}
    ppfn = get_pp_fn(1, 2, inkey="k_in", outkey="k_out")  # pylint: disable=unexpected-keyword-arg
    self.assertEqual({"k_in": 2, "k_out": 5, "other": 3}, ppfn(data))

    data = {"k": 6, "other": 3}
    ppfn = get_pp_fn(1, key="k")  # pylint: disable=unexpected-keyword-arg
    self.assertEqual({"k": 1, "other": 3}, ppfn(data))

  def test_inkeyoutkey_fails(self):
    @pp_utils.InKeyOutKey()
    def get_pp_fn(shift, scale=0):
      def _pp_fn(x):
        return scale * x + shift
      return _pp_fn

    with self.assertRaises(AssertionError):
      get_pp_fn(5, 2)


if __name__ == "__main__":
  tf.test.main()
