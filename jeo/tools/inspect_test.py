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

"""Tests for google3.experimental.brain.jeo.tools.inspect."""
from jeo.tools import inspect
import tensorflow as tf

from google3.testing.pybase import googletest


class InspectTest(googletest.TestCase):

  def test_tree_info_with_none_tensor_spec_shape(self):
    d = (tf.TensorSpec(shape=(None,)), tf.TensorSpec(shape=(10,)))
    out = inspect.tree_info(d)
    self.assertIsNone(out)  # Mark that it did not fail.

  def test_pytree_paths_split(self):
    d = {"a/b/c": 1, "a/b/d/e": 3}
    expected = {"a": {"b": {"c": 1, "d": {"e": 3}}}}
    out = inspect.pytree_paths_split(d)
    self.assertDictEqual(out, expected)

  def test_pytree_paths_flatten(self):
    d = {"a": {"b": {"c": 1, "d": {"e": 3}}}}
    expected = {"a/b/c": 1, "a/b/d/e": 3}
    out = inspect.pytree_paths_flatten(d)
    self.assertDictEqual(out, expected)

  def test_pytree_paths_flatten_with_list(self):
    d = {"a": {"b": [1, 2, 3]}}
    expected = {"a/b/0": 1, "a/b/1": 2, "a/b/2": 3}
    out = inspect.pytree_paths_flatten(d)
    self.assertDictEqual(out, expected)

  def test_pytree_paths_flatten_with_list_as_leaf(self):
    d = {"a": {"b": [1, 2, 3]}}
    expected = {"a.b": [1, 2, 3]}
    out = inspect.pytree_paths_flatten(d, ".", seq_is_leaf=True)
    self.assertDictEqual(out, expected)

  def test_pytree_list_to_dict(self):
    list_of_dicts = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
    expected = {"a": [1, 3], "b": [2, 4]}
    out = inspect.pytree_list_to_dict(list_of_dicts)
    self.assertCountEqual(out, expected)
    for k in expected:
      self.assertCountEqual(out[k], expected[k])


if __name__ == "__main__":
  googletest.main()
