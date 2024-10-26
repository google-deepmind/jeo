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

"""Tests for google3.experimental.brain.jeo.tools.decorators."""
import io
from unittest import mock

from jeo.tools import decorators

from google3.testing.pybase import googletest


class DecoratorsTest(googletest.TestCase):

  def test_deprecated(self):
    @decorators.deprecated("Testing dummy fn deprecation.")
    def dummy_fn():
      pass
    with mock.patch("sys.stderr", new=io.StringIO()) as fake_std_out:
      dummy_fn()
      returned_message = fake_std_out.getvalue()
      print(returned_message)
      self.assertIn(
          "Deprecated `DecoratorsTest.test_deprecated.<locals>.dummy_fn`"
          " (from __main__): Testing dummy fn deprecation", returned_message)

  def test_deprecated_no_message(self):
    @decorators.deprecated
    def dummy_fn():
      pass
    with mock.patch("sys.stderr", new=io.StringIO()) as fake_std_out:
      dummy_fn()
      returned_message = fake_std_out.getvalue()
      print(returned_message)
      self.assertIn(
          "Deprecated `DecoratorsTest.test_deprecated_no_message.<locals>."
          "dummy_fn` (from __main__)", returned_message)


if __name__ == "__main__":
  googletest.main()
