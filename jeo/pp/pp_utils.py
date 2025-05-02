# Copyright 2024 DeepMind Technologies Limited.
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

"""Preprocessing utils (not preprocessing ops)."""

from collections.abc import Sequence
import functools
import os
from typing import Any

from jeo.tools import geeflow_utils
import tensorflow as tf


def maybe_repeat(arg: Sequence[Any] | Any, n_reps: int) -> Sequence[Any]:
  if not isinstance(arg, Sequence) or isinstance(arg, str):
    arg = (arg,) * n_reps
  return arg


def get_lookup_table(label_map: dict[Any, Any] | Sequence[Any],  # pytype: disable=annotation-type-mismatch
                     default_value: Any = -1,
                     key_dtype: bool = None) -> tf.lookup.StaticHashTable:
  """Converts list of string labels or dict to a lookup hashtable."""
  if not isinstance(label_map, dict):
    label_map = {k: i for i, k in enumerate(label_map)}
  if isinstance(next(iter(label_map.values())), (list, tuple)):
    # Given reverse mapping {value: [list of keys]}.
    label_map, reversed_map = {}, label_map
    for value, keys in reversed_map.items():
      for k in keys:
        assert k not in label_map, f"Duplicate key {k} in {reversed_map}."
        label_map[k] = value

  lookup = tf.lookup.StaticHashTable(
      tf.lookup.KeyValueTensorInitializer(
          tf.constant(list(label_map.keys()), dtype=key_dtype),
          tf.constant(list(label_map.values())),
      ), default_value=default_value)
  return lookup


@functools.cache
def load_normalization_ranges(
    path: str,
    split_name: str = "train",
    postfix: str | None = None,
    center: str | None = "bins_median",
    scale: str | None = "bins_mad_std",
    as_tf: bool = True,
    data_dir: str | None = None,
) -> tuple[list[float] | None, list[float] | None]:
  """Returns center and scale parameters for normalization."""
  assert bool(center) == bool(scale)
  if center is None or center.lower() == "none":
    return None, None
  if data_dir:  # Using new locations.
    path = os.path.join(data_dir, path.replace(":", "/"), "stats")
  stats = geeflow_utils.load_json(path, split_name, postfix, as_cd=False)
  if not stats:
    raise ValueError(f"No stats loaded for {postfix} in {data_dir} {path}.")
  # Assuming that it's a multi-band variable.
  n_bands = len(stats)
  centers, scales = [], []
  for i in range(n_bands):
    if i not in stats:
      raise ValueError("Expected to have a dict of bands in stats.")
    centers.append(stats[i][center])
    scales.append(stats[i][scale])
  if as_tf:
    return tf.constant(centers, dtype=tf.float32), tf.constant(scales,
                                                               dtype=tf.float32)
  return centers, scales


class InKeyOutKey(object):
  """Decorator for preprocessing for single-input sinle-output ops.

  From http://github.com/google-research/big_vision/tree/HEAD/big_vision/pp/utils.py;l=12;rcl=675470346

  Attributes:
    indefault: The default input key. If None, `key` or `inkey` must be
      specified.
    outdefault: The default output key. If None, `key` or `outkey` must be
      specified.
    with_data: If True, the function will be called with the full data dict
      as an additional input.
  """

  def __init__(self, indefault: str | None = None,
               outdefault: str | None = None, with_data: bool = False):
    self.indefault = indefault
    self.outdefault = outdefault
    self.with_data = with_data

  def __call__(self, orig_get_pp_fn):

    def get_ikok_pp_fn(*args, key=None, inkey=self.indefault,
                       outkey=self.outdefault, **kw):
      assert key or inkey, "Input key must be specified."
      assert key or outkey, "Output key must be specified."

      orig_pp_fn = orig_get_pp_fn(*args, **kw)
      def _ikok_pp_fn(data):
        # Optionally allow the function to get the full data dict as aux input.
        if self.with_data:
          data[key or outkey] = orig_pp_fn(data[key or inkey], data=data)
        else:
          data[key or outkey] = orig_pp_fn(data[key or inkey])
        return data

      return _ikok_pp_fn

    return get_ikok_pp_fn
