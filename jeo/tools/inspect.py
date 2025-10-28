# Copyright 2025 DeepMind Technologies Limited.
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

"""Inspect tools/utils for programmatic and interactive usage."""
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import flax
import jax
import numpy as np

PyTree = Sequence[Any] | Mapping[Any, Any]


def pytree_paths(
    param_tree: PyTree,
    annotate_types: bool = False,
    as_kv: bool = False,
) -> Iterable[str | tuple[str, Any]]:
  """Returns a generator for pytree path names."""
  if annotate_types:
    type_path_names = {list: "LIST:", tuple: "TUPLE:", dict: "DICT:",
                       flax.core.FrozenDict: "FLAX:"}
  else:
    type_path_names = {list: "", tuple: "", dict: "", flax.core.FrozenDict: ""}
  path_name = type_path_names.get(type(param_tree), "LEAF")
  return_val = lambda k, v: (k, v) if as_kv else k
  if isinstance(param_tree, (tuple, list)):
    for i, x in enumerate(param_tree):
      for path, v in pytree_paths(x, annotate_types, as_kv=True):
        yield return_val(f"{path_name}{i}/{path}", v)
  elif isinstance(param_tree, dict):
    for key in sorted(param_tree.keys()):
      for path, v in pytree_paths(param_tree[key], annotate_types, as_kv=True):
        yield return_val(f"{path_name}{key}/{path}", v)
  elif isinstance(param_tree, flax.core.FrozenDict):
    for key in sorted(param_tree.keys()):
      for path, v in pytree_paths(param_tree[key], annotate_types, as_kv=True):
        yield return_val(f"{path_name}{key}/{path}", v)
  else:
    yield return_val(path_name, param_tree)  # LEAF


def pytree_paths_split(d: Mapping[str, Any]) -> dict[str, Any]:
  """Given dict with compressed keys, expand to a nested dict."""
  # {"a/b/c": 1, "a/b/d/e": 3} --> {"a": {"b": {"c": 1, "d": {"e": 3}}}}
  new_d = {}
  for k, v in d.items():
    sub_k = k.split("/")
    sub_d = new_d
    for m in sub_k[:-1]:
      if m not in sub_d: sub_d[m] = {}
      sub_d = sub_d[m]
    sub_d[sub_k[-1]] = v
  return new_d


def pytree_paths_flatten(
    d: Mapping[Any, Any], sep: str = "/", seq_is_leaf: bool = False
) -> dict[str, Any]:
  """Given dict with nested keys, flatten to a dict with compressed keys."""
  # {"a": {"b": {"c": 1, "d": {"e": 3}}}} --> {"a/b/c": 1, "a/b/d/e": 3}
  # {"a": {"b": [1, 2]}} --> {"a/b/0": 1, "a/b/1": 2}
  # seq_is_leaf=True: {"a": {"b": [1, 2]}} --> {"a/b": [1, 2]}
  is_leaf = None
  if seq_is_leaf:
    is_leaf = lambda x: isinstance(x, (list, tuple))
  flattened, _ = jax.tree_util.tree_flatten_with_path(d, is_leaf)
  return {_flattened_key(k, sep): v for k, v in flattened}


def _flattened_key(k: Sequence[Any], sep: str) -> str:
  return sep.join(x.key if hasattr(x, "key") else str(x.idx) for x in k)


def pytree_list_to_dict(list_of_dicts: list[Any]) -> dict[Any, Any]:
  """Given list of dicts, convert to dict of lists."""
  #  [{"a": 1, "b": 2}, {"a": 3, "b": 4}] --> {"a": [1, 3], "b": [2, 4]}
  if not list_of_dicts: return {}
  stack_args = lambda *args: np.stack(args)
  return jax.tree.map(stack_args, *list_of_dicts)


def tree_info(tree: PyTree, annotate_types: bool = False,
              type_and_counts: bool = True):
  """Prints info (path, dtype, shape) per leaf."""
  num_params, num_vars = 0, 0
  for k, v in pytree_paths(tree, annotate_types=annotate_types, as_kv=True):
    dtype = v.dtype if hasattr(v, "dtype") else type(v)
    shape = v.shape if hasattr(v, "shape") else v
    print(f"{k:<40} \t{str(dtype)} \t{shape} \t{type(v).__name__}")
    if type_and_counts:
      # TensorSpec(shape=(None,) has shape, but np.prod() returns None.
      num_params += (np.prod(shape) if hasattr(v, "shape") else 0) or 0
      num_vars += 1
  if type_and_counts:
    print(f"  {type(tree)}")
    print(f"  Number of variables: {num_vars}; total size: {num_params:,}")


def normalize(arr: np.ndarray,
              channel_dependent: bool = False,
              bottom_perc: int = 1,
              top_perc: int = 99,
              verbose: bool = False,
              clip: bool = True):
  """Normalizes numpy array per channel or globally.

  Args:
    arr: np array with last dimension representing the channels.
    channel_dependent: whether normalization should be performed per channel or
      per entire array.
    bottom_perc: where to clip the low values [0..100].
    top_perc: top percentile, where to clip at the high values [0..100].
    verbose: whether to print the selected value threshold.
    clip: whether the values should be clipped to the given range.

  Returns:
    Rescaled array of same size and dtype.
  """

  def _rescale(x):
    thresholds = np.percentile(x.flatten(), [bottom_perc, top_perc, 0, 100])
    if verbose:
      print(f"Rescaling thresholds: {thresholds[:2]} (full: {thresholds[2:]})")
    return normalize_with_minmax(x, thresholds[0], thresholds[1], clip=clip)

  if channel_dependent:
    return np.stack([_rescale(arr[..., i]) for i in range(arr.shape[-1])], -1)
  else:
    return _rescale(arr)


def normalize_with_minmax(arr: np.ndarray, min_value: float, max_value: float,
                          clip: bool = True) -> np.ndarray:
  """Normalizes array to [0, 1] with given min/max values."""
  if clip:
    arr = np.clip(arr, min_value, max_value)
  arr = (arr - min_value) / (max_value - min_value)
  return arr


def stats_str(arr: Any, f: str | None = None, with_median: bool = False,
              with_count: bool = False) -> str:
  """Returns a string with main stats info about the given array.

  By default, the string has the form: "mean +/- standard_deviation [min..max]"
  values of the data array.

  Args:
    arr: array-like.
    f: Optional format string.
    with_median: boolean.
    with_count: boolean.
  Returns:
    Stats string.
  """
  if arr is None or (isinstance(arr, (list, tuple)) and not arr):
    return "[empty]"
  if not isinstance(arr, np.ndarray):
    try:
      arr = arr.numpy()  # If arr is a TF-2 tensor.
    except AttributeError:
      pass
    try:
      arr = np.concatenate(arr).ravel()  # to deal with different length lists
    except ValueError:
      arr = np.array(arr)
  if f is None:
    f = "{:.3f}"
  if with_median:
    median = (" median: " + f).format(np.median(arr))
  else:
    median = ""
  count = " n: {:,}".format(len(arr)) if with_count else ""
  pm = "+/-"  # this one doesn't work: u' \u00B1'
  if arr.dtype.kind in ["i", "u"]:
    return (f + pm + f + " [{}..{}]").format(arr.mean(), arr.std(), arr.min(),
                                             arr.max()) + median + count
  return (f + pm + f + " [" + f + ".." + f + "]").format(
      arr.mean(), arr.std(), arr.min(), arr.max()) + median + count


def stats(arr: Any, *args, title: str | None = None, **kwargs):
  """Prints out stats about given array."""
  s = stats_str(arr, *args, **kwargs)
  if title:
    print(f"{title}: {s}")
  else:
    print(s)


def to_np(x: Any) -> np.ndarray | dict[Any, np.ndarray]:
  """Converts `x` to numpy (including bytes to strings and nested dicts)."""
  if isinstance(x, dict):
    return jax.tree.map(to_np, x)
  x = np.array(x)
  if x.dtype == "O":
    x = x.tolist()
    if isinstance(x, list):
      x = [y.decode() for y in x]
    else:
      x = x.decode()
  return np.asarray(x)


def iter_to_np(iterable: Iterable[Any], n: int = 10) -> list[np.ndarray]:
  """Returns first n elements from an iterable as numpy arrays."""
  # Useful eg. for quickly getting examples from a tf.data.Dataset or iterator.
  return [jax.tree.map(to_np, x) for _, x in zip(range(n), iterable)]
