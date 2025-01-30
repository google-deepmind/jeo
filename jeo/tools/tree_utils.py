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

"""Utility function for jax trees.

Based on
https://github.com/google-research/big_vision/blob/main/big_vision/utils.py
"""

import collections
import dataclasses
import re
from typing import Mapping

from absl import logging
import flax
import jax
import ml_collections as mlc
import numpy as np


def tree_flatten_with_names(tree):
  """Populates tree_flatten with leaf names.

  This function populates output of tree_flatten with leaf names, using a
  custom traversal that produces names is provided. The custom traversal does
  NOT have to traverse tree in the same order as jax, as we take care of
  automatically aligning jax' and custom traversals.

  Args:
    tree: python tree.

  Returns:
    A list of values with names: [(name, value), ...]
  """
  vals, tree_def = jax.tree.flatten(tree)

  # "Fake" token tree that is use to track jax internal tree traversal and
  # adjust our custom tree traversal to be compatible with it.
  tokens = range(len(vals))
  token_tree = tree_def.unflatten(tokens)
  val_names, perm = zip(*_traverse_with_names(token_tree))
  inv_perm = np.argsort(perm)

  # Custom traverasal should visit the same number of leaves.
  assert len(val_names) == len(vals)

  return [(val_names[i], v) for i, v in zip(inv_perm, vals)], tree_def


def tree_unflatten(names_and_vals):
  """Reverses `tree_flatten_with_names(tree)[0]`."""
  return recover_tree(*zip(*names_and_vals))


def tree_map_with_names(f, tree, *rest):
  """Like jax.tree.map but with a filter on the leaf path name.

  Args:
    f: A function with first parameter `name` (path-like "a/b/c") and remaining
      parameters values of `tree` and `*rest` corresponding to the given `name`
      Should return a new value for parameter `name`.
    tree: The tree of parameters `f` should be applied to.
    *rest: more trees of the exact same structure.

  Returns:
    A tree identical in structure to `tree` and `*rest` but with the leaves the
    result of calling `f` on corresponding name/leaves in `tree` and `*rest`.
  """
  names_and_vals, tree_def = tree_flatten_with_names(tree)
  names, vals = zip(*names_and_vals)
  rest_vals = [list(zip(*tree_flatten_with_names(t)[0]))[1] for t in rest]
  vals = [f(*name_and_vals) for name_and_vals in zip(names, vals, *rest_vals)]
  return tree_def.unflatten(vals)


def tree_map_with_regex(f, tree, regex_rules, not_f=lambda x: x, name=None):
  """Apply jax-style tree_map based on regex rules.

  Args:
    f: a function that is being applied to every variable.
    tree: jax tree of arrays.
    regex_rules: a list of tuples `(pattern, args)`, where `pattern` is a regex
      which used for variable matching and `args` are positional arguments
      passed to `f`. If some variable is not matched, we apply `not_f` transform
      which is id by default. If multiple patterns match, then only the first
      rule is applied.
    not_f: optional function which is applied to variables that do not match any
      pattern.
    name: a name of transform for logging purposes.

  Returns:
    a tree, transformed by `f` according to the given rules.
  """

  def _f(vname, v):
    for pattern, arg in regex_rules:
      if re.fullmatch(pattern, vname):
        if name and jax.process_index() == 0:
          logging.info(
              "Applying %s to %s with %s due to `%s`", name, vname, arg, pattern
          )
        return f(v, arg)
    return not_f(v)

  return tree_map_with_names(_f, tree)


def tree_get(tree, name):
  """Get an entry of pytree by flattened key name, eg a/b/c, with nice error.

  Args:
    tree: the pytree to be queried.
    name: the path to extract from the tree, see below for examples.

  Returns:
    A few examples:
      tree = {'a': 1, 'b': {'c': 2, 'd': 3}}
      tree_get(tree, 'a') == 1
      tree_get(tree, 'b/c') == 2
      tree_get(tree, 'b') == {'c': 2, 'd': 3}
  """
  flattened = dict(_traverse_with_names(tree, with_inner_nodes=True))
  try:
    return flattened[name]
  except KeyError as e:

    class Msg(str):  # Reason: https://stackoverflow.com/a/70114007/2366315

      def __repr__(self):  # pylint: disable=g-wrong-blank-lines
        return str(self)

    msg = "\n".join([name, "Available keys:", *flattened, ""])
    # Turn into configdict to use its "did you mean?" error message!
    msg = mlc.ConfigDict(flattened)._generate_did_you_mean_message(name, msg)  # pylint: disable=protected-access
    raise KeyError(Msg(msg)) from e


def tree_replace(tree, replacements):
  """Renames/removes (nested) keys.

  Example usage:

    tree = {'a': {'b': 2, 'c': 3}, 'c': 4}
    replacements = {
        'a/b': 'a/b/x',  # replaces 'a/b' with 'a/b/x'
        '.*c': 'C',      # replaces 'c' with 'C' ('a/c' is removed)
        'C': 'D',        # replaces 'C' (which was 'c') with 'D'
        '.*/c': None,    # removes 'a/c'
    }
    tree2 = rename_remove(tree, replacements)
    assert tree2 == {'D': 4, 'a': {'b': {'x': 2}}}

  Args:
    tree: A nested dictionary.
    replacements: Rules specifying `regex` as keys and `replacement` as values
      to be used with `m = re.match(regex, key)` and `m.expand(replacement)` for
      every `key` independently.  Note that: 1. If any rule matches with
      `replacement=None`, then the key is removed. 2. The rules are applied in
      order. It's possible to have multiple transformations on a single key.

  Returns:
    Updated `tree` according to rules defined in `replacements`.
  """
  replacements = {re.compile(kk): vv for kk, vv in replacements.items()}

  def rename(k):
    for kk, vv in replacements.items():
      m = kk.match(k)
      if m:
        k = k[: m.start()] + m.expand(vv) + k[m.end() :]
    return k

  def should_remove(k):
    return any(vv is None and kk.match(k) for kk, vv in replacements.items())

  names_and_vals, _ = tree_flatten_with_names(tree)
  names_and_vals = [
      (rename(k), v) for k, v in names_and_vals if not should_remove(k)
  ]
  return tree_unflatten(names_and_vals)


def tree_compare(tree1, tree2):
  """Returns `(tree1_only, tree2_only, dtype_shape_mismatch)`."""
  tree1 = flax.traverse_util.flatten_dict(tree1, sep="/")
  tree2 = flax.traverse_util.flatten_dict(tree2, sep="/")
  return (
      set(tree1) - set(tree2),
      set(tree2) - set(tree1),
      {
          k: [(v.dtype, v.shape), (tree2[k].dtype, tree2[k].shape)]
          for k, v in tree1.items()
          if k in tree2
          and (v.dtype != tree2[k].dtype or v.shape != tree2[k].shape)
      },
  )


def tree_filter(tree, mask):
  """Returns nested dict structure with only a subset of children."""
  # TODO: The code below only works for nested-dict and only when they
  # have same structure. Consider relax this.
  if not isinstance(tree, dict):
    assert isinstance(mask, bool), f"Mask leaves must be boolean! {mask}"
    return tree
  assert sorted(tree.keys()) == sorted(
      mask.keys()
  ), f"Keys in tree and mask are not equal! {tree.keys()} != {mask.keys()}"
  return {
      k: tree_filter(v, mask[k])
      for k, v in tree.items()
      if mask[k] is not False  # pylint: disable=g-bool-id-comparison
  }


def _traverse_with_names(tree, with_inner_nodes=False):
  """Traverses nested dicts/dataclasses and emits (leaf_name, leaf_val)."""
  if dataclasses.is_dataclass(tree):
    tree = flax.serialization.to_state_dict(tree)
  # Don't output the non-leaf nodes. If the optimizer doesn't have a state
  # the tree leaves can be Nones which was interpreted as a leaf by this
  # function but not by the other functions (like jax.tree.map).
  if tree is None:
    return
  elif isinstance(tree, Mapping):
    keys = sorted(tree.keys())
    for key in keys:
      for path, v in _traverse_with_names(tree[key], with_inner_nodes):
        yield (key + "/" + path).rstrip("/"), v
    if with_inner_nodes:
      yield "", tree
  elif isinstance(tree, (list, tuple)):
    for idx in range(len(tree)):
      for path, v in _traverse_with_names(tree[idx], with_inner_nodes):
        yield (str(idx) + "/" + path).rstrip("/"), v
    if with_inner_nodes:
      yield "", tree
  else:
    yield "", tree


def recover_tree(keys, values):
  """Recovers a tree as a nested dict from flat names and values.

  This function is useful to analyze checkpoints that are saved by our programs
  without need to access the exact source code of the experiment. In particular,
  it can be used to extract an reuse various subtrees of the scheckpoint, e.g.
  subtree of parameters.

  Args:
    keys: a list of keys, where '/' is used as separator between nodes.
    values: a list of leaf values.

  Returns:
    A nested tree-like dict.
  """
  tree = {}
  sub_trees = collections.defaultdict(list)
  for k, v in zip(keys, values):
    if "/" not in k:
      tree[k] = v
    else:
      k_left, k_right = k.split("/", 1)
      sub_trees[k_left].append((k_right, v))
  for k, kv_pairs in sub_trees.items():
    k_subtree, v_subtree = zip(*kv_pairs)
    tree[k] = recover_tree(k_subtree, v_subtree)
  return tree


def tree_broadcast(prefix, target):
  """Broadcasts a prefix tree to a full tree.

  Input-output examples:
  1. prefix: {"x": 10, "y": 20}
     target: {"x": {"a": 1, "b": 2}, "y": 3}

     Result: {"x": {"a": 10, "b": 10}, "y": 20}

  2. prefix: 100
     target: {"x": {"a": 1, "b": 2}, "y": 3}

     Result: {"x": {"a": 100, "b": 100}, "y": 100}

  3. prefix: {"x": 10}
     target: {"x": {"a": 1, "b": 2}, "y": 3}

     Result: ValueError

  Args:
    prefix: prefix pytree.
    target: boradcast target for a prefix tree.

  Returns:
    prefix tree broadcasted to a target tree.
  """

  def _broadcast(leaf, subtree):
    return jax.tree.map(lambda _: leaf, subtree)

  return jax.tree.map(_broadcast, prefix, target)


def make_mask_trees(tree, patterns, *, log=None):
  """Returns a boolean mask tree for every pattern (only first match)."""
  compiled_patterns = check_and_compile_patterns(patterns)

  def matchfirst(name, _):
    matches = []
    for pattern in compiled_patterns:
      matches.append(not any(matches) and bool(pattern.fullmatch(name)))
    if log is not None and True in matches and jax.process_index() == 0:
      logging.info(
          "%s: %s - matched by %s", log, name, patterns[matches.index(True)]
      )
    return np.array(matches)

  multimask = tree_map_with_names(matchfirst, tree)
  return [
      jax.tree.map(lambda matches, i=idx: matches[i], multimask)
      for idx in range(len(patterns))
  ]


def merge_params(loaded, inited, dont_load=(), match_dtype=False):
  """Makes `loaded` pytree match `init`, warning or failing on mismatch.

  Args:
    loaded: pytree of parameters, typically loaded from a checkpoint.
    inited: pytree of parameter, typically coming from model init.
    dont_load: List of regexes for parameters which shall not be taken from
      `loaded`, either because they should remain at their init value, or
      because they are missing on either side.
    match_dtype: returned pytree as leaves converted to dtype from `inited`.

  Returns:
    If successful, a new pytree which matches the structure of `init`
    but contains values from `loaded`, except for `dont_load`.

    If structures don't match and mismatches are not covered by regexes in
    `dont_load` argument, then raises an exception with more information.
  """
  if inited is None:  # A useful shortcut for example for colabs.
    return loaded

  dont_load = check_and_compile_patterns(dont_load)

  def should_merge(name):
    return not any(pattern.fullmatch(name) for pattern in dont_load)

  loaded_flat, _ = tree_flatten_with_names(loaded)
  inited_flat, _ = tree_flatten_with_names(inited)
  loaded_flat = {k: v for k, v in loaded_flat}
  inited_flat = {k: v for k, v in inited_flat}

  # Let's first build the pytree from all common keys.
  merged = {}
  for name, init_val in inited_flat.items():
    # param is present in both. Load or ignore it!
    if name in loaded_flat and should_merge(name):
      merged[name] = loaded_flat[name]
      if match_dtype:
        merged[name] = loaded_flat[name].astype(init_val.dtype)
    else:
      logging.info("Ignoring checkpoint and using init value for %s", name)
      merged[name] = init_val

  def pp(title, names, indent="  "):  # Just pretty-printing
    if names:
      return f"{title}:\n" + "\n".join(f"{indent}{k}" for k in sorted(names))
    else:
      return ""

  # Now, if there are keys that only exist in inited or loaded, be helpful:
  not_in_loaded = inited_flat.keys() - loaded_flat.keys()
  not_in_inited = loaded_flat.keys() - inited_flat.keys()
  logging.info(pp("Parameters in model but not in checkpoint", not_in_loaded))
  logging.info(pp("Parameters in checkpoint but not in model", not_in_inited))

  # And now see if any of them are not explicitly ignored => an error
  not_in_loaded = {k for k in not_in_loaded if should_merge(k)}
  not_in_inited = {k for k in not_in_inited if should_merge(k)}

  if not_in_loaded or not_in_inited:
    raise ValueError(
        pp("Params in checkpoint", loaded_flat.keys())
        + "\n"
        + pp("Params in model (code)", inited_flat.keys())
        + "\n"
        + pp(
            "Params in model (code) but not in checkpoint and not"
            " `dont_load`ed",
            not_in_loaded,
            indent=" - ",
        )
        + "\n"  # Special indent for tests.
        + pp(
            "Params in checkpoint but not in model (code) and not"
            " `dont_load`ed",
            not_in_inited,
            indent=" + ",
        )
    )  # Special indent for tests.

  return recover_tree(merged.keys(), merged.values())


def check_and_compile_patterns(patterns):
  """Validates and compiles a list of param-patterns.

  The validation consists of checking for common mistakes, currently only that
  the pattern does not start with a slash, because unlike FLAX, our parameter
  names don't start with a slash.

  Args:
    patterns: a single (string) pattern (regex), or a list of patterns.

  Returns:
    A list of compiled and verified regexes.
  """
  if isinstance(patterns, str):
    patterns = [patterns]

  assert isinstance(patterns, (list, tuple)), patterns

  def check_and_compile(pattern):
    assert not pattern.startswith(
        "/"
    ), f"Big vision parameter names never start with '/': '{pattern}"
    return re.compile(pattern)

  return list(map(check_and_compile, patterns))


def split_frozen(masks, scheds):
  """Computes `frozen_mask` and updates `masks` and `scheds`."""
  # Specifying `None` as a scheduler freezes params.
  all_false = jax.tree.map(lambda *bools: not any(bools), *masks)
  not_covered = tree_flatten_with_names(all_false)[0]
  not_covered = [k for k, v in not_covered if v]
  assert (
      not not_covered
  ), f"All params must be covered (use `None` for freezing): {not_covered}"
  frozen_masks = [mask for mask, sched in zip(masks, scheds) if sched is None]
  frozen_mask = jax.tree.map(
      lambda *bools: any(bools), *frozen_masks, all_false
  )  # `all_false` is required when `frozen_masks==[]`.
  masks, scheds = zip(*(
      (mask, sched) for mask, sched in zip(masks, scheds) if sched is not None
  ))
  return frozen_mask, masks, scheds
