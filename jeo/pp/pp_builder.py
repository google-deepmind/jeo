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

"""Preprocessing builder with registry from big_vision.

Based on http://github.com/google-research/big_vision/tree/HEAD/big_vision/pp/builder.py

Authors: Joan Puigcerver, Alexander Kolesnikov.
"""
import ast
from collections.abc import Callable
import contextlib
import functools
from typing import Any

from absl import logging
import tensorflow as tf


def get_preprocess_fn(pp_pipeline: str | None, log_data: bool = True,
                      log_steps: bool = False) -> Callable[[Any], Any]:
  """Transform an input string into the preprocessing function.

  The minilanguage is as follows:

    fn1|fn2(arg, arg2,...)|...

  And describes the successive application of the various `fn`s to the input,
  where each function can optionally have one or more arguments, which are
  either positional or key/value, as dictated by the `fn`.

  The output preprocessing function expects a dictionary as input. This
  dictionary should have a key "image" that corresponds to a 3D tensor
  (height x width x channel).

  Args:
    pp_pipeline: A string describing the pre-processing pipeline. If empty or
      None, no preprocessing will be executed.
    log_data: Whether to log the data before and after preprocessing. Can also
      be a string to show in the log for debugging, for example dataset name.
    log_steps: Whether to log the steps of the preprocessing pipeline.

  Returns:
    preprocessing function.

  Raises:
    ValueError: if preprocessing function name is unknown
  """

  names, ops, spec_strings = [], [], []
  if pp_pipeline:
    for op_spec in pp_pipeline.split("|"):
      if not op_spec: continue  # Skip empty section instead of error.
      try:
        ops.append(Registry.lookup(f"preprocess_ops.{op_spec}")())
        names.append(parse_name(op_spec)[0])
        spec_strings.append(op_spec)
      except SyntaxError as err:
        raise ValueError(f"Syntax error on: {op_spec}") from err

  def _preprocess_fn(data):
    """The preprocessing function that is returned."""
    nonlocal log_data, log_steps

    # Apply all the individual steps in sequence.
    if log_data:
      logging.info("Data before pre-processing (%s):\n%s", log_data, data)
    for name, op, spec in zip(names, ops, spec_strings):
      if log_steps:
        logging.info("Pre-processing step (%s): %s\n%s", name, spec, data)
      with tf.name_scope(name):
        data = op(data)

    # Validate input
    if not isinstance(data, dict):
      raise ValueError("Argument `data` must be a dictionary, "
                       "not %s" % str(type(data)))

    if log_data:
      logging.info("Data after pre-processing (%s):\n%s", log_data, data)
    log_data = False  # For eager&pygrain: only log first one of each pipeline.
    return data

  return _preprocess_fn


def parse_name(string_to_parse):
  """Parses input to the registry's lookup function.

  Args:
    string_to_parse: can be either an arbitrary name or function call
      (optionally with positional and keyword arguments).
      e.g. "multiclass", "resnet50_v2(filters_factor=8)".

  Returns:
    A tuple of input name, argument tuple and a keyword argument dictionary.
    Examples:
      "multiclass" -> ("multiclass", (), {})
      "resnet50_v2(9, filters_factor=4)" ->
          ("resnet50_v2", (9,), {"filters_factor": 4})
  """
  expr = ast.parse(string_to_parse, mode="eval").body  # pytype: disable=attribute-error
  if not isinstance(expr, (ast.Attribute, ast.Call, ast.Name)):
    raise ValueError(
        "The given string should be a name or a call, but a {} was parsed from "
        "the string {!r}".format(type(expr), string_to_parse))

  # Notes:
  # name="some_name" -> type(expr) = ast.Name
  # name="module.some_name" -> type(expr) = ast.Attribute
  # name="some_name()" -> type(expr) = ast.Call
  # name="module.some_name()" -> type(expr) = ast.Call

  if isinstance(expr, ast.Name):
    return string_to_parse, (), {}
  elif isinstance(expr, ast.Attribute):
    return string_to_parse, (), {}

  def _get_func_name(expr):
    if isinstance(expr, ast.Attribute):
      return _get_func_name(expr.value) + "." + expr.attr
    elif isinstance(expr, ast.Name):
      return expr.id
    else:
      raise ValueError(
          "Type {!r} is not supported in a function name, the string to parse "
          "was {!r}".format(type(expr), string_to_parse))

  def _get_func_args_and_kwargs(call):
    args = tuple([ast.literal_eval(arg) for arg in call.args])
    kwargs = {
        kwarg.arg: ast.literal_eval(kwarg.value) for kwarg in call.keywords
    }
    return args, kwargs

  func_name = _get_func_name(expr.func)
  func_args, func_kwargs = _get_func_args_and_kwargs(expr)

  return func_name, func_args, func_kwargs


class Registry(object):
  """Implements global Registry.
  """

  _GLOBAL_REGISTRY = {}

  @staticmethod
  def global_registry():
    return Registry._GLOBAL_REGISTRY

  @staticmethod
  def register(name, replace=False):
    """Creates a function that registers its input."""

    def _register(item):
      if name in Registry.global_registry() and not replace:
        raise KeyError("The name {!r} was already registered.".format(name))

      Registry.global_registry()[name] = item
      return item

    return _register

  @staticmethod
  def lookup(lookup_string, kwargs_extra=None):
    """Lookup a name in the registry."""

    try:
      name, args, kwargs = parse_name(lookup_string)
    except ValueError as e:
      raise ValueError(f"Error parsing:\n{lookup_string}") from e
    if kwargs_extra:
      kwargs.update(kwargs_extra)
    item = Registry.global_registry()[name]
    return functools.partial(item, *args, **kwargs)

  @staticmethod
  def knows(lookup_string):
    try:
      name, _, _ = parse_name(lookup_string)
    except ValueError as e:
      raise ValueError(f"Error parsing:\n{lookup_string}") from e
    return name in Registry.global_registry()


@contextlib.contextmanager
def temporary_ops(**kw):
  """Registers specified pp ops for use in a `with` block.

  Example use:

    with pp_registry.temporary_ops(
        pow=lambda alpha: lambda d: {k: v**alpha for k, v in d.items()}):
      pp = pp_builder.get_preprocess_fn("pow(alpha=2.0)|pow(alpha=0.5)")
      features = pp(features)

  Args:
    **kw: Names are preprocess string function names to be used to specify the
      preprocess function. Values are functions that can be called with params
      (e.g. the `alpha` param in above example) and return functions to be used
      to transform features.

  Yields:
    A context manager to be used in a `with` statement.
  """
  reg = Registry.global_registry()
  kw = {f"preprocess_ops.{k}": v for k, v in kw.items()}
  for k in kw:
    assert k not in reg
  for k, v in kw.items():
    reg[k] = v
  try:
    yield
  finally:
    for k in kw:
      del reg[k]
