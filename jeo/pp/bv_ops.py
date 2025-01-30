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

"""Preprocessing ops forked from big_vision code base."""
from collections.abc import Sequence
from typing import Any

import jax
from jeo.pp import pp_utils
from jeo.pp.pp_builder import Registry  # pylint: disable=g-importing-member
from jeo.tools import tree_utils
import numpy as np
import tensorflow as tf


@Registry.register("preprocess_ops.value_range")
@pp_utils.InKeyOutKey(indefault="image", outdefault="image")
def get_value_range(
    vmin: int | float = -1,
    vmax: int | float = 1,
    in_min: int | float = 0,
    in_max: int | float = 255.0,
    clip_values: bool = False,
):
  """Transforms a [in_min,in_max] image to [vmin,vmax] range.

  Input ranges in_min/in_max can be equal-size lists to rescale the invidudal
  channels independently.

  Args:
    vmin: A scalar. Output max value.
    vmax: A scalar. Output min value.
    in_min: A scalar or a list of input min values to scale. If a list, the
      length should match to the number of channels in the image.
    in_max: A scalar or a list of input max values to scale. If a list, the
      length should match to the number of channels in the image.
    clip_values: Whether to clip the output values to the provided ranges.

  Returns:
    A function to rescale the values.
  """

  def _value_range(image):
    """Scales values in given range."""
    in_min_t = tf.constant(in_min, tf.float32)
    in_max_t = tf.constant(in_max, tf.float32)
    image = tf.cast(image, tf.float32)
    image = (image - in_min_t) / (in_max_t - in_min_t)
    image = vmin + image * (vmax - vmin)
    if clip_values:
      image = tf.clip_by_value(image, vmin, vmax)
    return image

  return _value_range


@Registry.register("preprocess_ops.lookup")
@pp_utils.InKeyOutKey(indefault="image", outdefault="image")
def get_lookup(mapping: str, npzkey: str = "fnames", sep: str | None = None):
  """Map string to number."""

  # For NumPy files, we use the `npzkey` array in that file as the list of
  # strings which are mapped to their index in that array.
  # This is especially useful when other data (eg precomputed predictions)
  # goes along with this mapping, to have everything in one place (the npz).
  if mapping.endswith(".npz"):
    with tf.io.gfile.GFile(mapping, "rb") as f:
      keys = np.array(np.load(f, allow_pickle=False)[npzkey])
    vals = np.arange(len(keys))

  # Otherwise, we simply use the file as a text file, with either of:
  # - a string per line, mapped to its line-number
  # - a pair, separated by `sep` per line, first value being the string, second
  #   value being the integer that the string is mapped to.
  else:
    with tf.io.gfile.GFile(mapping, "r") as f:
      buf = f.read()
    if sep is None:  # values are the line numbers
      keys = buf.splitlines()
      vals = np.arange(len(keys))
    else:  # each line is key<sep>val, also make val int
      keys, vals = zip(*[l.split(sep) for l in buf.splitlines()])
      vals = [int(v) for v in vals]

  def _do_the_mapping(needle):
    """Map string to number."""
    with tf.init_scope():  # (Originally added for performance reasons.)
      table = tf.lookup.StaticHashTable(
          tf.lookup.KeyValueTensorInitializer(keys, vals), -1)
    return table.lookup(needle)

  return _do_the_mapping


@Registry.register("preprocess_ops.onehot")
def get_onehot(depth: int,
               key: str = "labels",
               key_result: str | None = None,
               multi: bool = True,
               on: float = 1.0,
               off: float = 0.0):
  """One-hot encodes the input.

  Args:
    depth: Length of the one-hot vector (how many classes).
    key: Key of the data to be one-hot encoded.
    key_result: Key under which to store the result (same as `key` if None).
    multi: If there are multiple labels, whether to merge them into the same
      "multi-hot" vector (True) or keep them as an extra dimension (False).
    on: Value to fill in for the positive label (default: 1).
    off: Value to fill in for negative labels (default: 0).

  Returns:
    Data dictionary.
  """

  def _onehot(data):
  # When there's more than one label, this is significantly more efficient
    # than using tf.one_hot followed by tf.reduce_max; we tested.
    labels = data[key]
    labels = tf.cast(labels, tf.int64)  # both scatter and one_hot expect this
    if labels.shape.rank > 0 and multi:
      x = tf.scatter_nd(labels[:, None], tf.ones(tf.shape(labels)[0]), (depth,))
      x = tf.clip_by_value(x, 0, 1) * (on - off) + off
    else:
      x = tf.one_hot(labels, depth, on_value=on, off_value=off)
    data[key_result or key] = x
    return data

  return _onehot


@Registry.register("preprocess_ops.keep")
def get_keep(*keys: Sequence[str]):
  """Keeps only the given keys."""

  def _keep(data):
    return {k: v for k, v in data.items() if k in keys}

  return _keep


@Registry.register("preprocess_ops.drop")
def get_drop(*keys: Sequence[str]):
  """Drops the given keys."""

  def _drop(data):
    return {k: v for k, v in data.items() if k not in keys}

  return _drop


@Registry.register("preprocess_ops.copy")
def get_copy(inkey: str, outkey: str):
  """Copies value of `inkey` into `outkey`."""

  def _copy(data):
    # A "semi-deep" copy. deepcopy doesn't work when tf tensors are part of the
    # game. What we want, is to only copy the python structure (dicts, lists)
    # and keep tensors as they are, since we never modify them in-place anyways.
    # The following achieves exactly that.
    data[outkey] = jax.tree.map(lambda x: x, data[inkey])
    return data

  return _copy


@Registry.register("preprocess_ops.squeeze_last_dim")
@pp_utils.InKeyOutKey(indefault="image", outdefault="image")
def get_squeeze_last_dim():
  def _squeeze_last_dim(x):
    return tf.squeeze(x, axis=-1)
  return _squeeze_last_dim


@Registry.register("preprocess_ops.concat")
def get_concat(inkeys: Sequence[str], outkey: str | None = None,
               axis: int = -1):
  """Concatenates elements along some axis."""

  def _concat(data):
    data[outkey or inkeys[0]] = tf.concat([data[k] for k in inkeys], axis)
    return data

  return _concat


@Registry.register("preprocess_ops.rag_tensor")
@pp_utils.InKeyOutKey(indefault="image", outdefault="image")
def get_rag_tensor():
  """Converts the specified feature to ragged tensor."""

  def rag_tensor(raw_tensor):
    # Note: Add one more dimension as `from_tensor` requires at least rank 2.
    return tf.RaggedTensor.from_tensor(raw_tensor[None])

  return rag_tensor


@Registry.register("preprocess_ops.pad_to_shape")
@pp_utils.InKeyOutKey(indefault="image", outdefault="image")
def get_pad_to_shape(shape: Sequence[int], pad_value: int = 0,
                     where: str = "after"):
  """Pads tensor to specified `shape`."""

  def _pads(cur, tgt):
    if tgt is None:
      return [0, 0]
    diff = tgt - cur
    return {
        "before": [diff, 0],
        "after": [0, diff],
        "both": [diff // 2, diff - diff // 2],
    }[where]

  def _pad_to_shape(x):
    assert len(x.shape.as_list()) == len(shape)
    paddings = [_pads(tgt=shape[i], cur=tf.shape(x)[i])
                for i in range(len(shape))]
    constant_value = tf.constant(pad_value, x.dtype)
    ret = tf.pad(x, paddings, constant_values=constant_value)
    ret.set_shape(shape)
    return ret

  return _pad_to_shape


@Registry.register("preprocess_ops.flatten")
def get_flatten():
  """Flattens the keys of data with separator '/'."""

  def flatten(data):
    flat, _ = tree_utils.tree_flatten_with_names(data)
    return dict(flat)

  return flatten


@Registry.register("preprocess_ops.reshape")
@pp_utils.InKeyOutKey(indefault="image", outdefault="image")
def get_reshape(new_shape: Sequence[int]):
  """Reshapes tensor to a given new shape.

  Args:
    new_shape: new shape for the tensor.

  Returns:
    A function for reshaping a tensor.

  """

  def _reshape(tensor):
    """Reshapes a tensor to a given shape."""
    dtype = tensor.dtype
    tensor = tf.reshape(tensor, new_shape)
    return tf.cast(tensor, dtype)

  return _reshape


@Registry.register("preprocess_ops.setdefault")
def get_setdefault(key: str, value: Any):
  """If `key` is an empty tensor, set it to `value`."""
  def _setdefault(data):
    x = data[key]
    v = tf.constant(value, dtype=x.dtype)
    v = tf.broadcast_to(v, [s or 1 for s in x.shape])
    data[key] = tf.cond(tf.size(x) > 0, lambda: x, lambda: v)
    return data
  return _setdefault


@Registry.register("preprocess_ops.choice")
def get_choice(n: int | str = "single", inkey: str | None = None,
               outkey: str | None = None, key: str | None = None):
  """Chooses the same `n` random entries of all `keys`.

  Args:
    n: int or "single": how many entries to randomly sample (without repeat).
       if the string "single", then only choose one and drop the leading 1 dim.
    inkey: str or list of str: See Note.
    outkey: str or list of str: See Note.
    key: str or list of str: See Note.

  Note:
    If key/inkey/outkey is a list, then the same random entries are chosen for
    all of the keys. Other than that, they function the same as InKeyOutKey.

    The outkey can also contain the placeholder `{key}` that'll be .

  Examples:
    choice(key="alt_text/text")
    choice(n=128, key=["patches", "positions"])
    choice(inkey=["questions_i18n", "answers_i18n"], outkey=["q", "a"])

  Returns:
    The pp op.
  """

  # Normalize keys:
  inkeys = pp_utils.maybe_repeat(inkey or key, 1)
  outkeys = pp_utils.maybe_repeat(outkey or key, 1)
  outkeys = [ok.format(key=ik) for ok, ik in zip(outkeys, inkeys)]

  def _choice(data):
    nitems = tf.shape(data[inkeys[0]])[0]

    # Sanity check that all keys have same leading dimension.
    with tf.control_dependencies([
        tf.debugging.assert_equal(tf.shape(data[k])[0], nitems)
        for k in inkeys]):
      nitems = tf.identity(nitems)

    if n == "single":
      index = tf.random.uniform([], 0, nitems, dtype=tf.int32)
    else:
      indices = tf.random.shuffle(tf.range(nitems))[:n]

    for ik, ok in zip(inkeys, outkeys):
      if n == "single":
        result = data[ik][index]  # pylint: disable=undefined-variable
      else:
        result = tf.gather(data[ik], indices, axis=0)  # pylint: disable=undefined-variable
        result = tf.ensure_shape(result, [n] + [None] * (result.ndim - 1))
      data[ok] = result

    return data
  return _choice
