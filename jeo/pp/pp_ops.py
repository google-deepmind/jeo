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

"""Preprocessing ops.

Some preprocessing ops are forked and slightly modified from big_vision or
grand_vision ops in order to make them more flexible to the use of multi-modal
multi-band satellite data.

Prefer not to use `*` or `**` in pp fn arguments, in order to allow
programmatic, not hard-coded arguments for easier sweeping across the modalities
and bands.
"""
from collections.abc import Sequence
from typing import Any

from jeo.pp import pp_utils
from jeo.pp.pp_builder import Registry  # pylint: disable=g-importing-member
import tensorflow as tf


@Registry.register("preprocess_ops.cast", replace=True)
def get_cast(key: str, dtype: str = "int32"):
  """Casts a given feature to a given type."""

  def _pp(features):
    features[key] = tf.cast(features[key], dtype)
    return features

  return _pp


@Registry.register("preprocess_ops.transpose", replace=True)
def get_transpose(key: str, perm: Sequence[int]):
  """Transposes a given feature."""

  def _pp(features):
    features[key] = tf.transpose(features[key], perm=perm)
    return features

  return _pp


@Registry.register("preprocess_ops.string_to_int", replace=True)
def get_string_to_int(key: str):
  """Converts string key to int."""

  def _pp(data):
    data[key] = tf.strings.to_number(data[key], tf.int32)
    return data

  return _pp


@Registry.register("preprocess_ops.skai_concat", replace=True)
def get_skai_concat(key1: str = "pre_image_png", key2: str = "post_image_png",
                    outkey: str = "image", axis: int = -1,
                    pop_origs: bool = True):
  """Concatenates images across a given axis."""

  def _pp(features):
    if pop_origs:
      to_concat = [features.pop(key1), features.pop(key2)]
    else:
      to_concat = [features[key1], features[key2]]
    features[outkey] = tf.concat(to_concat, axis=axis)
    return features

  return _pp


@Registry.register("preprocess_ops.jeo_concat", replace=True)
def get_concat(keys: Sequence[str], outkey: str = "image", axis: int = -1,
               pop_origs: bool = True, broadcast: bool = False):
  """Concatenates images across a given axis."""

  def _pp(features):
    if pop_origs:
      to_concat = [features.pop(k) for k in keys]
    else:
      to_concat = [features[k] for k in keys]
    if broadcast:
      # assume first feature has the target shape to broadcast to
      target_shape = tf.shape(to_concat[0])
      broadcasted = []
      for x in to_concat[1:]:
        broadcast_shape = tf.concat([
            target_shape[:axis],
            tf.shape(x)[axis:][:1],
            target_shape[axis:][1:],
        ], axis=-1)
        broadcasted.append(tf.broadcast_to(x, broadcast_shape))
      to_concat = [to_concat[0]] + broadcasted

    features[outkey] = tf.concat(to_concat, axis=axis)
    return features

  return _pp


@Registry.register("preprocess_ops.jeo_keep", replace=True)
def get_keep(keys: Sequence[str]):
  """Keeps only the given keys."""

  def _keep(data):
    return {k: v for k, v in data.items() if k in keys}

  return _keep


@Registry.register("preprocess_ops.jeo_drop", replace=True)
def get_drop(keys: Sequence[str]):
  """Drops the given keys."""

  def _drop(data):
    return {k: v for k, v in data.items() if k not in keys}

  return _drop


@Registry.register("preprocess_ops.squeeze_dim", replace=True)
def get_squeeze_dim(*keys: Sequence[str], axis: int = 0):
  """Squeezes by one dimension at given axis (can provide multiple keys)."""
  if len(keys) == 1 and isinstance(keys[0], (list, tuple)):
    keys = keys[0]

  def _pp(features):
    for k in keys:
      features[k] = tf.squeeze(features[k], axis=axis)
    return features

  return _pp


@Registry.register("preprocess_ops.max", replace=True)
def get_max(*keys: Sequence[str], axis: int = 0):
  """Runs reduce_max on a given axis (can provide multiple keys)."""
  if len(keys) == 1 and isinstance(keys[0], (list, tuple)):
    keys = keys[0]

  def _pp(features):
    for k in keys:
      features[k] = tf.math.reduce_max(features[k], axis=axis)
    return features

  return _pp


@Registry.register("preprocess_ops.reduce", replace=True)
def get_reduce(keys: Sequence[str], outkey: str, reduction: str = "prod"):
  """Combines multiple features into one with a specified reduction."""
  assert reduction in ("prod", "sum", "mean", "max", "min", "logsumexp",
                       "any", "all")

  def _pp(features):
    concat = tf.concat([features[k][..., None] for k in keys], axis=-1)
    features[outkey] = getattr(tf, f"reduce_{reduction}")(concat, axis=-1)
    return features

  return _pp


@Registry.register("preprocess_ops.expand_dim", replace=True)
def get_expand_dim(*keys: Sequence[str], axis: int = 0):
  """Expands by one dimension at given axis (can provide multiple keys)."""
  if len(keys) == 1 and isinstance(keys[0], (list, tuple)):
    keys = keys[0]

  def _pp(features):
    for k in keys:
      features[k] = tf.expand_dims(features[k], axis=axis)
    return features

  return _pp


@Registry.register("preprocess_ops.ensure_4d", replace=True)
def get_ensure_4d(keys: Sequence[str] | str, extra_axis: int = 0):
  """Ensures number of dimension for given key field."""
  keys = [keys] if isinstance(keys, str) else keys
  def _pp(features):
    for k in keys:
      if features[k].shape.rank == 3:
        features[k] = tf.expand_dims(features[k], axis=extra_axis)
      elif features[k].shape.rank != 4:
        raise ValueError(f"{k} has invalid shape: {features[k].shape}.")
    return features
  return _pp


@Registry.register("preprocess_ops.rename", replace=True)
def get_rename(**kv_pairs: dict[str, str]):
  """Renames selected keys."""

  def _pp(features):
    return {kv_pairs.get(k, k): v for k, v in features.items()}

  return _pp


@Registry.register("preprocess_ops.ensure_shape", replace=True)
@pp_utils.InKeyOutKey()
def get_ensure_shape(shape: Sequence[int]):
  """Ensures shape for given key field."""
  # This op is added for debugging. Don't leave it in final preprocessing.
  # If you have to use this op, most likely your previous preprocessing
  # ops are incorrect and sometimes you don't get the expected shape.
  def _pp(tensor):
    return tf.ensure_shape(tensor, shape)
  return _pp


@Registry.register("preprocess_ops.invert_bool", replace=True)
def get_invert_bool(keys: str | Sequence[str]):
  """Inverts input values and converts the result to its original dtype.

  Args:
    keys: list of keys to invert.

  Returns:
    A preprocessing function.
  """
  if isinstance(keys, str):
    keys = [keys]

  def _pp(features):
    for key in keys:
      x = features[key]
      features[key] = tf.cast(tf.logical_not(tf.cast(x, bool)), x.dtype)
    return features
  return _pp


@Registry.register("preprocess_ops.select_channels", replace=True)
def get_select_channels(
    key_channels: int | Sequence[int] | dict[str, int | Sequence[int]],
    axis: int = -1,
):
  """Selects channels (scalars or list of indices)."""
  # If channels is a scalar, then the axis is reduced.
  if isinstance(key_channels, (list, tuple, int)):
    key_channels = {"image": key_channels}
  def _pp(features):
    for key, ind in key_channels.items():
      features[key] = tf.gather(features[key], ind, axis=axis)
    return features
  return _pp


@Registry.register("preprocess_ops.select_channels_by_name", replace=True)
def get_select_channels_by_name(key: str, channels: Sequence[str],
                                names: Sequence[str], axis: int = -1):
  """Selects channels by name for a given key."""
  indices = [names.index(c) for c in channels]
  return get_select_channels({key: indices}, axis=axis)


@Registry.register("preprocess_ops.extract_channels", replace=True)
def get_extract_channels(
    from_key: str, key_channels: dict[str, int | Sequence[int]], axis=-1
):
  """Extracts channels to separate tensors."""

  def _pp(features):
    outputs = {}
    for key, ind in key_channels.items():
      outputs[key] = tf.gather(features[from_key], ind, axis=axis)
    return features | outputs

  return _pp


@Registry.register("preprocess_ops.add_meta_as_channel", replace=True)
def get_add_meta_as_channel(meta_key: str, to_key: str = "image"):
  """Appends scalar value as an additional channel."""
  def _pp(features):
    broadcast_shape = features[to_key].shape.as_list()[:-1] + [1]
    as_channel = tf.broadcast_to(features[meta_key], broadcast_shape)
    as_channel = tf.cast(as_channel, features[to_key].dtype)
    features[to_key] = tf.concat([features[to_key], as_channel], axis=-1)
    return features
  return _pp


@Registry.register("preprocess_ops.remap_ints", replace=True)
@pp_utils.InKeyOutKey()
def get_remap_ints(mapping: list[int]):
  """Mapping from a set of integer labels to another set of integer labels."""

  mapping = tf.constant(mapping, "int32")

  def _pp(data):
    in_dtype = data.dtype

    # Map the labels to the new values based on the index -> value mapping.
    data = tf.map_fn(lambda t: tf.gather(mapping, t), tf.cast(data, "int32"))

    # Ensure that the output dtype is the same as the input dtype.
    return tf.cast(data, in_dtype)

  return _pp


@Registry.register("preprocess_ops.remap", replace=True)
@pp_utils.InKeyOutKey()
def get_remap(mapping: dict[Any, Any], default_value: Any = 0):
  """Remapping values from a feature."""

  mapping = pp_utils.get_lookup_table(mapping, default_value)

  def _pp(data):
    return mapping[data]

  return _pp
