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

"""Checkpoint handling.

Based on
https://github.com/google-research/big_vision/blob/main/big_vision/utils.py
"""

import io
import os
import re
import shutil

import jax
from jeo.tools import tree_utils
import numpy as np


def save_checkpoint_oss(
    checkpoint, path, step_copy=None, compressed=False, group=None
):
  """Util for checkpointing: saves jax pytree objects to the disk.

  Args:
    checkpoint: arbitrary jax pytree to be saved.
    path: a path to save the checkpoint.
    step_copy: creates a copy of the checkpoint with `path-{step_copy}` name.
    compressed: whether to use np.savez or np.savez_compressed.
    group: not supported yet, will be ignored.
  """
  del group  # not supported yet.

  names_and_vals, _ = tree_utils.tree_flatten_with_names(checkpoint)
  io_buffer = io.BytesIO()

  if compressed:
    np.savez_compressed(io_buffer, **{k: v for k, v in names_and_vals})
  else:
    np.savez(io_buffer, **{k: v for k, v in names_and_vals})

  # In order to be robust to interruptions we first save checkpoint to the
  # temporal file and then move to actual path name.
  path_tmp = path + "-TEMPORARY"

  try:
    if os.path.exists(path_tmp):
      os.remove(path_tmp)

    with open(path_tmp, "wb") as f:
      f.write(io_buffer.getvalue())

    if step_copy is not None:
      copy_path = f"{path}-{step_copy:09d}"
      if os.path.exists(copy_path):
        os.remove(copy_path)
      shutil.copyfile(path_tmp, copy_path)

    os.replace(path_tmp, path)
  except OSError as e:
    print(f"Error saving checkpoint to {path}: {e}")
    if os.path.exists(path_tmp):
      try:
        os.remove(path_tmp)
      except OSError as e2:
        print(f"Error cleaning up temporary file {path_tmp}: {e2}")
    raise


def npload(fname):
  """Loads `fname` and returns an np.ndarray or dict thereof."""
  # Load the data; use local paths directly if possible:
  if os.path.exists(fname):
    loaded = np.load(fname, allow_pickle=False)
  else:
    raise FileNotFoundError(f"Checkpoint file not found: {fname!r}")

  # Support loading both single-array files (np.save) and zips (np.savez).
  if isinstance(loaded, np.ndarray):
    return loaded
  else:
    return dict(loaded)


def load_checkpoint(npz, tree=None):
  """Loads a jax pytree from a npz file.

  Args:
    npz: Either path to the checkpoint file (.npz), or a dict-like.
    tree: deprecated, use None. Bwd-compat for old format that only stored
      values, the pytree structure.

  Returns:
    A pytree that is the checkpoint.
  """
  if isinstance(npz, str):  # If not already loaded, then load.
    npz = npload(npz)
  keys, values = zip(*list(npz.items()))
  if tree:
    checkpoint = tree.unflatten(values)
  else:
    checkpoint = tree_utils.recover_tree(keys, values)
  return checkpoint


def load_params(ckpt, **kw):
  """Loads the parameters of a big_vision checkpoint, both old or new format.

  Args:
    ckpt: Path to the checkpoint (.npz) or dict-like.
    **kw: forwarded to the underlying load function (_np).

  Returns:
    A pytree that is the checkpoint, potentially sharded.

  Notes:
    The `ckpt` string can contain an colon-separated "submodel" indicator, like
    `img` in the example `/path/to/file.npz:img`.
    This is used to load sub-parts of a model, for example the image load the
    image encoder out of a two_tower (SigLIP) checkpoint, or distillation.
    This way, ANY model that uses this function can load itself from a
    checkpoint that contains multiple sub-models.
  """
  if isinstance(ckpt, str):  # Most common case of passing a checkpoint path.
    # Potentially read out the sub-part to load from after the colon
    # '/path/to/file:img/head' => '/path/to/file', 'img/head'
    # 'gs://path/to/file' => 'gs://path/to/file', None
    if match := re.match(r"^(.*?/.*?)(?::([\w/]+))?$", ckpt):
      ckpt, key = match.groups()
    else:
      raise ValueError(f"Weird ckpt path: {ckpt} ; Maybe prepend ./ ?")

    # Use the checkpoint filename to detect when we're loading old-style .npz
    # checkpoints, as opposed to new-style tensorstore checkpoint folders.
    if ".npz" in ckpt:  # Not a perfect heuristic, but good enough.
      checkpoint = load_checkpoint(ckpt, **kw)
      checkpoint = jax.tree.map(recover_dtype, checkpoint)
      if "params" in checkpoint:
        # Checkpoint with optax state (after (internal link)).
        params = checkpoint["params"]
      elif "opt" in checkpoint:
        # Checkpoint with Flax optimizer.
        params = checkpoint["opt"]["target"]
      else:
        # When open-sourcing, we often shared only the params directly.
        params = checkpoint
    else:
      raise ValueError(f"Unsupported checkpoint format: {ckpt}")
  else:
    raise ValueError(f"Unsupported checkpoint type: {type(ckpt)}")

  if key is not None:
    params = tree_utils.tree_get(params, key)

  return params


def recover_dtype(a):
  """Numpy's `save` stores bfloat16 type as "void" type, so we recover it."""
  if hasattr(a, "dtype") and a.dtype.type is np.void:
    assert a.itemsize == 2, "Unknown dtype!"
    return a.view(jax.numpy.bfloat16)
  else:
    return a
