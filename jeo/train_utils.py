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

"""Trainer utils."""
import importlib
import inspect
import os
from typing import Any, Callable, Mapping, Optional, Sequence, Union

from absl import logging
import jax
from jeo.tools import bv_optax
from jeo.tools import bv_utils
import ml_collections
import numpy as np

from tensorflow.io import gfile

ParamsT = Mapping[str, Any]

BASEDIR = BASEDIR_EXP = "jeo"
BV_BASEDIR = "big_vision"
SCENIC_BASEDIR = "scenic"


def import_module(module_name, subdir="models", reload=False):
  """Imports module specified by name and subdir."""
  if subdir and not subdir.endswith("."):
    subdir += "."

  # Handle special import cases.
  if module_name.startswith("BV:"):
    path, cls_name = split_path_cls(
        f"{BV_BASEDIR}.{subdir}{module_name.replace('BV:', '')}")
  elif subdir.startswith("models") and module_name.startswith("SCENIC:"):
    path, cls_name = split_path_cls(
        f"{SCENIC_BASEDIR}.projects.{module_name.replace('SCENIC:', '')}")
  else:
    if not module_name.startswith("proj."):
      module_name = subdir + module_name
    path, cls_name = split_path_cls(f"{BASEDIR_EXP}.{module_name}")
    try:
      if importlib.util.find_spec(path) is None:
        path, cls_name = split_path_cls(f"{BASEDIR}.{module_name}")
    except ModuleNotFoundError:
      path, cls_name = split_path_cls(f"{BASEDIR}.{module_name}")

  module = importlib.import_module(path)
  if cls_name:
    if subdir.startswith("models"):  # Set the expected Model class.
      setattr(module, "Model", getattr(module, cls_name))
    else:
      module = getattr(module, cls_name)
  if reload:
    module = importlib.reload(module)
  return module


def split_path_cls(path) -> tuple[str, Optional[str]]:
  elems = path.split(".")
  if elems[-1][0].isupper():
    return ".".join(elems[:-1]), elems[-1]
  return path, None


def load(
    load_fn: Callable[..., Any],
    init_params: ParamsT,
    init_file: str,
    model_params: ml_collections.ConfigDict | None,
    init_state: ParamsT,
    load_state: bool = True,
    **kwargs,  # Usually includes/expects `dont_load` sequence.
) -> tuple[ParamsT, ParamsT]:
  """Loads model params and optionally state if supported by load function."""
  if "init_state" in inspect.signature(load_fn).parameters:
    params = load_fn(init_params, init_file, model_params,
                     init_state=init_state, **kwargs)
  else:
    params = load_fn(init_params, init_file, model_params, **kwargs)
  if isinstance(params, (tuple, list)):
    params, state = params
    if load_state:
      logging.info("Loaded params and state (%s)", state.keys())
      return params, state
  return params, init_state


def get_files(file_patterns: Union[str, Sequence[str]]) -> list[str]:
  """Returns list of files based on file_patterns."""
  if isinstance(file_patterns, str):
    if "," in file_patterns:  # list as str for config sweeps.
      file_patterns = [f.strip() for f in file_patterns.split(",") if f]
    else:
      file_patterns = [file_patterns]
  files = [f for fp in file_patterns for f in gfile.Glob(fp)]  # pylint: disable=g-complex-comprehension
  if not files:
    raise ValueError(f"No files found for given patterns: {file_patterns}")
  return files


def sync():
  """Syncs hosts and empties async computation queue."""
  x = jax.numpy.ones([jax.local_device_count()])
  x = jax.device_get(jax.pmap(lambda x: jax.lax.psum(x, "i"), "i")(x))
  assert x[0] == jax.device_count()


def finalize_and_cleanup(workdir, cleanup, error_msg):
  """Finalizes trainer flow (syncs hosts and removes workdir if requested)."""
  # Before cleanup, as cleanup should only run for successful jobs.
  if error_msg is not None:
    raise RuntimeError(error_msg)
  else:
    # Make sure all hosts stay up until the end of main.
    sync()

  if cleanup and workdir and jax.process_index() == 0:
    gfile.rmtree(workdir)
    try:  # Only need this on the last work-unit, if already empty.
      gfile.remove(os.path.join(workdir, ".."))
    except gfile.GOSError:
      pass


def validate_and_update_config_inplace(config):
  """Checks some config options and renames according to recent changes."""
  def copy_if_set(old_name, new_name):
    if old_name in config:
      assert new_name not in config, f"Delete unused/deprecated {old_name}."
      config[new_name] = config[old_name]
  # For backward compatibility (old names will be removed at some point).
  copy_if_set("num_epochs", "total_epochs")
  copy_if_set("checkpoint_steps", "ckpt_steps")
  copy_if_set("keep_checkpoint_steps", "keep_ckpt_steps")

  # Verify settings to make sure no checkpoints are accidentally missed.
  if config.get("keep_ckpt_steps"):
    assert config.get("ckpt_steps"), "Specify `ckpt_steps`."
    assert config.keep_ckpt_steps % config.ckpt_steps == 0, (
        f"`keep_ckpt_steps` ({config.ckpt_steps}) should be"
        f"divisible by `ckpt_steps ({config.ckpt_steps}).`")
  logging.info("Updated config: \n%s", config)


def save_metrics(workdir, metrics, step=None):
  path = os.path.join(workdir, "metrics.npz")
  with gfile.GFile(path, "wb") as f:
    np.savez(f, **metrics)
  if step is not None:
    step_path = os.path.join(workdir, f"_metrics-{step:09d}.npz")
    gfile.copy(path, step_path, overwrite=True)


def get_frozen_mask(params, schedule, log=None):
  """Returns frozen mask boolean pytree based on config.schedule."""
  if not isinstance(schedule, (tuple, list)):
    schedule = [(".*", schedule)]
  patterns, scheds = zip(*schedule)
  masks = bv_utils.make_mask_trees(params, patterns, log=log)
  frozen_mask, _, _ = bv_optax._split_frozen(masks, scheds)  # pylint: disable=protected-access
  return frozen_mask


def log_frozen(params, schedule):
  """Logs freeze state of params based on config.schedule."""
  fm = get_frozen_mask(params, schedule)
  frozen = jax.tree.leaves(fm)
  num_freeze, num_learn = sum(frozen), len(frozen) - sum(frozen)
  logging.info("Frozen params layers: %s frozen, %s not frozen", num_freeze,
               num_learn)
  # TODO: Incorporate this in paramter_overview.
  for i, (k, v) in enumerate(bv_utils.tree_flatten_with_names(fm)[0]):
    logging.info(f"{i:3} {k:<40}: {'Frozen' if v else 'not frozen'}")  # pylint: disable=logging-fstring-interpolation
  return num_freeze, num_learn
