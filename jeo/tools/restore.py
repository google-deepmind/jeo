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

"""Checkpoint and model restoration related functions."""
import functools
import os
from typing import Any

from absl import logging
import flax
import flax.linen as nn
from jeo import train_utils
import ml_collections
import numpy as np

from google3.learning.deepmind.xmanager2.client import xmanager_api as xm
from google3.pyglib import gfile


def load_variables(ckpt_path: str) -> dict[str, Any]:
  """Loads all variables from numpy .npz checkpoint file."""
  with gfile.Open(ckpt_path, "rb") as fid:
    npz_dict = np.load(fid)
    d = dict(zip(npz_dict.keys(), npz_dict.values()))
    variables = flax.traverse_util.unflatten_dict(d, sep="/")
  # variables is a dictionary containing {params, extra, opt} and
  # potentially other state collections. Instead of returning only params, we
  # prefer to return all collections.
  return variables


def load_params_and_states(ckpt_path: str
                           ) -> tuple[dict[str, Any], dict[str, Any]]:
  """Loads params and states from numpy .npz checkpoint file."""
  variables = load_variables(ckpt_path)
  params = variables.pop("params")
  states = {k: v for k, v in variables.items() if k not in ["opt", "extra"]}
  return params, states


# TODO(mnn): Remove or switch to return only params.
def load_params(ckpt_path: str) -> dict[str, Any]:
  """Loads params from numpy .npz checkpoint file."""
  logging.warning("Deprecated: use load_variables() or "
                  "load_params_and_states() instead.")
  return load_variables(ckpt_path)


@functools.cache
def get_workdir(wu: xm.WorkUnit) -> str:
  """Returns workdir from work unit configuration or artifacts."""
  if "workdir" in wu.configuration:
    return wu.configuration["workdir"]
  for a in wu.get_artifacts():
    if a.description == "Workdir on CNS":
      return a.artifact
  raise ValueError(f"Could not identify workdir in {wu}.")


@functools.cache
def get_exp_workdir(xid: int) -> str:
  """Returns the workdir for the given experiment."""
  experiment = xm.XManagerApi().get_experiment(experiment_id=xid)
  for a in experiment.get_artifacts():
    if a.description == "Workdir on CNS":
      return a.artifact
  raise ValueError(f"Could not identify workdir in experiment {xid}.")


@functools.cache
def get_xm_client(force_remote=False):
  # force_remote will use real XM in localruns (eg for from_xid).
  return xm.XManagerApi(xm_deployment_env="alphabet", force_remote=force_remote)


def cache_data(src, dst_dir=None, sub_dir="", **kwargs):
  """Caches file or dir locally and returns path to new local location."""
  dst_dir = dst_dir or "/tmp/jeo_cache"
  if sub_dir:
    dst_dir = os.path.join(dst_dir, sub_dir)
  return _copy(src, dst_dir, **kwargs)


def _copy(src: str, dst_dir: str) -> str:
  """Copies file or directory to destination dir and returns dst path."""
  # TODO(mnn): add overwrite/force keyword to force overwriting files.
  # dst_dir is expected to be a directory which is created if it doesn't exist.
  if src.endswith("/"):
    src = src[:-1]
  if dst_dir.endswith("/"):
    dst_dir = dst_dir[:-1]
  gfile.MakeDirs(dst_dir, mode=gfile.LEGACY_GROUP_WRITABLE_WORLD_READABLE)
  if not gfile.IsDirectory(src):
    dst = os.path.join(dst_dir, os.path.basename(src))
    if not gfile.Exists(dst):
      logging.info("Copy %s to %s", src, dst)
      if src.endswith(".npz"):
        with gfile.Open(src, "rb") as f_src:
          with gfile.Open(dst, "wb") as f_dest:
            f_dest.write(f_src.read())
      else:
        gfile.Copy(src, dst)
    else:
      logging.info("Skip copy (file exists): %s", dst)
  else:
    dst = os.path.join(dst_dir, os.path.basename(src))
    for src_file in gfile.Glob(os.path.join(src, "*")):
      _copy(src_file, dst)
  return dst


@functools.cache
def get_model_and_ckpt_path(xid: int, wid: int = 1) -> tuple[nn.Module, str]:
  """Returns model and checkpoint path for given experiment work unit."""
  xm_client = get_xm_client()
  wu = xm_client.get_experiment(xid).get_work_unit(wid)
  config = ml_collections.ConfigDict(wu.configuration["config"])
  model_mod = train_utils.import_module(config["model_name"], "models")
  model = model_mod.Model(**config["model"])
  workdir = get_workdir(wu)
  ckpt_path = os.path.join(workdir, "checkpoint.npz")
  return model, ckpt_path


@functools.cache
def get_config(xid: int, wid: int = 1) -> ml_collections.ConfigDict:
  """Returns config for given experiment work unit."""
  wu = get_xm_client().get_experiment(xid).get_work_unit(wid)
  return ml_collections.ConfigDict(wu.configuration["config"])
