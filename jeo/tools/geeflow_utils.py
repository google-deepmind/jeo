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

"""Utils related to geeflow."""

import json
import os
from typing import Any

import ml_collections

from tensorflow.io import gfile

BAND_PREFIX = "_band_"


def load_json(
    path: str,
    split_name: str | None = None,
    postfix: str | None = None,
    as_cd: bool = False,
    drop_support: bool = True,
) -> dict[Any, Any] | ml_collections.ConfigDict:
  """Returns dict of accumulated stats."""
  full_path = standardized_path(path, split_name, postfix)
  if gfile.exists(full_path):
    with gfile.GFile(full_path, "r") as f:
      d = dict(json.load(f))
  else:
    full_path = standardized_path(path, split_name, postfix + BAND_PREFIX + "*")
    d = {}
    for filename in gfile.Glob(full_path):
      band_position = filename.rindex(BAND_PREFIX) + len(BAND_PREFIX)
      band_id = filename[band_position: -5]  # -5 is for ".json"
      with gfile.GFile(filename, "r") as f:
        d[band_id] = json.load(f)

  for k in list(d):
    if isinstance(d[k], dict):  # For bands accumulators.
      for kk in list(d[k]):
        if kk.startswith("~"):
          if drop_support:
            del d[k][kk]
          else:
            d[k][kk[1:]] = d[k].pop(kk)
    if k.startswith("~"):  # rename support variables to original names.
      if drop_support:
        del d[k]
      else:
        d[k[1:]] = d.pop(k)
    if k.isnumeric():
      d[int(k)] = d.pop(k)  # Keep band keys as numeric for now.
  if as_cd:
    return ml_collections.ConfigDict(d)
  return d


def standardized_path(path: str, split_name: str | None = None,
                      postfix: str | None = None) -> str:
  """Constructs/adjusts full path for json file."""
  if split_name:
    path = os.path.join(path, split_name)
  if postfix:
    if path.endswith("/"):
      path = os.path.join(path, postfix)
    else:
      path = f"{path}_{postfix}"
  if not path.endswith(".json"):
    path += ".json"
  return path
