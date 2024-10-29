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

"""Checkpoint handling.

Interface to the good old simple BigVision checkpointing.
To be updated later.
"""
from jeo.tools import bv_utils


def load_params(path, **kwargs):
  return bv_utils.load_params(path, **kwargs)


def load_checkpoint(path, tree=None):
  return bv_utils.load_checkpoint_np(path, tree)


def save_checkpoint(ckpt, path, step_copy=None, compressed=False):
  return bv_utils.save_checkpoint(ckpt, path, step_copy, compressed)
