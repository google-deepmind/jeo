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

"""Metric writers.

Based on
https://github.com/google-research/big_vision/blob/main/big_vision/utils.py
"""

import json
import multiprocessing.pool
import os
import shutil

from absl import logging
from clu import metric_writers
import jax
import numpy as np


class MetricWriter:
  """A class for logging metrics."""

  def __init__(self, xid=-1, wid=-1, workdir=None, config=None):
    self.step_start(0)
    if jax.process_index() != 0:
      return  # Only one host shall write stuff.

    self.pool = multiprocessing.pool.ThreadPool(1)  # 1 is important here.
    self.fname = None
    if workdir:
      if xid != -1 and wid != -1:
        self.fname = os.path.join(workdir, f"jeo_{xid}_{wid}_metrics.txt")
      else:
        self.fname = os.path.join(workdir, "jeo_metrics.txt")
      if config:
        with open(os.path.join(workdir, "config.json"), "w") as f:
          f.write(config.to_json())

  def step_start(self, step):
    self.step = step
    self.step_metrics = {}

  def measure(self, name, value):
    """Logs the metric value."""
    if jax.process_index() != 0:
      return  # Only one host shall write stuff.

    # Convenience for accepting scalar np/DeviceArrays, as well as N-d single
    # scalars, like [[[123]]] or similar, avoiding silly mistakes.
    value = np.array(value).squeeze()

    # If the value is a scalar, we keep it in mind to append a line to the logs.
    # If it has any structure, we instead just log its shape.
    value = float(value) if value.ndim == 0 else value.shape

    logging.info(f"\u001b[35m[{self.step}]\u001b[0m {name} = {value}")  # pylint: disable=logging-fstring-interpolation
    logging.flush()
    self.step_metrics[name] = value

    return value  # Just for convenience

  def step_end(self):
    """Ends a training step, write its full row."""
    if not self.step_metrics:
      return

    def write(metrics):
      with open(self.fname, "a") as f:
        f.write(json.dumps({"step": self.step, **metrics}) + "\n")

    if self.fname:
      self.pool.apply(lambda: None)  # Potentially wait for past writes.
      self.pool.apply_async(write, (self.step_metrics,))

  def close(self):
    self.step_end()
    if jax.process_index() == 0:
      self.pool.close()
      self.pool.join()


def maybe_cleanup_workdir(workdir, cleanup, info):
  """Potentially removes workdirs at end of run for cleanup."""
  if not workdir:
    return

  if not cleanup:
    info("Logs/checkpoints are in %s", workdir)
  elif jax.process_index() == 0:
    shutil.rmtree(workdir, ignore_errors=True)
    # Only need this on the last work-unit, if already empty.
    shutil.rmtree(os.path.join(workdir, ".."), ignore_errors=True)
