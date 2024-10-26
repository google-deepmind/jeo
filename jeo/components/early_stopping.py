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

"""Early stopping."""
import dataclasses
from typing import Optional

from absl import logging
import numpy as np


@dataclasses.dataclass
class EarlyStopping:
  """Early stopping monitor.

  The durations can be provided in steps or epochs. Additionally, consider to
  set `min_delta` and `mode`. Mode specifies whether the monitor metric is
  maximized or minimized.

  Example minimal config:
    config.early_stopping.monitor = "val/acc"
    config.early_stopping.patience_epochs = 10
    config.early_stopping.start_from_epoch = 50

  Attributes:
    monitor: Quantity to be monitored (KPI).
    min_delta: Minimum change in the monitored quantity to qualify as an
      improvement, i.e. an absolute change of less than min_delta, will count as
      no improvement.
    patience_epochs: Number of epochs with no improvement after which training
      will be stopped. Either patience_epochs or patience_steps should be given.
    patience_steps: Number of steps with no improvement after which training
      will be stopped. Either patience_epochs or patience_steps should be given.
    mode: One of `{"auto", "min", "max"}`. In `min` mode, training will stop
      when the quantity monitored has stopped decreasing; in `"max"` mode it
      will stop when the quantity monitored has stopped increasing; in `"auto"`
      mode, the direction is automatically inferred from the name of the
      monitored quantity. Defaults to "max" if monitoring mentric is not loss.
    start_from_epoch: Number of epochs to wait before starting to monitor
      improvement. This allows for a warm-up period in which no improvement is
      expected and thus training will not be stopped.
    start_from_step: See start_from_epoch. One of the two should be given.
    steps_per_epoch: Number of steps per epoch (if epoch quantities given).
  """

  monitor: str
  min_delta: float = 0.0
  mode: str = "auto"  # {auto, max, min}
  patience_steps: Optional[int] = None
  patience_epochs: Optional[int] = None
  start_from_step: Optional[int] = None
  start_from_epoch: Optional[int] = None
  steps_per_epoch: Optional[int] = None

  def __post_init__(self):
    # Initial validity checks and setup.
    assert self.mode in ("auto", "max", "min")
    assert bool(self.patience_steps) ^ bool(self.patience_epochs)
    assert bool(self.start_from_step) ^ bool(self.start_from_epoch)
    if not self.patience_steps:
      self.patience_steps = self.patience_epochs * self.steps_per_epoch
    if not self.start_from_step:
      self.start_from_step = self.start_from_epoch * self.steps_per_epoch
    if self.mode == "auto":
      self.mode = "min" if "loss" in self.monitor.lower() else "max"
    self._compare_fn = np.greater if self.mode == "max" else np.less
    self.min_delta = np.abs(self.min_delta)
    if self.mode != "max":
      self.min_delta *= -1.0
    logging.info("EarlyStopping setup: %s", self.__repr__())

    # Set up running variables.
    self._best = -np.inf if self.mode == "max" else np.inf
    self._best_step = 0

  def should_stop(self, step: int, metric: str, value: float) -> bool:
    """Determines whether early stopping criteria are met."""
    if metric != self.monitor:
      return False

    # Maximize: value > best+delta; Minimize: value < best-delta.
    is_improving = self._compare_fn(value, self._best + self.min_delta)
    if is_improving:
      self._best = value
      self._best_step = step
      return False

    if (step < self.start_from_step) or (
        step < self._best_step + self.patience_steps):
      return False

    return True


class NoStopping:
  """Dummy class to simplify wiring logic."""

  def should_stop(self, *_, **__) -> bool:
    return False


def from_config(config, steps_per_epoch):
  """Creates an early stopping monitor."""
  if "early_stopping" in config:
    return EarlyStopping(
        **config.early_stopping, steps_per_epoch=steps_per_epoch)
  logging.info("No EarlyStopping specified and set up.")
  return NoStopping()
