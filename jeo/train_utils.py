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

"""Trainer utils."""

import collections
from collections.abc import Callable, Mapping, Sequence
import contextlib
import importlib
import inspect
import multiprocessing
import os
import time
from typing import Any

from absl import logging
import jax
from jeo.tools import tree_utils
import ml_collections
import numpy as np

from tensorflow.io import gfile

ParamsT = Mapping[str, Any]

BASEDIR = BASEDIR_EXP = "jeo"
BV_BASEDIR = "big_vision"
SCENIC_BASEDIR = "scenic"


def import_module(module_name: str, subdir: str = "models",
                  reload: bool = False):
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


def split_path_cls(path: str) -> tuple[str, str | None]:
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


def get_files(file_patterns: str | Sequence[str]) -> list[str]:
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


def finalize_and_cleanup(
    workdir: str | None, cleanup: bool, error_msg: str | None
):
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


def validate_and_update_config_inplace(config: ml_collections.ConfigDict):
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


def _disabled_seek(*_):
  raise AttributeError("seek() is disabled on this object.")


def save_metrics(workdir: str, metrics: dict[str, Any],
                 step: int | None = None):
  """Saves metrics as npz."""
  path = os.path.join(workdir, "metrics.npz")
  with gfile.GFile(path, "wb") as f:
    setattr(f, "seek", _disabled_seek)
    np.savez(f, **metrics)
  if step is not None:
    gfile.makedirs(
        os.path.join(workdir, "intermediate_metrics"),
        mode=gfile.LEGACY_GROUP_WRITABLE_WORLD_READABLE,
    )
    step_path = os.path.join(
        workdir, "intermediate_metrics", f"metrics-{step:09d}.npz"
    )
    gfile.copy(path, step_path, overwrite=True)


def get_frozen_mask(params: ParamsT, schedule: Any, log: str | None = None):
  """Returns frozen mask boolean pytree based on config.schedule."""
  if not isinstance(schedule, (tuple, list)):
    schedule = [(".*", schedule)]
  patterns, scheds = zip(*schedule)
  masks = tree_utils.make_mask_trees(params, patterns, log=log)
  frozen_mask, _, _ = tree_utils.split_frozen(masks, scheds)  # pylint: disable=protected-access
  return frozen_mask


def log_frozen(params: ParamsT, schedule: Any):
  """Logs freeze state of params based on config.schedule."""
  fm = get_frozen_mask(params, schedule)
  frozen = jax.tree.leaves(fm)
  num_freeze, num_learn = sum(frozen), len(frozen) - sum(frozen)
  logging.info("Frozen params layers: %s frozen, %s not frozen", num_freeze,
               num_learn)
  # TODO: Incorporate this in paramter_overview.
  for i, (k, v) in enumerate(tree_utils.tree_flatten_with_names(fm)[0]):
    logging.info(f"{i:3} {k:<40}: {'Frozen' if v else 'not frozen'}")  # pylint: disable=logging-fstring-interpolation
  return num_freeze, num_learn


def itstime(
    step,
    every_n_steps,
    total_steps,
    host=None,
    last=True,
    first=True,
    drop_close_to_last=0.25,
):
  """Returns True if it's time to execute an action.

  Based on
  https://github.com/google-research/big_vision/blob/main/big_vision/utils.py

  Args:
    step: the current step representing "now".
    every_n_steps: the action should run every this many steps.
    total_steps: the step number of the last step of training.
    host: host number. If provided, only run if we are this process.
    last: whether to run on the last step or not.
    first: whether to run on the first step or not.
    drop_close_to_last: if a step would run, but is this close (in terms of
      fraction of every_n_step) to the last one, skip.

  Returns:
    True if the action should be executed, False if not.
  """

  # This logic avoids running `itstime` "a few" steps before the last step.
  # Canonical example: don't save checkpoint 2 steps before the last, and then
  # at the last again; it's pointless and checkpoint timing will time out.
  close_to_last = False
  if drop_close_to_last and every_n_steps:
    close_to_last = abs(step - total_steps) < drop_close_to_last * every_n_steps

  is_host = host is None or jax.process_index() == host
  is_step = every_n_steps and (step % every_n_steps == 0) and not close_to_last
  is_last = every_n_steps and step == total_steps
  is_first = every_n_steps and step == 1
  return is_host and (is_step or (last and is_last) or (first and is_first))


def checkpointing_timeout(writer, timeout):
  # Make sure checkpoint writing is not a bottleneck
  if writer is not None:
    try:
      # Note: `writer` is a multiprocessing.AsyncResult, and
      # timeout is in seconds.
      writer.get(timeout=timeout)
    except multiprocessing.TimeoutError as e:
      raise TimeoutError(
          "Checkpoint writing seems to be a bottleneck. Make sure you do "
          "not do something wrong, like writing checkpoints to a distant "
          "cell. In a case you are OK with checkpoint writing being a "
          "bottleneck, you can configure `ckpt_timeout` parameter"
      ) from e


def hms(s):
  """Format time in hours/minutes/seconds."""
  if s < 60:
    return f"{s:.0f}s"
  m, s = divmod(s, 60)
  if m < 60:
    return f"{m:.0f}m{s:.0f}s"
  h, m = divmod(m, 60)
  if h < 25:
    return f"{h:.0f}h{m:.0f}m"  # Seconds intentionally omitted.
  d, h = divmod(h, 24)
  return f"{d:.0f}d{h:.0f}h{m:.0f}m"  # Seconds intentionally omitted.


class Chrono:
  """Measures time and reports progress, hyper-specific to our train loops.

  Initially based on
  https://github.com/google-research/big_vision/blob/main/big_vision/utils.py

  Some concepts:
  1. This differentiates between three "types" of time:
    - training time: the time spent on actual training (fprop/bprop/update)
    - program time: overall time the program runs, including all overheads
    - pause time: the chronometer can be paused (eg during evals).
  2. This handles a "warmup": the first step is skipped for training time
      purposes, as it includes significant compilation overheads, which distort
      estimates.
  3. `accum`ulates (i.e. integrates) timings, and save/load them across
      restarts.
  """

  def __init__(self):
    self._timing_history = collections.defaultdict(list)
    self._measure = None
    self._write_note = None

    self.program_start_time = time.monotonic()
    self.train_start_time = None
    self.train_start_step = None  # When we started timing (after warmup)

    self.prev_time = None
    self.prev_step = None

    self.pause_start = None
    self.paused_time = 0

    self.total_steps = None
    self.global_bs = None
    self.steps_per_epoch = None

    self.warmup = 2  # How many calls to `tick` to skip.
    self.load()  # Inits accum integrators.
    self.note = "Chrono n/a"

  def inform(
      self,
      *,
      first_step=None,
      total_steps=None,
      global_bs=None,
      steps_per_epoch=None,
      measure=None,
      write_note=None,
  ):
    """Provide some extra info that's only known later in the program."""
    # The pattern of `self.x = x or self.x` allows one to call `inform` various
    # times with various subset of information (args), as they become available.
    # Except for `first_step` which can be 0 so is a bit more verbose.
    self.prev_step = first_step if first_step is not None else self.prev_step
    self.total_steps = total_steps or self.total_steps
    self.steps_per_epoch = steps_per_epoch or self.steps_per_epoch
    self.global_bs = global_bs or self.global_bs
    self._measure = measure or self._measure
    self._write_note = write_note or self._write_note
    if self.total_steps and self.prev_step is not None:
      self.note = (
          f"Steps:{self.prev_step}/{self.total_steps} "
          f"[{self.prev_step/self.total_steps:.1%}]"
      )

  def tick(self, step, measure=None, write_note=None):
    """A chronometer tick."""
    if step == self.prev_step:
      return  # Can happen from evals for example.

    measure = measure or self._measure
    write_note = write_note or self._write_note

    now = time.monotonic()
    measure("uptime", now - self.program_start_time)
    self.flush_timings()

    # We do always count examples, regardless of the timing-related warmup that
    # happens a few lines below.
    ds = step - self.prev_step  # Steps between ticks
    self.prev_step = step
    self.accum_examples_seen += ds * self.global_bs
    measure("examples_seen", self.accum_examples_seen)
    measure("progress", step / self.total_steps)
    if self.steps_per_epoch:
      measure("epoch", step / self.steps_per_epoch)

    # We take the start as the second time `tick` is called, so we avoid
    # measuring the overhead of compilation and don't include it in time
    # estimates.
    if self.warmup > 1:
      self.warmup -= 1
      write_note(self.note)  # This can help debugging.
      return
    if self.warmup == 1:
      self.train_start_time = self.prev_time = now
      self.train_start_step = step
      self.accum_program_time += now - self.program_start_time
      self.paused_time = 0  # Drop pauses that happened before timing starts.
      self.warmup = 0
      write_note(self.note)  # This can help debugging.
      return

    # Measurement with micro-timings of current training steps speed.
    # Time between ticks (ignoring pause)
    dt = now - self.prev_time - self.paused_time
    ncores = jax.device_count()  # Global device count
    measure("img/sec/core", self.global_bs * ds / dt / ncores)

    # Accumulate (integrate) times, good for plots.
    self.accum_train_time += dt
    self.accum_pause_time += self.paused_time
    self.accum_program_time += dt + self.paused_time

    # Convert to, and log as, core hours.
    core_hours = self.accum_train_time * ncores / 60 / 60
    devtype = jax.devices()[0].device_kind
    measure(f"core_hours_{devtype}", core_hours)
    measure("core_hours", core_hours)  # For convenience as x-axis in sweeps.

    # Progress note with "global" full-program average timings
    # (eg in program-time minus warmup)
    dt = now - self.train_start_time  # Time elapsed since end of warmup.
    steps_timed = step - self.train_start_step
    steps_todo = self.total_steps - step
    self.note = f"Steps:{step}/{self.total_steps} [{step/self.total_steps:.1%}]"
    self.note += f"\nWalltime:{hms(self.accum_program_time)}"
    self.note += f" ({hms(self.accum_pause_time)} eval)"
    self.note += f"\nETA:{hms(dt / steps_timed*steps_todo)}"
    self.note += f"\nTotal train time:{hms(dt / steps_timed*self.total_steps)}"
    write_note(self.note)

    log_memory(measure)

    self.prev_time = now
    self.paused_time = 0

  def pause(self, wait_for=()):
    assert self.pause_start is None, "Don't pause twice."
    jax.block_until_ready(wait_for)
    self.pause_start = time.monotonic()

  def resume(self):
    self.paused_time += time.monotonic() - self.pause_start
    self.pause_start = None

  def save(self):
    return dict(
        accum_program_time=self.accum_program_time,
        accum_train_time=self.accum_train_time,
        accum_pause_time=self.accum_pause_time,
        accum_examples_seen=self.accum_examples_seen,
    )

  def load(self, ckpt={}):  # pylint: disable=dangerous-default-value
    self.accum_program_time = float(ckpt.get("accum_program_time", 0.0))
    self.accum_train_time = float(ckpt.get("accum_train_time", 0.0))
    self.accum_pause_time = float(ckpt.get("accum_pause_time", 0.0))
    self.accum_examples_seen = int(ckpt.get("accum_examples_seen", 0))

  @contextlib.contextmanager
  def log_timing(self, name, *, noop=False):
    """Use this when you time sth once per step and want instant flushing."""
    t0 = time.monotonic()
    yield
    dt = time.monotonic() - t0
    if not noop:
      if self._measure:  # So that timed things still work in colab.
        self._measure(name, dt)
      logging.info("TIMING[%s]: %s", name, dt)
      logging.flush()

  @contextlib.contextmanager
  def log_timing_avg(self, name, *, noop=False):
    """Use this when you time sth multiple times per step (eg in a loop)."""
    t0 = time.monotonic()
    yield
    dt = time.monotonic() - t0
    if not noop:
      self._timing_history[name].append(dt)
      logging.info(
          "TIMING[%s]: avg %s current %s",
          name,
          np.mean(self._timing_history[name]),
          dt,
      )
      logging.flush()

  def flush_timings(self):
    assert self._measure is not None
    for name, times in self._timing_history.items():
      self._measure(name, np.mean(times))
    self._timing_history.clear()


# Singleton to use from everywhere. https://stackoverflow.com/a/6760726/2366315
chrono = Chrono()


def log_memory(measure):
  """Log a bunch of memory-related measurements."""
  try:
    import psutil  # pylint: disable=g-import-not-at-top
  except ImportError:
    psutil = None

  if psutil is not None:
    # Note that total != available + used, see psutil docs.
    vmem = psutil.virtual_memory()
    measure("y/hostmem/total", vmem.total)
    measure("y/hostmem/available", vmem.available)
    measure("y/hostmem/used", vmem.used)

  # We show only device 0 and 1 to avoid spam. The reason to show two and not
  # just one, if multiple are available, is because a frequent mistake is to
  # create arrays on the default device, which is device 0.
  for i, d in zip([0, 1], jax.local_devices()):
    for k, v in (d.memory_stats() or {}).items():
      measure(f"y/devmem/dev{i}/{k}", v)


def steps(
    prefix,
    config,
    data_size=None,
    batch_size=None,
    total_steps=None,
    default=ValueError,
):
  """Gets duration named `prefix` out of `config` and converts it to steps.

  Based on
  https://github.com/google-research/big_vision/blob/main/big_vision/utils.py

  Using this function to access a configuration value that denotes some kind
  of duration (eg training time, warmup, checkpoint frequency, ...) allows the
  duration to be specified in terms of steps, epochs, examples, or percent of
  training time, and converts any of these into steps, such that the training
  code only deals with steps.
  If the result is not an integer step number, it is rounded to the nearest one.

  Args:
    prefix: The name of the duration to query. The actual config fields can then
      be one of `prefix_steps`, `prefix_examples`, or `prefix_epochs`.
    config: The dictionary (config) from which to read the duration.
    data_size: The total number of training examples in one epoch.
    batch_size: The number of examples processed per step.
    total_steps: The total number of training steps to run.
    default: The default value to return when no duration of the name `prefix`
      is found in the `config`. Set to `ValueError` (the default) to raise an
      error instead of returning a default value.

  Returns:
    The number of steps from the config, or the default value.

  Raises:
    ValueError if there is no such duration in the config and no default is set.
  """
  # Be helpful and make sure only match one of the following suffixes.
  suffixes = {"steps", "examples", "epochs", "percent"}
  matches = {
      f"{prefix}_{s}"
      for s in suffixes
      if (x := config.get(f"{prefix}_{s}")) is not None and x >= 0
  }
  # Note that steps=0 is also a valid value (e.g. to only run evaluators).
  assert len(matches) <= 1, f"Only one of '{matches}' should be defined."

  if f"{prefix}_steps" in matches:
    return config[f"{prefix}_steps"]

  def to_integer(x):
    # Round to nearest but always executed at least one step unless explicitly
    # asked for 0. E.g. total_epochs=0 vs total_epochs=0.0001
    return max(1, round(x)) if x else 0

  if batch_size and f"{prefix}_examples" in matches:
    return to_integer(config[f"{prefix}_examples"] / batch_size)

  if batch_size and data_size and f"{prefix}_epochs" in matches:
    steps_per_epoch = data_size / batch_size
    return to_integer(config[f"{prefix}_epochs"] * steps_per_epoch)

  if total_steps and f"{prefix}_percent" in matches:
    pct = config[f"{prefix}_percent"]
    assert (
        0.0 <= pct <= 1.0
    ), (  # Be helpful, since it's not obvious.
        f"Percents should lie in [0.0, 1.0], but {prefix}_percent is {pct}"
    )
    return to_integer(pct * total_steps)

  if default is ValueError:
    raise ValueError(
        f"Cannot convert {prefix} to steps, due to missing batch_size "
        f"({batch_size}), data_size ({data_size}), total_steps ({total_steps})"
        ", or corresponding entry in config:\n"
        + "\n".join(config.keys())
    )

  return default


def startstop_prof(
    sess, step=None, first_step=0, log_steps=1, surround=10, **kw
):
  """Runs the profiler for `surround` steps around the next `log_steps`."""
  first_log = first_step + log_steps - (first_step % log_steps)
  # don't start before first!
  start = max(first_log - surround // 2, first_step + 1)
  return startstop_prof_at_steps(sess, step, start, start + surround, **kw)


def startstop_prof_at_steps(
    sess,
    step=None,
    first_step=None,
    last_step=None,
    name="steps",
    ttl=3 * 365 * 24 * 3600,
):
  del sess, step, first_step, last_step, name, ttl
  pass  # TODO: implement using `jax.profiler` API. Needs workdir.
