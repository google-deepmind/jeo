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

"""Metric writers."""

from clu import metric_writers  # Likely soon replacing by direct datatables API
import jax
import numpy as np


class MetricWriter:
  """Combined metric writer for big_vision-style writing.

  The high-level idea is that we log something only every N steps (eg 50).
  Then, within a step that logs, we collect the various "measure" calls that
  happen throughout the step, and send an actual, async "write" at the end of
  the step.
  Then the next step that has non-empty write (eg 50 steps later) actually waits
  for the last write to finish. This provides optimal async, while avoiding
  "infinite build-up" of pending writes, which happens for example when logging
  every single step and cause many other issues.
  """

  def __init__(self, xid, wid, xmeasurements=True, flush_immediately=True):
    self.step = 0
    # For experiments that at multiple steps per second, it's advisable to
    # disable flush_immediately. Otherwise waiting for DataTable service
    # becomes a bottleneck.
    self.flush_immediately = flush_immediately
    self.step_metrics = {}

    noop = metric_writers.MultiWriter([])
    self.writer_log = noop
    self.writer_xm = noop
    self.writer_dt = noop
    self.writer_dt_arrays = noop

    if jax.process_index() != 0: return  # Only one host shall write stuff.

    self.writer_log = metric_writers.LoggingWriter()

    if xid <= 0 or wid <= 0: return  # When locally run, only logging writer.

    if xmeasurements:
      # First test if the user is still able to write XM anymore.
      try:
        metric_writers.XmMeasurementsWriter().write_scalars(0, {"zzz/test": 0.})
      except Exception as e:  # pylint: disable=broad-exception-caught
        if "XM Measurements has been deprecated" not in str(e):
          raise
      else:
        # If it worked, then the user can still write to XM.
        # XM writer is async with large worker pool, because it's writing
        # metrics one-by-one which would get horribly slow otherwise.
        # Multithread is only a problem if there's no sync when crossing steps.
        self.writer_xm = metric_writers.AsyncWriter(
            metric_writers.XmMeasurementsWriter(), num_workers=None)

    # This is in beta forever. See go/brain-datatable-doc
    if metric_writers.DatatableWriter.user_has_beta_access():
      self.writer_dt = metric_writers.DatatableWriter(
          f"/datatable/xid/{xid}/data", [("wid", wid)])
      self.writer_dt_arrays = metric_writers.DatatableWriter(
          f"/datatable/xid/{xid}/arrays", [("wid", wid)])
      # NOTE: we use a separate datatable for arrays than for scalars, because
      # if we interleave large arrays into the scalar one, it takes forever to
      # load in flatboard even if we never look at the arrays!

  def step_start(self, step):
    self.step = step

  def measure(self, name, value):
    """Records measurement `name` with `value`, which may be scalar or img."""
    # Convenience for accepting scalar np/DeviceArrays, as well as N-d single
    # scalars, like [[[123]]] or similar, avoiding silly mistakes.
    value = np.array(value).squeeze()

    # For logging and XMeasurements scalars, we can shoot them off immediately:
    if value.shape == ():  # Scalar pylint: disable=g-explicit-bool-comparison
      self.writer_log.write_scalars(self.step, {name: float(value)})
      self.writer_xm.write_scalars(self.step, {name: float(value)})
    else:  # use this to log shape to help with debugging.
      self.writer_log.write_images(self.step, {name: value})

    # For datatable, we append to the collection of measurements for the current
    # step, and we'll only write the full row at the end of step. This is what
    # datatables really want and makes many flatboard features work.
    self.step_metrics[name] = value

    return value  # Just for convenience

  def step_end(self):
    """Ends a training step, write its full row, and prepares for next step."""
    if not self.step_metrics: return

    if self.flush_immediately:
      # Potentially waits for past writes to finish.
      if hasattr(self.writer_dt, "_writer"):  # Guard needed for no-op writer.
        self.writer_dt._writer.wait()  # pytype: disable=attribute-error # pylint: disable=protected-access
      if hasattr(self.writer_dt_arrays, "_writer"):
        self.writer_dt_arrays._writer.wait()  # pytype: disable=attribute-error # pylint: disable=protected-access

    self.writer_dt.write_scalars(self.step, {
        n: v for n, v in self.step_metrics.items() if v.shape == ()})  # pylint: disable=g-explicit-bool-comparison
    array_data = {n: v for n, v in self.step_metrics.items() if v.shape}
    if array_data:
      self.writer_dt_arrays.write_summaries(self.step, array_data)  # pylint: disable=g-explicit-bool-comparison
    self.step_metrics = {}

    if self.flush_immediately:
      # Async call that triggers writing in the background. We'll wait for the
      # writes to finish on the next non-empty call of `step_end` above.
      if hasattr(self.writer_dt, "_writer"):
        self.writer_dt._writer.flush()  # pytype: disable=attribute-error # pylint: disable=protected-access
      if hasattr(self.writer_dt_arrays, "_writer"):
        self.writer_dt_arrays._writer.flush()  # pytype: disable=attribute-error # pylint: disable=protected-access

  def close(self):
    self.step_end()
    self.writer_log.close()
    self.writer_xm.close()
    self.writer_dt.close()
    self.writer_dt_arrays.close()

