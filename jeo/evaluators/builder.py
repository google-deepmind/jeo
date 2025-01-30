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

"""Evaluators builder."""
import abc
from typing import Any, Callable, Sequence

from absl import logging
import jax
from jeo import input_pipeline
from jeo import train_utils
from jeo.metrics import clu_metrics_builder as jeo_metrics
from jeo.pp import pp_builder
import ml_collections

ConfigDict = ml_collections.ConfigDict


class EvaluatorBase(abc.ABC):
  """Evaluator interface."""

  @abc.abstractmethod
  def __init__(self, predict_fn, batch_size, metrics, **data_config):
    """Initializes the evaluator."""

  @abc.abstractmethod
  def _get_eval_fn(self, predict_fn):
    """Produces eval function, also applies pmap."""

  @abc.abstractmethod
  def run(self, params):
    """Runs across evaluation data and computes all metrics."""

  def _setup_metrics_and_eval_iter(
      self, metrics: Sequence[str], predict_fn: Callable[..., Any]
  ):
    self.metrics = jeo_metrics.MetricsCollection(metrics)
    self.metrics_fn = self.metrics.get_gather_fn()
    self.eval_fn = self._get_eval_fn(predict_fn)

  def _setup_dataset(self, batch_size, **data_config):
    """Constructs dataset and sets steps and iter attributes."""
    batch_size_per_host = batch_size // jax.process_count()
    pp_fn = pp_builder.get_preprocess_fn(data_config["pp"])
    ds, self.steps = input_pipeline.get_data(
        train=False,
        dataset=data_config.get("dataset"),
        split=data_config["split"],
        data_dir=data_config.get("dataset_dir"),
        dataset_module=data_config.get("dataset_module"),
        **data_config.get("dataset_kwargs", {}),
        batch_size=batch_size_per_host,
        preprocess_fn=pp_fn,
        filter_fn=data_config.get("filter_fn", None),
        cache_raw=data_config.get("cache_raw", False),
        cache_final=data_config.get("cache_final", False),
        prefetch=(0 if data_config.get("cache_final", False)
                  else data_config.get("prefetch_to_host", 2)),
        val_steps=data_config.get("steps", None),
        skip_decode=data_config.get("skip_decode", ("image",)))
    logging.info("Running validation for %d steps for %s, %s", self.steps,
                 data_config["dataset"], data_config["split"])
    if self.steps <= 0:
      raise ValueError(f"Insufficient val steps. steps: {self.steps} "
                       f"batch_size_eval: {batch_size}")
    self.data_iter = input_pipeline.start_input_pipeline(
        ds, data_config.get("prefetch_to_device", 1))


def from_config(
    config: ConfigDict,
    predict_fn: Callable[[Any], Any],
    write_note: Callable[[str], None] = lambda s: s,
    get_steps: Callable[[str, int, ConfigDict], int] | None = None,
    workdir: str | None = None,
) -> list[tuple[str, Any, int, str]]:
  """Creates a list of evaluators based on `config`."""
  evaluators = []
  specs = config.get("evals", {})
  is_dict = not isinstance(specs, (list, tuple))

  for spec in specs:
    if is_dict:
      name = spec
      cfg = specs[name].to_dict()
      module = cfg.pop("type", spec)
    else:
      if isinstance(spec, str):  # A shortcut when name == module
        spec = (spec, spec)
      name, module = spec
      if ":" in name:
        name = name.split(":")[-1]
      if name not in config:
        raise ValueError(f"Could not find configuration for evaluator {name}.")
      cfg = config.get(name).to_dict()  # Sanitize the config.

    write_note(f"{name} (module: {module})")
    # Use same batch_size as eval by default, to reduce fragmentation.
    cfg["batch_size"] = (cfg.get("batch_size") or config.get("batch_size_eval")
                         or config.get("batch_size"))
    # Evaluator log steps have precedence over global config log_eval steps.
    if get_steps is None:  # Legacy usage of log_eval_steps.
      logsteps = cfg.pop("log_steps", config.get("log_eval_steps", 1000))
    else:
      logsteps = get_steps("log", get_steps("log_eval", 1000, config), cfg)
    prefix = cfg.pop("prefix", f"{name}/")
    # Use global val_steps if steps not specified (only JEO evaluators).
    if ":" not in module:
      cfg["steps"] = cfg.get("steps", config.get("val_steps", None))
    # Pass skip_decode arg to all evaluators.
    cfg["skip_decode"] = config.get("skip_decode", ("image",))

    module_cls = train_utils.import_module(module, "evaluators")
    if not callable(module_cls):
      module_cls = getattr(module_cls, "Evaluator")
    evaluator = module_cls(predict_fn, **cfg)
    evaluators.append((name, evaluator, logsteps, prefix))

  return evaluators
