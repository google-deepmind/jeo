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

r"""Config for quick testing on Cifar10.
"""
from jeo.configs import config_utils
import ml_collections


def get_arg(arg):
  return config_utils.parse_arg(arg, runlocal=False, test=False)


def get_config(arg=None):
  """Returns config."""
  arg = get_arg(arg)
  config = ml_collections.ConfigDict()
  config.task_type = "classification"

  # Data.
  config.dataset = "cifar10"
  config.train_split = "train"
  config.shuffle_buffer_size = 10_000  # Per host.
  config.download_and_prepare = True
  config.num_classes = 10

  # Preprocessing.
  pp_pre = "decode|resize(64)"
  pp_post = (f"|onehot({config.num_classes}, key='label', key_result='labels')"
             "|value_range(-1,1)|keep('image','labels')")
  config.pp_train = pp_pre + pp_post
  pp_eval = pp_pre + pp_post

  # General training.
  config.seed = 0
  config.batch_size = 4096
  config.total_epochs = 200
  config.log_training_steps = 50
  config.log_eval_steps = 200
  config.ckpt_steps = 1000

  # Model.
  config.model_name = "bit"
  config.model = dict(
      depth=[2, 2, 2, 2],  # R26. Needs to be list to be overridable.
      width=1.0,
      num_classes=config.get_ref("num_classes"),
  )
  config.loss = "softmax_xent"

  # Optimizer.
  config.optax_name = "big_vision.momentum_hp"
  config.grad_clip_norm = 1.0
  config.wd = (1e-4 / 256) * config.batch_size
  config.lr = (0.1 / 256) * config.batch_size
  config.schedule = dict(decay_type="cosine",
                         warmup_steps=50_000 // config.batch_size * 5)

  # Eval.
  config.evals = ml_collections.ConfigDict()
  config.evals.val = ml_collections.config_dict.create(
      type="classification",
      dataset=config.dataset,
      dataset_dir=config.get("dataset_dir"),
      dataset_module=config.get("dataset_module"),
      dataset_kwargs=config.get("dataset_kwargs", {}),
      split="test",
      pp=pp_eval,
      loss_name=config.loss,
      metrics=("acc", "f1", "aucpr", "prec", "recall", "loss"),
      cache_final=True)

  if arg.runlocal:
    config.batch_size = 2
    config.total_epochs = None
    config.total_steps = 2
    config.schedule.warmup_steps = 1
    config.shuffle_buffer_size = None
    config.log_training_steps = 1
    config.log_eval_steps = 2
    config.evals.val.steps = 2
    config.evals.val.cache_final = False
    if "timing" in config.evals: config.evals.timing.steps = 2
    if "fewshot" in config.evals: config.evals.fewshot.steps = 2
    config.model.depth = [1, 1, 1, 1]
    config.xprof = False
    del config.ckpt_steps

  if arg.test:
    config.batch_size = 2
    config.total_epochs = None
    config.total_steps = 2
    config.schedule.warmup_steps = 1
    config.shuffle_buffer_size = None
    config.log_training_steps = 1
    config.log_eval_steps = 2
    config.evals.val.steps = 2
    config.evals.val.cache_final = False
    if "timing" in config.evals: config.evals.timing.steps = 2
    if "fewshot" in config.evals: del config.evals.fewshot
    config.model.depth = [1, 1, 1, 1]
    config.xprof = False
    del config.ckpt_steps

  return config


def nosweep(add):
  """Sweep."""
  del add


def metrics(*_):
  return ["val/acc", "val/prec", "val/loss", "a/imagenet_10shot",
          "training_loss", "l2_params", "l2_grads", "l2_updates",
          "img/sec/core", "global_schedule",]
