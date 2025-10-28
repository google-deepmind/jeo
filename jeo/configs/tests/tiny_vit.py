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

r"""Config for e2e testing.
"""
from jeo.configs import config_utils
import ml_collections


def get_config(arg=None):
  """Returns config."""
  arg = config_utils.parse_arg(
      arg, runlocal=False, colab=False, test=False, bs=4096)
  config = ml_collections.ConfigDict()
  config.task_type = "classification"

  # Data.
  config.dataset = "cifar10"
  config.train_split = "train"
  config.shuffle_buffer_size = 10_000  # Per host.
  config.num_classes = 10

  # Preprocessing.
  pp_pre = "decode|resize(64)"
  pp_post = (f"|onehot({config.num_classes}, key='label', key_result='labels')"
             "|value_range(-1,1)|keep('image','labels')")
  config.pp_train = pp_pre + pp_post
  pp_eval = pp_pre + pp_post

  # General training.
  config.seed = 0
  config.batch_size = arg.bs
  config.total_epochs = 200
  config.log_training_steps = 50
  config.log_eval_steps = 200
  config.ckpt_steps = 1000

  # Model.
  config.model_name = "vit"
  config.model = dict(
      variant="S/16",
      rep_size=True,
      pool_type="gap",
      num_classes=config.get_ref("num_classes"),
  )
  config.loss = "softmax_xent"

  # Optimizer.
  config.optax_name = "big_vision.momentum_hp"
  config.grad_clip_norm = 1.0
  config.wd = (1e-4 / 256) * config.batch_size
  config.lr = (0.1 / 256) * config.batch_size
  # config.schedule = dict(decay_type="cosine", warmup_epochs=5)
  config.schedule = dict(decay_type="cosine",
                         warmup_steps=50_000 // config.batch_size * 5)

  # Eval.
  config.evals = {"val": ml_collections.config_dict.create(
      type="classification",
      dataset=config.dataset,
      dataset_dir=config.get("dataset_dir"),
      dataset_module=config.get("dataset_module"),
      dataset_kwargs=config.get("dataset_kwargs", {}),
      split="test",
      steps=None,
      pp=pp_eval,
      loss_name=config.loss,
      metrics=("acc", "f1", "aucpr", "prec", "recall", "loss"),
      cache_final=True)}
  config.early_stopping = {
      "monitor": "val/acc",
      "patience_epochs": 50,
      "start_from_epoch": 100,
      "min_delta": 0.001,
      }

  if arg.runlocal:
    config.model.variant = "Ti/16"
    config.batch_size = 2
    config.total_epochs = None
    config.total_steps = 2
    del config.schedule.warmup_steps
    del config.shuffle_buffer_size
    config.log_training_steps = 1
    config.log_eval_steps = 2
    config.val_steps = 2
    config.cache_raw = False
    config.xprof = False
    del config.ckpt_steps
    config.evals.val.steps = 2
    config.evals.val.cache_final = False

  if arg.colab:
    config.batch_size = 16
    config.total_epochs = 2
    config.schedule.warmup_steps = 1_000
    # config.schedule.warmup_epochs = 1
    config.shuffle_buffer_size = 10

  if arg.test:
    config.batch_size = 2
    config.total_epochs = None
    config.total_steps = 2
    config.schedule.warmup_steps = 1
    # config.schedule.warmup_epochs = None
    config.shuffle_buffer_size = None
    config.log_training_steps = 1
    config.log_eval_steps = 2
    config.val_steps = 2
    config.model.variant = "Ti/16"
    config.xprof = False
    del config.ckpt_steps

  return config


def nosweep(add):
  """Sweep."""
  del add


def metrics(*_):
  return ["val/acc", "val/recall", "val/prec", "val/loss",
          "training_loss", "l2_params", "l2_grads", "l2_updates",
          "img/sec/core", "global_schedule",]
