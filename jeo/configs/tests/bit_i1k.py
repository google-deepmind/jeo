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

r"""A config for pre-training BiT on ILSVRC-2012 ("ImageNet-1k").

Training of a BiT-ResNet-50x1 variant.
Based on: http://google3/third_party/py/big_vision/configs/bit_i1k.py

gxm third_party/py/jeo/launch.py -- \
--config tests/bit_i1k.py --t=df=8x8 -a=em \
--tags=jeo_regression_test

# Quick local run.
time third_party/py/jeo/runlocal.sh --config=configs/tests/bit_i1k.py:runlocal

Regression test: go/jeo-test-colab and go/jeo-test-xm
http://xm2a/experiments?searchParams=tag:jeo_regression_test%20name:bit_i1k
"""
from big_vision.configs import common as bvcc
import ml_collections as mlc


def get_pp(train):
  pp_common = ("|onehot(1000, key='label', key_result='labels')"
               "|value_range(-1, 1)"
               "|keep('image', 'labels')")
  if train:
    return ("decode_jpeg_and_inception_crop(224)"
            "|flip_lr") + pp_common
  else:
    return ("decode"
            "|resize_small(256)"
            "|central_crop(224)") + pp_common


def get_config(arg=None):
  """Config for training on ImageNet-1k."""
  arg = bvcc.parse_arg(arg, runlocal=False)
  runlocal = arg.runlocal
  config = mlc.ConfigDict()
  config.task_type = "classification"

  # Data.
  config.dataset = "imagenet2012"
  config.train_split = "train[:99%]"
  config.shuffle_buffer_size = 250_000 if not runlocal else 10_000  # Per host.
  config.num_classes = 1000
  config.cache_raw = not runlocal  # Needs up to 120GB of RAM!

  # Preprocessing.
  config.pp_train = get_pp(True)
  pp_eval = get_pp(False)

  # General training.
  config.seed = 0
  config.batch_size = 4096  if not runlocal else 32
  config.total_epochs = 90
  config.log_training_steps = 50
  config.ckpt_steps = 1000

  # Model.
  config.model_name = "bit"
  config.model = dict(
      depth=50,  # You can also pass e.g. [3, 5, 10, 2]
      width=1.0,
      num_classes=config.get_ref("num_classes"),
  )
  config.loss = "softmax_xent"

  # Optimizer.
  config.optax_name = "big_vision.momentum_hp"
  config.grad_clip_norm = 1.0
  # linear scaling rule. Don"t forget to sweep if sweeping batch_size.
  config.lr = (0.1 / 256) * config.batch_size
  config.wd = (1e-4 / 256) * config.batch_size
  config.schedule = dict(decay_type="cosine", warmup_steps=1000)

  # Eval.
  def get_eval(split, dataset="imagenet2012"):
    return dict(
        type="classification",
        dataset=dataset,
        split=split,
        steps=None,
        pp=pp_eval.format(lbl="label"),
        loss_name=config.loss,
        log_steps=1000,
        metrics=("acc", "prec", "recall", "loss"),
        cache_final=not runlocal
    )
  config.evals = {}
  config.evals.train = get_eval("train[:2%]")
  config.evals.minival = get_eval("train[99%:]")
  config.evals.val = get_eval("validation")

  if arg.runlocal:
    config.batch_size = 2
    config.total_epochs = None
    config.total_steps = 10
    config.shuffle_buffer_size = 10
    config.schedule.warmup_steps = 2
    config.log_training_steps = 1
    config.evals.val.log_steps = 5
    del config.evals.train
    del config.evals.minival
    config.evals.val.steps = 2
    config.xprof = False

  return config


def nosweep(add):
  add(total_epochs=90)
  add(total_epochs=300)


def metrics():
  return ["val/acc", "minival/acc", "minival/prec", ("core_hours", "val/prec"),
          "training_loss"]
