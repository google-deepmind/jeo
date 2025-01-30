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

# pylint: disable=line-too-long
"""Utils and common structures across various models."""
import ml_collections as mlc

TRANSFORMERS = {
    # Default MAE decoder.
    "mae_dec": dict(num_layers=8, num_heads=16, mlp_dim=2048, emb_dim=512),
    # VTT enc-dec.
    "detr": dict(num_layers=6, num_heads=8, mlp_dim=1024, emb_dim=256),
    "small": dict(num_layers=8, num_heads=8, mlp_dim=2048, emb_dim=512),
    "base": dict(num_layers=12, num_heads=12, mlp_dim=3072, emb_dim=768),
    "large": dict(num_layers=24, num_heads=16, mlp_dim=4096, emb_dim=1024),
    # ViT: http://github.com/google-research/big_vision/tree/HEAD/big_vision/models/vit.py?l=243-246
    # Rules (up until L): mlp=emb*4; num_heads=emb/64.
    "TinyToy": dict(num_layers=3, num_heads=1, mlp_dim=192, emb_dim=48),
    "Toy": dict(num_layers=6, num_heads=2, mlp_dim=384, emb_dim=96),
    "Ti": dict(num_layers=12, num_heads=3, mlp_dim=768, emb_dim=192),
    "S": dict(num_layers=12, num_heads=6, mlp_dim=1536, emb_dim=384),
    "M": dict(num_layers=12, num_heads=8, mlp_dim=2048, emb_dim=512),
    "B": dict(num_layers=12, num_heads=12, mlp_dim=3072, emb_dim=768),  # B/16 (="base")
    "L": dict(num_layers=24, num_heads=16, mlp_dim=4096, emb_dim=1024),  # L/16
    "H": dict(num_layers=32, num_heads=16, mlp_dim=5120, emb_dim=1280),  # H/14
    "g": dict(num_layers=40, num_heads=16, mlp_dim=6144, emb_dim=1408),  # mlp4x: 5632
    "G": dict(num_layers=48, num_heads=16, mlp_dim=8192, emb_dim=1664),  # mlp4x: 6656
}


def get_fewshot(target_resolution=224, resize_resolution=256, runlocal=False,
                use_imagenet_subset=True, **kw):
  """Returns a standard-ish big_vision based fewshot eval config."""
  kw.setdefault("representation_layer", "pre_logits")
  kw.setdefault("prefix", "")  # No prefix as we already use a/ z/ and zz/.

  config = mlc.ConfigDict(kw)
  config.type = "fewshot"
  config.datasets = {
      "pets": ("oxford_iiit_pet", "train", "test"),
      "uc_merced": ("uc_merced", "train[:1000]", "train[1000:]"),
      }
  # imagenet2012_subset/10pct performs nearly exactly the same as imagenet2012,
  # but sometimes it has slightly better results.
  if use_imagenet_subset:
    config.datasets.imagenet = (
        "imagenet2012_subset/10pct", "train", "validation")
  else:
    # The first 65000 ImageNet samples have at least 30 shots per any class.
    config.datasets.imagenet = (
        "imagenet2012", "train[:65000]", "validation")

  config.pp_eval = (
      f"decode|resize({resize_resolution},key='image')"
      f"|central_crop({target_resolution},key='image')"
      f"|value_range(-1,1,key='image')|keep('image', 'label')")
  config.display_first = [("imagenet", 10)]

  if runlocal:
    config.datasets = {"pets": ("oxford_iiit_pet", "train", "test")}

  return config


def get_linear_probe_imagenet(runlocal=False, **kw):
  """Returns a standard-ish linear probe config of ImageNet."""
  kw.setdefault("representation_layer", "pre_logits")

  pp_common = ("|onehot(1000, key='label', key_result='labels')"
               "|value_range(-1, 1)"
               "|keep('image', 'labels')")
  pp_train = "decode_jpeg_and_inception_crop(224)|flip_lr" + pp_common
  pp_val = "decode|resize_small(256)|central_crop(224)" + pp_common

  config = mlc.ConfigDict(kw)
  config.type = "linear_probe"
  config.loss_name = "softmax_xent"
  config.log_steps = 1000
  config.metrics = ("acc", "prec", "recall", "loss")
  # Eval dataset.
  config.dataset = "imagenet2012"
  config.split = "train"
  config.cache_final = not runlocal
  config.pp = pp_val
  # Train config.
  config.train_config = mlc.ConfigDict(dict(
      task_type="classification",
      loss="softmax_xent",
      total_epochs=5,  # 60,
      optax_name="big_vision.momentum_hp",
      lr=0.01,  # ref_lr * batch_size / 256
      schedule={"decay_type": "cosine"},
      dataset="imagenet2012",
      split="train",
      num_classes=1000,
      pp_train=pp_train))

  return config
