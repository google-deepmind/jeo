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

"""Utils and common structures across various models."""
from collections.abc import Mapping, MutableMapping
from typing import Any
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
    "B": dict(num_layers=12, num_heads=12, mlp_dim=3072, emb_dim=768),  # B/16
    "L": dict(num_layers=24, num_heads=16, mlp_dim=4096, emb_dim=1024),  # L/16
    "H": dict(num_layers=32, num_heads=16, mlp_dim=5120, emb_dim=1280),  # H/14
    "g": dict(num_layers=40, num_heads=16, mlp_dim=6144, emb_dim=1408),
    "G": dict(num_layers=48, num_heads=16, mlp_dim=8192, emb_dim=1664),
}


def convert_dict_to_string(d: dict[str, Any]) -> str:
  return ",".join(f"{k}=\"{v}\"" if isinstance(v, str) else f"{k}={v}"
                  for k, v in d.items())


def parse_arg(arg, lazy=False, **spec):
  """Parses a string of comma-separated key=value pairs.

  Adapted from http://github.com/google-research/big_vision/tree/HEAD/big_vision/configs/config_utils.py.

  Ways that values can be passed when launching:
    --config amazing.py:runlocal,schedule=long,res=128
    --config amazing.py:res=128
    --config amazing.py:runlocal  # A boolean needs no value for "true".
    --config amazing.py:runlocal=False  # Explicit false boolean.
    --config amazing.py:128  # The first spec entry may be passed unnamed alone.

  Uses strict bool conversion (converting 'True', 'true' to True, and 'False',
    'false', '' to False).

  Args:
    arg: the string argument that's passed to get_config.
    lazy: allow lazy parsing of arguments, which are not in spec. For these,
      the type is auto-extracted in dependence of most complex possible type.
    **spec: the name and default values of the expected options.
      If the value is a tuple, the value's first element is the default value,
      and the second element is a function called to convert the string or a
      sequence separator string.
      Otherwise the type is automatically extracted from the default value.

  Returns:
    ConfigDict object with extracted type-converted values.
  """
  # Normalize arg and spec layout.
  arg = arg or ""  # Normalize None to empty string
  spec = {k: get_type_with_default(v) for k, v in spec.items()}

  result = mlc.ConfigDict(type_safe=False)  # For convenient dot-access only.

  # Expand convenience-cases for a single parameter without = sign.
  if arg and "," not in arg and "=" not in arg:
    # (think :runlocal) If it's the name of sth in the spec (or there is no
    # spec), it's that in bool.
    if arg in spec or not spec:
      arg = f"{arg}=True"
    # Otherwise, it is the value for the first entry in the spec.
    else:
      arg = f"{list(spec.keys())[0]}={arg}"
      # Yes, we rely on Py3.7 insertion order!

  # Now, expand the `arg` string into a dict of keys and values:
  raw_kv = {raw_arg.split("=")[0]:
                raw_arg.split("=", 1)[-1] if "=" in raw_arg else "True"
            for raw_arg in arg.split(",") if raw_arg}

  # And go through the spec, using provided or default value for each:
  for name, (default, type_fn) in spec.items():
    val = raw_kv.pop(name, None)
    result[name] = type_fn(val) if val is not None else default

  if raw_kv:
    if lazy:  # Process args which are not in spec.
      for k, v in raw_kv.items():
        result[k] = autotype(v)
    else:
      raise ValueError(f"Unhandled config args remain: {raw_kv}")

  return result


def get_type_with_default(v):
  """Returns (v, string_to_v_type) with lenient bool parsing."""
  # For bool, do safe string conversion.
  if isinstance(v, bool):
    def strict_bool(x):
      assert x.lower() in {"true", "false", ""}
      return x.lower() == "true"
    return (v, strict_bool)
  # If already a (default, type) tuple, use that.
  if isinstance(v, (tuple, list)):
    assert len(v) == 2, (
        "To have a list or tuple argument, spec should be a two-element tuple. "
        "First element is the value, second is a type, function, or sequence "
        "separator."
    )
    if isinstance(v[1], str):
      # Second element is interpreted as a separator for a sequence.
      # The type of the sequence is the type of the first default element.
      val, sep = v
      if not isinstance(val, (tuple, list)):
        val = (val,)
      return val, lambda x: tuple(type(val[0])(y) for y in x.split(sep))
    return (v[0], v[1])
  # Otherwise, derive the type from the default value.
  return (v, type(v))


def autotype(x):
  """Auto-converts string to bool/int/float if possible."""
  assert isinstance(x, str)
  if x.lower() in {"true", "false"}:
    return x.lower() == "true"  # Returns as bool.
  try:
    return int(x)  # Returns as int.
  except ValueError:
    try:
      return float(x)  # Returns as float.
    except ValueError:
      return x  # Returns as str.


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


def get_ds(arg, ds=None, v=None, parent=None, default_parent="forest_typology"):
  """Constructs full dataset name.

  Args:
    arg: ConfigDict with `ds` and optionally `ds_suffix` and `parent_dir` keys.
    ds: Dataset name (if not specified, uses the one from arg).
    v: Version of the dataset, of the form "samples.sources.labels". Can use
      wildcard "*" to use the version from the training dataset.
    parent: Parent directory/name of the dataset. If not specified, uses
      `parent_dir` from arg if specified, or the default_parent otherwise.
    default_parent: Default parent directory of the dataset (lowest priority).
  Returns:
    Full dataset name of the form: {parent}/{ds}{ds_suffix}:{version}.
  """
  ds = ds or arg.ds
  ds = ds.split("#")[0]
  if ":" in ds:
    ds, version = ds.split(":")
  else:
    _, version = arg.ds.split("#")[0].split(":")
  if v is not None:
    samples_v, sources_v, labels_v = v.split(".")
    samples_v = samples_v if samples_v != "*" else version.split(".")[0]
    sources_v = sources_v if sources_v != "*" else version.split(".")[1]
    labels_v = labels_v if labels_v != "*" else version.split(".")[2]
    version = f"{samples_v}.{sources_v}.{labels_v}"
  parent = parent or arg.get("parent_dir", default_parent)
  return f"{parent}/{ds}{arg.get('ds_suffix', '')}:{version}"


def update_config(cfg: MutableMapping[str, Any], updates: Mapping[str, Any]):
  """Updates a nested config with a flat dictionary of updates.

  Args:
    cfg: The config to update. Can be a dict or a ConfigDict.
    updates: A dictionary of updates to apply to the cfg. The keys are
      dot-separated paths to the leaves to update, and the values are the
      update values.
  """
  for k, v in updates.items():
    target = cfg
    levels = k.split(".")
    for level in levels[:-1]:
      if level not in target:
        target[level] = {}
      target = target[level]
    try:
      target[levels[-1]] = v
    except TypeError as e:
      # Pay attention to proper types if updating a ConfigDict.
      raise e
