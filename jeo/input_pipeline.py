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

"""Input pipeline (initially based on BV)."""
from collections.abc import Callable, Iterator, Sequence
import functools
import math
from typing import Any

from absl import logging
import einops
import flax.jax_utils as flax_utils
import jax
from jeo import train_utils
from jeo.pp import pp_builder
# Importing default preprocessing ops in order to register them.
# If using ops not imported here, provide module names via config.pp_modules.
import jeo.pp.bv_ops  # pylint: disable=unused-import
import jeo.pp.image_ops  # pylint: disable=unused-import
import jeo.pp.pp_ops  # pylint: disable=unused-import
import jeo.pp.rand_det_ops  # pylint: disable=unused-import
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

FilterFn = Callable[[Any], bool]
PreprocessFn = Callable[[Any], Any]


def get_num_examples(
    dataset: str,
    split: str,
    data_dir: str | None = None,
    dataset_module: str | None = None,
    **kwargs
) -> int:
  """Returns number of examples for given split."""
  if dataset_module:
    ds_module = train_utils.import_module(dataset_module, "datasets")
    return ds_module.get_num_examples(dataset, split, data_dir=data_dir,
                                      **kwargs)
  else:
    builder = get_builder(dataset, data_dir)
    return builder.info.splits[split].num_examples


def _get_filter_fn(filter_fn: FilterFn | str | None) -> FilterFn | None:
  """Returns filter function."""
  if isinstance(filter_fn, str):
    if not filter_fn:
      return None
    pos = filter_fn.find("=")
    assert pos >= 0
    key = filter_fn[:pos]
    val = filter_fn[pos + 1:]
    if key.startswith("int:"):
      key = key[4:]
      val = int(val)
    if key.startswith("bool:"):
      key = key[5:]
      val = (str(val).lower() == "true")
    return lambda x: x[key] == val
  return filter_fn


def get_data(
    dataset: str,
    split: str,
    preprocess_fn: PreprocessFn | str,
    batch_size: int,
    data_dir: int | None,
    train: bool,
    *,
    cache_raw: bool = False,
    cache_raw_keys: Sequence[str] = (),
    # For train==True only:
    shuffle_buffer_size: int | None = None,
    prefetch: int = 2,
    filter_fn: FilterFn | str | None = None,
    filter_final_fn: FilterFn | str | None = None,
    num_parallel_calls: int = 100,
    # For train==False only:
    cache_final: bool = False,
    val_steps: int | None = None,  # num_batches per eval epoch.
    # For custom dataset loaders:
    dataset_module: str | None = None,
    skip_decode: Sequence[str] = ("image",),
    download_and_prepare: bool = False,
    **kwargs
) -> tuple[tf.data.Dataset, int]:
  """Returns dataset with the number of train examples or evaluation steps.

  Args:
    dataset: Name of the dataset.
    split: Name of the split.
    preprocess_fn: Preprocessing spec string or preprocessing function.
    batch_size: Batch size.
    data_dir: Path to the data (can be None).
    train: Whether to prepare data for training or evaluation.
    cache_raw: Whether to cache the raw data before preprocessing.
    cache_raw_keys: List of keys to keep in the cached dataset.
    shuffle_buffer_size: Shuffle buffer size.
    prefetch: Number of batches to prefetch.
    filter_fn: Filter function performed during preprocessing.
    filter_final_fn: Filter function applied after preprocessing.
    num_parallel_calls: Number of parallel calls for dataset map operations.
    cache_final: Whether to cache the final (preprocessed) data.
    val_steps: Number of batches to take for evaluation. If None, will be
      derived from the dataset and batch size.
    dataset_module: Optional dataset module name (if not using TFDS).
    skip_decode: List of features to skip decoding.
    download_and_prepare: Download and prepare TFDS dataset.
    **kwargs: Additional arguments passed to dataset loading functions.
  Returns:
    Prefetched tf.data.Dataset object of the training or evaluation split.
    Number of train examples in a single epoch if train=True, or the number of
      evaluation steps with the given batch size if train=False.
  """
  filter_fn = _get_filter_fn(filter_fn)
  filter_final_fn = _get_filter_fn(filter_final_fn)
  if dataset_module:
    ds_module = train_utils.import_module(dataset_module, "datasets")
    # data should already be sharded by host process_id.
    data = ds_module.get_dataset(
        dataset=dataset, split=split, shuffle_files=train,
        data_dir=data_dir, **kwargs)
  else:
    data, _ = get_dataset_tfds(
        dataset=dataset, split=split, shuffle_files=train,
        data_dir=data_dir, skip_decode=skip_decode,
        download_and_prepare=download_and_prepare)

  if not callable(preprocess_fn):
    preprocess_fn = pp_builder.get_preprocess_fn(preprocess_fn, log_steps=True)
  data = _add_tpu_host_options(data)
  # Use data filtering at your own risk: the actual split sizes won't be known
  # in advance, so many things can go wrong in the code.
  if filter_fn:
    data = data.filter(filter_fn)
  if cache_raw:
    # Only keep a subset of the data. This significanlty reduces the memory
    # usage when caching a raw dataset that containes more keys than those being
    # used during model training.
    if cache_raw_keys:
      fn = lambda x: {k: v for k, v in x.items() if k in cache_raw_keys}
      data = data.map(fn)
    data = data.cache()
  if train:  # Reuse new API from big_vision.
    if filter_fn:
      # Potentially a costly operation over large datasets.
      num_examples = data.reduce(0, lambda x, _: x+1).numpy()
    else:
      num_examples = get_num_examples(dataset, split, data_dir,
                                      dataset_module, **kwargs)
    # num_examples might be wrong if filter_final_fn is given.

    # Final caching of training data is not recommended (only valid/useful if
    # preprocessing is deterministic and cache_raw is too memory heavy).
    if cache_final:  # First, do preprocessing and filtering once.
      data = data.map(preprocess_fn, num_parallel_calls=num_parallel_calls)
      if filter_final_fn:
        data = data.filter(filter_final_fn)
      data = data.cache()
      data = data.repeat(None)  # repeat data indefinetely
    else:
      data = data.repeat(None)  # repeat data indefinetely
      data = data.map(preprocess_fn, num_parallel_calls=num_parallel_calls)
      if filter_final_fn:
        data = data.filter(filter_final_fn)
    data = data.shuffle(shuffle_buffer_size) if shuffle_buffer_size else data
    # Drop remainder makes shape fully static, so we can later use it if needed.
    data = data.batch(batch_size, drop_remainder=True)
    return data.prefetch(prefetch), num_examples
  else:
    # Filtering eval data can break down the logic and should be avoided.
    assert filter_final_fn is None
    # We need to make sure that all hosts process all data and exactly the same
    # number of batches. Below we take max per-host num examples and use it on
    # all hosts to derive the number of batches.
    if not val_steps:
      if dataset_module is None:  # TFDS dataset.
        splits = tfds.even_splits(split, jax.process_count())
        split_examples = [get_num_examples(dataset, s, data_dir)
                          for s in splits]
        if filter_fn:
          # Potentially a costly operation over large datasets.
          num_examples = data.reduce(0, lambda x, _: x+1).numpy()
        else:
          num_examples = sum(split_examples)
        max_num_examples_per_host = max(split_examples)
      else:
        num_examples = get_num_examples(dataset, split, data_dir,
                                        dataset_module, **kwargs)
        # Approximation, assuming the examples were evenly distributed across
        # all hosts. This is not always true, but is not critical for train.
        # TODO: Get exact number of examples per host for auto_dataset.
        max_num_examples_per_host = math.ceil(num_examples/jax.process_count())
      val_steps = math.ceil(max_num_examples_per_host / batch_size)
      logging.info("Non-train dataset %s split %s: val_steps=%s "
                   "num_examples=%s max_num_examples_per_host=%s", dataset,
                   split, val_steps, num_examples, max_num_examples_per_host)
    else:
      logging.info("Non-train dataset %s split %s: given config.val_steps=%s",
                   dataset, split, val_steps)

    data = data.map(_add_internal_fields(preprocess_fn),
                    num_parallel_calls=num_parallel_calls)
    data = data.concatenate(_get_pad_data(data))
    # Since we do 'infinite' padding it is safe to drop the remainder.
    data = data.batch(batch_size, drop_remainder=True)
    data = data.take(val_steps)

    # Note we cache data after a finite number of batches is taken.
    data = data.cache() if cache_final else data
    data = data.repeat()
    logging.info("Non-train dataset %s split %s: val_steps=%s",
                 dataset, split, val_steps)
    return data.prefetch(1), val_steps


def start_input_pipeline(
    data: tf.data.Dataset, n_prefetch: int, shard: bool = True
) -> Iterator[Any]:
  """Return iterator with prefetching and numpy conversion."""
  def to_numpy(x):
    if hasattr(x, "_numpy"):
      # _numpy() call avoids redundant copy when converting tf.Tensor to numpy.
      return x._numpy()  # pylint: disable=protected-access
    else:
      # Transforms x into read-only numpy array without copy if possible, see:
      # https://github.com/tensorflow/tensorflow/issues/33254#issuecomment-542379165
      x = np.asarray(memoryview(x))
    return x

  it = (jax.tree.map(to_numpy, b) for b in iter(data))  # pylint: disable=protected-access
  if shard:
    d = jax.local_device_count()
    shard_fn = lambda x: einops.rearrange(x, "(d b) ... -> d b ...", d=d)
    it = (jax.tree.map(shard_fn, x) for x in it)
  if shard and n_prefetch:  # Only works for pmap.
    it = flax_utils.prefetch_to_device(it, n_prefetch)
  return it


@functools.lru_cache(maxsize=None)
def get_builder(dataset: str, data_dir: str | None) -> tfds.core.DatasetBuilder:
  return tfds.builder(dataset, data_dir=data_dir or None, try_gcs=True)


def get_dataset_tfds(
    dataset: str,
    split: str = "train",
    shuffle_files: bool = True,
    data_dir: str | None = None,
    skip_decode: Sequence[str] = ("image",),
    download_and_prepare: bool = False,
) -> tuple[tf.data.Dataset, tfds.core.DatasetBuilder]:
  """Returns TFDS dataset split."""
  builder = get_builder(dataset, data_dir)
  if download_and_prepare:
    builder.download_and_prepare()
  split = tfds.even_splits(split, jax.process_count())[jax.process_index()]
  skip_decoders = {
      f: tfds.decode.SkipDecoding()
      for f in skip_decode if f in builder.info.features
  }
  # Each host is responsible for a fixed subset of data
  return builder.as_dataset(
      split=split,
      shuffle_files=shuffle_files,
      read_config=tfds.ReadConfig(
          skip_prefetch=True,  # We prefetch after pipeline.
          try_autocache=False,  # We control this, esp. for few-shot.
          add_tfds_id=True,
      ),
      decoders=skip_decoders), builder


# Forked from third_party/py/big_vision/input_pipeline.py
def _get_pad_data(data: tf.data.Dataset) -> tf.data.Dataset:
  def zeros_like_spec(spec):
    # For unknown/flexible dimensions (None), just use 0 instead.
    return tf.zeros([x or 0 for x in spec.shape], spec.dtype)

  zero = jax.tree.map(zeros_like_spec, data.element_spec)
  return tf.data.Dataset.from_tensors(zero).repeat()


# Forked from third_party/py/big_vision/input_pipeline.py
def _add_internal_fields(pp_fn: PreprocessFn) -> PreprocessFn:
  """Wraps pp_fn to add _mask and _id keys."""
  # Adds internal keys, that we either, in this order of preference:
  # 1. keep from result of pp_fn,
  # 2. carry over from raw (not pp_fn'd) example, or
  # 3. add, if that makes sense.
  def _pp_fn(example):
    result = pp_fn(example)
    # _mask will be False on padded examples (see _get_pad_data).
    result.setdefault("_mask", example.get("_mask", tf.constant(True)))
    # Not all data-sources can provide an ID. Only carry-over if it can:
    if "_id" in example and "_id" not in result:
      result["_id"] = example["_id"]
    return result
  return _pp_fn


# Forked from third_party/py/big_vision/input_pipeline.py
def _add_tpu_host_options(data: tf.data.Dataset) -> tf.data.Dataset:
  options = tf.data.Options()
  options.threading.private_threadpool_size = 48
  options.threading.max_intra_op_parallelism = 1

  # Stop a whole bunch of magic stuff that eats up all RAM:
  options.experimental_optimization.inject_prefetch = False

  return data.with_options(options)
