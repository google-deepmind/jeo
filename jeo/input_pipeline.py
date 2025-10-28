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

"""Input pipeline (initially based on BV)."""
from collections.abc import Callable, Iterator, Sequence
import functools
import math
import multiprocessing.pool
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


def split_dataset_spatially(
    data: tf.data.Dataset, n: int, shift: int, size: int,
    keep_keys: Sequence[str], split_keys: Sequence[str]
    ) -> tuple[tf.data.Dataset, int]:
  """Splits each example in the dataset into n^2 examples.

  Args:
    data: Dataset to split.
    n: Number of splits (along both x and y axes).
    shift: X and Y pixelwise shift.
    size: Size of each sub example.
    keep_keys: All keys that need to be preserved.
    split_keys: Keys that need to be split(cropped).

  Returns:
    A new dataset and a multiplier for the number of examples.
  """
  dx = tf.data.Dataset.range(n)
  dy = tf.data.Dataset.range(n)

  dxy = dx.window(1).map(lambda x: x.repeat())
  dxy = dxy.flat_map(lambda x: tf.data.Dataset.zip((x, dy)))

  data = data.window(1).map(lambda ds: tf.data.Dataset.zip(
      tuple([ds[k] for k in keep_keys])).repeat())
  data = data.flat_map(lambda x: tf.data.Dataset.zip((x, dxy)))

  data = data.map(lambda x, y: ({k: x[i] for i, k in enumerate(keep_keys)} |
                                {"dx": y[0] * shift, "dy": y[1] * shift}))
  def crop(d):
    for k in split_keys:
      d[k] = d[k][..., d["dy"] : d["dy"] + size, d["dx"]: d["dx"] + size, :]
      d[k] = tf.reshape(d[k],
                        tuple(d[k].shape[:-3]) + (size, size, d[k].shape[-1]))
    return d
  data = data.map(crop)

  return data, n * n


def _get_single_ds(
    dataset: str | None,
    split: str,
    preprocess_fn: PreprocessFn | str,
    batch_size: int,
    data_dir: int | None,
    train: bool,
    *,
    cache_raw: bool = False,
    keep_keys: Sequence[str] = (),
    batch_preprocess_fn: PreprocessFn | str = "",
    # For train==True only:
    shuffle_buffer_size: int | None = None,
    prefetch: int = 2,
    filter_fn: FilterFn | str | None = None,
    filter_final_fn: FilterFn | str | None = None,
    num_parallel_calls: int = 100,
    # For train==False only:
    split_dataset_args: (
        tuple[int, int, int, Sequence[str], Sequence[str]] | None
    ) = None,
    cache_final: bool = False,
    val_steps: int | None = None,  # num_batches per eval epoch.
    # For custom dataset loaders:
    dataset_module: str | None = None,
    skip_decode: Sequence[str] = ("image",),
    download_and_prepare: bool = False,
    interleave_cycle_length: int | None = None,
    wid: int | None = None,
    data_service_address: str | None = None,
    data_service_max_outstanding_requests: int | None = None,
    data_service_batch_size: int | None = None,
    **kwargs,
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
    keep_keys: Subset of keys that are read from the tfds dataset. This avoids
      reading features that are not used later on. This works for both cached
      and non cached datasets.
    batch_preprocess_fn: Preprocessing spec string or preprocessing function
      applied on entire batches.
    shuffle_buffer_size: Shuffle buffer size.
    prefetch: Number of batches to prefetch.
    filter_fn: Filter function performed during preprocessing.
    filter_final_fn: Filter function applied after preprocessing.
    num_parallel_calls: Number of parallel calls for dataset map operations.
    split_dataset_args: Arguments for split_dataset function.
    cache_final: Whether to cache the final (preprocessed) data.
    val_steps: Number of batches to take for evaluation. If None, will be
      derived from the dataset and batch size.
    dataset_module: Optional dataset module name (if not using TFDS).
    skip_decode: List of features to skip decoding.
    download_and_prepare: Download and prepare TFDS dataset.
    interleave_cycle_length: Interleave cycle length.
    wid: XManager work unit id (used for data service).
    data_service_address: If using tf data service, this should be set to the
      address of the service.
    data_service_max_outstanding_requests: Maximum number of outstanding
      concurrent requests to the data service. This effectively controls the
      number of batches that are cached on the trainer from the data service at
      any one point in time (important as sometimes this being too high or
      autotuned can cause OOMs).
    data_service_batch_size: The batch size to use for the TF data service.
      Large batch sizes result in OOMs and proto serialization errors. The proto
      size limit is 2GB so the entire batch must fit in this limit. This is
      optional and can be set to None if we just want to use the default batch
      size for the data service too.
    **kwargs: Additional arguments passed to dataset loading functions.

  Returns:
    Prefetched tf.data.Dataset object of the training or evaluation split.
    Number of train examples in a single epoch if train=True, or the number of
      evaluation steps with the given batch size if train=False.
  """
  if data_service_address and (cache_raw or cache_final):
    raise ValueError("tf.data service is not supported for data caching.")
  filter_fn = _get_filter_fn(filter_fn)
  filter_final_fn = _get_filter_fn(filter_final_fn)
  if dataset_module:
    ds_module = train_utils.import_module(dataset_module, "datasets")
    # data should already be sharded by host process_id.
    data = ds_module.get_dataset(
        dataset=dataset, split=split, shuffle_files=train,
        data_dir=data_dir, keep_keys=keep_keys, **kwargs)
  else:
    data, _ = get_dataset_tfds(
        dataset=dataset, split=split, shuffle_files=train,
        data_dir=data_dir, skip_decode=skip_decode,
        download_and_prepare=download_and_prepare,
        interleave_cycle_length=interleave_cycle_length,
        keep_keys=keep_keys)

  if not callable(preprocess_fn):
    preprocess_fn = pp_builder.get_preprocess_fn(preprocess_fn, log_steps=True)
  if not callable(batch_preprocess_fn):
    batch_preprocess_fn = pp_builder.get_preprocess_fn(
        batch_preprocess_fn, log_steps=True
    )
  data = _add_tpu_host_options(data)

  num_examples_multiplier = 1
  if split_dataset_args:
    data, num_examples_multiplier = split_dataset_spatially(data,
                                                            *split_dataset_args)

  # Use data filtering at your own risk: the actual split sizes won't be known
  # in advance, so many things can go wrong in the code.
  if filter_fn:
    data = data.filter(filter_fn)
  if cache_raw:
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
      data = data.repeat(None)  # repeat data indefinitely
    else:
      data = data.repeat(None)  # repeat data indefinitely
      data = data.map(preprocess_fn, num_parallel_calls=num_parallel_calls)
      if filter_final_fn:
        data = data.filter(filter_final_fn)
    data = data.shuffle(shuffle_buffer_size) if shuffle_buffer_size else data

    # If the batch_size is large, then protos serialized during the tf data
    # service can exceed the 2GB proto size limit. To avoid this we can reduce
    # the batch size temporarily over here.
    intermediate_batch_size = data_service_batch_size or batch_size

    if intermediate_batch_size > 0:
      # Drop remainder makes shape fully static, so we can later use it if
      # needed.
      data = data.batch(intermediate_batch_size, drop_remainder=True)

    # Preprocessing applied on entire batches, e.g. reshaping or cut-mix.
    data = data.map(batch_preprocess_fn, num_parallel_calls)

    if data_service_address:
      data = data.apply(
          tf.data.experimental.service.distribute(
              # TODO: DYNAMIC sharding would be more correct here
              # as it would mix the data better. But currently it doesn't work
              # with the rest of the jeo data pipeline, giving this issue:
              # (internal link)
              processing_mode=tf.data.experimental.service.ShardingPolicy.OFF,
              service=data_service_address,
              job_name=f"train_data/{wid}/{dataset}/{split}".replace(
                  ":", "_").replace(".", "_"),
              max_outstanding_requests=data_service_max_outstanding_requests,
          )
      )
      if intermediate_batch_size != batch_size:
        data = data.unbatch().batch(batch_size, drop_remainder=True)
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
        split_examples = [
            get_num_examples(dataset, s, data_dir) * num_examples_multiplier
            for s in splits]
        if filter_fn:
          # Potentially a costly operation over large datasets.
          num_examples = data.reduce(0, lambda x, _: x+1).numpy()
        else:
          num_examples = sum(split_examples)
        max_num_examples_per_host = max(split_examples)
      else:
        num_examples = get_num_examples(
            dataset, split, data_dir, dataset_module,
            **kwargs) * num_examples_multiplier
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

    # Preprocessing applied on entire batches, e.g. reshaping or mix-cut.
    data = data.map(batch_preprocess_fn, num_parallel_calls)

    # Note we cache data after a finite number of batches is taken.
    data = data.cache() if cache_final else data
    data = data.repeat()
    logging.info("Non-train dataset %s split %s: val_steps=%s",
                 dataset, split, val_steps)
    return data.prefetch(1), val_steps


def _get_mixture_ds(
    dataset: Sequence[str],
    split: str | Sequence[str],
    batch_size: int,
    dataset_weights: Sequence[float] | None = None,
    prefetch: int = 2,
    **kwargs) -> tuple[tf.data.Dataset, int]:
  """Returns dataset with the number of train examples or evaluation steps.

  Args:
    dataset: Name of the dataset (or a list).
    split: Name of the split (or a list).
    batch_size: Batch size.
    dataset_weights: Weights of each dataset (sample) or None if just weighting
      by dataset sizes (each sample from any dataset has the same
      weight/probability).
    prefetch: Number of batches to prefetch.
    **kwargs: Additional arguments passed to dataset loading functions.
  Returns:
    Prefetched tf.data.Dataset object of the training or evaluation split.
    Number of train examples in a single epoch if train=True, or the number of
      evaluation steps with the given batch size if train=False.
  """
  assert kwargs["train"]
  if isinstance(split, str):
    split = [split] * len(dataset)
  assert len(dataset) == len(split)
  if dataset_weights is None:
    dataset_weights = [1] * len(dataset)
  assert len(dataset_weights) == len(dataset)

  names, datasets, totals = [], [], []
  pool = multiprocessing.pool.ThreadPool(len(dataset))

  def _make(dataset_and_split):
    dataset, split = dataset_and_split
    ds, total = _get_single_ds(dataset=dataset,
                               split=split,
                               batch_size=0,
                               prefetch=0,
                               **kwargs)
    return f"{dataset}:{split}", ds, total  # pylint: disable=bad-whitespace

  for name, dataset, total in pool.map(
      _make, ((ds, sp) for ds, sp in zip(dataset, split))):
    names.append(name)
    datasets.append(dataset)
    totals.append(total)

  totals = [
      total * weight for total, weight in zip(totals, dataset_weights)]

  # Normalize the weights such that they sum up to 1.
  weights = [x / sum(totals) for x in totals]

  logging.info(
      "NOTE: Total dataset mix size: %d\nContributions:\n%s", sum(totals),
      "\n".join(f"{ds}: {n} ({w * 100:.2g}%)"  # pylint: disable=bad-whitespace
                for ds, n, w in zip(names, totals, weights))
  )

  train_ds = tf.data.Dataset.sample_from_datasets(
      datasets, weights, stop_on_empty_dataset=True)

  train_ds = train_ds.batch(batch_size, drop_remainder=True)
  return train_ds.prefetch(prefetch), sum(totals)


def get_data(
    dataset: str | Sequence[str] | None, **kwargs
) -> tuple[tf.data.Dataset, int]:
  """Returns dataset with the number of train examples or evaluation steps.

  Args:
    dataset: Name of the dataset (or a list).
    **kwargs: Additional arguments passed to dataset loading functions.
  Returns:
    Prefetched tf.data.Dataset object of the training or evaluation split.
    Number of train examples in a single epoch if train=True, or the number of
      evaluation steps with the given batch size if train=False.
  """
  if dataset is None or isinstance(dataset, str):
    return _get_single_ds(dataset, **kwargs)

  return _get_mixture_ds(dataset, **kwargs)


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


@functools.cache
def get_builder(dataset: str, data_dir: str | None) -> tfds.core.DatasetBuilder:
  try:
    return tfds.builder(dataset, data_dir=data_dir or None, try_gcs=True)
  except tfds.core.registered.DatasetNotFoundError as exc:
    raise ValueError(f"Dataset {dataset} not found in {data_dir}") from exc


def get_dataset_tfds(
    dataset: str,
    split: str = "train",
    shuffle_files: bool = True,
    data_dir: str | None = None,
    skip_decode: Sequence[str] = ("image",),
    keep_keys: Sequence[str] | None = None,
    download_and_prepare: bool = False,
    interleave_cycle_length: int | None = None,
) -> tuple[tf.data.Dataset, tfds.core.DatasetBuilder]:
  """Returns TFDS dataset split."""
  builder = get_builder(dataset, data_dir)
  if download_and_prepare:
    builder.download_and_prepare()
  split = tfds.even_splits(split, jax.process_count())[jax.process_index()]
  decoders = {
      f: tfds.decode.SkipDecoding()
      for f in skip_decode if f in builder.info.features
  }
  if keep_keys:
    decoders = tfds.decode.PartialDecoding(
        {f: True for f in keep_keys if f in builder.info.features},
        {f: decoder for f, decoder in decoders.items() if f in keep_keys},
    )
  read_config = tfds.ReadConfig(
      skip_prefetch=True,  # We prefetch after pipeline.
      try_autocache=False,  # We control this, esp. for few-shot.
      add_tfds_id=True,
  )
  if interleave_cycle_length:
    # NOTE: Value "None" is different from the default value "MISSING". So we
    # only overwrite it if it is explicitly set.
    read_config.interleave_cycle_length = interleave_cycle_length
  # Each host is responsible for a fixed subset of data
  return builder.as_dataset(
      split=split,
      shuffle_files=shuffle_files,
      read_config=read_config,
      decoders=decoders), builder


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
