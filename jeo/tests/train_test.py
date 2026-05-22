# Copyright 2026 DeepMind Technologies Limited.
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

"""Tests for train."""

from unittest import mock

from absl.testing import absltest
import jax
from jeo import input_pipeline
from jeo import train as train_lib
from jeo.configs.tests import tiny_bit
import numpy as np
import tensorflow as tf


def mocked_get_data(batch_size, train=True, val_steps=2, **kwargs):
  del kwargs
  ds = tf.data.Dataset.from_generator(
      lambda: ({  # pylint: disable=g-long-lambda,g-complex-comprehension
          "image": np.ones(shape=(32, 32, 3), dtype=np.float32),
          "labels": np.ones((10,)),  # i % 10,
          "_mask": tf.constant(1),
      } for _ in range(100)),
      output_types={"image": tf.float32, "labels": tf.int64, "_mask": tf.int64},
      output_shapes={"image": (32, 32, 3), "labels": (10), "_mask": ()}
  )
  ds = ds.batch(batch_size).repeat()
  if train:
    return ds, 100
  return ds, val_steps or 2


def mocked_get_num_examples(dataset, split, data_dir=None, **kwargs):
  del dataset, data_dir, kwargs
  if not isinstance(split, str) and hasattr(split, "split"):
    split = split.split
  return ({"train": 100_000, "test": 2_000})[split]


class TrainTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.workdir = self.create_tempdir().full_path
    # Make sure that TF does not allocate GPU memory:
    tf.config.experimental.set_visible_devices([], "GPU")

  def test_trainer_e2e_no_eval(self):
    config = tiny_bit.get_config("test=True")
    config.batch_size = 2 * jax.device_count()
    del config.evals["val"]
    train_lib.FLAGS.cleanup = False
    train_lib.FLAGS.config = config
    train_lib.FLAGS.workdir = self.workdir
    if hasattr(train_lib.FLAGS, "xm_runlocal"):
      train_lib.FLAGS.xm_runlocal = True
    with mock.patch.object(input_pipeline, "get_data", mocked_get_data):
      with mock.patch.object(
          input_pipeline, "get_num_examples", mocked_get_num_examples
      ):
        train_lib.main(None)

  def test_trainer_e2e_with_eval(self):
    config = tiny_bit.get_config("test=True")
    config.batch_size = 2 * jax.device_count()
    train_lib.FLAGS.cleanup = False
    train_lib.FLAGS.config = config
    train_lib.FLAGS.workdir = self.workdir
    if hasattr(train_lib.FLAGS, "xm_runlocal"):
      train_lib.FLAGS.xm_runlocal = True
    with mock.patch.object(input_pipeline, "get_data", mocked_get_data):
      with mock.patch.object(input_pipeline, "get_num_examples",
                             mocked_get_num_examples):
        train_lib.main(None)


if __name__ == "__main__":
  absltest.main()
