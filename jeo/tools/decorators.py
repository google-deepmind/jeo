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

"""Decorators for deprecation and other use cases."""
import functools
from typing import Callable, Union

from absl import logging


def deprecated(msg_or_fn: Union[str, Callable]) -> Callable:  # pylint: disable=g-bare-generic
  """Decorates a function or method or class as depricated."""

  if isinstance(msg_or_fn, str):
    # Reason message provided.
    def depricated_wrapper(fn):
      @functools.wraps(fn)
      def new_fn(*args, **kwargs):
        logging.warning("Deprecated `%s` (from %s): %s",
                        fn.__qualname__, fn.__module__, msg_or_fn)
        return fn(*args, **kwargs)
      return new_fn
    return depricated_wrapper

  # No message provided.
  fn: Callable = msg_or_fn  # pylint: disable=g-bare-generic
  @functools.wraps(fn)
  def new_fn(*args, **kwargs):
    logging.warning("Deprecated `%s` (from %s)",
                    fn.__qualname__, fn.__module__)
    return fn(*args, **kwargs)
  return new_fn
