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

"""Visualizing satellite data.

Suggestions for visualizing various satellite data sources.
- Magrathean and the like already rescaled into uint8: Sometimes the contrast is
  low, and for better visualization it is good set max_perc to a bit below 100
  (to avoid a few highly reflective outliers), eg. norm(..., max_perc=99.8)
"""

from typing import Sequence

import matplotlib as mpl
import numpy as np


def norm(
    arr,
    *,
    per_channel=False,
    min_perc=0,
    max_perc=100,
    min_value=None,
    max_value=None,
    verbose=False,
    clip=True,
):
  """Normalizes numpy array per channel or globally.

  Note that the arg names and default values changed from inspect.normalize.

  Args:
    arr: np array with last dimension representing the channels.
    per_channel: whether normalization should be performed independently per
      channel or per entire array.
    min_perc: where to clip the low values [0..100].
    max_perc: top percentile, where to clip at the high values [0..100].
    min_value: min value for normalization. If not given, based on min_perc
      percentile value.
    max_value: max value for normalization. If not given, based on max_perc
      percentile value.
    verbose: whether to print the selected value threshold.
    clip: whether the values should be clipped to the given range.

  Returns:
    Rescaled array of same size to the value range [0, 1].
  """

  def _rescale(x):
    nonlocal min_value, max_value
    if min_value is None or max_value is None:
      if min_perc == 0 and max_perc == 100:
        thresholds = [x.min(), x.max()] * 2
      else:
        thresholds = np.percentile(x.flatten(), [min_perc, max_perc, 0, 100])
      min_value = thresholds[0] if min_value is None else min_value
      max_value = thresholds[1] if max_value is None else max_value
      if verbose:
        print(
            f"Rescaling from [{min_value}, {max_value}] to [0, 1] "
            f"(full: {thresholds[2:]})"
        )
    return normalize_with_minmax(x, min_value, max_value, clip=clip)

  if per_channel:
    return np.stack([_rescale(arr[..., i]) for i in range(arr.shape[-1])], -1)
  else:
    return _rescale(arr)


def normalize_with_minmax(arr, min_value, max_value, clip=True):
  """Normalizes array to [0, 1] with given min/max values."""
  if clip:
    arr = np.clip(arr, min_value, max_value)
  arr = (arr - min_value) / (max_value - min_value)
  return arr


def render_hillshade_dem(dem: np.ndarray) -> np.ndarray:
  """Visualize digital elevation model using hillshade visualization.

  Visualization code borrowed from the Candid sunroof team's colab at:
  http://google3/energy/suncatcher/dsm/candid/notebooks/sunroof_detailed_eval.ipynb;l=36;rcl=516631789

  Args:
    dem: Input dem to visualize, in units of meters. Assumed dimensions: [H, W].

  Returns:
    A [H, W, 3] uint8 rendering of the dsm using hillshade normalization.
  """

  ls = mpl.colors.LightSource(azdeg=315, altdeg=45)
  cmap = mpl.colormaps["gist_earth"]

  # DSMs might contain 0s or negative values due to reprojection. We don't
  # want these to skew the entire hillshade visualization.
  dsm_min = np.ma.masked_equal(dem, dem <= 0, copy=False).min()
  dsm = np.clip(dem - dsm_min, 0, dem.max() - dsm_min)

  hillshade = ls.shade(
      dsm.squeeze(), cmap=cmap, blend_mode="hsv", dx=0.25, dy=0.25
  )

  # Drop alpha channel and convert to uint8.
  return (255 * hillshade[:, :, :-1]).astype(np.uint8)


def overlay(
    back: np.ndarray,
    front: np.ndarray,
    mask: np.ndarray | None = None,
    alpha: float = 1.0,
) -> np.ndarray:
  """Overlays an image on top of another.

  Args:
    back: The image used as background.
    front: The image to put on top.
    mask: A boolean mask for the pixels where the overlay should happen.
    alpha: The mixing weight.

  Returns:
    The overlaid image.
  """

  # Ensure that back and front can be summed together.
  back, front = map(np.atleast_3d, (back, front))

  if mask is None:
    mask = np.ones(back.shape[:-1], np.bool_)
  mask = mask.squeeze()

  assert back.shape[:-1] == front.shape[:-1] == mask.shape
  assert mask.dtype == np.bool_
  assert back.dtype.kind == front.dtype.kind

  indices, image = np.where(mask), back.copy()
  image[indices] = image[indices] * (1 - alpha) + front[indices] * alpha
  return image.astype(back.dtype)


def panshap(rgb: np.ndarray, p: np.ndarray) -> np.ndarray:
  """Pansharpen rgb with pancromatic channel."""
  assert np.issubdtype(rgb.dtype, float) and 1 >= rgb.max() >= rgb.min() >= 0
  assert np.issubdtype(p.dtype, float) and 1 >= p.max() >= p.min() >= 0
  hs = mpl.colors.rgb_to_hsv(rgb)[..., :2]
  ps = mpl.colors.hsv_to_rgb(np.concatenate([hs, np.atleast_3d(p)], axis=-1))
  return ps


def labels_to_color(labels: np.ndarray, colors: Sequence[str]) -> np.ndarray:
  """Convert an image with integer labels to [0, 1] rgb values."""
  assert (
      len(colors) > labels.max()
  ), f"There are more labels {labels.max()} than colors {len(colors)}."
  return np.take(
      np.array([mpl.colors.to_rgb(c) for c in colors]), labels, axis=0
  ).squeeze()
