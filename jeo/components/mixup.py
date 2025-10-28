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

"""Mixup augmentation for per-pixel segmentation."""
from collections.abc import Sequence
from typing import Any

import jax
import jax.numpy as jnp


def get_mixup(
    rng: jax.Array,
    batch: dict[str, Any],
    images_key: str = "image",
    labels_key: str = "labels",
    mix_type: str = "cut-mix",
    valid_masks_key: str | None = None,
    p: float = 1.0,
    num_iter: int = 1,
) -> tuple[jax.Array, dict[str, Any]]:
  """Perform mixup augmentation for per-pixel segmentation.

  Args:
    rng: A JAX PRNGKey.
    batch: A batch of data.
    images_key: The key of images in the batch (can be tuple/list of keys
        for multiple satellite modalities.).
    labels_key: The key of labels in the batch.
    mix_type: The type of mixup.
    valid_masks_key: The key of valid masks of labels in the batch.
    p: The beta distribution parameter.
    num_iter: The number of iterations to perform cut-mix.

  Returns:
    rng: A JAX PRNGKey.
    batch: A batch of data after mixup augmentation.
  """
  # Inserting a patch of one image to another image.
  if mix_type == "cut-mix":
    # Support mix-up for multiple input modalities.
    if not isinstance(images_key, (tuple, list)):
      images_key = (images_key,)
    for _ in range(num_iter):
      rng, batch = _cutmix(
          rng,
          batch,
          images_keys=images_key,
          labels_key=labels_key,
          valid_masks_key=valid_masks_key,
          p=p,
      )
    return rng, batch
  elif mix_type == "mix-up":
    raise NotImplementedError("Not implemented yet")
  else:
    raise ValueError(f"Unknown mix_type: {mix_type}")


def _cutmix(
    rng: jax.Array,
    batch: dict[str, Any],
    images_keys: Sequence[str],
    labels_key: str,
    valid_masks_key: str | None,
    p: float,
) -> tuple[jax.Array, dict[str, Any]]:
  """Perform Cut-Mix augmentation for per-pixel segmentation.

  Test colab for minimal usage and simple visualization:
    [(internal link)?usp=sharing]

  Args:
    rng: A JAX PRNGKey.
    batch: A batch of data.
    images_keys: The keys of images in the batch (can be tuple/list of keys
        for multiple satellite modalities.).
    labels_key: The key of labels in the batch.
    valid_masks_key: The key of valid masks of labels in the batch.
      The default key value is "labels_mask", where labels != -1.
    p: The beta distribution parameter.

  Returns:
    rng: A JAX PRNGKey.
    batch: A batch of data after mixup augmentation.
  """
  labels = batch[labels_key]
  if labels.ndim == 3:  # [B, H, W]
    labels = jnp.expand_dims(labels, axis=3)
  batch_size, height, width, _ = labels.shape  # [B, H, W, C]
  # Get labels permutations.
  idx = jax.random.permutation(rng, batch_size)
  labels_b = labels[idx]  # [B, H, W, C]
  if valid_masks_key is None:
    valid_masks = jnp.ones_like(labels)
  else:
    if valid_masks_key in list(batch.keys()):
      valid_masks = batch[valid_masks_key]
    else:
      raise ValueError(f"Invalid valid_masks_key: {valid_masks_key}")
  valid_masks_b = valid_masks[idx]  # [B, H, W, C]
  # Generate the size of masking bounding box.
  box_rng, lam_rng, rng = jax.random.split(rng, num=3)
  lam = jax.random.beta(lam_rng, p, p, shape=())
  cut_rat = jnp.sqrt(1.0 - lam)
  cut_w = jnp.array(width * cut_rat, dtype=jnp.int32)
  cut_h = jnp.array(height * cut_rat, dtype=jnp.int32)

  masks = None
  # Apply Cut-Mix for labels (new labels = (1-mask)* labelA + mask*labelB).
  batch[labels_key], masks = _compose_two_images(
      labels, labels_b, valid_masks_b, box_rng, height, width, cut_h, cut_w,
      masks
  )

  # Update valid mask of labels.
  if valid_masks_key is not None:
    batch[valid_masks_key], _ = _compose_two_images(
        valid_masks,
        valid_masks_b,
        valid_masks_b,
        box_rng,
        height,
        width,
        cut_h,
        cut_w,
        masks
    )
  # Apply Cut-Mix for images for all modalities.
  # Assume all modalities are already resized to same size.
  for k in images_keys:
    images = batch[k]
    images_b = batch[k][idx]
    batch[k], _ = _compose_two_images(
        images, images_b, valid_masks_b, box_rng, height, width, cut_h, cut_w,
        masks
    )
  return rng, batch


def _compose_two_images(
    images: jnp.ndarray,
    images_b: jnp.ndarray,
    valid_masks_b: jnp.ndarray,
    box_rng: jax.Array,
    height: int,
    width: int,
    cut_h: jax.Array,
    cut_w: jax.Array,
    masks: jnp.ndarray | None,
):
  """Inserting the second minibatch into the first at the target locations."""

  def _single_compose_two_images(image, image_b, valid_mask_b, mask):
    # Get bounding box.
    if mask is None:
      bbox = _random_box(box_rng, height, width, cut_h, cut_w, valid_mask_b)
      mask = _window_mask(bbox, (height, width))
      # Expand dims for mask to handle [H, W, C] & [T, H, W, C] inputs.
      mask = jnp.expand_dims(mask, axis=range(0, len(image.shape) - 3))
      valid_mask_b = jnp.expand_dims(
          valid_mask_b, axis=range(0, len(image.shape) - 3)
      )
      # Only use the valid pixels from image_b.
      mask = jnp.logical_and(mask, valid_mask_b)

    return image * (1.0 - mask) + image_b * mask, mask

  if images[0].shape[-3:-1] != (height, width):
    masks = jax.image.resize(masks, (len(images), images[0].shape[-3],
                                     images[0].shape[-2], 1),
                             method="nearest")

  return jax.vmap(_single_compose_two_images)(images, images_b, valid_masks_b,
                                              masks)


def _random_box(
    rng: jax.Array,
    height: int,
    width: int,
    cut_h: jax.Array,
    cut_w: jax.Array,
    valid_mask: jnp.ndarray,
) -> jnp.ndarray:
  """Sample a random box of shape [cut_h, cut_w]."""
  # Randomly select a center from valid pixels.
  center_hws = jnp.argwhere(valid_mask == 1, size=height*width)
  idx = jax.random.randint(
      rng, shape=(), minval=0, maxval=valid_mask.sum(), dtype=jnp.int32
  )
  center_h, center_w = center_hws[idx, 0], center_hws[idx, 1]

  bby1 = jnp.clip(center_h - cut_h // 2, 0, height)
  bbx1 = jnp.clip(center_w - cut_w // 2, 0, width)
  h = jnp.clip(center_h + cut_h // 2, 0, height) - bby1
  w = jnp.clip(center_w + cut_w // 2, 0, width) - bbx1
  return jnp.array([bby1, bbx1, h, w])


def _window_mask(
    destination_box: jax.Array, size: Sequence[int]
) -> jnp.ndarray:
  """Mask a part of the image."""
  bby1, bbx1, h, w = destination_box
  h_range = jnp.reshape(jnp.arange(size[0]), [size[0], 1, 1])
  w_range = jnp.reshape(jnp.arange(size[1]), [1, size[1], 1])
  return jnp.logical_and(
      jnp.logical_and(bby1 <= h_range,
                      h_range < bby1 + h),
      jnp.logical_and(bbx1 <= w_range,
                      w_range < bbx1 + w)).astype(jnp.float32)
