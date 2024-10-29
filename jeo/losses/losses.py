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

"""Losses.

Organizes losses coarsely into:
  - Classification losses.
  - Regression losses.
  - Probabilistic losses.
  - Super-resolution losses.
  - Self-supervision losses.
  - Other (non-characterized) losses.
  - Loss combiners.
  - Utils and wiring.
"""

import functools
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
from jeo import train_utils
from jeo.tools import bv_utils
import optax


################################################################################
#                            Classification losses                             #
################################################################################
def generalized_softmax_xent(
    *,
    logits,
    labels,
    reduction=True,
    weights=None,
    label_smoothing=0.0,
    normalize=True,
):
  """Compute generlized (weighted, normalized, smoothed) cross entropy.

  Extended from big_vision.utils.weighted_softmax_xent for per-pixel losses.

  Args:
   logits: [B, ..., num_classes] float array.
   labels: One-hot encoded labels (B, ..., N) or int labels (B, ...).
   reduction: reduce across batch dim.
   weights: None or array of shape [B, ...].
   label_smoothing: label smoothing constant, used to determine the on and off
     values.
   normalize: normalize each batch item loss by the number of elements (pixels,
     tokens) in it. Otherwise, each pixel/token losses are added together.

  Returns:
    Scalar loss () or per_batch_item loss (B,).
  """
  targets = get_soft_targets(labels, logits, label_smoothing)
  loss = -jnp.sum(targets * jax.nn.log_softmax(logits), axis=-1)
  if weights is not None:
    loss = loss * weights

  non_batch_axes = range(1, targets.ndim - 1)
  loss = loss.sum(axis=non_batch_axes)
  if normalize:
    if weights is None:
      normalizing_factor = jnp.prod(jnp.array(targets.shape[1:-1]))
    else:
      normalizing_factor = jnp.clip(weights.sum(axis=non_batch_axes), 2e-38)
    loss = loss / normalizing_factor

  return loss.mean() if reduction else loss


def sigmoid_xent(
    *,
    logits,
    labels,
    reduction=True,
    weights=None,
    normalize=True,
    label_smoothing=0.0,
):
  """Computes cross-entropy over binary/multilabel/regression tasks."""
  labels = get_soft_targets(labels, logits, label_smoothing)
  log_p = jax.nn.log_sigmoid(logits)
  log_not_p = jax.nn.log_sigmoid(-logits)
  loss = -(labels * log_p + (1 - labels) * log_not_p)

  non_batch_axes = range(1, labels.ndim)
  if weights is None and normalize:
    loss = loss.mean(axis=non_batch_axes)  # Over non-batch axes.
  else:
    if weights is not None:
      if weights.ndim == loss.ndim - 1:
        weights = weights[..., None]
      loss = loss * weights
    loss = loss.sum(axis=non_batch_axes)
    if normalize:
      if weights is None:
        norm_factor = jnp.prod(jnp.array(labels.shape[1:]))
      else:
        norm_factor = jnp.clip(weights.sum(axis=non_batch_axes), 2e-38)
      loss = loss / norm_factor
  return jnp.mean(loss) if reduction else loss


def generalized_dice(
    logits,
    labels,
    *,
    reduction: bool = True,
    weights: jnp.ndarray | None = None,
    label_smoothing: float = 0.0,
    norm_type: str = "tanimoto",
    eps: float = 1e-3,
    beta: float | list[float] = 0.7,
    dual: bool = False,
    class_weights: list[float] | None = None,
    activation: str = "softmax",
) -> jnp.ndarray:
  """Compute generlized dice loss.

  Args:
   logits: [B, ..., C] float array with per class logits.
   labels: One-hot encoded labels (B, ..., N) or int labels (B, ...).
   reduction: reduce across batch dim.
   weights: None or array of shape [B, ...].
   label_smoothing: label smoothing constant, used to determine the on and off
     values.
   norm_type: The type of denominator to use to compute the dice coefficient.
   eps: The larger the smoothing coefficient the less sensitive is the dice loss
     to single pixel value changes. The default value is set low and only to
     avoid division by zero.
   beta: The coefficient that controls the weight given to false positive and
     false negatives. Higher values put more weight on false negatives leading
     to higher recall. Used only for `norm_type='tversky'`.
   dual: If True, the loss is computed as the average between the true loss and
     the loss on the dual labels. Literature claims that improves convergence
     speed.
   class_weights: Weights for each class. If None, all classes are weighted
     equally.
   activation: The activation function to use on the logits. Softmax and sigmoid
     are supported.

  Returns:
    Scalar loss () or per_batch_item loss (B,).
  """

  targets = get_soft_targets(labels, logits, label_smoothing)

  if weights is None:
    weights = jnp.ones(targets.shape[:-1])
  # Expand last dimension to facilitate computation of denominator.
  weights = jnp.expand_dims(weights, axis=-1)
  assert weights.ndim == logits.ndim

  if class_weights is None:
    class_weights = [1.0 / logits.shape[-1]] * logits.shape[-1]
  assert len(class_weights := jnp.array(class_weights)) == logits.shape[-1]
  # Ensure that the class weights sum to 1.
  class_weights = class_weights / class_weights.sum()

  if isinstance(beta, float):
    beta = [beta] * logits.shape[-1]
  assert len(beta := jnp.array(beta)) == logits.shape[-1]

  # [B, ..., C]
  probs = getattr(jax.nn, activation)(logits)
  # shape: [B,]
  coeff = _generalized_dice_coefficient(
      probs, targets, weights, class_weights, norm_type, eps, beta
  )
  if dual:
    # Compute the dual coefficient by inverting labels and predictions.
    dual_coeff = _generalized_dice_coefficient(
        1 - probs, 1 - targets, weights, class_weights, norm_type, eps, beta
    )
    coeff = 0.5 * coeff + 0.5 * dual_coeff

  return (1 - coeff).mean() if reduction else (1 - coeff)


def _generalized_dice_coefficient(
    probs: jnp.ndarray,
    labels: jnp.ndarray,
    weights: jnp.ndarray,
    class_weights: jnp.ndarray,
    norm_type: str,
    eps: float,
    beta: jnp.ndarray,
) -> jnp.ndarray:
  """Compute generalized dice coefficient."""
  kwargs = dict(where=weights > 0, axis=range(1, probs.ndim - 1))

  # [B, C]
  intersection = 2 * jnp.sum(probs * labels, **kwargs)

  if norm_type == "standard":
    # Union [B, C].
    norm = jnp.sum(probs + labels, **kwargs)
  elif norm_type == "squared":
    # Union with squared terms [B, C].
    norm = jnp.sum(probs**2 + labels**2, **kwargs)
  elif norm_type == "tanimoto":
    # Union with squared terms minus interection [B, C].
    norm = 2 * jnp.sum(probs**2 + labels**2 - probs * labels, **kwargs)
  elif norm_type == "tversky":
    # Union with squared terms that penalises false negative proportionally to
    # `beta` and false positive to `1-beta` [B, C].
    norm = 2 * jnp.sum((1 - beta) * probs**2 + beta * labels**2, **kwargs)
  else:
    raise ValueError(f"Unknown norm type: {norm_type}")

  # Reduce over classes [B, C] -> [B, ].
  coeff = (intersection + eps) / (norm + eps)
  return (class_weights * coeff).sum(-1)


def softmax_focal_loss(
    logits,
    labels,
    *,
    reduction=True,
    weights=None,
    alpha=0.25,
    gamma=2.0,
    label_smoothing=0.0,
    normalize=True,
):
  """Computes softmax focal loss for multiclass problems.

  Focal loss: https://arxiv.org/pdf/1708.02002.pdf
  FL = -alpha * log(p_t) * (1-p_t)^gamma
  p_t = p if y==1 else 1-p
  p = softmax(logits) or sigmoid(logits)


  Args:
    logits: Logits (B, ..., N).
    labels: One-hot encoded labels (B, ..., N) or int labels (B, ...).
    reduction: Whether to reduce across samples or return per_example_loss.
    weights: Whether additional weights/masks should be applied.
    alpha: Focal alpha scaling parameter (non-focal: 1.).
    gamma: Focal loss focusing paramter (non-focal: 0.).
    label_smoothing: label smoothing constant, used to determine the on and off
      values.
    normalize: normalize each batch item loss by the number of elements (pixels,
      tokens) in it. Otherwise, each pixel/token losses are added together.

  Returns:
    Single scalar loss or per_example_loss.
  """
  targets = get_soft_targets(labels, logits, label_smoothing)
  log_prob = jax.nn.log_softmax(logits, axis=-1)
  log_p = jnp.sum(jnp.array(alpha) * log_prob * targets, axis=-1)
  loss = -log_p * (1.0 - jnp.exp(log_p)) ** gamma
  # equivalent:
  # log_pt = jax.nn.log_softmax(logits, axis=-1) * targets
  # loss = -alpha * log_pt * (1-jax.nn.softmax(logits)*targets)**gamma

  if weights is not None:
    loss = loss * weights
  loss = loss.sum(axis=range(1, loss.ndim))

  if normalize and logits.ndim > 2:
    if weights is None:
      norm = jnp.prod(jnp.array(targets.shape[1:-1]))
    else:
      norm = jnp.clip(weights.sum(axis=range(1, weights.ndim)), 2e-38)
    loss = loss / norm

  return loss.mean() if reduction else loss


def onehot(labels, num_classes, on_value=1.0, off_value=0.0):
  x = labels[..., None] == jnp.arange(num_classes)[None]
  x = jax.lax.select(
      x, jnp.full(x.shape, on_value), jnp.full(x.shape, off_value)
  )
  return x.astype(jnp.float32)


def get_soft_targets(
    labels: jnp.ndarray, logits: jnp.ndarray, label_smoothing: float
) -> jnp.ndarray:
  """Compute soft targets."""
  if labels.shape[: logits.ndim - 1] != logits.shape[:-1]:
    raise ValueError(
        f"Received wrong shapes. labels:{labels.shape} logits:{logits.shape}."
    )

  if labels.ndim == logits.ndim and label_smoothing == 0:
    return labels

  # Confidence of labels is decreased from `1` to `1-smoothing`.
  confidence = 1.0 - label_smoothing
  # The rest of the probability mass is spread evenly among the other classes.
  num_classes = logits.shape[-1]
  low_confidence = label_smoothing / (num_classes - 1)

  if labels.ndim == logits.ndim:
    return jnp.where(labels, confidence, low_confidence)
  return onehot(labels, num_classes, confidence, low_confidence)


################################################################################
#                              Regression losses                               #
################################################################################
def l2_loss(logits, labels, *, reduction=True, weights=None, as_mse=True):
  """Computes L2 loss."""
  if logits.ndim == labels.ndim + 1:
    logits = jnp.squeeze(logits, -1)
  loss = optax.l2_loss(logits, labels)
  if as_mse:  # As mean squared error (optax above multiplies result with 0.5).
    loss *= 2
  non_batch_axes = tuple(range(1, labels.ndim))
  if weights is None:
    loss = loss.mean(axis=non_batch_axes)
  else:
    norm_factor = jnp.clip(weights.sum(axis=non_batch_axes), 2e-38)
    loss = (loss * weights).sum(axis=non_batch_axes) / norm_factor
  return loss.mean() if reduction else loss


################################################################################
#                             Probabilistic losses                             #
################################################################################
def gnll_loss(
    logits, labels, log_variances, *, reduction=True, weights=None, eps=1e-8
):
  """Computes Gaussian negative log likelihood loss."""

  variances = jnp.exp(log_variances) + eps
  # As optax multiplies result with 0.5, optax.l2_loss * 2.
  loss = 0.5 * (optax.l2_loss(logits, labels) * 2 / variances + log_variances)

  non_batch_axes = range(1, labels.ndim)
  if weights is None:
    loss = loss.mean(axis=non_batch_axes)
  else:
    norm_factor = jnp.clip(weights.sum(axis=non_batch_axes), 2e-38)
    loss = (loss * weights).sum(axis=non_batch_axes) / norm_factor
  return loss.mean() if reduction else loss


def kld_loss(
    *, logits, labels, reduction=True, gamma: float = 1.0, activation="sigmoid"
):  # {sigmoid, softmax, none}
  """Computes Kullback-Leibler divergence (relative entropy) loss."""
  # https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
  # loss = y_true * log(y_true / y_pred)
  if logits.ndim == labels.ndim + 1:
    logits = jnp.squeeze(logits, -1)
  log_predictions = (
      logits
      if activation == "none"
      else getattr(jax.nn, f"log_{activation}")(logits)
  )
  # Based on optax.kl_divergence(), but avoiding loss.mean(-1).
  loss = labels * (jnp.where(labels == 0, 0, jnp.log(labels)) - log_predictions)
  # If activation is softmax, then we should sum over the class probabilities.
  if activation == "softmax":
    loss = jnp.sum(loss, axis=-1)
  if gamma != 1.0:
    # mmeka: We add a small constant to the loss before exponentation to guard
    # against the gradient becoming Inf when `kld` is 0 e.g. without this
    # constant if `kld` is 0 and `kld_gamma` is 0.25 then the gradient
    # would be: 0.25 * pow(0, -0.75) = +Inf.
    loss = (loss + 1e-7) ** gamma
  if loss.ndim > 1:  # Over non-batch axes.
    loss = loss.mean(axis=range(1, loss.ndim))
  return loss.mean() if reduction else loss


################################################################################
#                           Super-resolution losses                            #
################################################################################
def supres_losses(sr, sr_sigma, hr, hr_mask=None, base_cpsnr=None,
                  border=3, use_l1=True, with_sigma=False,
                  brightness_bias=True, reduce_batch=True, reduce_spatial=True):
  """Computes loss and metrics for superresolution task.

  - It can take misalignement into account and compute best metrics within small
    shift regions (controlled by border).
  - By default, it returns L1 loss. If l1=False, it returns L2 loss.
  - It performs brightness bias correction by default.

  Args:
    sr: Predicted super-resolved images. (B,H,W) or (B,H,W,C).
    sr_sigma: Predicted super-resolved uncertainty/sigma. (B,H,W,[C]) or None.
    hr: High-res target images. (B,H,W,[C]).
    hr_mask: High-res target mask (0 for corrupted hr pixels). (B,H,W) or None.
    base_cpsnr: Baseline cPSNR. (B,) or None.
    border: radius to search for best allignement between SR & HR.
    use_l1: Returns L1 loss if True, or L2 otherwise.
    with_sigma: Whether to use sigma in loss computation (only L1 for now).
    brightness_bias: Whether to correct for brightness bias.
    reduce_batch: Whether to reduce batch dim via mean.
    reduce_spatial: Whether to reduce spatial dims via mean.
  Returns:
    L1 or L2 loss. (B,) or ().
    aux: dict of metrics, each (B,) or ().
  """
  assert reduce_spatial, "per_pixel loss/metrics is not supported yet."

  batch_size, img_size = sr.shape[:2]
  assert hr.shape == sr.shape, f"hr.shape: {hr.shape}"
  if hr_mask is None:
    hr_mask = jnp.ones_like(hr)
  if base_cpsnr is not None:
    assert base_cpsnr.shape == (batch_size,)

  # L1 loss: sum(abs(hr - sr)), where sr brightness is corrected to
  # be the same as hr.
  # Computing with (2*border+1) shifts (7*7 in total) and returning lowest loss.
  # And looks like it completely discards the borders in the predictions.
  max_shift = 2 * border
  crop_size = img_size - max_shift
  nonbatch_axes = tuple(range(1, sr.ndim))

  sr = sr[:, border:img_size - border, border:img_size - border]
  if with_sigma:
    sr_sigma = sr_sigma[:, border:img_size - border, border:img_size - border]

  l1, mse, cpsnr = [], [], []
  for i in range(max_shift + 1):
    for j in range(max_shift + 1):
      hr_ij = hr[:, i:i+crop_size, j:j+crop_size]
      hr_mask_ij = hr_mask[:, i:i+crop_size, j:j+crop_size]
      num_pixels_masked = jnp.sum(hr_mask_ij, axis=nonbatch_axes)

      # Mask out HR corrupted pixels.
      sr_ij = sr * hr_mask_ij
      hr_ij = hr_ij * hr_mask_ij

      # Brightness bias correction.
      if brightness_bias:
        b = jnp.sum(hr_ij - sr_ij, axis=nonbatch_axes)/num_pixels_masked
        b = jnp.expand_dims(b, axis=nonbatch_axes)
        sr_ij = (sr_ij + b) * hr_mask_ij

      if with_sigma:
        sr_sigma_ij = sr_sigma * hr_mask_ij
        if brightness_bias:
          sr_sigma_ij += b
        sr_sigma_ij = sr_sigma_ij * hr_mask_ij
        l1_loss = compute_l1_loss(sr_ij, hr_ij, num_pixels_masked, sr_sigma_ij)
      else:
        l1_loss = compute_l1_loss(sr_ij, hr_ij, num_pixels_masked)
      l1.append(l1_loss)
      mse.append(jnp.sum((sr_ij - hr_ij)**2, axis=nonbatch_axes)
                 / num_pixels_masked)
      cpsnr.append(-10.0 * jnp.log10(mse[-1]))

  # Stack across different shifts and take best values.
  l1 = jnp.stack(l1, axis=0).min(axis=0)  # (B,)
  cpsnr = jnp.stack(cpsnr, axis=0).max(axis=0)  # (B,)
  mse = jnp.stack(mse, axis=0).min(axis=0)  # (B,)
  # Cannot use name "mse", as it will be confused with standard MSE metric.
  outputs = (l1 if use_l1 else mse), {"cpsnr": cpsnr, "sr_mse": mse, "l1": l1}
  if base_cpsnr is not None:
    outputs[1]["ncpsnr"] = base_cpsnr / jnp.maximum(cpsnr, 1e-9)
    outputs[1]["base_cpsnr"] = base_cpsnr
  if reduce_batch:
    return jax.tree.map(jnp.mean, outputs)  # ()
  return outputs


def ssim(
    x: jnp.ndarray,  # ([B], H, W, C)
    y: jnp.ndarray,  # ([B], H, W, C)
    c1: float = 0.01**2,
    c2: float = 0.03**2) -> jnp.ndarray:  # ([B], H-2, W-2, C)
  """Computes structual-similarity loss on images x and y.

  Based on haiku code in:
  (internal link)/deepmind/research/robots/representations/jax/projects/sfm/losses/photometric.py?l=67

  Args:
    x: ([B], H, W, C) image with (possibly) leading batch dimension.
    y: ([B], H, W, C) image of same shape as `x`.
    c1: Constant for SSIM numerator term.
    c2: Constant for SSIM denominator term.

  Returns:
    Array of shape ([B], H-2, W-2, C) containing the SSIM metric between x and y
  """
  mu_x = nn.avg_pool(x, (3, 3, 1))
  mu_y = nn.avg_pool(y, (3, 3, 1))
  sigma_x = nn.avg_pool(x**2, (3, 3, 1)) - mu_x**2
  sigma_y = nn.avg_pool(y**2, (3, 3, 1)) - mu_y**2
  sigma_xy = nn.avg_pool(x * y, (3, 3, 1)) - mu_x * mu_y
  ssim_n = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
  ssim_d = (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
  ssim_full = ssim_n / ssim_d

  return jnp.clip((1 - ssim_full) / 2, 0, 1)


def compute_supres_loss(pred_hr, pred_uncert, hr, hr_mask, **kwargs):
  # pred_hr, pred_uncert = model_outputs
  hr = jnp.squeeze(hr, 1)  # Remove single-elem T axis ==> (B,H,W).
  if hr_mask is None:
    hr_mask = jnp.ones_like(hr)
  else:
    hr_mask = jnp.squeeze(hr_mask, 1)  # Remove single-elem T axis ==> (B,H,W).
  return supres_losses(pred_hr, pred_uncert, hr, hr_mask, **kwargs)


def compute_l1_loss(pred, true, n_pixels, sigma=None):
  # pred, true, sigma: (B,H,W,[C])
  # n_pixels: number of masked pixels (other values are expected to be 0 in img)
  if sigma is None:
    per_pixel_loss = jnp.abs(true-pred)
  else:
    per_pixel_loss = sigma + jnp.abs(true-pred) * jnp.exp(-sigma)  # pylint: disable=invalid-unary-operand-type
  return jnp.sum(per_pixel_loss, axis=range(1, per_pixel_loss.ndim)) / n_pixels


def compute_normalized_cpsnr(pred_cpsnr, base_cpsnr):
  return base_cpsnr / pred_cpsnr


################################################################################
#                            Self-supervised losses                            #
################################################################################
@jax.default_matmul_precision("float32")
def nt_xent(logits, reduction=True, temperature: float = 1.,
            axis_name="batch"):
  """Computes Normalized Temperature-scaled Cross Entropy loss.

  It follows tf2 implementation in third_party/py/simclr/objective.py.

  Args:
    logits: Logits views of shape (batch_size, num_views, emb_dim). Expected to
      be already normalized.
    reduction: Whether the losses should be reduced to a scalar.
    temperature: The temperature for the exponential part.
    axis_name: If axis_name is given, it is assumed the function is called
      within pmap. Set to None otherwise.

  Returns:
    Scalar or per-example losses.
  """
  b, v, _ = logits.shape  # (batch_size_per_device, num_views, emb_dim)
  assert v == 2, f"Not fully adapted for num_views > 2 ({logits.shape})."
  if axis_name:
    # To compare local examples with negatives across the global batch, we
    # need to gather logits from all devices.
    logits = jnp.concatenate(jax.lax.all_gather(logits, axis_name), axis=0)
    # To prepare labels and same-view masks, we need to know the indices
    # of the current local batch within the global batch.
    axis_index = jax.lax.axis_index(axis_name)
    local_positives = jnp.arange(b) + (jnp.float32(axis_index) * b)
    global_batch_size = b * jax.device_count()
  else:
    local_positives = jnp.arange(b)
    global_batch_size = b
  # Positives are pair-views, while all others are negatives.
  labels = jax.nn.one_hot(local_positives, global_batch_size * v)
  # To mask out same-view elements (sim is always == 1 for them).
  masks = jax.nn.one_hot(local_positives, global_batch_size) * 1e10
  local_logits = logits[local_positives.astype(int)]
  same_sim = [jnp.matmul(local_logits[:, i], jnp.transpose(logits[:, i]))
              / temperature - masks for i in range(v)]  # [(b,b), ...]
  cross_sim = [[jnp.matmul(local_logits[:, i], jnp.transpose(logits[:, j]))
                / temperature for j in range(v) if i != j]
               for i in range(v)]
  losses = [optax.softmax_cross_entropy(jnp.concatenate(
      (cross_sim[i][0], same_sim[i]), axis=1), labels) for i in range(v)]
  loss = jnp.add(*losses)
  return jnp.mean(loss) if reduction else loss


################################################################################
#                                 Other losses                                 #
################################################################################


################################################################################
#                                Loss combiners                                #
################################################################################
def weighted_losses_sum(
    logits,
    labels,
    *,
    reduction: bool = True,
    weights: jnp.ndarray | None = None,
    losses_config: dict[str, dict[str, Any]],
    normalize_loss_weights: bool = True,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
  """Combines multiple losses with a weighted sum.

  Args:
    logits: [B, ..., C] float array with per class logits.
    labels: One-hot encoded labels (B, ..., N) or int labels (B, ...).
    reduction: reduce across batch dim.
    weights: None or array of shape [B, ...].
    losses_config: A list of tuples of (loss_name, loss_weight, loss_kwargs).
    normalize_loss_weights: If True, the loss weights are normalized to sum to
      1.0.

  Returns:
    Scalar loss () or per_batch_item loss (B,).
  """
  combined_loss = 0.0 if reduction else jnp.zeros(logits.shape[0])
  norm_weights = 0.0
  losses_values = {}
  for name, loss_config in losses_config.items():
    if "loss" not in loss_config:
      raise ValueError(f"Loss missing  from {name}. Received: {loss_config}.")
    loss_fn = get_loss_fn(loss_config["loss"], **loss_config.get("loss_kw", {}))
    losses_values[name] = loss_fn(
        logits=logits, labels=labels, reduction=reduction, weights=weights
    )
    norm_weights += (loss_weight := loss_config.get("weight", 1.0))
    combined_loss += loss_weight * losses_values[name]
  if normalize_loss_weights:
    combined_loss /= norm_weights
  return combined_loss, losses_values


################################################################################
#                                Utils & wiring                                #
################################################################################
CLASSIFICATION_LOSS_FNS = {
    "sigmoid_xent": sigmoid_xent,  # And for regression.
    "softmax_xent": bv_utils.softmax_xent,
    "generalized_softmax_xent": generalized_softmax_xent,
    "softmax_focal_loss": softmax_focal_loss,
}
REGRESSION_LOSS_FNS = {
    "l2_loss": l2_loss,
    "kld_loss": kld_loss,
}
SEMANTIC_SEGMENTATION_LOSS_FNS = {
    "generalized_dice": generalized_dice,
}
REGRESSION_SEGMENTATION_FNS = {  # Only for reference and testing.
    "l2_loss": l2_loss,
    "kld_loss": kld_loss,
    "sigmoid_xent": sigmoid_xent,
}
SELFSUP_LOSS_FNS = {
    "nt_xent": nt_xent,
}
LOSS_FNS = {
    **CLASSIFICATION_LOSS_FNS,
    **REGRESSION_LOSS_FNS,
    **SELFSUP_LOSS_FNS,
    **SEMANTIC_SEGMENTATION_LOSS_FNS,
    "supres": supres_losses,  # Doesn't follow current conventions.
    "weighted_losses_sum": weighted_losses_sum,
    "gnll_loss": gnll_loss,
    "unused": lambda *_, **__: None,
}


def get_loss_fn(loss_name, **kwargs):
  if loss_name in LOSS_FNS:
    loss_fn = LOSS_FNS[loss_name]
  else:  # given some.module.loss_fn_name
    *path, fn_name = loss_name.split(".")
    module = train_utils.import_module(".".join(path), "")
    loss_fn = getattr(module, fn_name)
  if kwargs:
    loss_fn = functools.partial(loss_fn, **kwargs)
  return loss_fn
