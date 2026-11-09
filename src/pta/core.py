"""PTA positive-set likelihood and permutation-reference calibration.

The loss is the paper's symmetric positive-set marginal likelihood. There is
no separate diagonal cross-entropy term or mixing coefficient. Inputs to
``pta_loss`` are already temperature-scaled student logits and teacher scores
in the same units as the calibrated threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING
import math
from numbers import Integral, Real
from typing import Optional, Sequence, Union

import torch


Scalar = Union[Real, torch.Tensor]


@dataclass(frozen=True)
class CalibrationResult:
    """A threshold computed from retained, equally weighted score entries.

    ``num_scores`` counts entries after fixed-point filtering, including all
    repetitions. It is not a count of independent observations. ``rank`` is
    one-based for ``finite_rank`` and absent for ``linear_quantile``.
    ``threshold`` is +inf when the finite rank exceeds the available entries;
    callers must preserve this outcome rather than substitute the maximum.
    """

    threshold: float
    alpha: float
    rule: str
    num_scores: int
    rank: Optional[int]
    empirical_exceedance: float
    sample_unit: str = "retained_permuted_scores"


def _scalar(value: Scalar, name: str, *, allow_infinity: bool = False) -> float:
    if isinstance(value, torch.Tensor):
        if value.ndim != 0 or value.is_complex() or value.dtype == torch.bool:
            raise ValueError(f"{name} must be a real scalar")
        result = float(value.detach().cpu().item())
    elif isinstance(value, Real) and not isinstance(value, bool):
        result = float(value)
    else:
        raise TypeError(f"{name} must be a real number or scalar tensor")
    if math.isnan(result) or (not allow_infinity and not math.isfinite(result)):
        qualifier = "non-NaN" if allow_infinity else "finite"
        raise ValueError(f"{name} must be {qualifier}")
    return result


def _matrix(value: torch.Tensor, name: str, *, square: bool) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != 2 or min(value.shape) == 0:
        raise ValueError(f"{name} must be a nonempty two-dimensional tensor")
    if square and value.shape[0] != value.shape[1]:
        raise ValueError(f"{name} must be square with aligned diagonal pairs")
    if not value.is_floating_point():
        raise TypeError(f"{name} must have a real floating-point dtype")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} contains non-finite values")


@torch.no_grad()
def positive_mask(teacher_scores: torch.Tensor, threshold: Scalar) -> torch.Tensor:
    """Return ``teacher_scores > threshold`` OR the paired diagonal.

    The comparison uses the exact values of the supplied scores, promoted to
    float64 to avoid rounding an interpolated scalar cutoff to BF16/FP16.
    This does not recompute scores at a different precision. Teacher and
    threshold tensors do not receive gradients. Infinite cutoffs are valid:
    +inf gives diagonal-only supervision and -inf selects the entire batch.
    """

    _matrix(teacher_scores, "teacher_scores", square=True)
    cutoff = _scalar(threshold, "threshold", allow_infinity=True)
    mask = teacher_scores.detach().to(dtype=torch.float64) > cutoff
    mask.fill_diagonal_(True)
    return mask


def _directional_set_loss(
    logits: torch.Tensor, mask: torch.Tensor, dim: int
) -> torch.Tensor:
    # Subtract a common per-anchor offset before the two logsumexp operations.
    # It cancels algebraically and avoids cancellation of large absolute logits.
    shifted = logits - logits.amax(dim=dim, keepdim=True).detach()
    all_mass = torch.logsumexp(shifted, dim=dim)
    positive_mass = torch.logsumexp(shifted.masked_fill(~mask, -torch.inf), dim=dim)
    return (all_mass - positive_mass).mean()


def pta_loss(
    student_logits: torch.Tensor,
    teacher_scores: torch.Tensor,
    threshold: Scalar,
) -> torch.Tensor:
    """Symmetric paper PTA loss: logsumexp(all) - logsumexp(positives).

    Both inputs must be finite, square, equally shaped matrices on the same
    device, with one observed image/text pair on each diagonal position. The
    teacher cutoff selects off-diagonal positives; the observed pairs are
    always retained. The text-to-image direction uses columns of this same
    mask, not a separately symmetrised teacher matrix.

    FP16/BF16 student logits are promoted to FP32 for the loss reduction;
    FP32/FP64 inputs keep their dtype. Gradients flow only through student
    logits. The function does not apply an additional temperature or scale.
    """

    _matrix(student_logits, "student_logits", square=True)
    if not isinstance(teacher_scores, torch.Tensor):
        raise TypeError("teacher_scores must be a torch.Tensor")
    if teacher_scores.shape != student_logits.shape:
        raise ValueError("student_logits and teacher_scores must have the same shape")
    if teacher_scores.device != student_logits.device:
        raise ValueError("student_logits and teacher_scores must be on the same device")
    mask = positive_mask(teacher_scores, threshold)
    logits = student_logits
    if logits.dtype in (torch.float16, torch.bfloat16):
        logits = logits.float()
    image_to_text = _directional_set_loss(logits, mask, dim=1)
    text_to_image = _directional_set_loss(logits, mask, dim=0)
    return 0.5 * (image_to_text + text_to_image)


def calibrate_from_scores(
    scores: Union[torch.Tensor, Sequence[float]],
    alpha: float,
    rule: str = "finite_rank",
) -> CalibrationResult:
    """Calibrate an upper-tail threshold from a 1-D retained score vector.

    ``finite_rank`` (public default) selects the k-th order statistic with
    k = ceil((M+1)*(1-alpha)). If k > M, the threshold is +inf. ``linear_quantile``
    is the legacy linearly interpolated empirical (1-alpha) quantile. Both
    rules use strict ``score > threshold`` selection downstream, including ties.

    Repeated entries are retained with equal weight. All supplied entries must
    be finite; none are silently filtered or deduplicated. Empty input raises
    ValueError, as can occur after all positions in small batches are fixed.
    Scores are detached and copied to CPU float64 only after score computation.
    Neither rule asserts exact statistical control for dependent pooled scores.
    """

    level = _scalar(alpha, "alpha")
    if not 0.0 < level < 1.0:
        raise ValueError("alpha must lie strictly between 0 and 1")
    if rule not in {"finite_rank", "linear_quantile"}:
        raise ValueError("rule must be 'finite_rank' or 'linear_quantile'")
    try:
        values = torch.as_tensor(scores)
    except (TypeError, ValueError, RuntimeError) as error:
        raise TypeError("scores must be a real numeric vector") from error
    if values.ndim != 1:
        raise ValueError("scores must be one-dimensional; do not pool a score matrix here")
    if values.numel() == 0:
        raise ValueError("calibration requires at least one retained permutation score")
    if values.is_complex() or values.dtype == torch.bool:
        raise TypeError("scores must be real numeric values, not complex or boolean")
    if isinstance(scores, torch.Tensor):
        values = values.detach().to(device="cpu", dtype=torch.float64)
    else:
        # Reconstruct from the original sequence, not the dtype-inference
        # tensor above: a Python float is double precision and must not first
        # be rounded to the global torch default (usually float32).
        values = torch.as_tensor(scores, device="cpu", dtype=torch.float64)
    if not bool(torch.isfinite(values).all()):
        raise ValueError("scores contains non-finite values")

    count = values.numel()
    rank: Optional[int] = None
    if rule == "finite_rank":
        # Evaluate the user-facing decimal alpha without floating-point ceil
        # overshoot at exact ranks (e.g. M=9, alpha=0.7 has rank 3, not 4).
        target = Decimal(count + 1) * (Decimal(1) - Decimal(str(level)))
        rank = int(target.to_integral_value(rounding=ROUND_CEILING))
        threshold = (
            float(values.kthvalue(rank).values.item()) if rank <= count else math.inf
        )
    else:
        threshold = float(
            torch.quantile(values, 1.0 - level, interpolation="linear").item()
        )
    exceedance = float((values > threshold).double().mean().item())
    return CalibrationResult(
        threshold=threshold,
        alpha=level,
        rule=rule,
        num_scores=count,
        rank=rank,
        empirical_exceedance=exceedance,
    )


@torch.no_grad()
def batch_permutation_scores(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    *,
    batch_size: int = 1000,
    score_scale: Scalar = 1.0,
    generator: Optional[torch.Generator] = None,
    compute_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """One uniform text permutation per consecutive batch, discarding fixed points.

    Features must already be L2-normalised paired encodings, with the same
    shape, dtype and device. For every batch, draw a full ``torch.randperm``
    exactly once. Discard positions i with permutation[i] == i afterwards;
    do not repair indices, draw a derangement, or resample identity batches.
    Repeated numerical scores remain repeated in the output.

    The default preserves feature/scale dtypes and uses the historical score
    operation order ``(scale * image * permuted_text).sum(-1)``. A precision
    change must be explicit via ``compute_dtype``; it can change calibration.
    Use the same teacher scale and precision policy for downstream scores.

    Input rows count paired records; output entries count *retained scores*.
    This helper does not enforce a budget or draw additional records. A caller
    targeting M retained scores must provide/consume additional batches, then
    truncate the retained vector, and record both counts. An all-fixed outcome
    returns an empty vector. A CPU generator works with CPU or GPU features.
    """

    _matrix(image_features, "image_features", square=False)
    _matrix(text_features, "text_features", square=False)
    if image_features.shape != text_features.shape:
        raise ValueError("image_features and text_features must have the same shape")
    if image_features.device != text_features.device:
        raise ValueError("image_features and text_features must be on the same device")
    if image_features.dtype != text_features.dtype:
        raise ValueError("image_features and text_features must have the same dtype")
    if not isinstance(batch_size, Integral) or isinstance(batch_size, bool) or batch_size < 2:
        raise ValueError("batch_size must be an integer of at least 2")
    scale_value = _scalar(score_scale, "score_scale")
    if scale_value <= 0.0:
        raise ValueError("score_scale must be positive")
    if generator is not None and not isinstance(generator, torch.Generator):
        raise TypeError("generator must be a torch.Generator or None")
    if compute_dtype is not None and compute_dtype not in {
        torch.float16, torch.bfloat16, torch.float32, torch.float64
    }:
        raise ValueError("compute_dtype must be a real floating-point torch dtype")

    images, texts = image_features.detach(), text_features.detach()
    if compute_dtype is not None:
        images, texts = images.to(compute_dtype), texts.to(compute_dtype)
    if isinstance(score_scale, torch.Tensor):
        scale: Scalar = score_scale.detach().to(device=images.device)
        if compute_dtype is not None:
            scale = scale.to(dtype=compute_dtype)
    else:
        scale = scale_value

    permutation_device = generator.device if generator is not None else torch.device("cpu")
    retained = []
    for start in range(0, images.shape[0], int(batch_size)):
        stop = min(start + int(batch_size), images.shape[0])
        size = stop - start
        permutation = torch.randperm(size, generator=generator, device=permutation_device)
        indices = torch.arange(size, device=permutation.device)
        mismatched = (permutation != indices).to(device=images.device)
        permutation = permutation.to(device=images.device)
        if not bool(mismatched.any()):
            continue
        scores = (
            scale
            * images[start:stop][mismatched]
            * texts[start:stop][permutation[mismatched]]
        ).sum(dim=-1)
        if not bool(torch.isfinite(scores).all()):
            raise ValueError("permutation scores are non-finite; check feature/scale precision")
        retained.append(scores)
    if not retained:
        return images.new_empty((0,))
    return torch.cat(retained)
