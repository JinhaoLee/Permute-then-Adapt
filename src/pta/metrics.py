"""Retrieval and classification metrics; all returned scores are percentages."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F


def retrieval_metrics(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    text_to_image: Sequence[int] | torch.Tensor,
    *,
    recall_ks: Sequence[int] = (1, 5, 10),
    batch_size: int = 256,
    device: str | torch.device = "cpu",
    normalize: bool = True,
) -> dict[str, float]:
    """Compute exact full-pool R@K with any number of captions per image.

    ``text_to_image[j]`` identifies the ground-truth image for caption j.
    I2T counts a hit if any retrieved caption belongs to the query image; T2I
    counts each caption separately. Similarities are evaluated in chunks;
    candidates are never subsampled. Ties follow ``torch.topk``.
    ``mean_recall`` averages both directions over the explicitly requested Ks.
    Set ``normalize=False`` for features already normalized by an encoder; this
    preserves the supplied encoder normalization before FP32 similarities.
    """
    ks = tuple(sorted(set(int(k) for k in recall_ks)))
    if not ks or min(ks) < 1 or batch_size < 1:
        raise ValueError("Positive recall Ks and batch_size are required")
    images = torch.as_tensor(image_features).detach().float().cpu()
    texts = torch.as_tensor(text_features).detach().float().cpu()
    mapping = torch.as_tensor(text_to_image, dtype=torch.long).cpu()
    if images.ndim != 2 or texts.ndim != 2 or images.shape[1] != texts.shape[1]:
        raise ValueError("Image/text features must be nonempty matrices with equal width")
    n, m = len(images), len(texts)
    if not n or not m or mapping.shape != (m,):
        raise ValueError("Supply at least one image/caption and one image ID per caption")
    if mapping.min() < 0 or mapping.max() >= n:
        raise ValueError("Caption-to-image IDs are outside the image feature matrix")
    if torch.bincount(mapping, minlength=n).eq(0).any():
        raise ValueError("Every retrieval image must have a ground-truth caption")
    if not torch.isfinite(images).all() or not torch.isfinite(texts).all():
        raise ValueError("Features contain non-finite values")
    if images.norm(dim=1).eq(0).any() or texts.norm(dim=1).eq(0).any():
        raise ValueError("Zero-length feature vectors cannot define cosine similarity")
    if normalize:
        images, texts = F.normalize(images, dim=1), F.normalize(texts, dim=1)
    target_device = torch.device(device)
    i_hits, t_hits = {k: 0 for k in ks}, {k: 0 for k in ks}
    with torch.inference_mode():
        candidates = texts.to(target_device)
        mapping_device = mapping.to(target_device)
        for start in range(0, n, batch_size):
            scores = images[start : start + batch_size].to(target_device) @ candidates.T
            top = scores.topk(min(max(ks), m), dim=1).indices
            correct = mapping_device[top].eq(
                torch.arange(start, start + len(scores), device=target_device)[:, None]
            )
            for k in ks:
                i_hits[k] += int(correct[:, : min(k, m)].any(dim=1).sum())
        del candidates
        candidates = images.to(target_device)
        for start in range(0, m, batch_size):
            scores = texts[start : start + batch_size].to(target_device) @ candidates.T
            top = scores.topk(min(max(ks), n), dim=1).indices
            correct = top.eq(mapping_device[start : start + len(scores), None])
            for k in ks:
                t_hits[k] += int(correct[:, : min(k, n)].any(dim=1).sum())
    result = {f"i2t_recall@{k}": 100.0 * i_hits[k] / n for k in ks}
    result.update({f"t2i_recall@{k}": 100.0 * t_hits[k] / m for k in ks})
    result.update({f"mean_recall@{k}": (result[f"i2t_recall@{k}"] + result[f"t2i_recall@{k}"]) / 2 for k in ks})
    result["mean_recall"] = sum(result[f"mean_recall@{k}"] for k in ks) / len(ks)
    return result


def classification_accuracy(predictions, targets) -> float:
    predictions = torch.as_tensor(predictions).detach().cpu()
    targets = torch.as_tensor(targets).detach().cpu()
    if predictions.ndim == 2:
        predictions = predictions.argmax(dim=1)
    if predictions.shape != targets.shape or targets.ndim != 1 or not targets.numel():
        raise ValueError("Predictions and labels must have matching nonempty vector shapes")
    return float(predictions.eq(targets).float().mean() * 100.0)
