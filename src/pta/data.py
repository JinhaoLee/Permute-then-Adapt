"""Local image--caption records and the deterministic CLIP preprocessing path.

The public input format is JSONL, one image per line: ``image`` is a path and
``captions`` is a nonempty list of strings. Relative paths are resolved against
``image_root`` or, if omitted, the manifest's directory. No image download is
performed by this module.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Sequence


def _captions(record: dict[str, Any], location: str) -> list[str]:
    value = record.get("captions", record.get("caption"))
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list) or not value or not all(
        isinstance(x, str) and x.strip() for x in value
    ):
        raise ValueError(f"{location}: provide caption or nonempty captions strings")
    return value


def load_jsonl_records(
    manifest: str | Path, *, image_root: str | Path | None = None,
    check_images: bool = True,
) -> list[dict[str, Any]]:
    """Read records without sorting or changing caption order."""
    manifest = Path(manifest).expanduser().resolve()
    root = Path(image_root).expanduser().resolve() if image_root else manifest.parent
    result = []
    seen = set()
    with manifest.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            location = f"{manifest.name}:{line_no}"
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{location}: invalid JSON") from exc
            if not isinstance(item, dict) or not isinstance(item.get("image"), str):
                raise ValueError(f"{location}: expected an image path")
            path = Path(item["image"]).expanduser()
            path = path.resolve() if path.is_absolute() else (root / path).resolve()
            if path in seen:
                raise ValueError(f"{location}: duplicate image; combine its captions in one record")
            if check_images and not path.is_file():
                raise FileNotFoundError(f"{location}: image does not exist: {path}")
            seen.add(path)
            result.append({
                "id": item.get("id", len(result)),
                "image": str(path),
                "image_key": item["image"],
                "captions": _captions(item, location),
            })
    if not result:
        raise ValueError(f"No image-caption records found in {manifest}")
    return result


def load_coco_records(
    root: str | Path, split: str, *, check_images: bool = True,
) -> list[dict[str, Any]]:
    """Read the original COCO captions JSON, preserving its image/annotation order."""
    if split not in {"train2014", "val2014", "train2017", "val2017"}:
        raise ValueError("Unsupported COCO split")
    root = Path(root).expanduser().resolve()
    path = root / "annotations" / f"captions_{split}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    grouped: dict[Any, list[str]] = {}
    for annotation in payload["annotations"]:
        caption = annotation.get("caption")
        if isinstance(caption, str) and caption.strip():
            grouped.setdefault(annotation["image_id"], []).append(caption)
    records = []
    seen = set()
    for image in payload["images"]:
        image_id = image["id"]
        if image_id in seen:
            raise ValueError(f"Duplicate COCO image id: {image_id}")
        seen.add(image_id)
        if image_id not in grouped:
            continue
        key = f"{split}/{image['file_name']}"
        image_path = root / key
        if check_images and not image_path.is_file():
            raise FileNotFoundError(f"Missing COCO image: {image_path}")
        records.append({
            "id": image_id, "image": str(image_path), "image_key": key,
            "captions": grouped[image_id],
        })
    if not records:
        raise ValueError(f"No captions found for COCO split {split}")
    return records


def records_sha256(records: Sequence[dict[str, Any]]) -> str:
    """Hash ordered record identities and captions, excluding the resolved root."""
    digest = hashlib.sha256()
    for record in records:
        payload = {"id": record["id"], "image": record.get("image_key", record["image"]),
                   "captions": record["captions"]}
        digest.update(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


class ImageCaptionDataset:
    """Map-style records; training draws one caption at each image presentation."""

    def __init__(self, records: Sequence[dict[str, Any]], caption_sampling: str = "random"):
        if caption_sampling not in {"random", "first"}:
            raise ValueError("caption_sampling must be random or first")
        self.records = records
        self.caption_sampling = caption_sampling

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        from PIL import Image, ImageFile
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        record = self.records[index]
        with Image.open(record["image"]) as image:
            decoded = image.copy()
        choices = record["captions"]
        caption = random.choice(choices) if self.caption_sampling == "random" else choices[0]
        return {"image": decoded, "caption": caption, "index": index}


def _to_rgb(image):
    return image.convert("RGB")


def clip_image_transform(processor):
    """Match the original Resize→CenterCrop→RGB→float32→Normalize sequence."""
    import torch
    from torchvision.transforms import InterpolationMode, v2
    image_processor = getattr(processor, "image_processor", processor)
    size = image_processor.size
    if "height" in size:
        resize = (size["height"], size["width"])
        crop = resize
    else:
        resize = size["shortest_edge"]
        crop = (resize, resize)
    return v2.Compose([
        v2.Resize(resize, interpolation=InterpolationMode.BICUBIC),
        v2.CenterCrop(crop),
        v2.Lambda(_to_rgb),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=image_processor.image_mean, std=image_processor.image_std),
    ])


class PairCollator:
    """Preprocess the same sampled caption for teacher and student."""

    def __init__(self, processor, teacher_processor=None):
        self.processor = processor
        self.teacher_processor = teacher_processor
        self.transform = clip_image_transform(processor)
        self.teacher_transform = (
            clip_image_transform(teacher_processor) if teacher_processor else None
        )

    @staticmethod
    def _encode(processor, transform, examples):
        import torch
        tokenizer = getattr(processor, "tokenizer", processor)
        tokens = tokenizer(
            [x["caption"] for x in examples], padding=True, truncation=True,
            max_length=tokenizer.model_max_length, return_tensors="pt",
        )
        return {
            "pixel_values": torch.stack([transform(x["image"]) for x in examples]),
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"],
        }

    def __call__(self, examples):
        result = {
            "inputs": self._encode(self.processor, self.transform, examples),
            "indices": [x["index"] for x in examples],
            "captions": [x["caption"] for x in examples],
        }
        if self.teacher_processor:
            result["teacher_inputs"] = self._encode(
                self.teacher_processor, self.teacher_transform, examples,
            )
        return result


def seed_worker(worker_id):
    """Seed Python caption selection and NumPy in each data worker."""
    import numpy as np
    import torch
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
