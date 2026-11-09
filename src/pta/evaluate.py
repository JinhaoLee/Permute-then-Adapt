"""Portable CLIP evaluation on explicit local manifests.

Retrieval JSONL contains one image and all its captions per row. Classification
JSONL contains ``image`` and a zero-based ``label``; class names must be supplied
in that exact label order. This module does not guess ImageNet label mappings or
download/redefine benchmark splits. Every output identifies its caption protocol.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
import json
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from PIL import Image

from .data import clip_image_transform, load_jsonl_records
from .metrics import retrieval_metrics


@contextmanager
def _evaluation_mode(model):
    states = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        with torch.inference_mode():
            yield
    finally:
        for module, state in states:
            module.training = state


def _device(model, requested=None):
    return torch.device(requested) if requested is not None else next(model.parameters()).device


def _autocast(device, precision):
    if precision == "bf16" and device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    if precision not in {"fp32", "bf16"}:
        raise ValueError("precision must be fp32 or bf16")
    return nullcontext()


def _encoding_precision(model, device, precision):
    if device.type == "cuda" and precision == "bf16":
        return "bf16"
    dtype = next(model.parameters()).dtype
    return {torch.bfloat16: "bf16", torch.float32: "fp32"}.get(dtype, str(dtype))


def _image(value, root=None):
    if isinstance(value, Image.Image):
        return value.copy()
    path = Path(value).expanduser()
    if not path.is_absolute() and root is not None:
        path = Path(root) / path
    with Image.open(path) as image:
        return image.copy()


def _features(value, normalization="fp32"):
    # CLIP returns tensors; recent Transformers wrappers may return pooled outputs.
    if not isinstance(value, torch.Tensor):
        value = value.pooler_output
    if normalization == "model":
        value = F.normalize(value, dim=-1)
    elif normalization == "fp32":
        value = F.normalize(value.float(), dim=-1)
    elif normalization != "none":
        raise ValueError("Unsupported feature normalization")
    return value.float().cpu()


class _ImageDataset(torch.utils.data.Dataset):
    def __init__(self, records, processor, image_root):
        self.records, self.image_root = records, image_root
        self.transform = clip_image_transform(processor)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.transform(_image(self.records[index]["image"], self.image_root))


def _encode_images(model, processor, records, device, batch_size, image_root, precision, num_workers=0, normalization="fp32"):
    dtype = next(model.parameters()).dtype
    outputs = []
    loader = torch.utils.data.DataLoader(
        _ImageDataset(records, processor, image_root), batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    for pixels in loader:
        pixels = pixels.to(device=device, dtype=dtype, non_blocking=device.type == "cuda")
        with _autocast(device, precision):
            outputs.append(_features(model.get_image_features(pixel_values=pixels), normalization))
    return torch.cat(outputs)


def _encode_texts(model, processor, texts, device, batch_size, precision, normalization="fp32"):
    tokenizer = processor.tokenizer
    max_length = getattr(tokenizer, "model_max_length", 77)
    if not isinstance(max_length, int) or not 0 < max_length < 100000:
        max_length = 77
    model_limit = getattr(getattr(model.config, "text_config", None), "max_position_embeddings", 77)
    max_length = min(max_length, model_limit)
    outputs = []
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(
            texts[start : start + batch_size], padding=True, truncation=True,
            max_length=max_length, return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with _autocast(device, precision):
            outputs.append(_features(model.get_text_features(**encoded), normalization))
    return torch.cat(outputs)


def eval_coco_validation(
    model, processor, records: Sequence[dict[str, Any]], *,
    device=None, batch_size: int = 128, text_batch_size: int = 512,
    image_root=None, caption_mode: str = "first", recall_ks=(1, 5, 10),
    precision: str = "bf16", protocol: str | None = None, num_workers: int = 0,
    numerics: str = "legacy",
) -> dict[str, Any]:
    """Evaluate complete candidate pools, preserving caption order.

    The historical checkpoint selector uses ``caption_mode='first'`` and the
    six-direction/K mean over (1,5,10). Set ``caption_mode='all'`` for genuine
    multi-caption evaluation. A Karpathy test requires a manifest containing
    the actual benchmark test split; the name alone does not select records.
    All recall values, including ``mean_recall``, are percentages.
    ``numerics='legacy'`` preserves the mixed-precision encoder path used by
    the first-caption checkpoint selector. CUDA BF16 autocast can promote its
    normalization to FP32. ``numerics='fp32'`` explicitly casts features before
    normalization (the multi-caption protocol). Ranking is FP32 in both modes.
    """
    if caption_mode not in {"first", "all"} or batch_size < 1 or text_batch_size < 1:
        raise ValueError("Use first/all captions and positive encoding batch sizes")
    if num_workers < 0:
        raise ValueError("num_workers must be nonnegative")
    if numerics not in {"legacy", "fp32"}:
        raise ValueError("numerics must be legacy or fp32")
    if not records:
        raise ValueError("No retrieval records")
    texts, mapping = [], []
    for index, row in enumerate(records):
        captions = row.get("captions", row.get("caption"))
        if isinstance(captions, str):
            captions = [captions]
        if not captions or not all(isinstance(x, str) and x.strip() for x in captions):
            raise ValueError(f"Retrieval record {index} has no valid captions")
        selected = captions[:1] if caption_mode == "first" else captions
        texts.extend(selected)
        mapping.extend([index] * len(selected))
    target_device = _device(model, device)
    normalization = "model" if numerics == "legacy" else "fp32"
    with _evaluation_mode(model):
        images = _encode_images(model, processor, records, target_device, batch_size, image_root, precision, num_workers, normalization)
        captions = _encode_texts(model, processor, texts, target_device, text_batch_size, precision, normalization)
        result = retrieval_metrics(
            images, captions, mapping, recall_ks=recall_ks,
            batch_size=batch_size, device=target_device, normalize=False,
        )
    result.update({
        "num_images": len(records), "num_captions": len(texts),
        "caption_mode": caption_mode, "recall_ks": list(recall_ks),
        "score_units": "percent", "precision": _encoding_precision(model, target_device, precision),
        "weight_dtype": str(next(model.parameters()).dtype),
        "embedding_dtype": "float32", "similarity_dtype": "float32",
        "feature_normalization": normalization, "numerics": numerics,
        "protocol": protocol or f"explicit-manifest-{caption_mode}-captions",
        "image_preprocessing": "bicubic-resize-center-crop-rgb-clip-normalization",
    })
    return result


validation_retrieval = eval_coco_validation


def evaluate_classification(
    model, processor, records, class_names: Sequence[str], *, device=None,
    batch_size: int = 128, text_batch_size: int = 512, image_root=None,
    prompt_template: str = "A photo of a {label}", precision: str = "bf16",
    num_workers: int = 0,
    numerics: str = "legacy",
):
    """Single-template, normalized-feature top-1 accuracy in supplied label order."""
    if not records or not class_names or "{label}" not in prompt_template:
        raise ValueError("Supply examples, ordered class names, and a {label} template")
    if min(batch_size, text_batch_size) < 1 or num_workers < 0:
        raise ValueError("Use positive batch sizes and nonnegative num_workers")
    if numerics not in {"legacy", "fp32"}:
        raise ValueError("numerics must be legacy or fp32")
    labels = torch.tensor([int(row["label"]) for row in records])
    if labels.min() < 0 or labels.max() >= len(class_names):
        raise ValueError("Labels do not index the supplied class_names list")
    prompts = [prompt_template.format(label=name.replace("_", " ")) for name in class_names]
    if precision not in {"fp32", "bf16"}:
        raise ValueError("precision must be fp32 or bf16")
    target_device = _device(model, device)
    # Historical classification calls the BF16-weight encoders directly, without
    # CUDA autocast. Autocast also promotes normalization to FP32 and changes
    # encoder rounding. The helper flag "fp32" disables autocast; it does not
    # convert model weights, which remain in their loaded dtype. Retrieval uses
    # its separate mixed-precision path and is intentionally unaffected.
    encoding_precision = "fp32" if numerics == "legacy" else precision
    with _evaluation_mode(model):
        text_features = _encode_texts(model, processor, prompts, target_device, text_batch_size, encoding_precision, "model" if numerics == "legacy" else "fp32")
        image_features = _encode_images(model, processor, records, target_device, batch_size, image_root, encoding_precision, num_workers, "none" if numerics == "legacy" else "fp32")
        predictions = []
        similarity_dtype = next(model.parameters()).dtype if numerics == "legacy" else torch.float32
        candidates = text_features.to(device=target_device, dtype=similarity_dtype)
        for start in range(0, len(records), batch_size):
            queries = image_features[start : start + batch_size].to(device=target_device, dtype=similarity_dtype)
            if numerics == "legacy":
                scores = F.cosine_similarity(queries[:, None, :], candidates[None, :, :], dim=-1)
            else:
                scores = queries @ candidates.T
            predictions.append(scores.argmax(dim=1).cpu())
        correct = torch.cat(predictions).eq(labels).sum().item()
    return {
        "accuracy": 100.0 * correct / len(records), "num_examples": len(records),
        "num_classes": len(class_names), "score_units": "percent",
        "prompt_template": prompt_template, "precision": _encoding_precision(model, target_device, precision),
        "weight_dtype": str(next(model.parameters()).dtype),
        "embedding_dtype": "float32", "similarity_dtype": str(similarity_dtype),
        "normalization_dtype": str(next(model.parameters()).dtype) if numerics == "legacy" else "torch.float32",
        "autocast_encoding": encoding_precision == "bf16" and target_device.type == "cuda",
        "numerics": numerics,
    }


def _classification_records(path):
    path = Path(path)
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records or any("image" not in x or "label" not in x for x in records):
        raise ValueError("Classification JSONL needs image and zero-based label on each row")
    return records


def _class_names(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, list) or not value or not all(isinstance(x, str) for x in value):
        raise ValueError("class_names JSON must be an ordered nonempty list of strings")
    return value


def evaluate_dataset(model, processor, *, task, records, class_names=None, **kwargs):
    if task == "retrieval":
        return eval_coco_validation(model, processor, records, **kwargs)
    if task == "classification":
        return evaluate_classification(model, processor, records, class_names, **kwargs)
    raise ValueError("task must be retrieval or classification")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--processor", default="openai/clip-vit-base-patch16")
    parser.add_argument("--task", choices=("retrieval", "classification", "suite"), default="retrieval")
    parser.add_argument("--records", help="Local JSONL manifest")
    parser.add_argument("--class-names", help="Ordered class-name JSON list for classification")
    parser.add_argument("--suite", help="JSON: datasets=[{name,task,records,class_names,image_root,caption_mode,protocol}]")
    parser.add_argument("--image-root")
    parser.add_argument("--caption-mode", choices=("first", "all"), default="all")
    parser.add_argument("--protocol", default="explicit-manifest")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--text-batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--numerics", choices=("legacy", "fp32"), default="legacy")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    from transformers import CLIPModel, CLIPProcessor

    dtype = torch.bfloat16 if args.precision == "bf16" and str(args.device).startswith("cuda") else torch.float32
    model = CLIPModel.from_pretrained(args.checkpoint, torch_dtype=dtype).to(args.device)
    processor = CLIPProcessor.from_pretrained(args.processor)
    if args.task == "suite":
        if not args.suite:
            parser.error("--task suite requires --suite")
        suite_path = Path(args.suite).resolve()
        config = json.loads(suite_path.read_text(encoding="utf-8"))
        specs = config.get("datasets", [])
        if not specs:
            raise ValueError("Suite has no datasets")
        root = suite_path.parent
    else:
        if not args.records:
            parser.error("--records is required")
        if args.task == "classification" and not args.class_names:
            parser.error("--task classification requires --class-names")
        specs = [{"name": args.task, "task": args.task, "records": args.records,
                  "class_names": args.class_names, "image_root": args.image_root,
                  "caption_mode": args.caption_mode, "protocol": args.protocol}]
        config, root = {}, Path.cwd()
    results = {}
    cls_names = []
    for spec in specs:
        name, task = spec["name"], spec["task"]
        if name in results:
            raise ValueError(f"Duplicate evaluation name: {name}")
        manifest = Path(spec["records"])
        manifest = manifest if manifest.is_absolute() else root / manifest
        image_root = spec.get("image_root")
        if image_root is not None:
            image_root = Path(image_root)
            image_root = image_root if image_root.is_absolute() else root / image_root
        common = dict(device=args.device, batch_size=args.batch_size,
                      text_batch_size=args.text_batch_size, precision=args.precision,
                      num_workers=args.num_workers, numerics=spec.get("numerics", args.numerics))
        if task == "retrieval":
            records = load_jsonl_records(manifest, image_root=image_root)
            result = eval_coco_validation(
                model, processor, records, caption_mode=spec.get("caption_mode", args.caption_mode),
                protocol=spec.get("protocol", "explicit-manifest"), **common,
            )
        elif task == "classification":
            names_path = Path(spec["class_names"])
            names_path = names_path if names_path.is_absolute() else root / names_path
            records = _classification_records(manifest)
            result = evaluate_classification(
                model, processor, records, _class_names(names_path),
                image_root=image_root or manifest.parent,
                prompt_template=spec.get("prompt_template", "A photo of a {label}"), **common,
            )
            cls_names.append(name)
        else:
            raise ValueError(f"Unsupported suite task: {task}")
        results[name] = result
        print(json.dumps({"dataset": name, **result}), flush=True)
    expected = config.get("expected_classification_datasets")
    if expected is not None and list(expected) != cls_names:
        raise ValueError("Evaluated classification tasks do not match the declared suite/order")
    output = {"datasets": results, "score_units": "percent"}
    if cls_names:
        output["classification_datasets"] = cls_names
        output["classification_average"] = sum(results[n]["accuracy"] for n in cls_names) / len(cls_names)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
