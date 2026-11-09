"""Single-GPU PTA training, with explicit calibration and validation protocols.

Run ``python -m pta.train --config configs/coco.json --seed 41``. This entry
point implements only the positive-set probability-mass objective from the
paper. It does not combine that objective with a separate diagonal CE term.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .data import ImageCaptionDataset, PairCollator, load_coco_records, load_jsonl_records
from .data import records_sha256, seed_worker
from .models import load_models, move_inputs


@dataclass
class TrainConfig:
    train_manifest: str | None = None
    validation_manifest: str | None = None
    image_root: str | None = None
    coco_root: str | None = None
    train_split: str = "train2014"
    validation_split: str = "val2014"
    teacher_model: str = "openai/clip-vit-base-patch32"
    student_model: str = "openai/clip-vit-base-patch16"
    teacher_checkpoint: str | None = None
    student_checkpoint: str | None = None
    teacher_revision: str | None = None
    student_revision: str | None = None
    cache_dir: str | None = None
    output_dir: str = "runs/pta_seed41"
    seed: int = 41
    epochs: int = 20
    batch_size: int = 512
    eval_batch_size: int = 128
    eval_steps: int = 50
    early_stopping_patience: int = 5
    learning_rate: float = 1.25e-5
    weight_decay: float = 1e-4
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    precision: str = "bf16"
    attention_implementation: str = "sdpa"
    gradient_checkpointing: bool = True
    last_text_blocks: int = 2
    vision_tuning: str = "all_blocks"
    train_logit_scale: bool = True
    caption_sampling: str = "random"
    alpha: float = 0.01
    calibration_size: int = 5000
    calibration_batch_size: int = 1000
    calibration_rule: str = "finite_rank"
    calibration_budget_unit: str = "retained_scores"
    calibration_rng: str = "shared"
    calibration_device: str = "train_device"
    calibration_seed: int | None = None
    calibration_manifest: str | None = None
    validation_caption_mode: str = "first"
    num_workers: int = 4
    pin_memory: bool = False
    persistent_workers: bool = True
    prefetch_factor: int = 1
    drop_last: bool = True
    max_steps: int | None = None
    max_train_records: int | None = None
    max_validation_records: int | None = None
    device: str = "cuda"
    logging_steps: int = 10

    def validate(self):
        for name in ("epochs", "batch_size", "eval_batch_size", "eval_steps", "calibration_size",
                     "calibration_batch_size", "prefetch_factor", "logging_steps"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.batch_size < 2 or self.calibration_batch_size < 2:
            raise ValueError("Contrastive and permutation batches must contain at least two records")
        if not (0 < self.alpha < 1 and 0 <= self.warmup_ratio < 1):
            raise ValueError("Require alpha in (0,1) and warmup_ratio in [0,1)")
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.max_grad_norm <= 0:
            raise ValueError("Invalid optimizer configuration")
        if self.num_workers < 0 or self.early_stopping_patience < 0:
            raise ValueError("num_workers and patience must be nonnegative")
        if self.precision not in {"fp32", "bf16"}:
            raise ValueError("precision must be fp32 or bf16")
        if self.attention_implementation not in {"eager", "sdpa"}:
            raise ValueError("attention_implementation must be eager or sdpa")
        if self.vision_tuning not in {"all_blocks", "all_parameters"}:
            raise ValueError("vision_tuning must be all_blocks or all_parameters")
        if self.calibration_rule not in {"finite_rank", "linear_quantile"}:
            raise ValueError("Unknown calibration_rule")
        if self.calibration_budget_unit not in {"records", "retained_scores"}:
            raise ValueError("calibration_budget_unit must be records or retained_scores")
        if self.calibration_rng not in {"shared", "isolated"}:
            raise ValueError("calibration_rng must be shared or isolated")
        if self.calibration_device not in {"train_device", "cpu"}:
            raise ValueError("calibration_device must be train_device or cpu")
        if self.validation_caption_mode not in {"first", "all"}:
            raise ValueError("validation_caption_mode must be first or all")
        if self.caption_sampling not in {"random", "first"}:
            raise ValueError("caption_sampling must be random or first")
        if self.max_steps is not None and self.max_steps <= 0:
            raise ValueError("max_steps must be positive when supplied")
        for name in ("max_train_records", "max_validation_records"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive when supplied")
        if not self.train_manifest and not self.coco_root:
            raise ValueError("Provide train_manifest or coco_root")
        if not self.validation_manifest and not self.coco_root:
            raise ValueError("Provide validation_manifest or coco_root; validation is required")


def _jsonable(value):
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return "+inf" if value > 0 else "-inf" if value < 0 else "nan"
    return value


def _write_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_jsonable(payload), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _seed_all(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _rng_state():
    import numpy as np
    import torch
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def _restore_rng(state):
    import numpy as np
    import torch
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def _environment(device):
    import torch
    packages = {}
    for package in ("torch", "torchvision", "transformers", "numpy", "Pillow", "safetensors"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    return {"python": platform.python_version(), "packages": packages,
            "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "device_type": device.type,
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "gpu_memory_bytes": torch.cuda.get_device_properties(device).total_memory if device.type == "cuda" else None,
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32}


def _load_records(config):
    if config.train_manifest:
        train = load_jsonl_records(config.train_manifest, image_root=config.image_root)
    else:
        train = load_coco_records(config.coco_root, config.train_split)
    if config.validation_manifest:
        validation = load_jsonl_records(config.validation_manifest, image_root=config.image_root)
    else:
        validation = load_coco_records(config.coco_root, config.validation_split)
    if config.max_train_records:
        train = train[:config.max_train_records]
    if config.max_validation_records:
        validation = validation[:config.max_validation_records]
    calibration = load_jsonl_records(config.calibration_manifest, image_root=config.image_root) if config.calibration_manifest else train
    validation_paths = {Path(x["image"]).resolve() for x in validation}
    overlap = {Path(x["image"]).resolve() for x in train} & validation_paths
    if overlap:
        raise ValueError("Training and validation image paths overlap")
    calibration_overlap = {Path(x["image"]).resolve() for x in calibration} & validation_paths
    if calibration_overlap:
        raise ValueError("Calibration and validation image paths overlap")
    return train, validation, calibration


def calibrate(config, bundle, records, output, device):
    """Consume records or retained scores explicitly; never silently conflate them."""
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from .core import batch_permutation_scores, calibrate_from_scores

    saved_rng = _rng_state() if config.calibration_rng == "isolated" else None
    teacher_device = next(bundle.teacher.parameters()).device
    calibration_device = torch.device("cpu") if config.calibration_device == "cpu" else device
    if saved_rng is not None:
        _seed_all(config.calibration_seed if config.calibration_seed is not None else config.seed)
    try:
        teacher = bundle.teacher
        teacher.to(calibration_device)
        teacher.eval()
        dtype = next(teacher.parameters()).dtype
        loader = DataLoader(
            ImageCaptionDataset(records, config.caption_sampling),
            batch_size=config.calibration_batch_size, shuffle=True,
            num_workers=0, collate_fn=PairCollator(bundle.teacher_processor),
        )
        rng_before = _rng_state()
        scores, indices = [], []
        caption_digest = hashlib.sha256()
        records_consumed = 0
        discarded = 0
        for batch in loader:
            if config.calibration_budget_unit == "records":
                remaining = config.calibration_size - records_consumed
                if remaining <= 0:
                    break
                count = min(remaining, len(batch["indices"]))
                inputs = {k: v[:count] for k, v in batch["inputs"].items()}
            else:
                count = len(batch["indices"])
                inputs = batch["inputs"]
            inputs = move_inputs(inputs, calibration_device, dtype)
            with torch.no_grad():
                image = F.normalize(teacher.get_image_features(pixel_values=inputs["pixel_values"]), dim=-1)
                text = F.normalize(teacher.get_text_features(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]), dim=-1)
                values = batch_permutation_scores(
                    image, text, batch_size=count, score_scale=teacher.logit_scale.exp(),
                ) if count > 1 else image.new_empty(0)
            scores.append(values.detach().float().cpu())
            discarded += count - values.numel()
            records_consumed += count
            indices.extend(batch["indices"][:count])
            for caption in batch["captions"][:count]:
                caption_digest.update(caption.encode("utf-8") + b"\n")
            if config.calibration_budget_unit == "retained_scores" and sum(x.numel() for x in scores) >= config.calibration_size:
                break
        values = torch.cat(scores) if scores else torch.empty(0)
        retained_before_cap = values.numel()
        available = records_consumed if config.calibration_budget_unit == "records" else values.numel()
        if available < config.calibration_size:
            raise ValueError(f"Calibration budget cannot be met: requested {config.calibration_size} {config.calibration_budget_unit}, available {available}")
        if config.calibration_budget_unit == "retained_scores":
            values = values[:config.calibration_size]
        result = calibrate_from_scores(values, config.alpha, rule=config.calibration_rule)
        metadata = {**dataclasses.asdict(result), "budget_unit": config.calibration_budget_unit,
                    "budget_requested": config.calibration_size, "records_consumed": records_consumed,
                    "fixed_points_discarded": discarded, "retained_before_cap": retained_before_cap,
                    "records_sha256": records_sha256(records), "record_indices": indices,
                    "sampled_captions_sha256": caption_digest.hexdigest(),
                    "scores_float32_sha256": hashlib.sha256(values.numpy().tobytes()).hexdigest(),
                    "teacher_logit_scale_exp": float(teacher.logit_scale.exp().item()),
                    "rng_mode": config.calibration_rng, "device": str(calibration_device),
                    "permutation_device": "cpu"}
        _write_json(output / "calibration.json", metadata)
        torch.save(values, output / "calibration_scores.pt")
        torch.save({"before": rng_before, "after": _rng_state()}, output / "calibration_rng.pt")
        return float(result.threshold)
    finally:
        bundle.teacher.to(teacher_device)
        if saved_rng is not None:
            _restore_rng(saved_rng)


def _optimizer(model, config, device):
    import torch
    layer_norm_ids = {id(p) for module in model.modules() if isinstance(module, torch.nn.LayerNorm) for p in module.parameters(recurse=False)}
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (no_decay if id(parameter) in layer_norm_ids or name.endswith(".bias") else decay).append(parameter)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": config.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=config.learning_rate, betas=(0.9, 0.999), eps=1e-8,
        fused=(device.type == "cuda"),
    )


def train(config: TrainConfig):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Subset
    from transformers import get_cosine_schedule_with_warmup
    from .core import pta_loss
    from .evaluate import eval_coco_validation

    config.validate()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("This release entry point is single-GPU; launch one process")
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use device=cpu and precision=fp32 only for small smoke checks")
    output = Path(config.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "resolved_config.json").exists():
        raise FileExistsError("The output directory already contains a run; choose a new output_dir")
    lock = output / ".training.lock"
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(fd)
    started = time.monotonic()
    try:
        _seed_all(config.seed)
        _write_json(output / "resolved_config.json", dataclasses.asdict(config))
        _write_json(output / "environment.json", _environment(device))
        records, validation, calibration_records = _load_records(config)
        bundle = load_models(config, device)
        _write_json(output / "model_metadata.json", bundle.metadata)
        steps_per_epoch = len(records) // config.batch_size if config.drop_last else math.ceil(len(records) / config.batch_size)
        if steps_per_epoch < 1:
            raise ValueError("The training dataset is smaller than one complete batch")
        total_steps = config.max_steps if config.max_steps is not None else steps_per_epoch * config.epochs
        epochs = max(config.epochs, math.ceil(total_steps / steps_per_epoch)) if config.max_steps is not None else config.epochs
        order_arrays = [torch.randperm(len(records), generator=torch.Generator().manual_seed(config.seed + epoch)).numpy() for epoch in range(epochs)]
        np.savez_compressed(output / "epoch_orders.npz", **{f"epoch_{i}": values for i, values in enumerate(order_arrays)})
        _write_json(output / "data_metadata.json", {
            "implementation": "portable_single_gpu_pytorch_v1",
            "train_records": len(records), "validation_records": len(validation),
            "train_sha256": records_sha256(records), "validation_sha256": records_sha256(validation),
            "epoch_order_sha256": [hashlib.sha256(x.tobytes()).hexdigest() for x in order_arrays],
            "order_rule": "torch.randperm; independent CPU generator seeded seed+epoch",
            "validation_caption_mode": config.validation_caption_mode,
            "validation_metric": "mean of i2t/t2i recall@1,@5,@10 (percent)",
            "validation_schedule": "every eval_steps; end-only fallback if no scheduled validation occurred",
            "bitwise_legacy_rng_equivalence_claimed": False,
        })
        threshold = calibrate(config, bundle, calibration_records, output, device)
        optimizer = _optimizer(bundle.student, config, device)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=math.ceil(total_steps * config.warmup_ratio),
            num_training_steps=total_steps,
        )
        dtype = next(bundle.student.parameters()).dtype
        global_step = 0
        best_score = -float("inf")
        best_step = 0
        no_improvement = 0
        stop = False
        last_metrics = None
        last_eval_step = -1
        loss_window = []

        def save_checkpoint(name, epoch, batch_index):
            destination = output / name
            bundle.student.save_pretrained(destination)
            bundle.student_processor.save_pretrained(destination)
            torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                        "global_step": global_step, "epoch": epoch, "batch_index": batch_index,
                        "best_score": best_score, "best_step": best_step, "rng": _rng_state()},
                       destination / "training_state.pt")
            _write_json(destination / "pta_state.json", {
                "threshold": threshold, "step": global_step, "best_score": best_score,
                "objective": "symmetric_positive_set_marginal_likelihood",
                "calibration_file": "../calibration.json",
            })

        def validate(epoch, batch_index):
            nonlocal best_score, best_step, no_improvement, last_metrics, last_eval_step
            metrics = eval_coco_validation(
                bundle.student, bundle.student_processor, validation, device=device,
                batch_size=config.eval_batch_size, caption_mode=config.validation_caption_mode,
                recall_ks=(1, 5, 10), precision=config.precision,
                num_workers=config.num_workers,
            )
            last_metrics = metrics
            last_eval_step = global_step
            score = float(metrics["mean_recall"])
            if not math.isfinite(score):
                raise RuntimeError("Validation produced a non-finite selection metric")
            improved = score > best_score
            if improved:
                best_score, best_step, no_improvement = score, global_step, 0
                save_checkpoint("best", epoch, batch_index)
            else:
                no_improvement += 1
            event = {"event": "validation", "step": global_step, "epoch": epoch,
                     "metrics": metrics, "improved": improved, "best_step": best_step,
                     "best_mean_recall": best_score, "checks_without_improvement": no_improvement}
            with (output / "metrics.jsonl").open("a") as handle:
                handle.write(json.dumps(_jsonable(event), allow_nan=False) + "\n")
            print(json.dumps(_jsonable(event), allow_nan=False), flush=True)

        dataset = ImageCaptionDataset(records, config.caption_sampling)
        for epoch, order in enumerate(order_arrays):
            training_caption_digest = hashlib.sha256()
            presentations = 0
            kwargs = {"num_workers": config.num_workers, "pin_memory": config.pin_memory,
                      "worker_init_fn": seed_worker,
                      "generator": torch.Generator().manual_seed(config.seed + epoch),
                      "collate_fn": PairCollator(bundle.student_processor, bundle.teacher_processor),
                      "drop_last": config.drop_last}
            if config.num_workers:
                kwargs.update(persistent_workers=config.persistent_workers, prefetch_factor=config.prefetch_factor)
            loader = DataLoader(Subset(dataset, order.tolist()), batch_size=config.batch_size, **kwargs)
            bundle.student.train()
            bundle.teacher.eval()
            for batch_index, batch in enumerate(loader):
                for record_index, caption in zip(batch["indices"], batch["captions"]):
                    training_caption_digest.update(json.dumps([record_index, caption], ensure_ascii=False).encode("utf-8") + b"\n")
                presentations += len(batch["indices"])
                inputs = move_inputs(batch["inputs"], device, dtype)
                teacher_inputs = move_inputs(batch["teacher_inputs"], device, next(bundle.teacher.parameters()).dtype)
                optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    teacher_scores = bundle.teacher(**teacher_inputs).logits_per_image
                logits = bundle.student(**inputs).logits_per_image
                loss = pta_loss(logits, teacher_scores, threshold)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite loss at step {global_step + 1}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(bundle.student.parameters(), config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                global_step += 1
                loss_window.append(float(loss.detach()))
                if global_step % config.logging_steps == 0:
                    event = {"event": "train", "step": global_step, "epoch": epoch,
                             "loss": sum(loss_window) / len(loss_window),
                             "learning_rate": scheduler.get_last_lr()[0],
                             "elapsed_seconds": time.monotonic() - started}
                    print(json.dumps(event), flush=True)
                    _write_json(output / "progress.json", event)
                    loss_window = []
                if global_step % config.eval_steps == 0:
                    validate(epoch, batch_index)
                    stop = bool(config.early_stopping_patience and no_improvement >= config.early_stopping_patience)
                if stop or global_step >= total_steps:
                    break
            with (output / "caption_trace.jsonl").open("a") as handle:
                handle.write(json.dumps({"epoch": epoch, "record_presentations": presentations,
                                         "ordered_record_caption_sha256": training_caption_digest.hexdigest(),
                                         "last_step": global_step}) + "\n")
            if stop or global_step >= total_steps:
                break
        if last_eval_step < 0:
            validate(epoch, batch_index)
        save_checkpoint("last", epoch, batch_index)
        summary = {"status": "completed", "steps": global_step, "planned_steps": total_steps,
                   "early_stopped": stop, "best_step": best_step, "best_mean_recall": best_score,
                   "last_validation": last_metrics, "elapsed_seconds": time.monotonic() - started,
                   "last_validation_step": last_eval_step,
                   "objective": "symmetric_positive_set_marginal_likelihood",
                   "finished_configured_run": True,
                   "debug_limits_enabled": config.max_steps is not None or config.max_train_records is not None or config.max_validation_records is not None}
        _write_json(output / "summary.json", summary)
        print(json.dumps(_jsonable(summary), allow_nan=False), flush=True)
        return summary
    finally:
        lock.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="JSON object containing TrainConfig fields")
    for name in ("seed", "max_steps", "num_workers", "calibration_size"):
        parser.add_argument("--" + name.replace("_", "-"), type=int)
    for name in ("output_dir", "coco_root", "train_manifest", "validation_manifest"):
        parser.add_argument("--" + name.replace("_", "-"))
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="Override any config field; JSON values accepted")
    args = parser.parse_args(argv)
    payload = json.loads(Path(args.config).read_text())
    if not isinstance(payload, dict):
        parser.error("Config must contain a JSON object")
    for key in ("seed", "max_steps", "num_workers", "calibration_size", "output_dir", "coco_root", "train_manifest", "validation_manifest"):
        value = getattr(args, key)
        if value is not None:
            payload[key] = value
    for item in args.set:
        key, separator, raw = item.partition("=")
        if not separator:
            parser.error("--set expects KEY=VALUE")
        try:
            payload[key] = json.loads(raw)
        except json.JSONDecodeError:
            payload[key] = raw
    unknown = set(payload) - {field.name for field in dataclasses.fields(TrainConfig)}
    if unknown:
        parser.error("Unknown config fields: " + ", ".join(sorted(unknown)))
    return train(TrainConfig(**payload))


if __name__ == "__main__":
    main()
