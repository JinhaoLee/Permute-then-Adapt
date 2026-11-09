"""The fixed CLIP teacher and trainable CLIP student used by PTA."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ModelBundle:
    teacher: Any
    student: Any
    teacher_processor: Any
    student_processor: Any
    metadata: dict


def configure_student(model, *, last_text_blocks: int = 2, train_logit_scale: bool = True,
                      vision_tuning: str = "all_blocks"):
    """Train all vision transformer blocks and the final text blocks.

    The historical setting freezes vision embeddings/pre/post layer norms;
    text embeddings, earlier text blocks and the final text norm are also frozen.
    ``all_parameters`` explicitly enables a different vision-tuning setting.
    """
    layers = model.text_model.encoder.layers
    if not 0 <= last_text_blocks <= len(layers):
        raise ValueError("last_text_blocks is outside this model's text encoder")
    if vision_tuning not in {"all_blocks", "all_parameters"}:
        raise ValueError("vision_tuning must be all_blocks or all_parameters")
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    if vision_tuning == "all_blocks":
        for parameter in model.vision_model.parameters():
            parameter.requires_grad_(False)
        for layer in model.vision_model.encoder.layers:
            for parameter in layer.parameters():
                parameter.requires_grad_(True)
    for parameter in model.text_model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.text_projection.parameters():
        parameter.requires_grad_(last_text_blocks > 0)
    if last_text_blocks:
        for layer in list(layers)[-last_text_blocks:]:
            for parameter in layer.parameters():
                parameter.requires_grad_(True)
    model.logit_scale.requires_grad_(train_logit_scale)
    return {
        "trainable_parameters": sum(x.numel() for x in model.parameters() if x.requires_grad),
        "total_parameters": sum(x.numel() for x in model.parameters()),
        "trainable_parameter_names": [n for n, p in model.named_parameters() if p.requires_grad],
        "last_text_blocks": last_text_blocks,
        "vision_tuning": vision_tuning,
        "vision_trainable_parameters": sum(x.numel() for x in model.vision_model.parameters() if x.requires_grad),
        "vision_total_parameters": sum(x.numel() for x in model.vision_model.parameters()),
        "train_logit_scale": train_logit_scale,
    }


def _load_clip(source, canonical, *, dtype, revision=None, cache_dir=None, attention_implementation="sdpa"):
    from transformers import CLIPModel, CLIPProcessor
    kwargs = {"torch_dtype": dtype, "attn_implementation": attention_implementation}
    if revision:
        kwargs["revision"] = revision
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    model = CLIPModel.from_pretrained(source, **kwargs)
    processor_kwargs = {k: v for k, v in kwargs.items() if k in {"revision", "cache_dir"}}
    try:
        processor = CLIPProcessor.from_pretrained(source, **processor_kwargs)
    except (OSError, ValueError):
        # A model-only local checkpoint may omit its tokenizer/processor assets.
        processor = CLIPProcessor.from_pretrained(canonical, **processor_kwargs)
    return model, processor


def load_models(config, device) -> ModelBundle:
    import torch
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[config.precision]
    if config.precision == "bf16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This GPU does not support bf16; select precision=fp32 explicitly")
    teacher, teacher_processor = _load_clip(
        config.teacher_checkpoint or config.teacher_model, config.teacher_model,
        dtype=dtype, revision=config.teacher_revision, cache_dir=config.cache_dir,
        attention_implementation=config.attention_implementation,
    )
    student, student_processor = _load_clip(
        config.student_checkpoint or config.student_model, config.student_model,
        dtype=dtype, revision=config.student_revision, cache_dir=config.cache_dir,
        attention_implementation=config.attention_implementation,
    )
    teacher.requires_grad_(False)
    teacher.eval()
    student_metadata = configure_student(
        student, last_text_blocks=config.last_text_blocks,
        train_logit_scale=config.train_logit_scale, vision_tuning=config.vision_tuning,
    )
    if config.gradient_checkpointing:
        student.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    teacher.to(device)
    student.to(device)
    metadata = {
        "teacher_model": config.teacher_model,
        "student_model": config.student_model,
        "teacher_revision": getattr(teacher.config, "_commit_hash", None),
        "student_revision": getattr(student.config, "_commit_hash", None),
        "teacher_fixed": True,
        "teacher_logit_scale_exp": float(teacher.logit_scale.exp().item()),
        "weight_dtype": str(dtype),
        "attention_implementation": config.attention_implementation,
        **student_metadata,
    }
    return ModelBundle(teacher, student, teacher_processor, student_processor, metadata)


def move_inputs(inputs, device, dtype):
    return {
        key: value.to(device=device, dtype=dtype if key == "pixel_values" else value.dtype,
                      non_blocking=False)
        for key, value in inputs.items()
    }
