#!/usr/bin/env python3
"""Convert official COCO captions annotations to the local PTA JSONL format.

Uses only the Python standard library. All captions are retained in their
original annotation order; evaluation selects first/all captions explicitly.
Images follow the order of the annotation JSON's ``images`` list.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def _identifier(value, location):
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{location}: expected a nonnegative integer id")
    return value


def prepare_manifest(
    annotations, image_root, output, *, path_mode="manifest", expected_images=None,
    overwrite=False,
):
    """Validate a COCO captions split and write one JSONL row per image."""
    annotations = Path(annotations).expanduser().resolve()
    image_root = Path(image_root).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    if path_mode not in {"manifest", "image-root"}:
        raise ValueError("path_mode must be manifest or image-root")
    if expected_images is not None and expected_images <= 0:
        raise ValueError("expected_images must be positive")
    if not image_root.is_dir():
        raise FileNotFoundError(f"Image root is not a directory: {image_root}")
    if output == annotations:
        raise ValueError("Output must differ from the original annotations file")
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output}; use --overwrite to replace it")
    raw = annotations.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict) or not isinstance(payload.get("images"), list) or not isinstance(payload.get("annotations"), list):
        raise ValueError("Expected a COCO captions JSON object with images and annotations lists")
    if not payload["images"]:
        raise ValueError("The annotations contain no images")

    images = []
    ids = set()
    paths = set()
    for position, image in enumerate(payload["images"]):
        location = f"images[{position}]"
        if not isinstance(image, dict):
            raise ValueError(f"{location}: expected an object")
        image_id = _identifier(image.get("id"), location)
        if image_id in ids:
            raise ValueError(f"Duplicate image id: {image_id}")
        ids.add(image_id)
        name = image.get("file_name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{location}: missing file_name")
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"{location}: file_name must stay inside image_root")
        path = (image_root / relative).resolve()
        if not path.is_relative_to(image_root):
            raise ValueError(f"{location}: image path escapes image_root")
        if not path.is_file():
            raise FileNotFoundError(f"Missing image for id {image_id}: {path}")
        if path in paths:
            raise ValueError(f"Duplicate image path for id {image_id}: {name}")
        paths.add(path)
        stored = (Path(os.path.relpath(path, output.parent)).as_posix()
                  if path_mode == "manifest" else relative.as_posix())
        images.append((image_id, stored))

    captions = {image_id: [] for image_id in ids}
    annotation_ids = set()
    for position, annotation in enumerate(payload["annotations"]):
        location = f"annotations[{position}]"
        if not isinstance(annotation, dict):
            raise ValueError(f"{location}: expected an object")
        annotation_id = _identifier(annotation.get("id"), location)
        if annotation_id in annotation_ids:
            raise ValueError(f"Duplicate caption annotation id: {annotation_id}")
        annotation_ids.add(annotation_id)
        image_id = _identifier(annotation.get("image_id"), location)
        if image_id not in ids:
            raise ValueError(f"{location}: unknown image_id {image_id}")
        caption = annotation.get("caption")
        if not isinstance(caption, str) or not caption.strip():
            raise ValueError(f"{location}: missing or empty caption")
        captions[image_id].append(caption)
    missing = [image_id for image_id, _ in images if not captions[image_id]]
    if missing:
        raise ValueError(f"Images without captions: {missing[:10]}")
    if expected_images is not None and len(images) != expected_images:
        raise ValueError(f"Expected {expected_images} images, found {len(images)}")

    # Write only after all records have been validated, so a bad split does not
    # leave an apparently complete manifest behind.
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for image_id, image_path in images:
                row = {"id": image_id, "image": image_path, "captions": captions[image_id]}
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "images": len(images), "captions": sum(map(len, captions.values())),
        "path_mode": path_mode, "explicit_image_root_required": path_mode == "image-root",
        "annotation_sha256": hashlib.sha256(raw).hexdigest(),
        "manifest_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True, help="Official captions_train/val2014.json")
    parser.add_argument("--image-root", required=True, help="Directory containing this split's image files")
    parser.add_argument("--output", required=True, help="Destination JSONL manifest")
    parser.add_argument("--path-mode", choices=("manifest", "image-root"), default="manifest",
                        help="Store paths relative to the manifest (default) or to an explicit evaluator image root")
    parser.add_argument("--expected-images", type=int, help="Optional full-split count assertion")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = prepare_manifest(
            args.annotations, args.image_root, args.output, path_mode=args.path_mode,
            expected_images=args.expected_images, overwrite=args.overwrite,
        )
    except (OSError, ValueError) as error:
        parser.exit(2, f"error: {error}\n")
    print(json.dumps({"output": args.output, **result}, indent=2))


if __name__ == "__main__":
    main()
