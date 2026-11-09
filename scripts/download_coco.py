#!/usr/bin/env python3
"""Download the official COCO 2014 archives used by the PTA recipe.

The URLs are published by the COCO project. Image archives stay in the user's
data directory; this script does not redistribute images or checkpoints.
"""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import urllib.request
import zipfile


ARCHIVES = {
    "train2014": ("http://images.cocodataset.org/zips/train2014.zip", 13510573713),
    "val2014": ("http://images.cocodataset.org/zips/val2014.zip", 6645013297),
    "annotations": ("http://images.cocodataset.org/annotations/annotations_trainval2014.zip", 252872794),
}


def download(url, destination, expected_size):
    if destination.exists():
        if destination.stat().st_size != expected_size:
            raise RuntimeError(f"Existing archive has an unexpected size: {destination}")
        print(f"Using existing archive: {destination}", flush=True)
    else:
        partial = destination.with_suffix(destination.suffix + ".part")
        if partial.exists():
            raise FileExistsError(f"Incomplete download found; inspect or remove it before retrying: {partial}")
        request = urllib.request.Request(url, headers={"User-Agent": "PTA-dataset-download/0.1"})
        received = 0
        with urllib.request.urlopen(request, timeout=90) as response, partial.open("wb") as output:
            while True:
                block = response.read(8 * 1024 * 1024)
                if not block:
                    break
                output.write(block)
                received += len(block)
                if received // (256 * 1024 * 1024) != (received-len(block)) // (256 * 1024 * 1024):
                    print(f"{destination.name}: {received:,}/{expected_size:,} bytes", flush=True)
        if received != expected_size:
            raise RuntimeError(f"Unexpected archive size: received {received}, expected {expected_size}; retained {partial}")
        partial.rename(destination)
    digest = hashlib.sha256()
    with destination.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract(archive_path, root):
    root = root.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        members = archive.infolist()
        for member in members:
            name = PurePosixPath(member.filename)
            if name.is_absolute() or ".." in name.parts or "\\" in member.filename:
                raise ValueError(f"Unsafe archive member: {member.filename}")
            if stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError(f"Unexpected archive symlink: {member.filename}")
            destination = root.joinpath(*name.parts)
            if not destination.resolve().is_relative_to(root):
                raise ValueError(f"Archive member escapes output root: {member.filename}")
        for member in members:
            destination = root.joinpath(*PurePosixPath(member.filename).parts)
            if member.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                if destination.stat().st_size != member.file_size:
                    raise FileExistsError(f"Refusing to replace existing data with a different size: {destination}")
                continue
            temporary = destination.with_name(destination.name + ".extracting")
            if temporary.exists():
                raise FileExistsError(f"Incomplete extracted file found: {temporary}")
            # Reading each member to EOF also checks its ZIP CRC.
            with archive.open(member) as source, temporary.open("wb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
            temporary.rename(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/coco"))
    parser.add_argument("--parts", nargs="+", choices=list(ARCHIVES), default=list(ARCHIVES))
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--list-only", action="store_true", help="Show sources and archive sizes without downloading.")
    args = parser.parse_args()
    if args.list_only:
        print(json.dumps({key: {"url": ARCHIVES[key][0], "bytes": ARCHIVES[key][1]} for key in args.parts}, indent=2))
        return
    archive_root = args.root / "archives"
    archive_root.mkdir(parents=True, exist_ok=True)
    provenance = []
    for key in dict.fromkeys(args.parts):
        url, size = ARCHIVES[key]
        path = archive_root / url.rsplit("/", 1)[-1]
        sha256 = download(url, path, size)
        provenance.append({"part": key, "url": url, "bytes": size, "downloaded_sha256": sha256})
        if args.extract:
            extract(path, args.root)
        (args.root / "download-manifest.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()

