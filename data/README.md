# Training data

The COCO recipe uses **COCO Captions 2014 train2014** for adaptation and
calibration, and **val2014** for checkpoint selection. Acquire the images and
annotations from the [official COCO download page](https://cocodataset.org/#download).
The [official download-page source](https://github.com/cocodataset/cocodataset.github.io/blob/master/dataset/download.htm)
lists the same archive URLs used by the helper below.

From the repository root:

```bash
python scripts/download_coco.py --list-only
python scripts/download_coco.py --root data/coco --extract
```

This downloads roughly 20 GB of archives, then extracts the files. Allow space
for both the archives and extracted data. The helper records downloaded archive
hashes and checks ZIP CRCs during extraction. It uses the URLs published by the
provider; a locally computed hash is a provenance record, not a provider-signed
checksum. Existing files of a different size are not overwritten.

Expected layout:

```text
data/coco/
  train2014/
    COCO_train2014_000000000009.jpg
    ...
  val2014/
    COCO_val2014_000000000042.jpg
    ...
  annotations/
    captions_train2014.json
    captions_val2014.json
    ...
```

The recipe samples one training caption per image while retaining the original
caption lists. The configuration records calibration-budget and validation
caption policies because first-caption validation and all-caption retrieval are
different evaluation protocols. Keep the image split and annotation version
fixed when comparing results.

Actual images, archives, model checkpoints and local manifests belong in ignored
data/run directories. The repository contains acquisition instructions and
loaders. Dataset photographs and annotations have separate terms; see
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

## COCO validation retrieval

Convert the downloaded validation annotations into the JSONL format used by
the evaluation entry point:

```bash
python scripts/prepare_coco_manifest.py \
  --annotations data/coco/annotations/captions_val2014.json \
  --image-root data/coco/val2014 \
  --output data/manifests/coco-val2014.jsonl \
  --expected-images 40504
```

The helper keeps the annotation file's image order and each image's original
caption order. It retains **all captions**; it does not randomly choose one or
reorder them. Missing images/captions, duplicate image or caption-annotation IDs,
and invalid image paths cause an error. By default the stored image paths are
relative to the output manifest, so the following command needs no additional
`--image-root`. The helper also reports annotation and manifest hashes.

After training, evaluate the selected checkpoint with the historical
**first-caption** validation policy:

```bash
python -m pta.evaluate \
  --checkpoint runs/pta-coco-seed41/best \
  --processor runs/pta-coco-seed41/best \
  --task retrieval \
  --records data/manifests/coco-val2014.jsonl \
  --caption-mode first --numerics legacy --precision bf16 \
  --batch-size 1024 --text-batch-size 1024 --num-workers 8 \
  --protocol coco-val2014-first-caption \
  --output runs/pta-coco-seed41/coco-val2014-first.json
```

This uses all **40,504 val2014 images**, one caption per image, and reports
Recall@1/5/10 in both directions. `mean_recall` is the average of those six
percentage values. These are validation measurements used for checkpoint
selection, not a Karpathy 1K/5K test protocol.

An optional full-validation, all-caption diagnostic can use FP32 model weights,
feature normalization and ranking:

```bash
python -m pta.evaluate \
  --checkpoint runs/pta-coco-seed41/best \
  --processor runs/pta-coco-seed41/best \
  --task retrieval \
  --records data/manifests/coco-val2014.jsonl \
  --caption-mode all --numerics fp32 --precision fp32 \
  --batch-size 128 --text-batch-size 512 --num-workers 8 \
  --protocol coco-val2014-all-captions-fp32 \
  --output runs/pta-coco-seed41/coco-val2014-all-fp32.json
```

All-caption and first-caption recall have different candidate pools and correct
matches, and the numerical policies above also differ. Keep their results
separate when comparing runs. For training manifests, the same converter accepts
`captions_train2014.json` with `train2014` as the image root. If instead using
`--path-mode image-root`, supply that same image root explicitly to the evaluator.

## A new image-text dataset

1. Obtain the image files and captions from the dataset's original provider.
2. Put the images in a local directory and create one JSONL record per image.
3. Use image-disjoint training and validation files. Keep all captions belonging
   to an image in that image's record.
4. Set the manifest paths in a training configuration; select the teacher and
   student checkpoints and recalibrate the threshold on this dataset.

A record has this form:

```json
{"image": "images/example.jpg", "captions": ["A dog playing with a ball.", "A dog outdoors."]}
```

Image paths are resolved relative to the manifest unless an image root is
supplied. The loader checks that each image has at least one nonempty caption.
See [example.jsonl](example.jsonl) for a format example; it is not a dataset and
does not contain the example photographs. New data need not use COCO's JSON
annotation schema.

For evaluation datasets, retain the original label order and split definitions.
The benchmark sources and distributions used by the paper are listed in the
third-party notices. ImageNet requires obtaining the data through its provider's
access process.

