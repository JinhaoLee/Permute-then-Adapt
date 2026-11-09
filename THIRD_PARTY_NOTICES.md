# Third-party asset notices

Resources used in **Permute-then-Adapt: Weak-to-Strong Contrastive Image–Text Adaptation**. Official sources and published terms checked on **7 November 2026**.

This register credits the original providers and identifies dataset variants, model checkpoints, software, and available licensing information. Images, annotations, loader software, and model checkpoints can have different terms. A license for a software repository or scientific paper is not assigned here to the underlying photographs. Where the inspected original source does not state a named license, that limitation is recorded explicitly.

## Datasets

The 18 datasets below cover the main classification/retrieval experiments and the additional CC3M, Caltech101, and SUN397 experiments. Dataset variants identify the evaluated benchmarks; an unrecorded archive hash or upstream revision is not implied.

| Dataset and variant | Original attribution and official source | License or published usage notice |
| --- | --- | --- |
| **Microsoft COCO Captions, 2014** — train2014/val2014; adaptation, calibration and retrieval | Lin et al. / COCO Consortium; [project](https://cocodataset.org/), [official terms](https://cocodataset.org/dataset/termsofuse.htm) | **CC BY 4.0 applies to annotations and the website.** The Consortium does not own the image copyrights; image use is subject to Flickr terms and the original image rights. The annotation license is not a blanket license for image files. |
| **Flickr30K** — original image/caption corpus; retrieval and calibration | Young et al.; [University of Illinois project and terms](https://shannon.cs.illinois.edu/DenotationGraph/) | Official image access is for **non-commercial research and educational purposes**, subject to Flickr terms; original image copyrights remain with their owners. No separate named standard caption/dataset license was identified on the inspected official page. |
| **Conceptual Captions / CC3M, 2018** — 1M/2M/3M training subsets | Sharma et al.; [original paper](https://aclanthology.org/P18-1238/), [Google dataset repository](https://github.com/google-research-datasets/conceptual-captions), [terms](https://github.com/google-research-datasets/conceptual-captions/blob/master/LICENSE) | **Custom Google dataset terms**, permitting dataset use for any purpose and requesting acknowledgement. The release comprises captions and image URLs; these terms do not establish a blanket grant over third-party image files downloaded from those URLs. |
| **CIFAR-10** — original 10-class benchmark, test split | Dataset creators Alex Krizhevsky, Vinod Nair and Geoffrey Hinton; [official source](https://cave.cs.toronto.edu/kriz/cifar.html), [Krizhevsky's 2009 report](https://cave.cs.toronto.edu/kriz/learning-features-2009-TR.pdf) | No independent named dataset license stated on the inspected official page. The provider requests citation of the 2009 technical report. No general image redistribution grant is inferred here. |
| **CIFAR-100** — original 100-class benchmark, fine labels, test split | Dataset creators Alex Krizhevsky, Vinod Nair and Geoffrey Hinton; [official source](https://cave.cs.toronto.edu/kriz/cifar.html), [Krizhevsky's 2009 report](https://cave.cs.toronto.edu/kriz/learning-features-2009-TR.pdf) | Same original-provider notice as CIFAR-10; no independent named dataset license identified in the inspected official source. |
| **Food-101** — original 101-category benchmark, validation split | Bossard et al.; [ETH Zurich project](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) | No named dataset/image license established from the inspected official material. Original source-image rights and any terms supplied with the acquired data remain applicable. |
| **ImageNet-1K** — 1,000-class validation benchmark | Deng et al.; [official source](https://image-net.org/), [access agreement](https://image-net.org/accessagreement) | **Custom ImageNet Terms of Access** specify non-commercial research and educational use. ImageNet does not own all image copyrights; conditional access-sharing provisions do not establish a blanket public image redistribution grant. |
| **ImageNet-V2** — matched-frequency variant | Recht et al.; [official repository and image-license explanation](https://github.com/modestyachts/ImageNetV2) | The repository's **MIT license explicitly excludes the actual image data**. Image use follows corresponding Flickr/original-image terms. |
| **ImageNet-A** — original natural adversarial examples benchmark | Hendrycks et al.; [official source](https://github.com/hendrycks/natural-adv-examples), [repository license](https://github.com/hendrycks/natural-adv-examples/blob/master/LICENSE) | Repository software is **MIT**. No separate uniform image-data license was established from the inspected official README/LICENSE; the software license is not asserted as a blanket photo license. |
| **ImageNet-R** — original rendition benchmark | Hendrycks et al.; [official source](https://github.com/hendrycks/imagenet-r), [repository license](https://github.com/hendrycks/imagenet-r/blob/master/LICENSE) | Repository software is **MIT**. No separate uniform image-data license was established from the inspected official README/LICENSE. |
| **Describable Textures Dataset (DTD)** — original 47-category benchmark | Cimpoi et al.; [Oxford project](https://www.robots.ox.ac.uk/~vgg/data/dtd/), [provider README](https://thor.robots.ox.ac.uk/dtd/README.txt) | The provider states availability to the computer-vision community **for research purposes**. No named standard dataset license was identified; original image rights remain separate. |
| **Stanford Cars** — original 196-class benchmark, test split | Krause et al.; [original project](https://ai.stanford.edu/~jkrause/cars/car_dataset.html), [original paper](https://openaccess.thecvf.com/content_iccv_workshops_2013/W19/html/Krause_3D_Object_Representations_2013_ICCV_paper.html) | Original dataset/image license not independently recovered from the linked primary sources. The original project endpoint was unavailable at the check date; mirror or loader licenses are not substituted for photo rights. |
| **Caltech-UCSD Birds, CUB-200-2011** — original test split | Wah et al.; [Caltech project and usage notice](https://www.vision.caltech.edu/datasets/cub_200_2011/) | Provider restricts image use to **non-commercial research and educational purposes** and states that original image copyrights are not owned by the dataset authors. |
| **FGVC-Aircraft 2013b** — variant classification, test split | Maji et al.; [Oxford project, copyright and release notes](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/) | Photographs are provided **exclusively for non-commercial research**. Original photographers retain copyright; the provider directs other uses to them. |
| **Oxford Flowers-102** — 102-category release, supplied test split | Nilsback and Zisserman; [Oxford project](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/), [provider README](https://thor.robots.ox.ac.uk/flowers/102/README.txt) | No explicit named dataset/image license identified in the inspected project page or README. Public download availability is not treated as a new photo-rights grant. |
| **Oxford-IIIT Pet** — original 37-category release, supplied test split | Parkhi et al.; [official source and License section](https://www.robots.ox.ac.uk/~vgg/data/pets/), [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) | Official page specifies **CC BY-SA 4.0**, with commercial/research availability, and also states that image copyrights remain with the original owners. Retain both the named license and that provider qualification. |
| **Caltech101** — original 101-category collection, appendix evaluation | Fei-Fei, Fergus and Perona; [official CaltechDATA record](https://data.caltech.edu/records/mzrjq-6wc02), DOI 10.22002/D1.20086 | No named dataset/image license established from the inspected official record. An empty license field is not treated as permission for unrestricted reuse. |
| **SUN397** — 397-category scene benchmark, appendix evaluation | Xiao et al.; [original project](https://vision.princeton.edu/projects/2010/SUN/), [MIT original-paper record](https://dspace.mit.edu/entities/publication/092e56e9-7936-4ce3-91d4-43ae7ec24708) | Original image-data license/terms not independently recovered; the project endpoint was unavailable at the check date. The scientific paper's license is not assigned to dataset photographs. |

### Distribution and evaluation details

Some loaders use Hugging Face distributions: [`Multimodal-Fatima/StanfordCars_test`](https://huggingface.co/datasets/Multimodal-Fatima/StanfordCars_test), [`jxie/dtd`](https://huggingface.co/datasets/jxie/dtd), [`uoft-cs/cifar10`](https://huggingface.co/datasets/uoft-cs/cifar10), [`uoft-cs/cifar100`](https://huggingface.co/datasets/uoft-cs/cifar100), and [`ethz/food101`](https://huggingface.co/datasets/ethz/food101). These identifiers record distribution provenance; the original-provider and image terms above remain distinct from loader/library licenses.

COCO and Flickr30K retrieval results also use the Karpathy split protocol. This does not create a new license for the underlying image/caption corpus. No unverified provenance or independent license for an additional split-annotation download is asserted here.

## Models and class-name metadata

| Asset / version identifier | Original source and attribution | License / scope notice |
| --- | --- | --- |
| **OpenAI CLIP ViT-B/32**, `openai/clip-vit-base-patch32` — weak teacher | Radford et al. / OpenAI; [official checkpoint card](https://huggingface.co/openai/clip-vit-base-patch32), [CLIP project](https://github.com/openai/CLIP) | The upstream CLIP reference repository has an [**MIT license**](https://github.com/openai/CLIP/blob/main/LICENSE). The official checkpoint card supplies checkpoint provenance and intended-use information; no separate weight-specific license declaration was located in the inspected card. The Transformers software license is not substituted for checkpoint terms. |
| **OpenAI CLIP ViT-B/16**, `openai/clip-vit-base-patch16` — strong student | Radford et al. / OpenAI; [official checkpoint card](https://huggingface.co/openai/clip-vit-base-patch16), [CLIP project](https://github.com/openai/CLIP) | Same upstream reference-repository **MIT** notice and checkpoint-card scope as ViT-B/32. The checkpoint identifier does not imply a recorded immutable revision. |
| **CLIP_benchmark English class names**, `clip_benchmark/datasets/en_classnames.json` — ImageNet-1K and Flowers labels | Mehdi Cherti / CLIP_benchmark contributors; [metadata source](https://github.com/LAION-AI/CLIP_benchmark/blob/main/clip_benchmark/datasets/en_classnames.json) | [**MIT**](https://github.com/LAION-AI/CLIP_benchmark/blob/main/LICENSE), copyright 2022 Mehdi Cherti. This notice covers repository metadata, separately from the image datasets. |

The CLIP checkpoints, tokenizer and processor are loaded through Hugging Face Transformers. The upstream OpenAI reference implementation is credited for provenance; it is not presented as the runtime package imported by the training loader.

## Software dependencies

The table identifies direct projects used by the training/evaluation and plotting code, plus the conditional logging SDK. Follow the linked upstream license and incorporated-software notices for the installed version. It does not replace an environment lockfile or a package's complete bundled notices.

| Project | License / upstream notice |
| --- | --- |
| Python interpreter and standard library | [PSF License Version 2, with incorporated-software notices](https://docs.python.org/3/license.html) |
| PyTorch | [BSD-3-Clause and bundled notices](https://github.com/pytorch/pytorch/blob/main/LICENSE) |
| torchvision | [BSD-3-Clause](https://github.com/pytorch/vision/blob/main/LICENSE) |
| Hugging Face Transformers | [Apache-2.0](https://github.com/huggingface/transformers/blob/main/LICENSE) |
| Hugging Face Datasets | [Apache-2.0](https://github.com/huggingface/datasets/blob/main/LICENSE) |
| NumPy | [BSD-3-Clause and bundled notices](https://github.com/numpy/numpy/blob/main/LICENSE.txt) |
| Pillow / PIL | [Pillow license, currently named MIT-CMU upstream](https://github.com/python-pillow/Pillow/blob/main/LICENSE); historical version metadata may use HPND |
| COCO API / pycocotools | [BSD-2-Clause](https://github.com/cocodataset/cocoapi/blob/master/license.txt) |
| tqdm | [MPL-2.0 and MIT per-file/contributor notices](https://github.com/tqdm/tqdm/blob/master/LICENCE) |
| Matplotlib | [Matplotlib's PSF-based license and bundled notices](https://github.com/matplotlib/matplotlib/blob/main/LICENSE/LICENSE) |
| Python Fire | [Apache-2.0](https://github.com/google/python-fire/blob/master/LICENSE) |
| Weights & Biases Python SDK, when logging is enabled | [MIT](https://github.com/wandb/wandb/blob/main/LICENSE); hosted-service terms are separate |

Version identifiers are reported where established. For the permutation-stability diagnostic, recorded versions are Python 3.11.5, NumPy 1.26.4 and PyTorch 2.9.1+cu128; these are not asserted as the environment of every historical experiment.

## Obtaining and reusing third-party resources

Use the providers' official acquisition routes and retain their applicable acknowledgements, license, copyright and usage notices. Preserve upstream LICENSE/NOTICE material when copying covered code or metadata. This repository's documentation supplies attribution and links; it does not grant new rights in third-party resources or redistribute their image archives or pretrained checkpoints.

Rows marked as having no established named license describe the inspected source information, without asserting that use is prohibited or that unrestricted permission has been granted. Consult the original provider's current terms for the intended use.
