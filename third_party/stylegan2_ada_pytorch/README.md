# stylegan2-ada-pytorch (modified)

This is a vendored and modified copy of NVIDIA's
[stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch) at commit
`d72cc7d041b42ec8e806021a205ed9349f87c6a4`. It was used to train the two StyleGAN2 generators of
"Low Latency Gaze Tracking via Latent Optical Sensing" (arXiv 2605.17990): a grayscale 256 x 256 generator and an
RGB 512 x 512 generator, released as `weights/stylegan2_gray256.pkl` and `weights/stylegan2_rgb512.pkl`.

## What changed

The changes add what the two paper recipes need:

* Grayscale training data (`--grayscale`).
* A gaze-preserving augmentation (`--gaze_augment`): random +-16 px crops of 288 px eye images, a circular aperture and brightness/contrast jitter, with G and D running at 256 px. The matching FID protocol is also available in `calc_metrics.py`.
* Options for the latent width and the learning rate (`--zdim`, `--lr`).
* An LMDB reader in `dataset_tool.py`.
* `torch_utils/ops` files backported from StyleGAN3 so the code runs on PyTorch >= 1.11.

Without the new flags, `train.py` behaves like upstream. [MODIFICATIONS.md](MODIFICATIONS.md) lists every changed file
and the reason for each change.

## Usage

Training zips are built from an eye-patch LMDB with `dataset_tool.py`, and
[`scripts/train_stylegan.sh`](../../scripts/train_stylegan.sh) runs the two training recipes (its header lists the commands). The rest of this repository only needs
this directory to unpickle the generators: `latentgaze.models.generator` adds it to `sys.path` and calls
`legacy.load_network_pkl`.

The networks run their 4 highest resolutions in FP16. On a CPU, pass `force_fp32=True`, for example
`G.synthesis(ws, noise_mode='const', force_fp32=True)`.

## Requirements

Training uses the upstream requirements: Linux, NVIDIA GPUs, a CUDA toolkit and a C++ compiler for the custom
`bias_act`/`upfirdn2d` plugins, plus `click requests tqdm psutil scipy ninja` and optionally `pyspng`.
`dataset_tool.py` also needs `lmdb` for LMDB sources, and `opencv-python` only for LMDBs that store encoded images.
The paper models were trained with Python 3.9, PyTorch 1.12.1 (CUDA 11.3 build), the CUDA 11.8 toolkit and GCC 11.3.

## License

This directory is **not** covered by the CC BY-NC-SA 4.0 license of the rest of the repository. It is distributed under the
NVIDIA Source Code License for StyleGAN2 with ADA ([LICENSE.txt](LICENSE.txt), unchanged). The three
`torch_utils/ops` files taken from NVlabs/stylegan3 are under the NVIDIA Source Code License for StyleGAN3
([LICENSE-stylegan3.txt](LICENSE-stylegan3.txt)). Both licenses allow use only for non-commercial purposes, which
they define as research or evaluation. The modifications listed in MODIFICATIONS.md are distributed under the same
license.

Network pickles produced by this code embed its source code through `torch_utils.persistence`, so the released
`.pkl` generators are also subject to these terms.

Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
