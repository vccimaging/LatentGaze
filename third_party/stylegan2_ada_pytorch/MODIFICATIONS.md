# Modifications

This directory is NVIDIA's [stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch) at commit
`d72cc7d041b42ec8e806021a205ed9349f87c6a4`, modified to train the two generators of
"Low Latency Gaze Tracking via Latent Optical Sensing" (arXiv 2605.17990). Every file that is not listed below is
byte-identical to upstream.

## Modified files

| File | Change | Reason |
|---|---|---|
| `train.py` | New options `--grayscale`, `--gaze_augment`, `--lr` and `--zdim`. With `--gaze_augment`, `--cfg=auto` derives its hyper-parameters for 256 px, and datasets smaller than 288 px are rejected. `--zdim=N` sets `z_dim = w_dim = channel_max = N` and `conv_clamp = max(N // 2, 64)`. Without `--zdim` the networks are upstream (`512 / 512 / 512 / 256`). The paper recipes are added to `--help`. | Expresses the paper recipes as command-line options. |
| `training/dataset.py` | `ImageFolderDataset(grayscale=True)` converts RGB images to one channel as `(R+G+B)/3`, truncated to uint8. | Grayscale generator. |
| `training/training_loop.py` | New `gaze_augment` argument. When set, G and D are built at 256 px and the reals go through `training/gaze_augment.augment_reals`. `reals.png` is centre-cropped and masked, and the flag is passed on to the metrics. | Gaze-preserving training on 288 px data. |
| `metrics/metric_utils.py` | `MetricOptions(gaze_augment=...)`. When set, the reals are centre-cropped from 288 to 256 px and multiplied by the circular aperture (outside = 0), and the fakes get the same aperture (outside = -1). `gaze_augment` is part of the cache key for the real-image statistics. | The paper's FID protocol. The cache key is the same as in the original runs. |
| `calc_metrics.py` | New `--gaze_augment` option. By default it is read from `training_options.json` next to the pickle, and is false otherwise. When set, the dataset resolution is not forced to the G resolution. For a 1-channel G, `grayscale=True` is added to the dataset options. | Recomputes the paper FID on its own. |
| `dataset_tool.py` | Reads LMDBs of pickled dict records `{'image': uint8 HxWx3, ...}` in key order. An LMDB is recognised by the `_lmdb` suffix or by a `data.mdb` file. A size that is not a power of two prints a warning instead of an error. The bare `except:` of the LMDB loop becomes `except Exception:`, so stopping the generator early no longer raises "generator ignored GeneratorExit". | Builds the 288 px and 512 px zips from the FLUX-augmented LMDB. |
| `generate.py` | Writes 1-channel generators as `L` PNGs. Falls back to the CPU with `force_fp32=True`. | Supports the grayscale generator. |
| `torch_utils/ops/upfirdn2d.py` | `_init()` sets `_inited = True`, as `bias_act.py` does. | After a failed build, later calls loaded a stale cached plugin (possibly compiled for another GPU) instead of using the reference implementation. |
| `torch_utils/ops/conv2d_gradfix.py`, `torch_utils/ops/conv2d_resample.py`, `torch_utils/ops/grid_sample_gradfix.py` | Replaced by the versions from NVlabs/stylegan3 at commit `c233a91`, which are under the StyleGAN3 license (see `LICENSE-stylegan3.txt`). The `pkg_resources` version check is replaced by a parse of `torch.__version__`. `grid_sample_gradfix` accepts the `(op, overloads)` tuple that newer PyTorch returns. | Runs on PyTorch >= 1.11. The paper models were trained with 1.12.1. |

## Added files

| File | Purpose |
|---|---|
| `training/gaze_augment.py` | Aperture mask, centre crop and gaze-preserving augmentation of the reals. See the next section. |
| `README.md` | Replaces the upstream README. |
| `MODIFICATIONS.md` | This file. |
| `LICENSE-stylegan3.txt` | License of the three backported `torch_utils/ops` files. |

### Gaze-preserving augmentation (`--gaze_augment=1`)

The reals are stored at 288 x 288 px and the networks run at 256 x 256 px.

* **Training reals.** Each sample is cropped to a 256 x 256 window. The window's offset from the centred crop is drawn uniformly from {-16, ..., 16} px on each axis. A circular aperture is then applied (radius 128 px, centre at 127.5, outside set to -1). Next comes a per-sample brightness shift of `+U[-0.2, 0.2]` and a contrast factor of `x U[0.8, 1.2]` around the per-channel mean inside the aperture. The result is clamped to [-1, 1] and the aperture is applied again. There are no flips or rotations, so the gaze direction is preserved.
* **FID.** The reals are centre-cropped and masked, the fakes are masked, and there is no photometric jitter.

The code consumes the random number generator in the same order as the implementation used to train the released generators, and its outputs are bit-identical to it for both 1-channel and 3-channel batches.

## Removed from the vendored copy

`.github/`, `.gitignore`, `Dockerfile`, `docker_run.sh`, `docs/` (help texts, `license.html`, images), the upstream
`README.md`, and `projector.py` and `style_mixing.py` (not used here).
