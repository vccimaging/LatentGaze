# Usage guide

[← Project overview](../README.md)

Follow the [installation steps](../README.md#getting-started), then run these commands
from the repository root.

<a id="models-and-results"></a>

## Pretrained weights

Download the [weights](https://drive.google.com/drive/folders/1nZYi9PGOghLQ0b_YegX2uc5saHjukf-1?usp=sharing)
into `weights/`. For prototype measurements and calibration, only `masks_decoder.pt` is needed.

| File | Model |
|:---|:---|
| `masks_decoder.pt` | Optical masks and measurement decoder |
| `inversion_teacher.pt` | Grayscale inversion teacher |
| `gaze_estimator_gray.pt` | Grayscale gaze estimator |
| `stylegan2_gray256.pkl` | Grayscale eye generator |
| `steering_compact_gray.pt` | Grayscale gaze steering |
| `stylegan2_rgb512.pkl` | Colour eye generator |
| `inversion_wplus_rgb.pt` | Colour inversion encoder |
| `steering_wplus_identity_rgb.pt` | Colour gaze steering |
| `gaze_estimator_rgb_resnet18.pt` | Colour gaze estimator |

<a id="measurement-to-gaze-pipeline"></a>

## Load the decoder

Predict gaze from the included prototype measurements:

```python
import torch
from latentgaze.data.human import load_subject
from latentgaze.models import load_decoder

decoder = load_decoder("weights/masks_decoder.pt").eval()
measurements, _ = load_subject("data/human/human_data_003.npz")

with torch.no_grad():
    latent, gaze, validity = decoder(measurements)

print(gaze[:5])
```

Each input row contains 16 measurements. The decoder returns a latent representation,
a 3D gaze vector and an eye-validity logit.

<a id="reproducibility-of-the-headline-results"></a>

## Calibration

Calibrate on the included recordings:

```bash
python scripts/calibrate.py --config configs/calibration.yaml 'budgets=[12]'
```

Outputs are written to `runs/calibration/`. To use your own recordings:

```bash
python scripts/calibrate.py --config configs/calibration.yaml \
    data.root=/path/to/recordings 'budgets=[12]'
```

Name each file `human_data_XXX.npz`, with arrays `measures` (N × 16) and `vectors` (N × 3).

## Evaluation

Build the test LMDB using [Data preparation](#data-preparation), then run:

```bash
python scripts/evaluate.py masks_decoder weights/masks_decoder.pt \
    --test-data data/lmdb/test_lmdb
```

The script also accepts `inversion` or `gaze_estimator` with the corresponding checkpoint.
Use `--save-predictions predictions.csv` to save per-sample predictions.

### Colour gaze steering

```bash
python scripts/eval_steering.py --crops data/crops --split data/splits/test.csv \
    --generator weights/stylegan2_rgb512.pkl \
    --inversion weights/inversion_wplus_rgb.pt \
    --controller weights/steering_wplus_identity_rgb.pt \
    --gaze-estimator weights/gaze_estimator_rgb_resnet18.pt
```

## Augmentation environment

Use a separate environment with a recent Python/PyTorch installation, `diffusers>=0.36`,
`transformers>=5.0`, `accelerate`, `sentencepiece`, `protobuf`, OpenCV, Pillow and NumPy.
This step needs access to the FLUX.1-dev weights and a GPU with about 40 GB of memory.
The main environment's `constraints.txt` does not apply here.

Run on a raw right-eye crop, with the inner eye corner on the right:

```bash
python scripts/augment_example.py --image eye_crop.png --out augmented.png
```

The output shows the input, eye mask, Canny control and generated image.

## Data preparation

Obtain ETH-XGaze from the [dataset provider](https://ait.ethz.ch/xgaze).
Use the subjects listed in `data/splits/train.csv` and `data/splits/test.csv`.

<a id="release-status"></a>

Prepare normalized eye crops of 576 × 576 pixels covering 45 × 45 mm. The crop-extraction
script is not included. Arrange the crops and gaze labels as:

```text
data/crops/<subject>/<frame>/<camera>_{r,l}.JPG
data/crops/<subject>.csv
```

CSV columns: `frame,camera.JPG,gr_x,gr_y,gr_z,gl_x,gl_y,gl_z,valid_r,valid_l`.
Use the two frontal views (`cam00`, `cam01`). The LMDB builder mirrors left-eye crops
and matches subject-folder names by their `subjectXXXX` prefix.

**Build the training and test LMDBs:**

```bash
python scripts/data/build_lmdb.py --data-root data/crops \
    --split data/splits/train.csv --output data/lmdb/train_lmdb
python scripts/data/build_lmdb.py --data-root data/crops \
    --split data/splits/test.csv --output data/lmdb/test_lmdb
```

**Augment each training subject** using the [augmentation environment](#augmentation-environment):

```bash
for s in $(tail -n +2 data/splits/train.csv | cut -d, -f1 | uniq); do
    python scripts/data/augment_lmdb.py --crops data/crops \
        --split data/splits/train.csv --subject "$s" \
        --out "data/lmdb/flux/${s}_aug.lmdb"
done
```

**Merge the training data:**

```bash
python scripts/data/merge_lmdbs.py --out data/lmdb/flux_lmdb \
    --tag-subject data/lmdb/flux/*_aug.lmdb
python scripts/data/merge_lmdbs.py --out data/lmdb/train_combined_lmdb \
    data/lmdb/flux_lmdb data/lmdb/train_lmdb
```

## Training

### Generators

Build the StyleGAN datasets and launch training:

```bash
python third_party/stylegan2_ada_pytorch/dataset_tool.py \
    --source=data/lmdb/flux_lmdb --dest=data/stylegan/gaze_eth_augmented.zip \
    --width=288 --height=288
python third_party/stylegan2_ada_pytorch/dataset_tool.py \
    --source=data/lmdb/flux_lmdb --dest=data/stylegan/gaze_eth_augmented_rgb_512.zip \
    --width=512 --height=512
bash scripts/train_stylegan.sh gray
bash scripts/train_stylegan.sh rgb
```

Use `DATA_ZIP=<path>` to supply a different dataset. StyleGAN2's CUDA kernels require `nvcc`.

### Models

Train in this order: `gaze_estimator_gray`, `gaze_estimator_rgb`, `inversion_gray`,
`inversion_rgb`, `masks_decoder`, `steering_gray`, `steering_rgb`.

| Model | Script | Configuration |
|:---|:---|:---|
| Gaze estimators | `scripts/train_gaze_estimator.py` | `gaze_estimator_gray.yaml` / `gaze_estimator_rgb.yaml` |
| Inversion encoders | `scripts/train_inversion.py` | `inversion_gray.yaml` / `inversion_rgb.yaml` |
| Masks and decoder | `scripts/train_masks.py` | `masks_decoder.yaml` |
| Gaze steering | `scripts/train_steering.py` | `steering_gray.yaml` / `steering_rgb.yaml` |

Pass a config from `configs/`, with optional overrides:

```bash
python scripts/train_masks.py --config configs/masks_decoder.yaml
python scripts/train_masks.py --config configs/masks_decoder.yaml train.max_iters=1000
```

Checkpoints are written to `runs/<config name>/`. Use `best.pt` for the gaze, inversion
and mask models, and the last `step_XXXXXX.pt` for steering. StyleGAN snapshots are in
`runs/stylegan/<run>/network-snapshot-*.pkl`.

Configs use the pretrained files in `weights/` by default. To use your own checkpoints,
override `generator`, `gaze_estimator`, `inversion`, `teacher` or `init_checkpoint` as needed:

```bash
python scripts/train_masks.py --config configs/masks_decoder.yaml \
    teacher=runs/inversion_gray/best.pt
```
