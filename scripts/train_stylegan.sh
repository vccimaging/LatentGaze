#!/usr/bin/env bash
# Train the StyleGAN2-ADA generators (Supp. F.1) with third_party/stylegan2_ada_pytorch.
#
#   bash scripts/train_stylegan.sh gray    # grayscale 256 px, 1 GPU
#   bash scripts/train_stylegan.sh rgb     # RGB 512 px, 2 GPUs
#
# DATA_ZIP is built from data/lmdb/flux_lmdb with third_party/stylegan2_ada_pytorch/dataset_tool.py
# (--width=288 --height=288 for gray, 512 for rgb). Optional: OUTDIR, PYTHON_BIN, EXTRA_ARGS.

set -euo pipefail

RECIPE="${1:-}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STYLEGAN_DIR="$REPO_ROOT/third_party/stylegan2_ada_pytorch"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTDIR="${OUTDIR:-runs/stylegan}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

case "$RECIPE" in
    gray)
        DATA_ZIP="${DATA_ZIP:-data/stylegan/gaze_eth_augmented.zip}"
        RECIPE_ARGS=(--gpus=1 --zdim=256 --aug=noaug --grayscale=true --gaze_augment=true)
        ;;
    rgb)
        DATA_ZIP="${DATA_ZIP:-data/stylegan/gaze_eth_augmented_rgb_512.zip}"
        RECIPE_ARGS=(--gpus=2 --zdim=512 --aug=noaug --allow-tf32=true --workers=4 --lr=0.001)
        ;;
    *)
        echo "usage: $0 gray|rgb" >&2
        exit 1
        ;;
esac

# shellcheck disable=SC2086  # EXTRA_ARGS is intentionally word-split
"$PYTHON_BIN" "$STYLEGAN_DIR/train.py" --outdir="$OUTDIR" --data="$DATA_ZIP" "${RECIPE_ARGS[@]}" \
    --snap=50 --metrics=fid50k_full $EXTRA_ARGS
