r"""Pack the ETH-XGaze eye crops of a split into an LMDB of eye patches (Supp. B).

    python scripts/data/build_lmdb.py --data-root data/crops --split data/splits/train.csv --output data/lmdb/train_lmdb
    python scripts/data/build_lmdb.py --data-root data/crops --split data/splits/test.csv --output data/lmdb/test_lmdb

The split lists the subjects, frames and cameras (cam00, cam01) and the validity of each eye; the crops are
<data-root>/<subject>/<frame>/<cam>_{r,l}.JPG with the gaze labels in <data-root>/<subject>.csv.
"""

import argparse
import csv
import os
import pickle
import re
import sys

import lmdb
import numpy as np
from PIL import Image


def natural_key(text):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", text)]


def read_rows(csv_path):
    """CSV rows as (frame, camera, image extension, gaze_r, gaze_l, valid_r, valid_l), sorted."""
    rows = []
    with open(csv_path) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 10 or parts[0] == "frame":
                continue
            camera, ext = os.path.splitext(parts[1])
            gaze = np.array([float(v) for v in parts[2:8]], dtype=np.float32)
            rows.append((parts[0], camera, ext, gaze[:3], gaze[3:], int(parts[8]), int(parts[9])))
    return sorted(rows, key=lambda r: (natural_key(r[0]), natural_key(r[1])))


def load_patch(path, size, flip):
    image = Image.open(path).convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    if flip:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    return np.asarray(image)


def read_split(path):
    """{subject: [(frame, camera, valid_r, valid_l), ...]} of a split list."""
    split = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            split.setdefault(r["subject"], []).append((r["frame"], r["camera"], int(r["valid_r"]), int(r["valid_l"])))
    return split


def subject_records(data_root, subject, rows, size):
    """Yield the records of one subject's split rows; gaze labels come from the subject's crop CSV."""
    folder = next(n[:-4] for n in sorted(os.listdir(data_root)) if n.endswith(".csv") and n[:-4].split("_")[0] == subject)
    labels = {(r[0], r[1]): r for r in read_rows(os.path.join(data_root, folder + ".csv"))}
    for frame, camera, valid_r, valid_l in sorted(rows, key=lambda r: (natural_key(r[0]), natural_key(r[1]))):
        _, _, ext, gaze_r, gaze_l, _, _ = labels[(frame, camera)]
        eyes = (("r", gaze_r, valid_r), ("l", gaze_l * np.float32([-1, 1, 1]), valid_l))
        for eye, label, valid in eyes:
            path = os.path.join(data_root, folder, frame, f"{camera}_{eye}{ext}")
            yield {"image": load_patch(path, size, flip=eye == "l"), "label": label, "valid": valid,
                   "subject": subject, "frame": frame, "camera": camera, "eye": eye}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True, help="folder with <subject>/ crop folders and <subject>.csv")
    ap.add_argument("--output", required=True, help="output LMDB directory")
    ap.add_argument("--split", required=True, help="split list, data/splits/train.csv or test.csv")
    ap.add_argument("--size", type=int, default=288, help="patch size in pixels (default 288)")
    ap.add_argument("--map-size-gb", type=float, default=1024)
    ap.add_argument("--commit-every", type=int, default=1000)
    args = ap.parse_args()

    split = read_split(args.split)
    subjects = sorted(split)
    if os.path.exists(os.path.join(args.output, "data.mdb")):
        sys.exit(f"{args.output} already exists")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    env = lmdb.open(args.output, map_size=int(args.map_size_gb * 2 ** 30))
    txn = env.begin(write=True)
    n = n_valid = 0
    for subject in subjects:
        for record in subject_records(args.data_root, subject, split[subject], args.size):
            txn.put(str(n).encode(), pickle.dumps(record, protocol=4))
            n += 1
            n_valid += int(record["valid"])
            if n % args.commit_every == 0:
                txn.commit()
                txn = env.begin(write=True)
        print(f"{subject}: {n} records so far")
    txn.commit()
    env.close()
    print(f"{args.output}: {n} records ({n_valid} valid) from {len(subjects)} subjects")


if __name__ == "__main__":
    main()
