r"""FLUX identity augmentation of the training crops of one subject into an LMDB (Sec. 3.4, Supp. C).

    python scripts/data/augment_lmdb.py --crops data/crops --split data/splits/train.csv --subject subject0000 \
        --out data/lmdb/flux/subject0000_aug.lmdb

Uses the cam00 right-eye crops of the subject's split rows. Every crop is stored as is; a valid crop is followed
by its augmented version (a prompt drawn from PROMPTS or None, which stores the crop again).
"""

import argparse
import csv
import os
import pickle
import random
import re
import sys

import lmdb
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from augment_example import LANCZOS, ORIENTATION_SUFFIX, generate, load_models, prepare

PROMPTS = [
    "raw photo, close-up of a person's face around the eye, very dark ebony skin on forehead and cheek, smooth skin texture, natural outdoor lighting, shot on DSLR",
    "raw photo, close-up of a person's face around the eye, rich dark brown African skin on forehead and brow area, well-defined eyebrow, natural lighting, shot on DSLR",
    "raw photo, close-up of a person's face around the eye, warm golden brown South Asian skin on cheek and forehead, thick dark eyebrow, visible pores, natural lighting, shot on DSLR",
    "raw photo, close-up of a person's face around the eye, medium olive Mediterranean skin on forehead and cheek, natural texture, soft diffused lighting, shot on DSLR",
    "raw photo, close-up of a person's face around the eye, light caramel Latin skin on cheek and temple area, natural lighting, shot on DSLR",
    "raw photo, close-up of a person's face around the eye, smooth East Asian skin on forehead and cheek, soft natural lighting, shot on DSLR",
    "raw photo, close-up of a person's face around the eye, olive Middle Eastern skin on forehead, bushy dark eyebrow, visible pores, natural lighting, shot on DSLR",
    "raw photo, close-up of a person's face around the eye, fair Nordic skin on cheek and brow area, light blonde eyebrow, faint freckles, natural lighting, shot on DSLR",
    "raw photo, close-up of a person's face around the eye, Southeast Asian skin on cheek and brow area, thin arched eyebrow, smooth skin, natural lighting, shot on DSLR",
    "raw photo, close-up of a person's face around the eye, pale East European skin on cheek and temple, overcast diffused lighting, shot on DSLR",
    "raw photo, close-up of a person's face around the eye, thin metal rectangular glasses frame over the brow, fair skin on cheek, indoor lighting, shot on DSLR",
    "raw photo, close-up of a person's face around the eye, thick black plastic glasses frame resting on cheek and brow, dark skin, natural lighting, shot on DSLR",
    "raw photo, close-up of a person's face around the eye, tortoiseshell rectangular glasses frame over forehead, olive skin, natural lighting, shot on DSLR",
    "raw photo, close-up of a person's face around the eye, round wire-rim glasses frame, wrinkled skin on cheek, warm indoor lighting, shot on DSLR",
    "raw photo, close-up of a person's face around the eye, rimless frameless glasses barely visible near brow, pale skin, soft natural lighting, shot on DSLR",
    "raw photo, close-up of a person's face around the eye, half-rim reading glasses frame on upper cheek, elderly skin, warm indoor light, shot on DSLR",
    "raw photo, close-up of a person's face around the eye, large square plastic glasses frame on forehead and cheek, dark skin, natural lighting, shot on DSLR",
    "raw photo, close-up of a person's face around the eye, cat-eye glasses frame visible near brow, fair skin on cheek, indoor lighting, shot on DSLR",
    "raw photo, close-up of a person's face around the eye, natural no-makeup look on forehead and cheek, visible pores, fair skin, soft natural lighting, shot on DSLR",
    "raw photo, close-up of a person's face around the eye, groomed filled-in brow above the eye, warm skin tone on cheek, natural lighting, shot on DSLR",
    "raw photo, close-up of a person's face around the eye, subtle eyeliner along lash line, medium skin tone on cheek and brow, indoor lighting, shot on DSLR",
    "raw photo, close-up of a person's face around the eye, neutral eyeshadow on brow bone and upper cheek, caramel skin, soft lighting, shot on DSLR",
    "raw photo, close-up of a person's face around the eye, concealer and foundation on undereye cheek area, tired pale skin, indoor lighting, shot on DSLR",
    "raw photo, close-up of a person's face around the eye, dark circles and shadow on undereye cheek skin, tired pale skin, natural lighting, shot on DSLR",
    "raw photo, close-up of a person's face around the eye, puffiness on upper cheek below brow, fair aging skin, soft morning light, shot on DSLR",
    "raw photo, close-up of a person's face around the eye, freckled fair skin on forehead and cheek, sparse light brow, outdoor natural lighting, shot on DSLR",
    "raw photo, close-up of a person's face around the eye, oily shiny forehead and cheek skin, enlarged pores, medium skin tone, natural lighting, shot on DSLR",
    "raw photo, close-up of a person's face around the eye, mole above the brow on forehead, olive skin on cheek, natural lighting, shot on DSLR",
    "raw photo, close-up of a person's face around the eye, age spots on temple and cheek, elderly dark skin, natural lighting, shot on DSLR",
    "raw photo, close-up of a person's face around the eye, acne scarring on cheek near temple, medium skin, natural lighting, shot on DSLR",
    "raw photo, close-up of a person's face around the eye, thick bushy eyebrow on forehead, medium skin tone, natural lighting, shot on DSLR",
    "raw photo, close-up of a person's face around the eye, sparse thin eyebrow on forehead, aging cheek skin, indoor warm lighting, shot on DSLR",
    "raw photo, close-up of a person's face around the eye, crow's feet wrinkles on temple skin, pale cheek, natural lighting, shot on DSLR",
    "raw photo, close-up of a person's face around the eye, medium brown skin, soft diffused overcast daylight, shot on DSLR",
    "raw photo, close-up of a person's face around the eye, dark skin, warm indoor lamp light, shot on DSLR",
    "raw photo, close-up of a person's face around the eye, fair skin, bright midday sunlight, slight overexposure on cheek, shot on DSLR",
    "raw photo, close-up of a person's face around the eye, warm skin tone, golden hour natural light, shot on DSLR",
    "raw photo, close-up of a person's face around the eye, pale skin, cool blue-tinted indoor fluorescent light, shot on DSLR",
    "raw photo, close-up of a person's face around the eye, dark skin, side window light, strong shadow on one side, shot on DSLR",
    "raw photo, close-up of a person's face around the eye, caramel skin, bright even studio box light, shot on DSLR",
    "raw photo, close-up of a person's face around the eye, dark skin, direct flash photography, high contrast, shot on DSLR",
    "raw photo, close-up of a person's face around the eye, slight sweat on brow, flushed cheeks, outdoor exertion, medium skin, shot on DSLR",
    "raw photo, close-up of a person's face around the eye, rosy cold-flushed cheeks, winter outdoor lighting, fair skin, shot on DSLR",
    "raw photo, close-up of a person's face around the eye, tired expression, slight shadow under eye, medium skin, natural lighting, shot on DSLR",
    "raw photo, close-up of a person's face around the eye, hijab framing the face with shadow on temple, olive skin on cheek, soft indoor lighting, shot on DSLR",
    "raw photo, close-up of a person's face around the eye, thick bangs partially covering forehead above the brow, fair skin on cheek, natural lighting, shot on DSLR",
    "raw photo, close-up of an elderly Black man's face around the eye, very dark ebony skin, deep forehead wrinkles, coarse thick gray bushy eyebrow, age spots on temple, no glasses, warm indoor lamp light, visible pores, shot on DSLR",
    "raw photo, close-up of a young East Asian woman's face around the eye, smooth pale skin on forehead and cheek, thin arched groomed eyebrow, no glasses, no makeup, soft diffused overcast daylight, shot on DSLR",
    "raw photo, close-up of a middle-aged South Asian man's face around the eye, warm golden brown skin, thick dark bushy eyebrow, visible enlarged pores on cheek, wearing thin rectangular metal glasses, natural outdoor sunlight, shot on DSLR",
    "raw photo, close-up of an elderly Japanese woman's face around the eye, aged East Asian skin with fine wrinkles on temple, thin sparse gray eyebrow, wearing round wire-rim glasses, warm indoor window light, shot on DSLR",
    "raw photo, close-up of a young African woman's face around the eye, rich dark chocolate skin on forehead and cheek, well-groomed defined dark eyebrow, subtle kohl eyeliner, no glasses, golden hour warm light, shot on DSLR",
    "raw photo, close-up of a middle-aged Mediterranean man's face around the eye, olive tanned skin, thick dark unibrow connecting at nose bridge, crow's feet wrinkles on temple, no glasses, bright midday sunlight, shot on DSLR",
    "raw photo, close-up of a young Nordic woman's face around the eye, very fair pale skin, light blonde thin eyebrow, faint freckles on cheek, wearing rimless frameless glasses, cool blue-tinted fluorescent indoor light, shot on DSLR",
    "raw photo, close-up of an elderly Indian man's face around the eye, deep brown weathered skin on forehead, coarse bushy salt-and-pepper eyebrow, deep wrinkles on temple and cheek, wearing thick black plastic glasses, warm natural lighting, shot on DSLR",
    "raw photo, close-up of a young Middle Eastern woman's face around the eye, olive skin on forehead and cheek, dark well-defined arched eyebrow, subtle neutral eyeshadow, no glasses, warm golden natural light, shot on DSLR",
    "raw photo, close-up of an elderly West African woman's face around the eye, deep mahogany dark skin, sparse thin gray eyebrow, loose sagging skin on cheek and forehead, no glasses, side window light with strong shadow, shot on DSLR",
    "raw photo, close-up of a young Slavic man's face around the eye, pale East European skin on temple and cheek, thick straight dark eyebrow, slight stubble shadow near temple, no glasses, overcast soft lighting, shot on DSLR",
    "raw photo, close-up of a middle-aged Southeast Asian woman's face around the eye, warm tan skin on forehead and cheek, thin groomed arched eyebrow, light concealer under eye, wearing tortoiseshell rectangular glasses, bright even indoor lighting, shot on DSLR",
    "raw photo, close-up of a young mixed-race man's face around the eye, warm caramel skin with visible pores on forehead, thick well-groomed dark eyebrow, mole above the brow, no glasses, direct flash photography high contrast, shot on DSLR",
    "raw photo, close-up of an elderly Scandinavian woman's face around the eye, very pale translucent aging skin, thin white eyebrow, prominent crow's feet and forehead wrinkles, wearing half-rim reading glasses, warm desk lamp indoor light, shot on DSLR",
    "raw photo, close-up of a young Black woman's face around the eye, smooth rich dark skin on forehead and cheek, defined eyebrow, subtle brown eyeshadow on brow bone, wearing cat-eye glasses, bright even studio box light, shot on DSLR",
]
PROMPTS_WITH_NONE = [p + ORIENTATION_SUFFIX for p in PROMPTS] + [None]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--crops", required=True, help="folder with <subject>/ crop folders and <subject>.csv")
    ap.add_argument("--split", required=True, help="split list, data/splits/train.csv")
    ap.add_argument("--subject", required=True, help="subject id of the split, e.g. subject0000")
    ap.add_argument("--out", required=True, help="output LMDB")
    ap.add_argument("--seed", type=int, default=None, help="prompt sampling seed (default: the subject number)")
    ap.add_argument("--camera", default="cam00")
    ap.add_argument("--limit", type=int, default=None, help="process only the first N crops")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    seed = args.seed if args.seed is not None else int(re.search(r"subject(\d+)", args.subject).group(1))
    random.seed(seed)
    models = load_models(args.device)
    folder = next(n[:-4] for n in sorted(os.listdir(args.crops)) if n.endswith(".csv") and n[:-4].split("_")[0] == args.subject)
    with open(os.path.join(args.crops, folder + ".csv"), newline="") as f:
        gaze_r = {(r[0], r[1].split(".")[0]): r[2:5] for r in csv.reader(f) if len(r) >= 10}
    with open(args.split, newline="") as f:
        rows = [(r["frame"], r["camera"], int(r["valid_r"])) for r in csv.DictReader(f)
                if r["subject"] == args.subject and r["camera"] == args.camera][:args.limit]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    env = lmdb.open(args.out, map_size=1 << 40)
    n = 0
    with env.begin(write=True) as txn:
        def put(record):
            nonlocal n
            txn.put(str(n).encode(), pickle.dumps(record))
            n += 1

        for frame, camera, valid in rows:
            gaze = np.array(gaze_r[(frame, camera)], dtype=np.float32)
            crop = Image.open(os.path.join(args.crops, folder, frame, f"{camera}_r.JPG")).convert("RGB")
            base = {"image": np.array(crop), "label": gaze, "valid": valid}
            put({**base, "aug_idx": 0, "prompt": ""})
            if not valid:
                continue
            try:
                prep = prepare(models, crop)
            except ValueError:
                continue
            prompt = random.choices(PROMPTS_WITH_NONE)[0]
            if prompt is None:
                put({**base, "aug_idx": 1, "prompt": "none"})
                continue
            out = generate(models, prep, prompt, seed=n)
            if out["accepted"]:
                put({**base, "image": np.array(out["generated"].resize(crop.size, LANCZOS)), "aug_idx": 1,
                     "prompt": prompt, "clip_score": out["clip_score"], "eye_area_orig": int(prep["eye"].sum()),
                     "eye_area_gen": out["eye_area_gen"]})
            else:
                put({**base, "aug_idx": -1, "prompt": "fallback_original"})
    env.close()
    print(f"{args.subject}: {n} records in {args.out}")


if __name__ == "__main__":
    main()
