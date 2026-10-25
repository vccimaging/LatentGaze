"""Concatenate eye-patch LMDBs into one, re-keyed 0..N-1 (sources in the given order).

    # the per-subject FLUX LMDBs, tagged with their subject
    python scripts/data/merge_lmdbs.py --out data/lmdb/flux_lmdb --tag-subject data/lmdb/flux/*_aug.lmdb
    # FLUX patches followed by the real crops: the training set of the released models
    python scripts/data/merge_lmdbs.py --out data/lmdb/train_combined_lmdb data/lmdb/flux_lmdb data/lmdb/train_lmdb
"""

import argparse
import os
import pickle

import lmdb


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sources", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag-subject", action="store_true",
                    help="add subject_id = the source name without '_aug.lmdb' to every record")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    out = lmdb.open(args.out, map_size=1 << 41)
    n = 0
    for src in args.sources:
        env = lmdb.open(src, readonly=True, lock=False, readahead=False)
        subject = os.path.basename(os.path.normpath(src)).replace("_aug.lmdb", "")
        with env.begin() as txn, out.begin(write=True) as out_txn:
            for _, value in txn.cursor():  # LMDB key order, as for the released training set
                if args.tag_subject:
                    record = pickle.loads(value)
                    record["subject_id"] = subject
                    value = pickle.dumps(record)
                out_txn.put(str(n).encode(), value)
                n += 1
        env.close()
        print(f"{src}: {n} records so far")
    out.close()


if __name__ == "__main__":
    main()
