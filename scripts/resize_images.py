"""Downscale the NPY 1024 cache into one uint8 array per split, small enough to keep in memory during training.

Usage: python scripts/resize_images.py [--size 256] [--clahe] [--workers N]

Writes npy_{size}/train.npy (npy_{size}_clahe/ with --clahe) in the row order of patients.csv and npy_{size}/test.npy in the row order of test.csv.
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
import pandas as pd
from tqdm import tqdm

from rsna.data.prepare import load_resized
from rsna.paths import load_paths


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--clahe", action="store_true", help="apply CLAHE at 1024 px before downscaling")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args()

    processed = load_paths()["processed_data_dir"]
    out_dir = processed / (f"npy_{args.size}" + ("_clahe" if args.clahe else ""))
    out_dir.mkdir(exist_ok=True)
    for split, table in [("train", "patients.csv"), ("test", "test.csv")]:
        ids = pd.read_csv(processed / table)["patient_id"]
        files = [processed / "npy_1024" / split / f"{pid}.npy" for pid in ids]
        images = np.empty((len(files), args.size, args.size), dtype=np.uint8)
        with ProcessPoolExecutor(args.workers) as pool:
            load = partial(load_resized, size=args.size, clahe=args.clahe)
            for i, image in enumerate(tqdm(pool.map(load, files, chunksize=64), total=len(files), desc=split)):
                images[i] = image
        np.save(out_dir / f"{split}.npy", images)
        print(f"{split}: {images.shape} -> {out_dir / f'{split}.npy'}")


if __name__ == "__main__":
    main()
