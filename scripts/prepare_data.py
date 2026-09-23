"""Convert all DICOMs to .npy and write patients.csv, boxes.csv and test.csv.

Usage: python scripts/prepare_data.py [--workers N]

Existing .npy files are skipped, so the script can be re-run after an interruption.
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
from tqdm import tqdm

from rsna.data.prepare import add_folds, build_boxes, build_patients, dicom_to_npy
from rsna.paths import load_paths


def convert_split(raw_dir, out_dir, split, workers):
    dcm_files = sorted((raw_dir / f"stage_2_{split}_images").glob("*.dcm"))
    npy_dir = out_dir / "npy_1024" / split
    npy_dir.mkdir(parents=True, exist_ok=True)
    npy_files = [npy_dir / f"{f.stem}.npy" for f in dcm_files]
    with ProcessPoolExecutor(workers) as pool:
        rows = list(tqdm(pool.map(dicom_to_npy, dcm_files, npy_files, chunksize=64), total=len(dcm_files), desc=split))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args()

    paths = load_paths()
    raw_dir, out_dir = paths["raw_data_dir"], paths["processed_data_dir"]

    train_meta = convert_split(raw_dir, out_dir, "train", args.workers)
    test_meta = convert_split(raw_dir, out_dir, "test", args.workers)

    labels = pd.read_csv(raw_dir / "stage_2_train_labels.csv")
    classes = pd.read_csv(raw_dir / "stage_2_detailed_class_info.csv")
    patients = add_folds(build_patients(labels, classes, train_meta))
    boxes = build_boxes(labels)

    patients.to_csv(out_dir / "patients.csv", index=False)
    boxes.to_csv(out_dir / "boxes.csv", index=False)
    test_meta.to_csv(out_dir / "test.csv", index=False)
    print(f"patients: {len(patients)} | boxes: {len(boxes)} | test: {len(test_meta)} -> {out_dir}")


if __name__ == "__main__":
    main()
