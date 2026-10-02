"""Write the downscaled training images and boxes in the Ultralytics YOLO format, plus train/val lists per fold.

Usage: python scripts/make_yolo_dataset.py [--size 256]

Writes yolo_{size}/images/<patient_id>.png (grayscale, from npy_{size}/train.npy), yolo_{size}/labels/<patient_id>.txt
(one line per box: class 0, x center, y center, width, height, all divided by the image size; no file for images
without boxes, which YOLO then uses as background), and fold{k}_train.txt / fold{k}_val.txt / fold{k}.yaml for every fold.
Test images are not written: train_yolo.py predicts them straight from npy_{size}/test.npy.
"""
import argparse

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from rsna.metrics import BOX_COLS
from rsna.paths import load_paths


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--size", type=int, default=256)
    args = parser.parse_args()

    processed = load_paths()["processed_data_dir"]
    out = processed / f"yolo_{args.size}"
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(exist_ok=True)
    patients = pd.read_csv(processed / "patients.csv")
    boxes = pd.read_csv(processed / "boxes.csv")
    images = np.load(processed / f"npy_{args.size}" / "train.npy", mmap_mode="r")

    for pid, image in tqdm(zip(patients["patient_id"], images), total=len(patients), desc="images"):
        cv2.imwrite(str(out / "images" / f"{pid}.png"), np.asarray(image))
    for pid, b in boxes.groupby("patient_id"):
        x, y, w, h = (b[BOX_COLS].to_numpy(float) / 1024).T  # boxes.csv is in 1024 px
        lines = [f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}" for xc, yc, bw, bh in zip(x + w / 2, y + h / 2, w, h)]
        (out / "labels" / f"{pid}.txt").write_text("\n".join(lines) + "\n")

    for fold in sorted(patients["fold"].unique()):
        for split, rows in [("train", patients["fold"] != fold), ("val", patients["fold"] == fold)]:
            paths = [str(out / "images" / f"{pid}.png") for pid in patients.loc[rows, "patient_id"]]
            (out / f"fold{fold}_{split}.txt").write_text("\n".join(paths) + "\n")
        (out / f"fold{fold}.yaml").write_text(f"path: {out.as_posix()}\ntrain: fold{fold}_train.txt\n"
                                              f"val: fold{fold}_val.txt\nnames:\n  0: opacity\n")
    print(f"{len(patients)} images, {boxes['patient_id'].nunique()} with boxes -> {out}")


if __name__ == "__main__":
    main()
