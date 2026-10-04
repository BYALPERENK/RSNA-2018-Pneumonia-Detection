"""Horizontal-flip predictions of trained detectors, for flip test-time augmentation (TTA).

Usage: python scripts/predict_flip.py grid_256/fasterrcnn_convnext_tiny_f0_s0 cv_256/yolo26s_f1_s0 ...

Each run (a train_detector.py or train_yolo.py output directory) predicts its validation fold and the test images
flipped left-right, with its saved checkpoint (YOLO: the one in summary.json); the boxes are flipped back
(x -> 1024 - x - width) and saved next to the original ones as val_preds_flip.csv and test_preds_flip.csv
(score >= 0.01, at most 20 per image, as before). As a check, the validation images are also predicted without the
flip and scored, with the saved val_preds.csv and the flipped boxes, at the run's best threshold (from summary.json).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

from rsna.metrics import BOX_COLS, score_images
from rsna.models.detector import build_detector
from rsna.paths import load_paths
import train_detector
import train_yolo


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+", help="run directories relative to outputs/detection")
    parser.add_argument("--max-gpu-frac", type=float, default=0.85)
    return parser.parse_args()


def load_predictor(run_dir):
    """The run's input size, its validation fold and a function (uint8 images (N, size, size) on the CPU) -> box
    table, for its saved checkpoint."""
    summary = json.loads((run_dir / "summary.json").read_text())
    if "checkpoint" in summary:  # YOLO
        args = dict(line.split(": ", 1) for line in (run_dir / "args.yaml").read_text().splitlines() if ": " in line)
        size, batch = int(args["imgsz"]), int(args["batch"])
        fold = int(Path(args["data"]).stem.removeprefix("fold"))  # data: .../yolo_256/fold0.yaml
        model = YOLO(str(run_dir / "weights" / f"{summary['checkpoint']}.pt"))
        return size, fold, lambda images: train_yolo.predict(model, images, size, batch)
    config = json.loads((run_dir / "config.json").read_text())
    model, in_chans = build_detector(config["arch"], config["backbone"], config["size"], pretrained=False)
    model.load_state_dict(torch.load(run_dir / "best.pt"))
    model.cuda()
    return config["size"], config["fold"], lambda images: train_detector.predict(
        model, torch.from_numpy(images).cuda(), config["batch_size"], in_chans, 1024 / config["size"])


def table(pred, ids, flipped):
    pred = pred.assign(patient_id=ids[pred.pop("image")])
    if flipped:
        pred["x"] = 1024 - pred["x"] - pred["width"]
    return pred[["patient_id", *BOX_COLS, "confidence"]]


def main():
    args = parse_args()
    paths = load_paths()
    processed = paths["processed_data_dir"]
    torch.cuda.init()
    free, total = torch.cuda.mem_get_info()
    torch.cuda.set_per_process_memory_fraction(args.max_gpu_frac - (total - free) / total)
    patients = pd.read_csv(processed / "patients.csv")
    true = pd.read_csv(processed / "boxes.csv")
    test_ids = pd.read_csv(processed / "test.csv")["patient_id"].to_numpy()

    for run in args.runs:
        run_dir = paths["outputs_dir"] / "detection" / run
        size, fold, predict = load_predictor(run_dir)
        train = np.load(processed / f"npy_{size}" / "train.npy", mmap_mode="r")
        val_idx = np.flatnonzero(patients["fold"] == fold)
        val_ids = patients["patient_id"].to_numpy()[val_idx]
        val_images = np.ascontiguousarray(train[val_idx])
        test_images = np.load(processed / f"npy_{size}" / "test.npy")

        threshold = json.loads((run_dir / "summary.json").read_text())["threshold"]
        score = lambda p: score_images(true, p[p["confidence"] >= threshold], val_ids).mean()  # noqa: E731
        check = table(predict(val_images), val_ids, False)
        saved = pd.read_csv(run_dir / "val_preds.csv")
        flip = lambda images: np.ascontiguousarray(images[:, :, ::-1])  # noqa: E731
        flipped = table(predict(flip(val_images)), val_ids, True)
        flipped.to_csv(run_dir / "val_preds_flip.csv", index=False)
        table(predict(flip(test_images)), test_ids, True).to_csv(run_dir / "test_preds_flip.csv", index=False)
        print(f"{run} at threshold {threshold}: saved {score(saved):.4f} ({len(saved)} boxes), predicted again "
              f"{score(check):.4f} ({len(check)}), flipped {score(flipped):.4f} ({len(flipped)})", flush=True)
        del predict
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
