"""Train an Ultralytics YOLO detector on one fold and score it with the competition metric, like train_detector.py.

Usage: python scripts/train_yolo.py --model yolo26s --batch-size 256
       python scripts/train_yolo.py --model yolo26s --batch-size 256 --probe   # 1 short epoch + validation, to measure memory

COCO-pretrained weights (downloaded by Ultralytics into models/yolo/), Ultralytics' default training and augmentation,
on the PNGs and labels from make_yolo_dataset.py at --size (no resizing). Ultralytics keeps the epoch with the best
validation mAP (best.pt) and the last epoch (last.pt). Both are scored with the competition metric at every box threshold,
as in train_detector.py, and the validation and test boxes (score >= 0.01, at most 20 per image) of the better one are saved.
"""
import argparse
import json
import os
import shutil
import time

import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

from rsna.metrics import BOX_COLS
from rsna.paths import load_paths
import train_detector
from train_detector import evaluate

# YOLO box scores are low (best threshold ~0.1 at 256 px): also try thresholds below 0.10
train_detector.THRESHOLDS = np.round(np.r_[np.arange(0.02, 0.10, 0.01), train_detector.THRESHOLDS], 2)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="Ultralytics model name, e.g. yolo26s, yolo11s, yolov8s")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--experiment", default="yolo_256", help="outputs/detection/<experiment>/<run name>")
    parser.add_argument("--probe", action="store_true", help="1 epoch on 10%% of the training images + full validation, then exit")
    return parser.parse_args()


def predict(model, images, size, batch_size):
    """Boxes of every image as a table (index into images, x, y, width, height, confidence) in 1024 px coordinates."""
    image, box, confidence = [], [], []
    for start in range(0, len(images), batch_size):
        x = [np.repeat(im[..., None], 3, axis=2) for im in images[start:start + batch_size]]
        for i, r in enumerate(model.predict(x, imgsz=size, conf=0.01, max_det=20, batch=len(x), verbose=False), start):
            image.append(np.full(len(r.boxes), i))
            box.append(r.boxes.xyxy.cpu().numpy())
            confidence.append(r.boxes.conf.cpu().numpy())
    b = np.concatenate(box) * (1024 / size)
    return pd.DataFrame({"image": np.concatenate(image), "x": b[:, 0], "y": b[:, 1], "width": b[:, 2] - b[:, 0],
                         "height": b[:, 3] - b[:, 1], "confidence": np.concatenate(confidence)})


def main():
    args = parse_args()
    run_name = f"{args.model}_f{args.fold}_s{args.seed}"
    paths = load_paths()
    processed = paths["processed_data_dir"]
    out_dir = paths["outputs_dir"] / "detection" / args.experiment / (run_name + ("_probe" if args.probe else ""))
    weights_dir = paths["models_dir"] / "yolo"
    weights_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(weights_dir)  # Ultralytics downloads weights (also the one for its AMP check) into the working directory
    free, total = torch.cuda.mem_get_info()
    others = total - free

    t0 = time.time()
    model = YOLO(str(weights_dir / f"{args.model}.pt"))
    model.train(data=str(processed / f"yolo_{args.size}" / f"fold{args.fold}.yaml"), imgsz=args.size,
                epochs=1 if args.probe else args.epochs, fraction=0.1 if args.probe else 1.0, batch=args.batch_size,
                workers=args.workers, seed=args.seed, project=str(out_dir.parent), name=out_dir.name, exist_ok=True,
                plots=False)
    peak = others + torch.cuda.max_memory_reserved()
    if args.probe:
        print(json.dumps({"run": run_name, "batch_size": args.batch_size, "peak_gpu_gib": round(peak / 2**30, 2),
                          "peak_gpu_frac": round(peak / total, 3), "minutes": round((time.time() - t0) / 60, 2)}))
        shutil.rmtree(out_dir)
        return
    train_minutes = (time.time() - t0) / 60

    patients = pd.read_csv(processed / "patients.csv")
    true = pd.read_csv(processed / "boxes.csv")
    val_idx = np.flatnonzero(patients["fold"] == args.fold)
    val_ids = patients["patient_id"].to_numpy()[val_idx]
    val_positive = patients["target"].to_numpy()[val_idx] == 1
    val_stage1 = val_ids[patients["is_stage1_test"].to_numpy()[val_idx]]
    images = np.load(processed / f"npy_{args.size}" / "train.npy", mmap_mode="r")
    val_images = np.asarray(images[val_idx])

    results = {}
    for which in ["best", "last"]:
        m = YOLO(str(out_dir / "weights" / f"{which}.pt"))
        pred = predict(m, val_images, args.size, args.batch_size)
        pred = pred.assign(patient_id=val_ids[pred.pop("image")])
        results[which] = (m, pred, evaluate(pred, true, val_ids, val_positive, val_stage1))
        print(which, " ".join(f"{k} {v:.4f}" for k, v in results[which][2].items()), flush=True)
    chosen = max(results, key=lambda w: results[w][2]["score"])
    m, val_pred, scores = results[chosen]
    val_pred[["patient_id", *BOX_COLS, "confidence"]].to_csv(out_dir / "val_preds.csv", index=False)
    test_ids = pd.read_csv(processed / "test.csv")["patient_id"].to_numpy()
    test_pred = predict(m, np.load(processed / f"npy_{args.size}" / "test.npy"), args.size, args.batch_size)
    test_pred.assign(patient_id=test_ids[test_pred.pop("image")])[["patient_id", *BOX_COLS, "confidence"]] \
        .to_csv(out_dir / "test_preds.csv", index=False)

    log = pd.read_csv(out_dir / "results.csv")
    best_epoch = int(log.loc[log["metrics/mAP50-95(B)"].idxmax(), "epoch"])
    summary = {"run": run_name, "checkpoint": chosen, "best_epoch": best_epoch if chosen == "best" else args.epochs,
               **scores, "score_best_pt": results["best"][2]["score"], "score_last_pt": results["last"][2]["score"],
               "batch_size": args.batch_size, "minutes": train_minutes, "peak_gpu_frac": peak / total}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"{chosen}.pt: score {scores['score']:.4f} at threshold {scores['threshold']}, auc {scores['auc']:.4f}")


if __name__ == "__main__":
    main()
