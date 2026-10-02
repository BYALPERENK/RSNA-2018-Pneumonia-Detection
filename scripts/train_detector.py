"""Train a Lung Opacity box detector on one fold and save per-epoch validation scores and box predictions.

Usage: python scripts/train_detector.py --arch retinanet --backbone convnext_tiny.fb_in22k_ft_in1k --fold 0 --batch-size 64
       python scripts/train_detector.py ... --probe   # a few steps only, to measure memory and speed

PyTorch's allocator is capped at --max-gpu-frac of the GPU minus what other programs use at start, so it frees its
cache or fails instead of spilling into shared memory (which is silent and very slow on Windows). The batch sizes are
chosen for a peak of ~76% (measured with --probe), so the cap is only a safety margin.

All training images are used, including those without boxes (empty targets), so the detector also learns to
predict nothing. Images stay on the GPU as uint8 (npy_{size}/train.npy) and are augmented there; the boxes are
moved with the same affine matrix (no rotation, so they stay exact). The learning rate is base_lr * sqrt(batch size / 16),
warms up linearly and then follows a cosine to zero.

Validation, each epoch, with the competition metric (src/rsna/metrics.py) on the 1024 px boxes: boxes below a score
threshold are dropped, and the threshold with the best score is reported. The epoch with the best score is kept;
its validation and test boxes (score >= 0.01, at most 20 per image) are saved.
The image score is the highest box score (0 without boxes); its AUC, and its F1 at the best threshold, are logged too.
"""
import argparse
import json
import math
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, roc_auc_score

from rsna.data.augment import augment, to_float, transform_boxes
from rsna.metrics import BOX_COLS, score_images
from rsna.models.detector import ARCHS, build_detector
from rsna.paths import load_paths

AUGMENTATION = dict(max_rotate=0, scale=(0.85, 1.15), max_shift=0.08, brightness=0.15, contrast=(0.85, 1.15))
THRESHOLDS = np.round(np.arange(0.10, 0.91, 0.05), 2)  # box score thresholds tried on the validation fold


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arch", choices=ARCHS, required=True)
    parser.add_argument("--backbone", required=True, help="timm model name with pretrained tag, or coco")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--base-lr", type=float, default=1e-4, help="learning rate at batch size 16")
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-epochs", type=float, default=1.0)
    parser.add_argument("--drop-path", type=float, default=0.1)
    parser.add_argument("--experiment", default="grid_256", help="outputs/detection/<experiment>/<run name>")
    parser.add_argument("--max-gpu-frac", type=float, default=0.85, help="cap on the whole GPU's memory use (other programs at start + this process)")
    parser.add_argument("--probe", action="store_true", help="train and predict a few batches, print peak memory and speed, exit")
    return parser.parse_args()


def predict(model, images, batch_size, in_chans, scale):
    """Boxes of every image as a table (index into images, x, y, width, height, confidence) in 1024 px coordinates."""
    model.eval()
    image, box, confidence = [], [], []
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            x = to_float(images[start:start + batch_size]).expand(-1, in_chans, -1, -1)
            for i, out in enumerate(model(list(x)), start):
                image.append(torch.full((len(out["boxes"]),), i))
                box.append(out["boxes"].float().cpu())
                confidence.append(out["scores"].float().cpu())
    b = torch.cat(box).numpy() * scale
    return pd.DataFrame({"image": torch.cat(image).numpy(), "x": b[:, 0], "y": b[:, 1], "width": b[:, 2] - b[:, 0],
                         "height": b[:, 3] - b[:, 1], "confidence": torch.cat(confidence).numpy()})


def evaluate(pred, true, ids, positive, stage1):
    """Competition score at every threshold; at the best one also the stage-1 score and the image-level F1, plus AUC."""
    scores = {t: score_images(true, pred[pred["confidence"] >= t], ids) for t in THRESHOLDS}
    best_t = max(scores, key=lambda t: scores[t].mean())
    image_score = pred.groupby("patient_id")["confidence"].max().reindex(ids, fill_value=0).to_numpy()
    return {"score": scores[best_t].mean(), "threshold": best_t, "score_stage1": scores[best_t].loc[stage1].mean(),
            "auc": roc_auc_score(positive, image_score), "f1": f1_score(positive, image_score >= best_t),
            "pos_rate": (image_score >= best_t).mean()}


def main():
    args = parse_args()
    run_name = f"{args.arch}_{args.backbone.split('.')[0]}_f{args.fold}_s{args.seed}"
    paths = load_paths()
    out_dir = paths["outputs_dir"] / "detection" / args.experiment / run_name
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.cuda.init()
    free, total = torch.cuda.mem_get_info()
    others = total - free  # other programs + this process's CUDA context
    torch.cuda.set_per_process_memory_fraction(args.max_gpu_frac - others / total)

    processed = paths["processed_data_dir"]
    patients = pd.read_csv(processed / "patients.csv")
    true = pd.read_csv(processed / "boxes.csv")
    images = torch.from_numpy(np.load(processed / f"npy_{args.size}" / "train.npy")).cuda()
    scale = 1024 / args.size
    xyxy = true[BOX_COLS].to_numpy(np.float32) / scale
    xyxy[:, 2:] += xyxy[:, :2]
    rows = true.groupby("patient_id").indices
    boxes = [torch.from_numpy(xyxy[rows[p]]).cuda() if p in rows else torch.zeros(0, 4, device="cuda")
             for p in patients["patient_id"]]
    train_idx = np.flatnonzero(patients["fold"] != args.fold)
    val_idx = np.flatnonzero(patients["fold"] == args.fold)
    val_ids = patients["patient_id"].to_numpy()[val_idx]
    val_positive = patients["target"].to_numpy()[val_idx] == 1
    val_stage1 = val_ids[patients["is_stage1_test"].to_numpy()[val_idx]]

    model, in_chans = build_detector(args.arch, args.backbone, args.size, drop_path=args.drop_path)
    model.cuda()
    lr = args.base_lr * math.sqrt(args.batch_size / 16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    steps_per_epoch = len(train_idx) // args.batch_size  # the last incomplete batch is dropped
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = round(args.warmup_epochs * steps_per_epoch)
    step = 0

    def set_lr():
        if step < warmup_steps:
            factor = (step + 1) / warmup_steps
        else:
            factor = 0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / max(1, total_steps - warmup_steps)))
        for group in optimizer.param_groups:
            group["lr"] = lr * factor

    order_rng = np.random.default_rng(args.seed)  # separate random streams: the data order and augmentation
    aug_gen = torch.Generator(device="cuda").manual_seed(args.seed)  # do not depend on the model's initialization

    def train_step(batch_idx):
        nonlocal step
        set_lr()
        x, theta = augment(images[torch.from_numpy(batch_idx).cuda()], aug_gen, **AUGMENTATION, return_theta=True)
        targets = []
        for i, t in zip(batch_idx, theta):
            b = transform_boxes(boxes[i], t, args.size) if len(boxes[i]) else boxes[i]
            b = b[((b[:, 2] - b[:, 0]) >= 2) & ((b[:, 3] - b[:, 1]) >= 2)]  # drop boxes pushed (almost) out of the image
            targets.append({"boxes": b, "labels": torch.ones(len(b), dtype=torch.int64, device="cuda")})
        losses = model(list(x.expand(-1, in_chans, -1, -1)), targets)
        loss = sum(losses.values())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        step += 1
        return loss.detach()

    if args.probe:  # training, prediction, then training again, as in an epoch; peak = others + PyTorch's peak
        for rounds in range(2):
            model.train()
            batches = order_rng.permutation(train_idx)[:args.batch_size * 10].reshape(10, -1)
            for i, batch_idx in enumerate(batches):
                if i == 3:
                    torch.cuda.synchronize()
                    start = time.time()
                train_step(batch_idx)
            torch.cuda.synchronize()
            speed = 7 * args.batch_size / (time.time() - start)
            predict(model, images[val_idx[:args.batch_size * 4]], args.batch_size, in_chans, scale)
        peak = others + torch.cuda.max_memory_reserved()
        print(json.dumps({"run": run_name, "batch_size": args.batch_size, "peak_gpu_gib": round(peak / 2**30, 2),
                          "peak_gpu_frac": round(peak / total, 3), "train_img_per_s": round(speed)}))
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args) | {"lr": lr, "steps_per_epoch": steps_per_epoch, "n_train": len(train_idx), "n_val": len(val_idx),
                           "augmentation": AUGMENTATION}
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(f"{run_name}: {len(train_idx)} train / {len(val_idx)} val, {steps_per_epoch} steps per epoch, lr {lr:.2e}")

    def val_predictions():
        pred = predict(model, images[val_idx], args.batch_size, in_chans, scale)
        return pred.assign(patient_id=val_ids[pred.pop("image")])

    log, best_score, best_epoch, t0 = [], -1.0, 0, time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_lr = optimizer.param_groups[0]["lr"]
        losses = [train_step(batch) for batch in
                  order_rng.permutation(train_idx)[:steps_per_epoch * args.batch_size].reshape(steps_per_epoch, -1)]
        row = {"epoch": epoch, "lr": epoch_lr, "train_loss": torch.stack(losses).mean().item(),
               **evaluate(val_predictions(), true, val_ids, val_positive, val_stage1), "minutes": (time.time() - t0) / 60}
        log.append(row)
        pd.DataFrame(log).to_csv(out_dir / "log.csv", index=False)
        print(" ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in row.items()), flush=True)
        if row["score"] > best_score:
            best_score, best_epoch = row["score"], epoch
            torch.save(model.state_dict(), out_dir / "best.pt")

    # best epoch: validation and test boxes
    model.load_state_dict(torch.load(out_dir / "best.pt"))
    val_pred = val_predictions()
    val_pred[["patient_id", *BOX_COLS, "confidence"]].to_csv(out_dir / "val_preds.csv", index=False)
    test_ids = pd.read_csv(processed / "test.csv")["patient_id"].to_numpy()
    test_images = torch.from_numpy(np.load(processed / f"npy_{args.size}" / "test.npy")).cuda()
    test_pred = predict(model, test_images, args.batch_size, in_chans, scale)
    test_pred.assign(patient_id=test_ids[test_pred.pop("image")])[["patient_id", *BOX_COLS, "confidence"]] \
        .to_csv(out_dir / "test_preds.csv", index=False)
    best = next(r for r in log if r["epoch"] == best_epoch)
    summary = {"run": run_name, "best_epoch": best_epoch, **{k: best[k] for k in best if k not in ("epoch", "lr", "minutes")},
               "minutes": (time.time() - t0) / 60, "peak_gpu_frac": (others + torch.cuda.max_memory_reserved()) / total}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"best epoch {best_epoch}: score {best['score']:.4f} at threshold {best['threshold']}, auc {best['auc']:.4f}")


if __name__ == "__main__":
    main()
