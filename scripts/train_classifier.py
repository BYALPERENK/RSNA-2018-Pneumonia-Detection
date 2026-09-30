"""Train the image-level pneumonia classifier on one fold and save per-epoch validation metrics and predictions.

Usage: python scripts/train_classifier.py --model densenet121.tv_in1k --fold 0 --batch-size 224
       python scripts/train_classifier.py ... --probe   # a few steps only, to measure memory and speed

All images stay on the GPU as uint8 (npy_{size}/train.npy, or npy_{size}_clahe with --clahe, from scripts/resize_images.py) and are augmented there.
The learning rate is base_lr * sqrt(batch size / 256) and warms up linearly. After that it is either halved when
the validation AUC has not improved for --lr-patience epochs (plateau), or follows a cosine to zero (cosine).
Training stops when the validation AUC has not improved for --early-stop epochs. The epoch with the best
validation AUC is kept; its validation and test predictions are saved with and without horizontal-flip TTA.
With --bootstrap (bagging) the model trains on a resample with replacement of the training images, of the same size,
so the number of steps is unchanged.

The 2 vs 3-class experiment (outputs/classification/fold0_256) used
--schedule cosine --base-lr 5e-4 --aug light --label-smoothing 0 --drop-path 0 --early-stop 0.
"""
import argparse
import json
import math
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from rsna.data.augment import augment, to_float
from rsna.models.classifier import CLASSES, Classifier
from rsna.paths import load_paths

AUGMENTATIONS = {
    "light": dict(max_rotate=10, scale=(0.9, 1.1), max_shift=0.05, brightness=0.1, contrast=(0.9, 1.1)),
    "main": dict(max_rotate=13, scale=(0.85, 1.15), max_shift=0.08, brightness=0.15, contrast=(0.85, 1.15), vflip=True),
}
AUGMENTATIONS["main_no_vflip"] = AUGMENTATIONS["main"] | {"vflip": False}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="timm model name with pretrained tag")
    parser.add_argument("--n-classes", type=int, choices=[2, 3], default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20, help="maximum number of epochs")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--base-lr", type=float, default=3e-4, help="learning rate at batch size 256")
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--schedule", choices=["plateau", "cosine"], default="plateau")
    parser.add_argument("--lr-patience", type=int, default=3, help="plateau: halve the lr after this many epochs without a better AUC")
    parser.add_argument("--early-stop", type=int, default=5, help="stop after this many epochs without a better AUC; 0 = off")
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--drop-path", type=float, default=0.1)
    parser.add_argument("--clahe", action="store_true", help="read the CLAHE images (npy_{size}_clahe)")
    parser.add_argument("--aug", choices=list(AUGMENTATIONS), default="main")
    parser.add_argument("--bootstrap", action="store_true", help="train on a bootstrap resample of the training images")
    parser.add_argument("--experiment", default="main_256", help="outputs/classification/<experiment>/<run name>")
    parser.add_argument("--probe", action="store_true", help="run a few steps, print memory use and speed, exit")
    return parser.parse_args()


def predict(model, images, batch_size, tta=False, labels=None):
    """Probabilities, (N, 1) or (N, 3), without augmentation; tta averages them with the horizontally flipped image.

    With labels, also returns the mean loss (without TTA).
    """
    model.eval()
    probs, loss_sum = [], 0.0
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for start in range(0, len(images), batch_size):
            x = to_float(images[start:start + batch_size]).contiguous(memory_format=torch.channels_last)
            logits = model(x)
            p = model.probabilities(logits)
            if labels is not None:
                loss_sum += model.loss(logits, labels[start:start + batch_size]).item() * len(x)
            if tta:
                p = (p + model.probabilities(model(x.flip(-1)))) / 2
            probs.append(p)
    probs = torch.cat(probs).cpu().numpy()
    return probs if labels is None else (probs, loss_sum / len(images))


def metrics(probs, classes):
    """AUC, F1 and accuracy of the pneumonia probability (threshold 0.5), plus 3-class scores when available."""
    target = classes == 2
    p = probs[:, -1]  # sigmoid output, or the Lung Opacity column of the softmax
    row = {"auc": roc_auc_score(target, p), "f1": f1_score(target, p >= 0.5), "acc": accuracy_score(target, p >= 0.5)}
    hard = classes > 0  # Lung Opacity vs Not Normal only: the hard negatives
    row["auc_lo_vs_not_normal"] = roc_auc_score(target[hard], p[hard])
    if probs.shape[1] == 3:
        row["acc_3class"] = accuracy_score(classes, probs.argmax(1))
    return row


def prediction_table(ids, probs, n_classes):
    table = pd.DataFrame({"patient_id": ids, "p_pneumonia": probs[:, -1]})
    if n_classes == 3:
        table[["p_normal", "p_not_normal", "p_lung_opacity"]] = probs
    return table


def main():
    args = parse_args()
    run_name = f"{args.model.split('.')[0]}_{args.n_classes}c_f{args.fold}_s{args.seed}" + ("_boot" if args.bootstrap else "")
    paths = load_paths()
    out_dir = paths["outputs_dir"] / "classification" / args.experiment / run_name
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    processed = paths["processed_data_dir"]
    patients = pd.read_csv(processed / "patients.csv")
    npy_dir = processed / (f"npy_{args.size}" + ("_clahe" if args.clahe else ""))
    images = torch.from_numpy(np.load(npy_dir / "train.npy")).cuda()
    classes = torch.tensor(patients["class"].map(CLASSES.index).to_numpy(), device="cuda")
    train_idx = np.flatnonzero(patients["fold"] != args.fold)
    val_idx = np.flatnonzero(patients["fold"] == args.fold)
    if args.bootstrap:
        train_idx = np.random.default_rng([args.seed, 1]).choice(train_idx, len(train_idx), replace=True)
    val_images, val_classes = images[val_idx], classes[val_idx]
    val_classes_np = val_classes.cpu().numpy()

    model = Classifier(args.model, args.n_classes, drop_path=args.drop_path,
                       label_smoothing=args.label_smoothing).cuda().to(memory_format=torch.channels_last)
    lr = args.base_lr * math.sqrt(args.batch_size / 256)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    steps_per_epoch = len(train_idx) // args.batch_size  # the last incomplete batch is dropped
    total_steps = steps_per_epoch * args.epochs
    if args.schedule == "plateau":
        plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5,
                                                             patience=args.lr_patience - 1,  # halves on the n-th bad epoch
                                                             threshold=0.0, threshold_mode="abs")
    step = 0

    def set_lr():
        """Linear warm-up for both schedules, then a cosine per step or nothing (the plateau scheduler acts per epoch)."""
        if step < args.warmup_steps:
            factor = (step + 1) / args.warmup_steps
        elif args.schedule == "cosine":
            factor = 0.5 * (1 + math.cos(math.pi * (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)))
        else:
            return
        for group in optimizer.param_groups:
            group["lr"] = lr * factor

    order_rng = np.random.default_rng(args.seed)  # separate random streams: the data order and augmentation
    aug_gen = torch.Generator(device="cuda").manual_seed(args.seed)  # do not depend on the model's initialization

    def train_step(batch_idx):
        nonlocal step
        set_lr()
        idx = torch.from_numpy(batch_idx).cuda()
        x = augment(images[idx], aug_gen, **AUGMENTATIONS[args.aug]).contiguous(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model.loss(model(x), classes[idx])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        step += 1
        return loss

    if args.probe:
        model.train()
        batches = order_rng.permutation(train_idx)[:args.batch_size * 8].reshape(8, -1)
        for i, batch_idx in enumerate(batches):
            if i == 3:
                torch.cuda.synchronize()
                start = time.time()
            train_step(batch_idx)
        torch.cuda.synchronize()
        speed = 5 * args.batch_size / (time.time() - start)
        predict(model, val_images[:args.batch_size], args.batch_size, tta=True)
        free, total = torch.cuda.mem_get_info()
        print(json.dumps({"batch_size": args.batch_size, "gpu_used_gib": round((total - free) / 2**30, 2),
                          "gpu_used_frac": round(1 - free / total, 3), "train_img_per_s": round(speed)}))
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args) | {"lr": lr, "steps_per_epoch": steps_per_epoch, "n_train": len(train_idx),
                           "n_train_unique": len(np.unique(train_idx)), "n_val": len(val_idx),
                           "augmentation": AUGMENTATIONS[args.aug]}
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(f"{run_name}: {len(train_idx)} train / {len(val_idx)} val, {steps_per_epoch} steps per epoch, lr {lr:.2e}")

    log, best_auc, best_epoch, t0 = [], -1.0, 0, time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_lr = optimizer.param_groups[0]["lr"] if step >= args.warmup_steps else lr
        losses = [train_step(batch) for batch in
                  order_rng.permutation(train_idx)[:steps_per_epoch * args.batch_size].reshape(steps_per_epoch, -1)]
        train_loss = torch.stack(losses).mean().item()
        probs, loss = predict(model, val_images, args.batch_size, labels=val_classes)
        row = {"epoch": epoch, "lr": epoch_lr, "train_loss": train_loss, "val_loss": loss,
               **metrics(probs, val_classes_np), "minutes": (time.time() - t0) / 60}
        log.append(row)
        pd.DataFrame(log).to_csv(out_dir / "log.csv", index=False)
        print(" ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in row.items()), flush=True)

        if row["auc"] > best_auc:
            best_auc, best_epoch = row["auc"], epoch
            torch.save(model.state_dict(), out_dir / "best.pt")
        if args.schedule == "plateau":
            plateau.step(row["auc"])
        if args.early_stop and epoch - best_epoch >= args.early_stop:
            print(f"early stop: no better AUC for {args.early_stop} epochs")
            break

    # best epoch: validation and test predictions with and without TTA
    model.load_state_dict(torch.load(out_dir / "best.pt"))
    val_ids = patients["patient_id"].to_numpy()[val_idx]
    test_ids = pd.read_csv(processed / "test.csv")["patient_id"].to_numpy()
    test_images = torch.from_numpy(np.load(npy_dir / "test.npy")).cuda()
    best = next(r for r in log if r["epoch"] == best_epoch)
    summary = {"run": run_name, "best_epoch": best_epoch, "epochs_run": len(log),
               **{k: best[k] for k in best if k not in ("epoch", "lr", "minutes")}}
    for suffix, tta in [("", False), ("_tta", True)]:
        val_probs = predict(model, val_images, args.batch_size, tta)
        val_table = prediction_table(val_ids, val_probs, args.n_classes)
        val_table.insert(1, "class", val_classes_np)
        val_table.to_csv(out_dir / f"val_preds{suffix}.csv", index=False)
        prediction_table(test_ids, predict(model, test_images, args.batch_size, tta), args.n_classes) \
            .to_csv(out_dir / f"test_preds{suffix}.csv", index=False)
        if tta:
            summary |= {f"{k}_tta": v for k, v in metrics(val_probs, val_classes_np).items()}
    summary |= {"minutes": (time.time() - t0) / 60, "peak_gpu_gib": torch.cuda.max_memory_reserved() / 2**30}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"best epoch {best_epoch}: auc {best['auc']:.4f}, with TTA {summary['auc_tta']:.4f}")


if __name__ == "__main__":
    main()
