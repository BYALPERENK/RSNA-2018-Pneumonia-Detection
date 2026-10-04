"""Post-processing of saved box predictions (patient_id, x, y, width, height, confidence; 1024 px):
stricter NMS, rank normalization of the scores, and weighted boxes fusion (WBF) of several models."""
import numpy as np
import pandas as pd
import torch
from torchvision.ops import batched_nms

from rsna.metrics import BOX_COLS


def nms(pred: pd.DataFrame, iou: float) -> pd.DataFrame:
    """NMS per image at the given IoU (on top of the model's own)."""
    xyxy = torch.tensor(pred[BOX_COLS].to_numpy(np.float32))
    xyxy[:, 2:] += xyxy[:, :2]
    keep = batched_nms(xyxy, torch.tensor(pred["confidence"].to_numpy(np.float32)),
                       torch.tensor(pd.factorize(pred["patient_id"])[0]), iou).numpy()
    return pred.iloc[np.sort(keep)].reset_index(drop=True)


def rank_scores(pred: pd.DataFrame, by: str | None = None) -> pd.DataFrame:
    """Scores replaced by their percentile rank among the model's boxes (within each `by` group, e.g. fold)."""
    conf = pred.groupby(by)["confidence"] if by else pred["confidence"]
    return pred.assign(confidence=conf.rank(pct=True))


def _iou_one(box, others):
    x1, y1 = np.maximum(box[0], others[:, 0]), np.maximum(box[1], others[:, 1])
    x2, y2 = np.minimum(box[2], others[:, 2]), np.minimum(box[3], others[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda b: (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])  # noqa: E731
    return inter / (area(box) + area(others) - inter)


def wbf(preds: list[pd.DataFrame], weights=None, iou_thr: float = 0.3) -> pd.DataFrame:
    """Weighted boxes fusion of several models' boxes, per image.

    The boxes of all models are sorted by weighted score (score x model weight); each one joins the fused box it
    overlaps most if their IoU > iou_thr, otherwise it starts a new one. A fused box's coordinates are the
    weighted-score mean of its members; its score is the weighted mean over the models of each model's top score in
    the cluster (0 for a model without a box there), so each model votes once.
    """
    w = np.ones(len(preds)) if weights is None else np.asarray(weights, float)
    table = pd.concat([p[["patient_id", *BOX_COLS, "confidence"]].assign(model=m) for m, p in enumerate(preds)],
                      ignore_index=True)
    xyxy = table[BOX_COLS].to_numpy(float, copy=True)  # copy: the input may be read-only (joblib memmap)
    xyxy[:, 2:] += xyxy[:, :2]
    score, model = table["confidence"].to_numpy(), table["model"].to_numpy()
    rows = []
    for pid, idx in table.groupby("patient_id").indices.items():
        ws = score[idx] * w[model[idx]]
        fused, members = [], []
        for i in idx[np.argsort(-ws)]:
            if fused:
                overlap = _iou_one(xyxy[i], np.array(fused))
                j = overlap.argmax()
                if overlap[j] > iou_thr:
                    members[j].append(i)
                    k = np.array(members[j])
                    kw = score[k] * w[model[k]]
                    if kw.sum() > 0:  # boxes with score 0 (e.g. below every rank of 07) keep the first box
                        fused[j] = (kw[:, None] * xyxy[k]).sum(0) / kw.sum()
                    continue
            fused.append(xyxy[i].copy())
            members.append([i])
        for box, k in zip(fused, members):
            top = np.zeros(len(preds))
            np.maximum.at(top, model[k], score[k])
            rows.append((pid, box[0], box[1], box[2] - box[0], box[3] - box[1], (w * top).sum() / w.sum()))
    return pd.DataFrame(rows, columns=["patient_id", *BOX_COLS, "confidence"])
