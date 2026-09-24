"""Competition metric: precision averaged over IoU thresholds 0.40-0.75, then averaged over images.

For one image and one threshold t, predictions are taken in order of confidence and each one is matched to the
unmatched true box with the highest IoU, if that IoU is above t. The score at t is TP / (TP + FP + FN).
An image with no true boxes scores 0 as soon as it has one prediction, and is skipped when it has none.
"""
import numpy as np
import pandas as pd

THRESHOLDS = np.round(np.arange(0.40, 0.751, 0.05), 2)
BOX_COLS = ["x", "y", "width", "height"]


def iou(a, b) -> np.ndarray:
    """IoU between every box in a (n, 4) and every box in b (m, 4); boxes are (x, y, width, height)."""
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    iw = np.minimum(a[:, None, 0] + a[:, None, 2], b[None, :, 0] + b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0])
    ih = np.minimum(a[:, None, 1] + a[:, None, 3], b[None, :, 1] + b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1])
    inter = np.clip(iw, 0, None) * np.clip(ih, 0, None)
    union = (a[:, 2] * a[:, 3])[:, None] + (b[:, 2] * b[:, 3])[None, :] - inter
    return inter / union


def image_score(true_boxes, pred_boxes, confidences=None, thresholds=THRESHOLDS) -> float | None:
    """Score of one image, or None when it has neither true nor predicted boxes (the metric skips such images)."""
    true_boxes = np.asarray(true_boxes, dtype=float).reshape(-1, 4)
    pred_boxes = np.asarray(pred_boxes, dtype=float).reshape(-1, 4)
    if len(true_boxes) == 0 and len(pred_boxes) == 0:
        return None
    if len(true_boxes) == 0 or len(pred_boxes) == 0:
        return 0.0
    if confidences is not None:
        pred_boxes = pred_boxes[np.argsort(-np.asarray(confidences, dtype=float), kind="stable")]
    ious = iou(pred_boxes, true_boxes)

    precisions = []
    for t in thresholds:
        matched = np.zeros(len(true_boxes), dtype=bool)
        for row in ious:  # one prediction at a time, most confident first
            candidates = np.where(matched, -1.0, row)
            j = candidates.argmax()
            if candidates[j] > t:
                matched[j] = True
        tp = matched.sum()
        fp = len(pred_boxes) - tp
        fn = len(true_boxes) - tp
        precisions.append(tp / (tp + fp + fn))
    return float(np.mean(precisions))


def score_images(true: pd.DataFrame, pred: pd.DataFrame, patient_ids) -> pd.Series:
    """Score of every image in patient_ids; NaN for skipped images, so .mean() gives the competition score.

    true has one row per box (patient_id + BOX_COLS), pred the same plus a confidence column.
    """
    empty = np.empty((0, 4))
    true_boxes, pred_boxes, confidences = true[BOX_COLS].to_numpy(float), pred[BOX_COLS].to_numpy(float), pred["confidence"].to_numpy(float)
    true_rows = true.groupby("patient_id").indices  # patient_id -> row positions
    pred_rows = pred.groupby("patient_id").indices
    scores = {}
    for pid in patient_ids:
        t, p = true_rows.get(pid), pred_rows.get(pid)
        score = image_score(empty if t is None else true_boxes[t], empty if p is None else pred_boxes[p],
                            None if p is None else confidences[p])
        scores[pid] = np.nan if score is None else score
    return pd.Series(scores, name="score", dtype=float).rename_axis("patient_id")
