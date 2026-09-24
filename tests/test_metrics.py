import numpy as np
import pandas as pd
import pytest

from rsna.metrics import THRESHOLDS, image_score, iou, score_images

BOX = [100, 100, 100, 100]
SHIFTED = [120, 100, 100, 100]  # IoU with BOX = 80 / 120 = 0.667, above 6 of the 8 thresholds
FAR = [600, 600, 100, 100]      # no overlap with BOX


def test_thresholds():
    assert THRESHOLDS.tolist() == [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]


def test_iou():
    assert iou(BOX, BOX)[0, 0] == pytest.approx(1.0)
    assert iou(BOX, FAR)[0, 0] == 0.0
    assert iou(BOX, SHIFTED)[0, 0] == pytest.approx(80 / 120)
    assert iou([0, 0, 10, 10], [10, 0, 10, 10])[0, 0] == 0.0  # touching edges
    assert iou([BOX, FAR], [BOX, SHIFTED, FAR]).shape == (2, 3)


def test_exact_match():
    assert image_score([BOX], [BOX]) == 1.0


def test_shifted_box_hits_some_thresholds():
    assert image_score([BOX], [SHIFTED]) == pytest.approx(6 / 8)


def test_extra_prediction_is_a_false_positive():
    assert image_score([BOX], [BOX, FAR]) == pytest.approx(1 / 2)


def test_missed_box_is_a_false_negative():
    assert image_score([BOX, FAR], [BOX]) == pytest.approx(1 / 2)


def test_duplicate_prediction_matches_only_once():
    assert image_score([BOX], [BOX, BOX]) == pytest.approx(1 / 2)


def test_no_prediction_on_positive_image():
    assert image_score([BOX], []) == 0.0


def test_prediction_on_negative_image_scores_zero():
    assert image_score([], [BOX]) == 0.0


def test_negative_image_without_prediction_is_skipped():
    assert image_score([], []) is None


def test_order_of_boxes_does_not_matter():
    assert image_score([BOX, FAR], [FAR, BOX]) == 1.0
    assert image_score([FAR, BOX], [BOX, FAR]) == 1.0


def test_most_confident_prediction_is_matched_first():
    true = [BOX, [160, 100, 100, 100]]
    between = [124, 100, 100, 100]  # IoU 0.613 with the first true box, 0.471 with the second
    # exact box first: it takes the first true box, `between` takes the second at 0.40-0.45 -> (2 x 1 + 6 x 1/3) / 8
    assert image_score(true, [between, BOX], confidences=[0.1, 0.9]) == pytest.approx(1 / 2)
    # `between` first: it takes the first true box at 0.40-0.60 and the exact box becomes a false positive -> 1/3 everywhere
    assert image_score(true, [between, BOX], confidences=[0.9, 0.1]) == pytest.approx(1 / 3)


def test_score_images():
    true = pd.DataFrame([["pos", *BOX]], columns=["patient_id", "x", "y", "width", "height"])
    pred = pd.DataFrame([["pos", 0.9, *BOX], ["neg_fp", 0.5, *BOX]],
                        columns=["patient_id", "confidence", "x", "y", "width", "height"])
    scores = score_images(true, pred, ["pos", "neg_fp", "neg_empty"])
    assert scores["pos"] == 1.0
    assert scores["neg_fp"] == 0.0
    assert np.isnan(scores["neg_empty"])
    assert scores.mean() == pytest.approx(1 / 2)  # the skipped image does not count
