"""Build the label tables, the cross-validation folds and the NPY image cache."""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
from sklearn.model_selection import StratifiedKFold

HEADER_FIELDS = {"ViewPosition": "view", "PatientSex": "sex", "PatientAge": "age", "PixelSpacing": "pixel_spacing"}


def dicom_to_npy(dcm_path: Path, npy_path: Path) -> dict:
    """Save the pixels of one DICOM as .npy and return its header fields."""
    ds = pydicom.dcmread(dcm_path)
    if not npy_path.exists():
        # write to a temporary file first, so an interrupted run never leaves a truncated .npy behind
        tmp_path = npy_path.with_suffix(".tmp")
        with open(tmp_path, "wb") as f:
            np.save(f, ds.pixel_array)
        os.replace(tmp_path, npy_path)
    row = {"patient_id": dcm_path.stem}
    for field, name in HEADER_FIELDS.items():
        row[name] = ds.get(field)
    row["age"] = int(row["age"])
    row["pixel_spacing"] = round(float(row["pixel_spacing"][0]), 3)
    return row


def build_patients(labels: pd.DataFrame, classes: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """One row per training patient: target, 3-class label, box count and header fields."""
    n_boxes = labels.groupby("patientId")["x"].count().rename("n_boxes")
    patients = (
        labels.groupby("patientId")["Target"].max().rename("target").to_frame()
        .join(classes.drop_duplicates("patientId").set_index("patientId"))
        .join(n_boxes)
        .rename_axis("patient_id")
        .reset_index()
    )
    return patients.merge(meta, on="patient_id", how="left")


def build_boxes(labels: pd.DataFrame) -> pd.DataFrame:
    """One row per box, in pixels of the 1024 x 1024 image."""
    cols = ["x", "y", "width", "height"]
    boxes = labels.dropna(subset=["x"]).drop(columns="Target").rename(columns={"patientId": "patient_id"})
    assert boxes[cols].notna().all().all(), "box with missing coordinates"
    assert (boxes[cols] % 1 == 0).all().all(), "non-integer box coordinates"
    boxes[cols] = boxes[cols].astype(int)
    assert ((boxes["width"] > 0) & (boxes["height"] > 0)).all(), "box with zero width or height"
    assert ((boxes["x"] >= 0) & (boxes["y"] >= 0) & (boxes["x"] + boxes["width"] <= 1024)
            & (boxes["y"] + boxes["height"] <= 1024)).all(), "box outside the image"
    return boxes.reset_index(drop=True)


def add_folds(patients: pd.DataFrame, n_folds: int = 5, seed: int = 42) -> pd.DataFrame:
    """Assign a fold to every patient, stratified on class x view."""
    patients = patients.reset_index(drop=True)  # StratifiedKFold returns positions; with a 0..n-1 index they equal the labels
    strata = patients["class"] + " | " + patients["view"]
    patients["fold"] = -1
    for fold, (_, val_idx) in enumerate(StratifiedKFold(n_folds, shuffle=True, random_state=seed).split(patients, strata)):
        patients.loc[val_idx, "fold"] = fold
    return patients
