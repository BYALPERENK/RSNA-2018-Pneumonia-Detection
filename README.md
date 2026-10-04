# RSNA Pneumonia Detection

End-to-end pipeline for the [RSNA Pneumonia Detection Challenge](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge) (Kaggle, 2018): find the lung opacities of pneumonia on chest X-rays and draw boxes around them. Built from scratch as a learning project.

**The project is meant to be followed through its [notebooks](#notebooks), in order from 00 to 07.** Each notebook is one step of the work: it states the question it answers, shows the code and the raw results of every experiment, and ends with the findings and the choices carried to the next step. Reading them in order shows what was tried, what worked, what did not, and why.

## Results

Private leaderboard (late submissions, scored on ~99% of the 3,000 stage-2 test images):

| Submission | Private LB | Chosen |
|---|---|---|
| 1-3 detectors + classifier gating + box shrink 0.85 × 0.90, box threshold 0.6 | **0.2067** | before any submission, on out-of-fold predictions |
| 2-Same, box threshold 0.65 | 0.2113 | on the private leaderboard itself (optimistic) |
| 3-The detector ensemble without post-processing | 0.1593 | – |

For scale: 1st place 0.2548, 10th about 0.225, about 50th 0.192 (344 teams).

Out-of-fold (5 folds, 26,684 training images):

| Model | Score | AUC | F1 |
|---|---|---|---|
| Classifier: stacking of 4 CNNs at 384 px (04) | – | 0.905 | 0.657 |
| Detectors: Faster R-CNN + FCOS + YOLO26s at 256 px, WBF, flip TTA (05) | 0.2547 | 0.897 | 0.666 |
| Same, on the 1,000 stage-1 test images, with the 06 post-processing | 0.2753 | 0.934* | 0.818 |

AUC and F1 are for the image-level decision (pneumonia or not); the score is the competition metric. *With gating the decision combines the box and the classifier score and has no single AUC; 0.934 is the classifier's alone.

## The key point: the test set is labelled differently

Most training images were read by one radiologist. The test images were read by three, and overlapping boxes were replaced by their **intersection** ([Shih et al., 2019](https://escholarship.org/content/qt53d65470/qt53d65470.pdf)). Test boxes are therefore smaller (mean area ×0.63) and the test set has more positives. A detector trained and validated on single-reader labels cannot see this; all of the top 4 teams shrank their boxes.

1,000 triple-read images (the stage-1 test set) are inside the training set. We found and marked them ([02](notebooks/02_data_preparation.ipynb), section 5) and used them, out-of-fold, as a local stand-in for the test labels ([06](notebooks/06_postprocessing.ipynb)): our boxes are the right size for single-reader labels and about 25% too large for consensus labels; shrinking them by 0.85 × 0.90 adds +0.035 there. On the test set the shrink and the classifier gating carried over as measured, but the lower box threshold chosen on these images did not, which explains most of the gap between 0.275 out-of-fold and 0.207 on the leaderboard ([07](notebooks/07_submission.ipynb), sections 5–7).

## Notebooks

| # | Notebook | Question |
|---|---|---|
| 00 | [overview](notebooks/00_overview.ipynb) | What is the competition, and how is the project organised? |
| 01 | [eda](notebooks/01_eda.ipynb) | What is in the data, and what does it mean for the models? |
| 02 | [data_preparation](notebooks/02_data_preparation.ipynb) | Image cache, label tables, folds; which images are triple-read? |
| 03 | [metric_and_baselines](notebooks/03_metric_and_baselines.ipynb) | The competition metric (unit-tested) and image-free baselines |
| 04 | [classification](notebooks/04_classification.ipynb) | 2 or 3 classes, which backbone, which ensemble? |
| 05 | [detection](notebooks/05_detection.ipynb) | Which detectors, NMS, resolution, ensemble, flip TTA? |
| 06 | [postprocessing](notebooks/06_postprocessing.ipynb) | Box shrink, threshold and classifier gating, chosen on the triple-read images |
| 07 | [submission](notebooks/07_submission.ipynb) | Test predictions, submissions, and why the test score is lower |

## Compared with the top solutions

- **The same:** a fixed box shrink (1st place 0.875, ours 0.85 × 0.90), a classifier to decide which images get boxes (1st place's ensemble reached AUC 0.93 on the stage-1 test images; ours 0.934), a threshold that puts boxes on 35–40% of the test images (theirs 37%, ours 38–41%), low resolution (224–512 px).
- **Where the gap is:** the detectors. The 1st place's own ablation scores 0.253 with 1 classifier and 5 Relation Networks. Ideas not tried here, all needing new training: negative and positive images concatenated side by side during training (1st), loss on images without boxes plus an auxiliary image-level class output and dropout in a RetinaNet at 512 px (2nd), averaging the raw outputs of several checkpoints, multi-scale TTA.

## Project structure

```
rsna-pneumonia/
├── notebooks/        # 00–07, read in order
├── src/rsna/         # package code: data, augmentation, models, metric, box post-processing
├── scripts/          # data preparation, training and prediction entry points
├── tests/            # unit tests of the competition metric
├── configs/          # paths
├── models/           # trained weights (not tracked)
└── outputs/          # predictions, logs and submissions (not tracked)
```

## Setup

```bash
pip install -r requirements.txt
pip install -e .
```

PyTorch and torchvision are not listed in `requirements.txt` because the right build depends on your CUDA version. Install them with the command from [pytorch.org](https://pytorch.org/get-started/locally/) (developed with PyTorch 2.14 + CUDA 13.2 on one GPU).

Download the competition data from Kaggle into `../data/raw` (see `configs/paths.yaml`); it is not redistributed here. To mark the 1,000 stage-1 test images inside the training set, put `stage_1_sample_submission.csv` from [pmcheng/rsna-pneumonia](https://github.com/pmcheng/rsna-pneumonia/tree/master/data) into `../data/external/stage1/`.

## Reproducing

```bash
python scripts/prepare_data.py                 # DICOM -> NPY cache, patients.csv (folds), boxes.csv, test.csv
python scripts/resize_images.py --size 256     # and --size 384 for the classifiers
python scripts/make_yolo_dataset.py --size 256 # YOLO images and labels
```

Then the models, one run per fold (`--fold 0` … `4`):
- **Classifiers:** `scripts/train_classifier.py` with the 4 backbones in 04's summary and the settings of its sections.
- **Detectors:** `scripts/train_detector.py --arch fasterrcnn --backbone convnext_tiny.fb_in22k_ft_in1k`, `--arch fcos --backbone mobilevitv2_150.cvnets_in22k_ft_in1k` and `scripts/train_yolo.py --model yolo26s`, at `--size 256`; then `scripts/predict_flip.py` on every run for the flip TTA.

Notebooks 04–07 read these runs from `outputs/`, choose the ensembles and the post-processing, and write the submission files to `outputs/submissions/`.

```bash
pytest tests
```

## License

The code is under the [MIT License](LICENSE). The competition data belongs to RSNA and is not included; see the competition's rules on Kaggle.
