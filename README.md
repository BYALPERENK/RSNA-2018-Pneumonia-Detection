# RSNA Pneumonia Detection

End-to-end pipeline for the [RSNA Pneumonia Detection Challenge](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge): exploratory data analysis, image classification and lung opacity detection on chest X-rays.

> 🚧 Work in progress

## Results

| Model | Task | CV score | Public LB | Private LB |
|-------|------|----------|-----------|------------|
| –     | –    | –        | –         | –          |

## Project structure

```
rsna-pneumonia/
├── notebooks/        # EDA, evaluation and error analysis
├── src/rsna/         # reusable package code (data, models, metrics, training)
├── scripts/          # data preparation entry points
├── configs/          # experiment and path configs
├── reports/figures/  # figures used in this README
├── models/           # trained weights (not tracked)
└── outputs/          # predictions and logs (not tracked)
```

## Setup

```bash
pip install -r requirements.txt
pip install -e .
```

PyTorch and torchvision are not listed in `requirements.txt` because the right build depends on your CUDA version. Install them with the command from [pytorch.org](https://pytorch.org/get-started/locally/) (developed with PyTorch 2.14 + CUDA 13.2).

Download the competition data from Kaggle into `../data/raw` (see `configs/paths.yaml`). The dataset is not redistributed in this repository. Optionally, to mark the 1,000 stage-1 test images (read by three radiologists) inside the training set, put `stage_1_sample_submission.csv` from [pmcheng/rsna-pneumonia](https://github.com/pmcheng/rsna-pneumonia/tree/master/data) into `../data/external/stage1/`.

Then build the image cache and label tables (~31 GB in `../data/processed`):

```bash
python scripts/prepare_data.py
```

## Notes

- **The test set is labelled differently from most of the training set.** Most training images were read by one radiologist, while the test images were read by three, and the final boxes are roughly where their boxes overlap. Test boxes are therefore smaller, and the top teams shrank their predicted boxes. The 1,000 stage-1 test images, which are now part of the training set, are labelled like the test set; we mark them and use them to tune box shrinking and thresholds. Details and evidence: [02_data_preparation](notebooks/02_data_preparation.ipynb), section 5.

## Roadmap

- [x] Exploratory data analysis
- [x] Data preparation: NPY image cache, label tables, patient-level stratified folds
- [x] Competition metric (unit-tested) and image-free baselines
- [ ] Classification baseline
- [ ] Detection model
- [ ] Evaluation and error analysis
- [ ] Ensembling and post-processing
