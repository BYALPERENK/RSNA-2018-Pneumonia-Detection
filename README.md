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

Download the competition data from Kaggle into `../data/raw` (see `configs/paths.yaml`). The dataset is not redistributed in this repository.

## Roadmap

- [x] Exploratory data analysis
- [ ] DICOM → PNG preprocessing
- [ ] Patient-level stratified splits
- [ ] Classification baseline
- [ ] Detection model
- [ ] Evaluation and error analysis
- [ ] Ensembling and post-processing
