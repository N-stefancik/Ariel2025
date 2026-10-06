# Ariel

Hey everyone viewing this project! I created this fo rthe 2025 Ariel Data Challenge Hosted by UCL and Nuerips on Kaggle. The orginal Kaggle notebook is given in `notebooks/best-scoring-ariel.original.ipynb`. The altered exploratory and analysis based notebook for viewing and understand the project is in `notebooks\ariel.ipynb`. 

## Setup

Open `notebooks/ariel.ipynb` for the step-by-step analysis: data inspection, light curves, smoothing, phase detection, depth fitting, small-batch comparisons, parameter experiments, PCA, and optional labels. Competition submission code is excluded from this notebook. The actual submission to Kaggle is given in `notebooks/best-scoring-ariel.original.ipynb` although this is not insightful to look at.

## Required data

Datasets are not included in the supplied notebook. Competition data is large and so I route you to the Kaggle page if you want to examine it.

Notebook evaluation requires `train.csv`. The array rows must match the selected training rows in planet order. Set `Config.DATASET = "train"` when calibrating raw training data. Raw calibration requires `adc_info.csv`, `<dataset>_star_info.csv`, and the competition sensor signal/calibration parquet directory structure.

## Analysis notebook

Run the sections sequentially. Change `PLANET_INDEX`, `EXPERIMENT_COUNT`, or `MIDDLE` to explore different cases. The notebook uses preprocessed data only; it does not need raw Kaggle calibration files or labels for its main workflow. 
