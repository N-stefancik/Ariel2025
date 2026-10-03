# Ariel

Hey everyone viewing this project! I created this fo rthe 2025 Ariel Data Challenge Hosted by UCL and Nuerips on Kaggle. The orginal Kaggle notebook is given in `notebooks/best-scoring-ariel.original.ipynb`. The altered exploratory and analysis based notebook for viewing and understand the project is in `notebooks\ariel.ipynb`. 

The rest is genereated by Codex for credit and is just setup information that I hope can be helpful if any issues come up!

## Setup

Run from the Ariel folder:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m jupyterlab
```

Open `notebooks/ariel.ipynb` for the step-by-step analysis lab: data inspection, light curves, smoothing, phase detection, depth fitting, small-batch comparisons, parameter experiments, PCA, and optional labels. Competition submission code is excluded from this notebook. The separate Python runner remains available. The exact original, including saved outputs, is `notebooks/best-scoring-ariel.original.ipynb`.

## Required data

Datasets are not included in the supplied notebook. Place `preprocessed_data.npy` in `data/` and the competition data in `data/ariel-data-challenge-2025/`. You can override these locations with `ARIEL_PREPROCESSED_PATH` and `ARIEL_DATA_PATH` before launching Jupyter.

Notebook evaluation requires `train.csv`. The array rows must match the selected training rows in planet order. Set `Config.DATASET = "train"` when calibrating raw training data. Raw calibration requires `adc_info.csv`, `<dataset>_star_info.csv`, and the competition sensor signal/calibration parquet directory structure.

## Python runner

```powershell
.\.venv\Scripts\python.exe ariel.py --preprocessed data/preprocessed_data.npy --output predictions.csv
```

For a submission, supply test signals in the same planet order as `sample_submission.csv`:

```powershell
.\.venv\Scripts\python.exe ariel.py --preprocessed data/test_preprocessed_data.npy --submission --output submission.csv
```

Omit `--preprocessed` to calibrate raw data; use `--dataset train` for training data. The baseline predicts one depth per planet and repeats it across spectral submission columns, as in the source.


## Local pqdm packages

The packages formerly stored in Kaggle's `ariel-2024-pqdm` folder are now in this project's `data/` folder. `requirements.txt` installs the exact supplied wheel files. Run pip from the project root. The notebook's first cell also runs pip from that root, even when Jupyter starts inside `notebooks/`.

## Analysis notebook

Run the sections sequentially. Change `PLANET_INDEX`, `EXPERIMENT_COUNT`, or `MIDDLE` to explore different cases. The notebook uses preprocessed data only; it does not need raw Kaggle calibration files or labels for its main workflow. Optional label evaluation is disabled until you confirm row alignment. Select the project `.venv` kernel; to register it, run `.\.venv\Scripts\python.exe -m ipykernel install --user --name ariel --display-name "Python (Ariel)"`.
