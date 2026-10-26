# Hard Drive Failure Prediction

Predicts which hard drives will fail within the next 7 days, using daily SMART sensor readings from the Backblaze Drive Stats dataset (Q1 2024). Built as a learning project that covers the full cycle: data, labeling, model, experiment tracking, serving, Docker and drift monitoring.

## Data

- 91 daily CSV files (2024-01-01 to 2024-03-31): 25,189,213 drive-day rows, 291,331 drives, 978 failures.
- Reduced to 8 columns and saved as `data/full_q1_2024.parquet`. The Parquet file was created in `notebooks/01_explore.ipynb`.
- Features: five SMART raw values: `smart_5_raw` (remapped sectors), `smart_9_raw` (hours powered on), `smart_194_raw` (temperature), `smart_197_raw` (pending sectors), `smart_198_raw` (offline uncorrectable sectors).
- Failed drives show bad sectors far more often than healthy ones (for example `smart_197_raw` above 0: 49.6% vs 1.2%).

## Label and split

- The raw `failure` column is 1 only on a drive's last day, so the label is "fails within the next 7 days" (7,293 positive rows).
- Time-based split, not random: train up to 2024-02-22, test 2024-03-01 to 2024-03-24. A one-week gap before the test period and a one-week tail after it prevent labels from leaking across the boundary.

## Model and results

XGBoost classifier (200 trees, depth 4) with `scale_pos_weight` to handle the extreme class imbalance (about 1 positive row per 3,000 negatives in training). Accuracy is meaningless here (always answering "no failure" scores 99.97%), so the metrics are PR-AUC, recall and precision.

| Metric (test set) | Value |
|---|---|
| PR-AUC | 0.0202 (random baseline 0.000243, about 83x higher) |
| Cut-off 0.5 | recall 0.606, precision 0.0027, 360,782 rows flagged |
| Cut-off 0.99 | recall 0.116, precision 0.054, 3,487 rows flagged |

Feature importance: `smart_5_raw` 0.656, `smart_197_raw` 0.224, `smart_9_raw` 0.054, `smart_194_raw` 0.034, `smart_198_raw` 0.032.

## Project layout

- `src/dataset.py`: loads the data, builds the 7-day label, makes the time split
- `src/train.py`: trains the model and logs parameters, metrics and the model to MLflow
- `src/export_model.py`: exports the trained model to `models/model.ubj`
- `src/serve.py`: FastAPI service with `/health` and `/predict`
- `src/monitor.py`: input drift check (PSI) between training data and a newer week
- `Dockerfile`, `requirements.txt`: container build for the service

## How to run

Serve the model locally (needs `requirements.txt` installed):

```
uvicorn serve:app --app-dir src --port 8000
```

Or with Docker:

```
docker build -t drive-failure-api .
docker run --rm -p 8000:8000 drive-failure-api
```

Then open http://127.0.0.1:8000/docs and send a request to `POST /predict`:

```
{"smart_5_raw": 500, "smart_9_raw": 40000, "smart_194_raw": 33, "smart_197_raw": 100, "smart_198_raw": 100}
```

The response is `{"failure_score": 0.9599}`. A drive with no bad sectors scored 0.2541 in the same test.

Training (`python src/train.py`) and drift monitoring (`python src/monitor.py`) also need `mlflow` and `pyarrow`, and the Parquet file in `data/`.

## Limitations

- The test set has only 272 failing drives, so small differences between runs should not be over-read.
- Precision is very low: at cut-off 0.5 the model flags many healthy drives. Some of these may be early warnings beyond 7 days, but that has not been tested.
- `scale_pos_weight` inflates the scores, so `failure_score` is a ranking score, not a true probability.
- The drift check looks at inputs only. Labels for the newest week are not trustworthy, so accuracy on new data is not measured.
- The Parquet file is not in Git. It is tracked with DVC, but no DVC remote is set up yet, so a fresh clone cannot download it.
- The first Docker build downloads large GPU packages pulled in by `xgboost`. A CPU-only variant could shrink the image, but it has not been tried.