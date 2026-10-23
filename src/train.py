import time

import mlflow
import mlflow.xgboost
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

from dataset import (HORIZON_DAYS, PROJECT_DIR, SMART_COLS, TEST_END,
                     TEST_START, TRAIN_END, load_labeled, time_split)

TRACKING_URI = "sqlite:///" + (PROJECT_DIR / "mlflow.db").as_posix()
EXPERIMENT_NAME = "drive-failure-7d"
CUTOFFS = [("0_5", 0.5), ("0_99", 0.99)]


def build_model(y_train):
    """Create the XGBoost model; return it together with its settings."""
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    params = dict(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        scale_pos_weight=n_neg / n_pos,
        tree_method="hist",
        eval_metric="aucpr",
        n_jobs=-1,
        random_state=42,
    )
    return XGBClassifier(**params), params


def evaluate(y_test, proba):
    """Compute PR-AUC, plus recall and precision at each cut-off."""
    y_arr = y_test.to_numpy()
    n_pos = int(y_arr.sum())
    results = {"pr_auc": average_precision_score(y_arr, proba)}
    for tag, t in CUTOFFS:
        flagged = proba >= t
        caught = int((flagged & (y_arr == 1)).sum())
        results["recall_at_" + tag] = caught / n_pos
        results["precision_at_" + tag] = caught / max(int(flagged.sum()), 1)
        results["rows_flagged_at_" + tag] = int(flagged.sum())
    return results


def log_to_mlflow(run_name, params, train_seconds, results, model):
    """Record settings, scores and the model as one MLflow run."""
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)
        mlflow.log_params({
            "horizon_days": HORIZON_DAYS,
            "train_end": str(TRAIN_END.date()),
            "test_start": str(TEST_START.date()),
            "test_end": str(TEST_END.date()),
            "features": ",".join(SMART_COLS),
        })
        mlflow.log_metric("train_seconds", train_seconds)
        for name, value in results.items():
            mlflow.log_metric(name, value)
        mlflow.xgboost.log_model(model, name="model")
    return run.info.run_id


if __name__ == "__main__":
    full = load_labeled()
    X_train, y_train, X_test, y_test = time_split(full)
    del full

    model, params = build_model(y_train)

    start = time.time()
    model.fit(X_train, y_train)
    train_seconds = round(time.time() - start)

    proba = model.predict_proba(X_test)[:, 1]
    results = evaluate(y_test, proba)

    print("train_seconds:", train_seconds)
    for name, value in results.items():
        print(name, round(value, 4))

    run_id = log_to_mlflow("train_py_baseline_xgb_200_trees", params,
                           train_seconds, results, model)
    print("run id:", run_id)