import time

from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

from dataset import load_labeled, time_split

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