import numpy as np
import pandas as pd

from dataset import DATA_PATH, SMART_COLS, TRAIN_END

NEW_START = pd.Timestamp("2024-03-25")
NEW_END = pd.Timestamp("2024-03-31")


def psi(reference, current, bins=10):
    """Population Stability Index: how much a feature's spread has shifted."""
    ref = reference.dropna().to_numpy()
    cur = current.dropna().to_numpy()

    inner = np.quantile(ref, np.linspace(0, 1, bins + 1)[1:-1])
    edges = np.unique(np.concatenate([[0.5], inner]))
    edges = np.concatenate([[-np.inf], edges, [np.inf]])

    ref_share = np.histogram(ref, bins=edges)[0] / len(ref)
    cur_share = np.histogram(cur, bins=edges)[0] / len(cur)
    ref_share = np.clip(ref_share, 1e-6, None)
    cur_share = np.clip(cur_share, 1e-6, None)
    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))


if __name__ == "__main__":
    data = pd.read_parquet(DATA_PATH, columns=["date"] + SMART_COLS)
    data["date"] = pd.to_datetime(data["date"])

    is_ref = data["date"] <= TRAIN_END
    is_new = (data["date"] >= NEW_START) & (data["date"] <= NEW_END)
    ref = data.loc[is_ref, SMART_COLS].astype("float32")
    new = data.loc[is_new, SMART_COLS].astype("float32")
    del data

    rows = []
    for col in SMART_COLS:
        rows.append({
            "feature": col,
            "missing_ref": ref[col].isna().mean(),
            "missing_new": new[col].isna().mean(),
            "nonzero_ref": (ref[col] > 0).mean(),
            "nonzero_new": (new[col] > 0).mean(),
            "psi": psi(ref[col], new[col]),
        })
    report = pd.DataFrame(rows).round(4)

    print(report.to_string(index=False))
    print("rows: reference", len(ref), "| new", len(new))