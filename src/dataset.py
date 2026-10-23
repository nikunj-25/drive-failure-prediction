from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "full_q1_2024.parquet"

SMART_COLS = ["smart_5_raw", "smart_9_raw", "smart_194_raw",
              "smart_197_raw", "smart_198_raw"]

HORIZON_DAYS = 7
TRAIN_END = pd.Timestamp("2024-02-22")
TEST_START = pd.Timestamp("2024-03-01")
TEST_END = pd.Timestamp("2024-03-24")


def load_labeled(path=DATA_PATH, horizon_days=HORIZON_DAYS):
    """Load the Parquet file and add the 'fails within N days' label."""
    full = pd.read_parquet(path)
    full["date"] = pd.to_datetime(full["date"])

    fail_date = full.loc[full["failure"] == 1].set_index("serial_number")["date"]
    days_left = (full["serial_number"].map(fail_date) - full["date"]).dt.days
    full["label"] = ((days_left >= 0) & (days_left <= horizon_days)).astype("int8")
    return full


def time_split(full):
    """Split by date: train on earlier rows, test on later rows."""
    is_train = full["date"] <= TRAIN_END
    is_test = (full["date"] >= TEST_START) & (full["date"] <= TEST_END)

    X_train = full.loc[is_train, SMART_COLS].astype("float32")
    y_train = full.loc[is_train, "label"]
    X_test = full.loc[is_test, SMART_COLS].astype("float32")
    y_test = full.loc[is_test, "label"]
    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    full = load_labeled()
    X_train, y_train, X_test, y_test = time_split(full)
    print("train:", X_train.shape, "positives:", int(y_train.sum()))
    print("test :", X_test.shape, "positives:", int(y_test.sum()))