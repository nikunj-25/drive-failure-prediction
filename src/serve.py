import mlflow
import mlflow.xgboost
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from dataset import PROJECT_DIR, SMART_COLS

TRACKING_URI = "sqlite:///" + (PROJECT_DIR / "mlflow.db").as_posix()
MODEL_URI = "runs:/d6c40d6f22f84d3294cfe724987f6014/model"

mlflow.set_tracking_uri(TRACKING_URI)
model = mlflow.xgboost.load_model(MODEL_URI)

app = FastAPI(title="Drive failure risk API")


class DriveReadings(BaseModel):
    smart_5_raw: float | None = None
    smart_9_raw: float | None = None
    smart_194_raw: float | None = None
    smart_197_raw: float | None = None
    smart_198_raw: float | None = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(readings: DriveReadings):
    row = pd.DataFrame([readings.model_dump()], columns=SMART_COLS)
    row = row.astype("float32")
    score = float(model.predict_proba(row)[0, 1])
    return {"failure_score": round(score, 4)}