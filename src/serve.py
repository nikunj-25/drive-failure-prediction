import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from xgboost import XGBClassifier

from dataset import PROJECT_DIR, SMART_COLS

MODEL_PATH = PROJECT_DIR / "models" / "model.ubj"

model = XGBClassifier()
model.load_model(str(MODEL_PATH))

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