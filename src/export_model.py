import mlflow
import mlflow.xgboost

from dataset import PROJECT_DIR

TRACKING_URI = "sqlite:///" + (PROJECT_DIR / "mlflow.db").as_posix()
RUN_ID = "d6c40d6f22f84d3294cfe724987f6014"

mlflow.set_tracking_uri(TRACKING_URI)
model = mlflow.xgboost.load_model(f"runs:/{RUN_ID}/model")

out_dir = PROJECT_DIR / "models"
out_dir.mkdir(exist_ok=True)
out_path = out_dir / "model.ubj"
model.save_model(str(out_path))
print("saved:", out_path)