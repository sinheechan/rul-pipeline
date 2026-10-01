import sys
import joblib
import numpy as np
from src.config import MODEL_DIR, DATASET
from src.db import read_sql, replace_rows

def predict_split(dataset, split):
    b = joblib.load(MODEL_DIR / "rul_latest.joblib")
    df = read_sql(f"SELECT * FROM feat.{dataset.lower()}_{split}")
    df["pred_rul"] = np.clip(b["model"].predict(df[b["features"]]), 0, None)
    df["model_version"] = b["version"]
    out = df[["dataset", "split", "unit_id", "cycle", "pred_rul", "model_version"]]
    n = replace_rows(out, "ml", "rul_predictions",
                     {"dataset": dataset, "split": split, "model_version": b["version"]})
    print(f"[ml.rul_predictions] {split}: {n} rows ({b['version']})")

if __name__ == "__main__":
    predict_split(DATASET, sys.argv[1])