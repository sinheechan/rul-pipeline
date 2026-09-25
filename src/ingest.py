from datetime import datetime
import pandas as pd
from src.config import RAW_DIR, FILE_COLS, DATASET
from src.db import replace_rows

def read_cmapss(dataset: str, split: str) -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / f"{split}_{dataset}.txt",
                     sep=r"\s+", header=None, names=FILE_COLS)
    df.insert(0, "split", split)
    df.insert(0, "dataset", dataset)
    return df

def read_rul_truth(dataset: str) -> pd.DataFrame:
    rul = pd.read_csv(RAW_DIR / f"RUL_{dataset}.txt",
                      sep=r"\s+", header=None, names=["rul"])
    rul.insert(0, "unit_id", range(1, len(rul) + 1))
    rul.insert(0, "dataset", dataset)
    return rul

def ingest_files(dataset: str = DATASET):
    batch_id = f"file_{datetime.now():%Y%m%d%H%M%S}"
    for split in ["train", "test"]:
        df = read_cmapss(dataset, split)
        df["batch_id"] = batch_id
        n = replace_rows(df, "raw", "sensor_readings",
                         {"dataset": dataset, "split": split})
        print(f"[raw.sensor_readings] {split}: {n} rows")
    n = replace_rows(read_rul_truth(dataset), "raw", "rul_truth",
                     {"dataset": dataset})
    print(f"[raw.rul_truth] {n} rows")

if __name__ == "__main__":
    ingest_files()