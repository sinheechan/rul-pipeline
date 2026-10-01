import sys
import pandas as pd
from src.config import USE_SENSORS, RUL_CAP, DATASET
from src.db import read_sql, get_engine

WINDOWS = [5, 10, 20]
SLOPE_W = 10

def feature_columns():
    cols = ["cycle"] + USE_SENSORS
    for s in USE_SENSORS:
        for w in WINDOWS:
            cols += [f"{s}_mean{w}", f"{s}_std{w}"]
        cols += [f"{s}_slope{SLOPE_W}", f"{s}_ewm"]
    return cols

def add_rolling(df):
    df = df.sort_values(["unit_id", "cycle"]).reset_index(drop=True)
    g = df.groupby("unit_id")
    feats = {}
    for s in USE_SENSORS:
        for w in WINDOWS:
            r = g[s].rolling(w, min_periods=1)
            feats[f"{s}_mean{w}"] = r.mean().reset_index(level=0, drop=True)
            feats[f"{s}_std{w}"] = r.std().reset_index(level=0, drop=True)
        feats[f"{s}_slope{SLOPE_W}"] = (df[s] - g[s].shift(SLOPE_W)) / SLOPE_W
        feats[f"{s}_ewm"] = g[s].transform(lambda x: x.ewm(span=10).mean())
    out = pd.concat([df, pd.DataFrame(feats)], axis=1)
    return out.fillna({c: 0 for c in feats})   # 초반 구간의 NaN은 0으로

def add_rul_train(df):
    max_c = df.groupby("unit_id")["cycle"].transform("max")
    df["rul_true"] = max_c - df["cycle"]
    return df

def add_rul_from_truth(df, dataset):
    truth = read_sql("""
        SELECT t.unit_id, t.rul AS rul_end, MAX(r.cycle) AS full_len
        FROM raw.rul_truth t
        JOIN raw.sensor_readings r
          ON r.dataset = t.dataset AND r.unit_id = t.unit_id AND r.split = 'test'
        WHERE t.dataset = :d
        GROUP BY t.unit_id, t.rul""", d=dataset)
    df = df.merge(truth, on="unit_id", how="left")
    df["rul_true"] = df["rul_end"] + df["full_len"] - df["cycle"]
    return df.drop(columns=["rul_end", "full_len"])

def build_features(dataset, split):
    df = read_sql("SELECT * FROM clean.sensor_readings "
                  "WHERE dataset=:d AND split=:s", d=dataset, s=split)
    df = add_rolling(df)
    df = add_rul_train(df) if split == "train" else add_rul_from_truth(df, dataset)
    df["rul"] = df["rul_true"].clip(upper=RUL_CAP)
    keep = ["dataset", "split", "unit_id", "is_imputed", "rul_true", "rul"] + feature_columns()
    df = df[keep]
    table = f"{dataset.lower()}_{split}"
    df.to_sql(table, get_engine(), schema="feat", if_exists="replace",
              index=False, method="multi", chunksize=1000)
    print(f"[feat.{table}] {df.shape}")
    return df

if __name__ == "__main__":
    build_features(DATASET, sys.argv[1])