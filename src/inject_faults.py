import numpy as np
import pandas as pd
from src.config import SENSORS, DATASET
from src.ingest import read_cmapss
from src.db import replace_rows

def corrupt(df, seed=42):
    rng = np.random.default_rng(seed)
    df = df.copy()
    # 1) 결측: 센서 값의 0.5%를 비움
    vals = df[SENSORS].to_numpy(dtype=float)
    vals[rng.random(vals.shape) < 0.005] = np.nan
    df[SENSORS] = vals
    # 2) 스파이크: 30개 값을 1.5배로
    rows = rng.choice(len(df), 30, replace=False)
    cols = rng.choice(["s2", "s3", "s4", "s7", "s11", "s12"], 30)
    for r, c in zip(rows, cols):
        df.iloc[r, df.columns.get_loc(c)] *= 1.5
    # 3) 누락: 첫 사이클이 아닌 행 20개 삭제
    df = df.drop(rng.choice(df.index[df["cycle"] > 1], 20, replace=False))
    # 4) 중복: 15행을 한 번 더 추가
    return pd.concat([df, df.sample(15, random_state=seed)])

if __name__ == "__main__":
    df = read_cmapss(DATASET, "train")
    df["split"] = "corrupt"
    df["batch_id"] = "fault_injection"
    bad = corrupt(df)
    n = replace_rows(bad, "raw", "sensor_readings",
                     {"dataset": DATASET, "split": "corrupt"})
    print(f"injected corrupt data: {n} rows")