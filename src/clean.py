import sys
import pandas as pd
from src.config import OPS, SENSORS, DATASET
from src.db import read_sql, replace_rows

META = ["dataset", "split", "unit_id", "batch_id"]

def hampel(s, window=7, n_sigmas=5):
    """주변 7개 값의 중앙값에서 크게 벗어난 튀는 값을 중앙값으로 대체"""
    med = s.rolling(window, center=True, min_periods=1).median()
    mad = (s - med).abs().rolling(window, center=True, min_periods=1).median()
    spike = ((s - med).abs() > n_sigmas * 1.4826 * mad) & (mad > 0)
    return s.where(~spike, med), int(spike.sum())

def clean_unit(u):
    u = (u.drop(columns=["loaded_at"], errors="ignore")
           .sort_values("cycle", kind="stable")
           .drop_duplicates("cycle", keep="last")      # 1) 중복 제거
           .set_index("cycle"))
    u = u.reindex(range(1, int(u.index.max()) + 1))    # 2) 빠진 사이클 복원
    u.index.name = "cycle"
    u["is_imputed"] = u[SENSORS].isna().any(axis=1)
    u[META] = u[META].ffill().bfill()
    num = OPS + SENSORS
    u[num] = u[num].interpolate(limit_direction="both")  # 3) 결측 보간
    spikes = 0
    for s in SENSORS:                                     # 4) 스파이크 보정
        u[s], k = hampel(u[s])
        spikes += k
    u = u.reset_index()
    u["unit_id"] = u["unit_id"].astype(int)
    return u, spikes

def build_clean(dataset, split):
    raw = read_sql("SELECT * FROM raw.sensor_readings "
                   "WHERE dataset=:d AND split=:s", d=dataset, s=split)
    parts, spikes = [], 0
    for _, u in raw.groupby("unit_id"):
        c, k = clean_unit(u)
        parts.append(c)
        spikes += k
    clean = pd.concat(parts, ignore_index=True)
    n = replace_rows(clean, "clean", "sensor_readings",
                     {"dataset": dataset, "split": split})
    print(f"[clean] {split}: raw {len(raw)} → clean {n} rows, "
          f"imputed {int(clean['is_imputed'].sum())}, spikes fixed {spikes}")
    return clean

if __name__ == "__main__":
    build_clean(DATASET, sys.argv[1])