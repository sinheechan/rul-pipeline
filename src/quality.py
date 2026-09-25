import sys
import uuid
import pandas as pd
from src.config import KEY, OPS, SENSORS, MODEL_DIR, DATASET
from src.db import read_sql, get_engine

REF_PATH = MODEL_DIR / "ref_stats.csv"

# ---------- 기준 통계 (정상 train 데이터의 센서별 최소·최대) ----------
def build_reference(dataset=DATASET):
    df = read_sql("SELECT * FROM raw.sensor_readings "
                  "WHERE dataset=:d AND split='train'", d=dataset)
    ref = pd.DataFrame({"min": df[SENSORS].min(), "max": df[SENSORS].max()})
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    ref.to_csv(REF_PATH)
    return ref

def load_reference():
    return pd.read_csv(REF_PATH, index_col=0)

# ---------- 개별 검사 ----------
def _res(name, failed, detail="", warn=False):
    status = "PASS" if failed == 0 else ("WARN" if warn else "FAIL")
    return {"check_name": name, "status": status,
            "failed_rows": int(failed), "detail": detail}

def check_schema(df):
    missing = sorted(set(KEY + OPS + SENSORS) - set(df.columns))
    return _res("schema", len(missing), f"missing={missing}")

def check_nulls(df):
    na = df[SENSORS].isna()
    top = na.sum()
    top = top[top > 0].sort_values(ascending=False).head(3).to_dict()
    return _res("null", na.any(axis=1).sum(), f"top_cols={top}")

def check_duplicates(df):
    return _res("duplicate_key", df.duplicated(KEY).sum())

def check_cycle_continuity(df):
    g = df.drop_duplicates(KEY).groupby("unit_id")["cycle"]
    missing = g.max() - g.min() + 1 - g.count()
    bad = missing[missing > 0]
    not_start1 = int((g.min() != 1).sum())
    return _res("cycle_continuity", bad.sum() + not_start1,
                f"units_with_gap={bad.index.tolist()[:10]}, not_start_at_1={not_start1}")

def check_constant(df):
    std = df[SENSORS].std()
    const = std[std < 1e-4].index.tolist()
    return _res("constant_sensor", len(const), f"cols={const}", warn=True)

def check_range(df, ref, margin=0.5):
    """train 최소·최대에서 범위 폭의 50%를 넘게 벗어나면 이상값"""
    width = ref["max"] - ref["min"]
    cols = [c for c in SENSORS if width[c] > 0]
    lo = ref.loc[cols, "min"] - margin * width[cols]
    hi = ref.loc[cols, "max"] + margin * width[cols]
    out = (df[cols] < lo) | (df[cols] > hi)
    top = out.sum()
    top = top[top > 0].sort_values(ascending=False).head(3).to_dict()
    return _res("range", out.any(axis=1).sum(), f"margin={margin} top_cols={top}")

# ---------- 실행 ----------
CHECKS = [check_schema, check_nulls, check_duplicates,
          check_cycle_continuity, check_constant]

def run_checks(df, dataset, split, stage):
    run_id = uuid.uuid4().hex[:12]
    results = [f(df) for f in CHECKS] + [check_range(df, load_reference())]
    out = pd.DataFrame(results).assign(run_id=run_id, dataset=dataset,
                                       split=split, stage=stage)
    out.to_sql("check_results", get_engine(), schema="quality",
               if_exists="append", index=False)
    return out

def run_checks_from_db(dataset, split, stage, fail_on_error=False):
    df = read_sql(f"SELECT * FROM {stage}.sensor_readings "
                  "WHERE dataset=:d AND split=:s", d=dataset, s=split)
    out = run_checks(df, dataset, split, stage)
    print(out[["check_name", "status", "failed_rows", "detail"]].to_string(index=False))
    failed = out.loc[out["status"] == "FAIL", "check_name"].tolist()
    if fail_on_error and failed:
        raise ValueError(f"품질검사 실패: {failed}")
    return out

if __name__ == "__main__":
    # python -m src.quality ref              → 기준 통계 생성
    # python -m src.quality <split> <stage>  → 예: train raw
    if sys.argv[1] == "ref":
        print(build_reference())
    else:
        run_checks_from_db(DATASET, sys.argv[1], sys.argv[2])