import sys
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
from sklearn.covariance import EmpiricalCovariance
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from src.config import USE_SENSORS, MODEL_DIR, DATASET
from src.db import read_sql, replace_rows, register_model

FEATS = [f"{s}_mean5" for s in USE_SENSORS]
HEALTHY_CYCLES = 30   # 수명 초반 30사이클을 정상 상태로 가정
ALARM_RUN = 3         # 3사이클 연속 이상일 때만 경보
FIT_UNITS = 70        # 엔진 1~70으로 학습, 71~100으로 평가

class MahalanobisMonitor:
    name = "mahalanobis"
    def fit(self, X):
        self.scaler = StandardScaler().fit(X)
        self.cov = EmpiricalCovariance().fit(self.scaler.transform(X))
        self.threshold = float(np.percentile(self.score(X), 99))
        return self
    def score(self, X):
        return self.cov.mahalanobis(self.scaler.transform(X))

class IForestMonitor:
    name = "iforest"
    def fit(self, X):
        self.model = IsolationForest(n_estimators=300, random_state=42).fit(X)
        self.threshold = float(np.percentile(self.score(X), 99))
        return self
    def score(self, X):
        return -self.model.score_samples(X)   # 클수록 이상

def first_alarm_cycle(cycles, flags, k=ALARM_RUN):
    f = pd.Series(flags, index=cycles)
    run = f.astype(int).groupby((~f).cumsum()).cumsum()   # 연속 True 길이
    hit = run[run >= k]
    return int(hit.index[0]) if len(hit) else np.nan

def evaluate(monitor, df):
    rows = []
    for u, d in df.groupby("unit_id"):
        flags = monitor.score(d[FEATS].to_numpy()) > monitor.threshold
        alarm = first_alarm_cycle(d["cycle"].to_numpy(), flags)
        rows.append({"unit_id": u, "alarm_cycle": alarm,
                     "lead_time": d["cycle"].max() - alarm})
    r = pd.DataFrame(rows)
    hit = r["alarm_cycle"].notna()
    metrics = {
        "detection_rate": round(float(hit.mean()), 3),
        "lead_time_median": float(r.loc[hit, "lead_time"].median()) if hit.any() else None,
        "early_alarm_rate": round(float((r["lead_time"] > 150).mean()), 3),
    }
    return metrics, r

def train_monitors(dataset=DATASET):
    df = read_sql(f"SELECT * FROM feat.{dataset.lower()}_train")
    fit_mask = df["unit_id"] <= FIT_UNITS
    healthy = df[fit_mask & (df["cycle"] <= HEALTHY_CYCLES)]
    stamp = datetime.now().strftime("%Y%m%d%H%M")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    candidates = []
    for M in [MahalanobisMonitor, IForestMonitor]:
        m = M().fit(healthy[FEATS].to_numpy())
        metrics, _ = evaluate(m, df[~fit_mask])
        version = f"anomaly_{m.name}_{stamp}"
        joblib.dump({"model": m, "version": version}, MODEL_DIR / f"{version}.joblib")
        register_model(version, "anomaly", m.name, metrics,
                       {"healthy_cycles": HEALTHY_CYCLES, "alarm_run": ALARM_RUN})
        print(version, metrics)
        candidates.append((metrics["detection_rate"], -metrics["early_alarm_rate"], m, version))
    # 탐지율이 높은 쪽, 같으면 조기(오)경보가 적은 쪽을 운영 모델로
    _, _, best, best_version = max(candidates, key=lambda c: (c[0], c[1]))
    joblib.dump({"model": best, "version": best_version}, MODEL_DIR / "anomaly_latest.joblib")
    print("selected:", best_version)

def score_split(dataset, split):
    bundle = joblib.load(MODEL_DIR / "anomaly_latest.joblib")
    m, version = bundle["model"], bundle["version"]
    df = read_sql(f"SELECT dataset, split, unit_id, cycle, {', '.join(FEATS)} "
                  f"FROM feat.{dataset.lower()}_{split}")
    df["score"] = m.score(df[FEATS].to_numpy())
    df["is_anomaly"] = df["score"] > m.threshold
    df["model_version"] = version
    out = df[["dataset", "split", "unit_id", "cycle", "score", "is_anomaly", "model_version"]]
    n = replace_rows(out, "ml", "anomaly_scores",
                     {"dataset": dataset, "split": split, "model_version": version})
    print(f"[ml.anomaly_scores] {split}: {n} rows ({version})")

def main(argv):
    if argv[1] == "fit":
        train_monitors()
    elif argv[1] == "score":
        score_split(DATASET, argv[2])

if __name__ == "__main__":
    import src.anomaly as mod   # 저장되는 클래스 경로를 src.anomaly로 고정 (부록 참고)
    mod.main(sys.argv)