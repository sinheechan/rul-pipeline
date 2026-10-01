import sys
from sqlalchemy import text
from src.config import DATASET
from src.db import get_engine
from src.ingest import read_cmapss

STEP = 5   # 한 번 실행에 엔진별 5사이클씩 유입

def replay_next_batch(dataset=DATASET, step=STEP):
    with get_engine().begin() as conn:   # 적재와 상태 갱신을 한 트랜잭션으로
        row = conn.execute(text("SELECT last_cycle FROM ops.replay_state "
                                "WHERE dataset=:d"), {"d": dataset}).fetchone()
        last = row[0] if row else 0
        src = read_cmapss(dataset, "test")
        batch = src[(src["cycle"] > last) & (src["cycle"] <= last + step)].copy()
        if batch.empty:
            print("replay finished")
            return 0
        batch["split"] = "stream"
        batch["batch_id"] = f"stream_{last + 1:03d}_{last + step:03d}"
        batch.to_sql("sensor_readings", conn, schema="raw", if_exists="append",
                     index=False, method="multi", chunksize=1000)
        conn.execute(text("""
            INSERT INTO ops.replay_state (dataset, last_cycle) VALUES (:d, :c)
            ON CONFLICT (dataset) DO UPDATE SET last_cycle = EXCLUDED.last_cycle"""),
            {"d": dataset, "c": last + step})
    print(f"[stream] cycles {last + 1}~{last + step}: {len(batch)} rows")
    return len(batch)

def reset(dataset=DATASET):
    """시연을 처음부터 다시 할 때 사용"""
    with get_engine().begin() as conn:
        for t in ["raw.sensor_readings", "clean.sensor_readings",
                  "ml.anomaly_scores", "ml.rul_predictions"]:
            conn.execute(text(f"DELETE FROM {t} WHERE dataset=:d AND split='stream'"),
                         {"d": dataset})
        conn.execute(text("DELETE FROM ops.replay_state WHERE dataset=:d"), {"d": dataset})

if __name__ == "__main__":
    reset() if sys.argv[1:] == ["reset"] else replay_next_batch()