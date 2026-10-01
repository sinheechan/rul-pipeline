from datetime import datetime, timedelta

try:                                  # Airflow 3.x
    from airflow.sdk import dag, task
except ImportError:                   # Airflow 2.x
    from airflow.decorators import dag, task

DATASET = "FD001"

@dag(
    dag_id="rul_stream_pipeline",
    schedule="*/10 * * * *",          # 10분마다 새 배치 유입
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,                # 이전 실행이 끝나야 다음 실행
    default_args={"retries": 1, "retry_delay": timedelta(minutes=1)},
    tags=["portfolio", "predictive-maintenance"],
)
def rul_stream_pipeline():

    @task.short_circuit
    def ingest() -> bool:
        from src.replay import replay_next_batch
        return replay_next_batch(DATASET) > 0     # 새 데이터 없으면 이후 태스크 건너뜀

    @task
    def quality_raw():
        from src.quality import run_checks_from_db
        run_checks_from_db(DATASET, "stream", "raw", fail_on_error=False)

    @task
    def clean():
        from src.clean import build_clean
        build_clean(DATASET, "stream")

    @task
    def quality_gate():
        from src.quality import run_checks_from_db
        run_checks_from_db(DATASET, "stream", "clean", fail_on_error=True)

    @task
    def features():
        from src.features import build_features
        build_features(DATASET, "stream")

    @task
    def anomaly():
        from src.anomaly import score_split
        score_split(DATASET, "stream")

    @task
    def rul():
        from src.predict import predict_split
        predict_split(DATASET, "stream")

    ingest() >> quality_raw() >> clean() >> quality_gate() >> features() >> [anomaly(), rul()]

rul_stream_pipeline()