CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS clean;
CREATE SCHEMA IF NOT EXISTS feat;
CREATE SCHEMA IF NOT EXISTS quality;
CREATE SCHEMA IF NOT EXISTS ml;
CREATE SCHEMA IF NOT EXISTS ops;

-- 원본 그대로 (중복·결측 포함 가능, PK 없음)
CREATE TABLE IF NOT EXISTS raw.sensor_readings (
    dataset  VARCHAR(10) NOT NULL,
    split    VARCHAR(10) NOT NULL,   -- train / test / stream / corrupt
    unit_id  INT NOT NULL,
    cycle    INT NOT NULL,
    op1 DOUBLE PRECISION, op2 DOUBLE PRECISION, op3 DOUBLE PRECISION,
    s1  DOUBLE PRECISION, s2  DOUBLE PRECISION, s3  DOUBLE PRECISION,
    s4  DOUBLE PRECISION, s5  DOUBLE PRECISION, s6  DOUBLE PRECISION,
    s7  DOUBLE PRECISION, s8  DOUBLE PRECISION, s9  DOUBLE PRECISION,
    s10 DOUBLE PRECISION, s11 DOUBLE PRECISION, s12 DOUBLE PRECISION,
    s13 DOUBLE PRECISION, s14 DOUBLE PRECISION, s15 DOUBLE PRECISION,
    s16 DOUBLE PRECISION, s17 DOUBLE PRECISION, s18 DOUBLE PRECISION,
    s19 DOUBLE PRECISION, s20 DOUBLE PRECISION, s21 DOUBLE PRECISION,
    batch_id  VARCHAR(50),
    loaded_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_raw_key
    ON raw.sensor_readings (dataset, split, unit_id, cycle);

CREATE TABLE IF NOT EXISTS raw.rul_truth (
    dataset VARCHAR(10) NOT NULL,
    unit_id INT NOT NULL,
    rul     INT NOT NULL,
    PRIMARY KEY (dataset, unit_id)
);

-- 검증·정제를 통과한 데이터 (키 중복 불가)
CREATE TABLE IF NOT EXISTS clean.sensor_readings
    (LIKE raw.sensor_readings INCLUDING DEFAULTS);
ALTER TABLE clean.sensor_readings ADD COLUMN is_imputed BOOLEAN DEFAULT false;
ALTER TABLE clean.sensor_readings ADD PRIMARY KEY (dataset, split, unit_id, cycle);

-- 품질검사 이력
CREATE TABLE IF NOT EXISTS quality.check_results (
    id          BIGSERIAL PRIMARY KEY,
    run_id      VARCHAR(40) NOT NULL,
    dataset     VARCHAR(10),
    split       VARCHAR(10),
    stage       VARCHAR(10),          -- raw / clean
    check_name  VARCHAR(50),
    status      VARCHAR(10),          -- PASS / WARN / FAIL
    failed_rows INT,
    detail      TEXT,
    checked_at  TIMESTAMPTZ DEFAULT now()
);

-- 모델 결과
CREATE TABLE IF NOT EXISTS ml.anomaly_scores (
    dataset VARCHAR(10), split VARCHAR(10), unit_id INT, cycle INT,
    score DOUBLE PRECISION, is_anomaly BOOLEAN,
    model_version VARCHAR(50), scored_at TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (dataset, split, unit_id, cycle, model_version)
);

CREATE TABLE IF NOT EXISTS ml.rul_predictions (
    dataset VARCHAR(10), split VARCHAR(10), unit_id INT, cycle INT,
    pred_rul DOUBLE PRECISION,
    model_version VARCHAR(50), predicted_at TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (dataset, split, unit_id, cycle, model_version)
);

CREATE TABLE IF NOT EXISTS ml.model_registry (
    model_version VARCHAR(50) PRIMARY KEY,
    task      VARCHAR(20),            -- anomaly / rul
    algorithm VARCHAR(30),
    metrics   JSONB,
    params    JSONB,
    trained_at TIMESTAMPTZ DEFAULT now()
);

-- 스트리밍 재현 진행 상태 (7단계에서 사용)
CREATE TABLE IF NOT EXISTS ops.replay_state (
    dataset    VARCHAR(10) PRIMARY KEY,
    last_cycle INT NOT NULL
);