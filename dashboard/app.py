import os
import sys
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from src.config import USE_SENSORS, DATASET

load_dotenv(ROOT / ".env")
st.set_page_config(page_title="설비 잔여수명 모니터링", layout="wide")

@st.cache_resource
def engine():
    return create_engine(os.environ["DB_URL"], pool_pre_ping=True)

@st.cache_data(ttl=60)   # 60초마다 DB 재조회
def q(sql, **params):
    return pd.read_sql(text(sql), engine(), params=params)

# ---------- 사이드바 ----------
st.sidebar.title("설정")
split = st.sidebar.radio("데이터", ["stream", "test"],
                         help="stream: Airflow로 유입 중인 데이터 / test: 전체 test 데이터")
crit = st.sidebar.slider("위험 기준 (잔여 사이클 미만)", 10, 50, 30)
warn = st.sidebar.slider("주의 기준 (잔여 사이클 미만)", 51, 120, 60)
feat_table = f"feat.{DATASET.lower()}_{split}"

rul_ver = q("SELECT model_version FROM ml.rul_predictions WHERE dataset=:d AND split=:s "
            "ORDER BY predicted_at DESC LIMIT 1", d=DATASET, s=split)
if rul_ver.empty:
    st.info("예측 결과가 없습니다. 파이프라인을 먼저 실행하세요.")
    st.stop()
rul_ver = rul_ver.iloc[0, 0]
ano_ver = q("SELECT model_version FROM ml.anomaly_scores WHERE dataset=:d AND split=:s "
            "ORDER BY scored_at DESC LIMIT 1", d=DATASET, s=split).iloc[0, 0]

st.title("설비 잔여수명(RUL) 모니터링")
st.caption(f"RUL 모델: {rul_ver} · 이상탐지 모델: {ano_ver}")
tab1, tab2, tab3, tab4 = st.tabs(["전체 현황", "설비 상세", "데이터 품질", "모델 성능"])

# ---------- 1. 전체 현황 ----------
with tab1:
    latest = q("""
        SELECT DISTINCT ON (p.unit_id) p.unit_id, p.cycle, p.pred_rul, a.is_anomaly
        FROM ml.rul_predictions p
        LEFT JOIN ml.anomaly_scores a
          ON a.dataset = p.dataset AND a.split = p.split
         AND a.unit_id = p.unit_id AND a.cycle = p.cycle AND a.model_version = :av
        WHERE p.dataset = :d AND p.split = :s AND p.model_version = :rv
        ORDER BY p.unit_id, p.cycle DESC""", d=DATASET, s=split, rv=rul_ver, av=ano_ver)
    latest["상태"] = pd.cut(latest["pred_rul"], [-1, crit, warn, 10_000],
                          labels=["위험", "주의", "정상"], right=False)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("모니터링 설비", len(latest))
    c2.metric("위험", int((latest["상태"] == "위험").sum()))
    c3.metric("주의", int((latest["상태"] == "주의").sum()))
    c4.metric("이상 징후 감지", int(latest["is_anomaly"].fillna(False).astype(bool).sum()))
    fig = px.bar(latest.sort_values("pred_rul"), x="unit_id", y="pred_rul", color="상태",
                 color_discrete_map={"위험": "#d62728", "주의": "#ff7f0e", "정상": "#2ca02c"},
                 labels={"unit_id": "설비 번호", "pred_rul": "예측 잔여수명 (사이클)"})
    fig.update_xaxes(type="category", categoryorder="total ascending")
    st.plotly_chart(fig, width="stretch")
    st.subheader("정비 우선순위 Top 10")
    st.dataframe(latest.sort_values("pred_rul").head(10), hide_index=True,
                 width="stretch")

# ---------- 2. 설비 상세 ----------
with tab2:
    units = latest["unit_id"].sort_values().tolist()
    u = int(st.selectbox("설비 번호", units))
    sensors = st.multiselect("센서", USE_SENSORS, default=["s4", "s11", "s15"])
    sd = q("SELECT * FROM clean.sensor_readings WHERE dataset=:d AND split=:s "
           "AND unit_id=:u ORDER BY cycle", d=DATASET, s=split, u=u)
    if sensors:
        st.plotly_chart(px.line(sd, x="cycle", y=sensors, title="센서 추이"),
                        width="stretch")
    an = q("SELECT cycle, score, is_anomaly FROM ml.anomaly_scores WHERE dataset=:d "
           "AND split=:s AND unit_id=:u AND model_version=:v ORDER BY cycle",
           d=DATASET, s=split, u=u, v=ano_ver)
    fig = px.line(an, x="cycle", y="score", title="이상 점수 (빨간 점 = 임계값 초과)")
    hits = an[an["is_anomaly"]]
    fig.add_scatter(x=hits["cycle"], y=hits["score"], mode="markers",
                    marker_color="#d62728", name="이상")
    st.plotly_chart(fig, width="stretch")
    pr = q(f"""
        SELECT p.cycle, p.pred_rul AS 예측, f.rul_true AS 실제
        FROM ml.rul_predictions p
        JOIN {feat_table} f ON f.unit_id = p.unit_id AND f.cycle = p.cycle
        WHERE p.dataset=:d AND p.split=:s AND p.unit_id=:u AND p.model_version=:v
        ORDER BY p.cycle""", d=DATASET, s=split, u=u, v=rul_ver)
    st.plotly_chart(px.line(pr, x="cycle", y=["예측", "실제"], title="잔여수명 예측 vs 실제"),
                    width="stretch")

# ---------- 3. 데이터 품질 ----------
with tab3:
    qc = q("""SELECT checked_at, run_id, split, stage, check_name, status, failed_rows, detail
              FROM quality.check_results WHERE dataset=:d
              ORDER BY checked_at DESC LIMIT 500""", d=DATASET)
    c1, c2 = st.columns(2)
    c1.metric("누적 검사 실행", qc["run_id"].nunique())
    c2.metric("누적 FAIL", int((qc["status"] == "FAIL").sum()))
    last_run = qc[(qc["split"] == split)].sort_values("checked_at").groupby("stage").tail(6)
    st.subheader("최근 검사 결과")
    st.dataframe(last_run.sort_values(["stage", "check_name"]), hide_index=True,
                 width="stretch")
    fails = qc[qc["status"] == "FAIL"].groupby(["split", "check_name"]).size().reset_index(name="건수")
    if not fails.empty:
        st.plotly_chart(px.bar(fails, x="check_name", y="건수", color="split",
                               title="검사 항목별 FAIL 누적"), width="stretch")

# ---------- 4. 모델 성능 ----------
with tab4:
    reg = q("SELECT model_version, task, algorithm, metrics, trained_at "
            "FROM ml.model_registry ORDER BY trained_at DESC")
    table = pd.concat([reg.drop(columns="metrics"), pd.json_normalize(reg["metrics"])], axis=1)
    for task in ["rul", "anomaly"]:
        st.subheader("RUL 예측 모델" if task == "rul" else "이상탐지 모델")
        st.dataframe(table[table["task"] == task].dropna(axis=1, how="all"),
                     hide_index=True, width="stretch")
    imp_path = ROOT / "models" / "lgbm_importance_top15.csv"
    if imp_path.exists():
        imp = pd.read_csv(imp_path, index_col=0).iloc[:, 0].sort_values()
        st.plotly_chart(px.bar(imp, orientation="h", title="LightGBM 특징 중요도 Top 15"),
                        width="stretch")