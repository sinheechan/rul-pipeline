import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = Path(os.getenv("RAW_DIR", ROOT / "data" / "raw" / "CMAPSSData"))
MODEL_DIR = Path(os.getenv("MODEL_DIR", ROOT / "models"))

DATASET = "FD001"
KEY = ["dataset", "split", "unit_id", "cycle"]
OPS = ["op1", "op2", "op3"]
SENSORS = [f"s{i}" for i in range(1, 22)]
FILE_COLS = ["unit_id", "cycle"] + OPS + SENSORS

# 1단계 EDA 결과로 정한 사용 센서 (본인 EDA 결과와 다르면 수정)
USE_SENSORS = ["s2", "s3", "s4", "s7", "s8", "s9", "s11",
               "s12", "s13", "s14", "s15", "s17", "s20", "s21"]
RUL_CAP = 125