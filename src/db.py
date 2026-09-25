import os
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()
_engine = None

def get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(os.environ["DB_URL"], pool_pre_ping=True)
    return _engine

def read_sql(sql: str, **params) -> pd.DataFrame:
    return pd.read_sql(text(sql), get_engine(), params=params)

def replace_rows(df: pd.DataFrame, schema: str, table: str, where: dict) -> int:
    """where 조건에 해당하는 행을 지우고 df를 넣는다. 한 트랜잭션이라 중간 실패 시 원상복구된다."""
    cond = " AND ".join(f"{k} = :{k}" for k in where)
    with get_engine().begin() as conn:
        conn.execute(text(f"DELETE FROM {schema}.{table} WHERE {cond}"), where)
        df.to_sql(table, conn, schema=schema, if_exists="append",
                  index=False, method="multi", chunksize=1000)
    return len(df)