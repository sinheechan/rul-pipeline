import copy
from datetime import datetime
import joblib
import lightgbm as lgb
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from src.config import MODEL_DIR, DATASET, USE_SENSORS
from src.db import read_sql, register_model
from src.features import feature_columns

FEATURES = feature_columns()
SEQ_LEN = 30

# ---------- 공통 ----------
def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))

def nasa_score(y, p):
    d = np.asarray(p) - np.asarray(y)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))

def load(split):
    return read_sql(f"SELECT * FROM feat.{DATASET.lower()}_{split} ORDER BY unit_id, cycle")

def last_rows(df):
    return df.groupby("unit_id").tail(1)

def eval_test(test, pred_last):
    y = last_rows(test)["rul_true"].to_numpy()
    return {"test_rmse": round(rmse(y, pred_last), 2),
            "test_nasa_score": round(nasa_score(y, pred_last), 1)}

# ---------- 1) 기준선: 선형회귀 ----------
def train_linear(train, test):
    cols = ["cycle"] + USE_SENSORS
    m = LinearRegression().fit(train[cols], train["rul"])
    pred = np.clip(m.predict(last_rows(test)[cols]), 0, None)
    return eval_test(test, pred)

# ---------- 2) LightGBM ----------
LGB_PARAMS = dict(learning_rate=0.03, num_leaves=31, min_child_samples=50,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                  random_state=42, verbose=-1)

def train_lgbm(train, test):
    X, y, groups = train[FEATURES], train["rul"], train["unit_id"]
    oof, best_iters = np.zeros(len(train)), []
    for tr, va in GroupKFold(n_splits=5).split(X, y, groups):   # 엔진 단위로 분할
        m = lgb.LGBMRegressor(n_estimators=2000, **LGB_PARAMS)
        m.fit(X.iloc[tr], y.iloc[tr], eval_set=[(X.iloc[va], y.iloc[va])],
              callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[va] = m.predict(X.iloc[va])
        best_iters.append(m.best_iteration_)
    n_est = int(np.mean(best_iters))
    final = lgb.LGBMRegressor(n_estimators=n_est, **LGB_PARAMS).fit(X, y)
    pred = np.clip(final.predict(last_rows(test)[FEATURES]), 0, None)
    metrics = {"cv_rmse": round(rmse(y, oof), 2), **eval_test(test, pred)}
    imp = (pd.Series(final.feature_importances_, index=FEATURES)
             .sort_values(ascending=False).head(15))
    imp.to_csv(MODEL_DIR / "lgbm_importance_top15.csv")
    return final, metrics, {"n_estimators": n_est, **LGB_PARAMS}

# ---------- 3) LSTM ----------
class LSTMRegressor(nn.Module):
    def __init__(self, n_feat, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, hidden, num_layers=2, batch_first=True, dropout=0.2)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1]).squeeze(-1)

def make_windows(df, cols, seq=SEQ_LEN):
    X, y = [], []
    for _, u in df.groupby("unit_id"):
        a = u[cols].to_numpy(np.float32)
        r = u["rul"].to_numpy(np.float32)
        for end in range(seq, len(a) + 1):
            X.append(a[end - seq:end])
            y.append(r[end - 1])
    return np.stack(X), np.array(y)

def last_windows(df, cols, seq=SEQ_LEN):
    X = []
    for _, u in df.groupby("unit_id"):
        a = u[cols].to_numpy(np.float32)
        if len(a) < seq:   # 기록이 짧으면 첫 값으로 앞을 채움
            a = np.vstack([np.repeat(a[:1], seq - len(a), axis=0), a])
        X.append(a[-seq:])
    return np.stack(X)

def train_lstm(train, test, epochs=40, patience=5):
    torch.manual_seed(42)
    cols = USE_SENSORS
    scaler = StandardScaler().fit(train[cols])
    tr = train[train["unit_id"] <= 80].copy()
    va = train[train["unit_id"] > 80].copy()
    for d in (tr, va):
        d[cols] = scaler.transform(d[cols])
    Xtr, ytr = make_windows(tr, cols)
    Xva, yva = make_windows(va, cols)
    loader = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
                        batch_size=256, shuffle=True)
    model = LSTMRegressor(len(cols))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    best, best_state, wait = float("inf"), None, 0
    for ep in range(epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val = rmse(yva, model(torch.from_numpy(Xva)).numpy())
        print(f"epoch {ep + 1:02d}  val_rmse {val:.2f}")
        if val < best:
            best, best_state, wait = val, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= patience:   # 조기 종료
                break
    model.load_state_dict(best_state)
    te = test.copy()
    te[cols] = scaler.transform(te[cols])
    with torch.no_grad():
        pred = model(torch.from_numpy(last_windows(te, cols))).numpy().clip(0)
    return model, scaler, {"val_rmse": round(best, 2), **eval_test(test, pred)}

# ---------- 실행 ----------
def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    train, test = load("train"), load("test")
    stamp = datetime.now().strftime("%Y%m%d%H%M")

    m_lin = train_linear(train, test)
    register_model(f"rul_linear_{stamp}", "rul", "linear", m_lin, {})
    print("linear  ", m_lin)

    lgbm, m_lgb, p_lgb = train_lgbm(train, test)
    v_lgb = f"rul_lgbm_{stamp}"
    bundle = {"model": lgbm, "version": v_lgb, "features": FEATURES}
    joblib.dump(bundle, MODEL_DIR / f"{v_lgb}.joblib")
    joblib.dump(bundle, MODEL_DIR / "rul_latest.joblib")   # 운영 모델
    register_model(v_lgb, "rul", "lightgbm", m_lgb, p_lgb)
    print("lightgbm", m_lgb)

    lstm, scaler, m_lstm = train_lstm(train, test)
    v_lstm = f"rul_lstm_{stamp}"
    torch.save({"state_dict": lstm.state_dict(), "scaler": scaler,
                "cols": USE_SENSORS, "seq_len": SEQ_LEN}, MODEL_DIR / f"{v_lstm}.pt")
    register_model(v_lstm, "rul", "lstm", m_lstm,
                   {"seq_len": SEQ_LEN, "hidden": 64, "layers": 2})
    print("lstm    ", m_lstm)

if __name__ == "__main__":
    main()