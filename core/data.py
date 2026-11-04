"""데이터 수집: Yahoo Finance(가격·VIX·채권 ETF), CNN(공포탐욕지수).

공포탐욕지수는 CNN 값을 우선 쓰고, 받지 못한 날은 CNN 방식으로 직접 계산한 값(core/feargreed.py)으로 채웁니다.
CNN에서 받은 값은 data/fear_greed.csv에 누적 저장합니다(매일 작업이 저장소에 커밋).
"""
import logging
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from . import feargreed
from .settings import DATA

log = logging.getLogger("alarm")
CNN_BASE = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
CNN_HEADERS = {"User-Agent": "invest-alarm/1.0 (personal dashboard)", "Accept": "application/json"}


def _path(name):
    DATA.mkdir(exist_ok=True)
    return DATA / f"{name.replace('^', '_')}.csv"


def _load(name):
    p = _path(name)
    if not p.exists():
        return None
    return pd.read_csv(p, parse_dates=["date"], index_col="date")["value"].astype(float)


def fetch_close(ticker: str) -> tuple[pd.Series, str]:
    last_err = None
    for attempt in range(3):
        try:
            import yfinance as yf
            h = yf.Ticker(ticker).history(period="max", interval="1d", auto_adjust=False)
            if h is None or h.empty:
                raise RuntimeError("빈 데이터")
            s = h["Close"].copy()
            s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
            s = s[~s.index.duplicated(keep="last")].dropna()
            return s, "Yahoo Finance"
        except Exception as e:
            last_err = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"{ticker} 데이터를 받지 못했습니다: {last_err}")


def _parse_cnn(js) -> pd.Series:
    s = pd.Series({pd.to_datetime(d["x"], unit="ms").normalize(): float(d["y"])
                   for d in js["fear_and_greed_historical"]["data"]})
    now = js.get("fear_and_greed", {})
    if "score" in now and "timestamp" in now:
        s.loc[pd.to_datetime(now["timestamp"]).tz_localize(None).normalize()] = float(now["score"])
    return s[~s.index.duplicated(keep="last")].sort_index()


def fetch_cnn(save: bool = False) -> tuple[pd.Series | None, str]:
    """CNN 공포탐욕지수. 성공하면 기존 저장본과 합쳐 반환. (시리즈, 상태 문구)"""
    import requests
    old = _load("fear_greed")
    start = (datetime.utcnow() - timedelta(days=360)).strftime("%Y-%m-%d")
    urls = [f"{CNN_BASE}/{start}", CNN_BASE, f"{CNN_BASE}/"]
    err = None
    for u in urls:
        try:
            r = requests.get(u, headers=CNN_HEADERS, timeout=20)
            r.raise_for_status()
            s = _parse_cnn(r.json())
            if old is not None:
                s = pd.concat([old[~old.index.isin(s.index)], s]).sort_index()
            if save:
                s.rename("value").to_csv(_path("fear_greed"), index_label="date", float_format="%.2f")
            return s, "CNN"
        except Exception as e:
            err = e
    log.warning("CNN 공포탐욕지수 수신 실패: %s", err)
    if old is not None:
        return old, f"CNN 수신 실패 → {old.index[-1].date()}까지 저장본 + 자체 계산"
    return None, "CNN 수신 실패 → 자체 계산값 사용"


def load_all(cfg, test: bool = False, save: bool = False):
    """{티커: 시리즈}, 공포탐욕 시리즈(CNN+자체계산 혼합), 알림 문구 목록, 공포탐욕 정보."""
    if test:
        syn = synthetic()
        closes = {t: syn[t] for t in syn if t != "fear_greed"}
        cnn, cnn_src = syn["fear_greed"], "CNN(가상)"
        notes = ["가상 데이터(테스트)"]
    else:
        notes, closes = [], {}
        need = list(cfg.indices.values()) + list(cfg.vol_index.values()) + feargreed.PROXY_TICKERS
        for t in dict.fromkeys(need):
            try:
                closes[t] = fetch_close(t)[0]
            except Exception as e:
                if t in feargreed.PROXY_TICKERS and t not in cfg.indices.values() and t not in cfg.vol_index.values():
                    notes.append(f"{t} 수신 실패(공포탐욕 자체 계산에서 제외)")
                else:
                    raise
        cnn, cnn_src = fetch_cnn(save=save)
        if cnn_src != "CNN":
            notes.append(f"공포탐욕지수: {cnn_src}")
    proxy, comp = feargreed.proxy_index(closes)
    fg, cnn_dates = feargreed.blend(cnn, proxy)
    info = {"cnn_dates": cnn_dates, "proxy": proxy, "components": comp,
            "compare": feargreed.compare(cnn, proxy), "cnn_status": cnn_src}
    return closes, fg, notes, info


def synthetic(seed: int = 7) -> dict:
    """인터넷 없이 화면·로직을 확인하기 위한 가상 데이터 (실제 시세 아님)."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("1999-01-01", datetime.now().strftime("%Y-%m-%d"))
    n = len(idx)

    def walk(mu, sigma, p0, shock=True):
        r = rng.normal(mu, sigma, n)
        if shock:
            for pos, size, length in [(0.06, -0.9, 250), (0.38, -0.6, 120), (0.78, -0.35, 25),
                                      (0.86, -0.4, 180), (0.96, -0.2, 20)]:
                a = int(pos * n)
                r[a:a + length] += size / length
        return pd.Series(p0 * np.exp(np.cumsum(r)), index=idx)

    out = {"SPY": walk(0.0004, 0.011, 120), "QQQ": walk(0.0006, 0.016, 90)}
    out["^GSPC"] = out["SPY"] * 10
    ret = out["SPY"].pct_change().fillna(0)
    S = lambda a: pd.Series(np.asarray(a), index=idx)
    out["RSP"] = S(out["SPY"].values * np.exp(np.cumsum(rng.normal(0, 0.002, n))))
    out["TLT"] = S(100 * np.exp(np.cumsum(-ret.values * 0.3 + rng.normal(0, 0.006, n))))
    out["LQD"] = S(100 * np.exp(np.cumsum(rng.normal(0.0001, 0.003, n))))
    out["HYG"] = S(out["LQD"].values * np.exp(np.cumsum(ret.values * 0.3 + rng.normal(0, 0.002, n))))
    vix = (15 + (-ret.rolling(20).sum().fillna(0)) * 120).clip(9, 80)
    out["^VIX"], out["^VXN"] = vix, vix * 1.2
    out["DIA"] = walk(0.0003, 0.010, 90)
    out["^KS11"] = walk(0.0002, 0.012, 800)
    out["005930.KS"] = walk(0.0004, 0.019, 9000)
    out["000660.KS"] = walk(0.0005, 0.027, 20000)
    fg = (50 + ret.rolling(30).sum().fillna(0) * 400).clip(1, 99)
    out["fear_greed"] = fg[fg.index >= pd.Timestamp.now() - pd.Timedelta(days=360)]
    return out
