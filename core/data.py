"""데이터 수집: Yahoo Finance(가격·VIX), CNN(공포탐욕지수).

받은 공포탐욕지수는 data/fear_greed.csv에 누적 저장합니다(클라우드 알람 작업이 매일 커밋).
CNN이 클라우드 서버 접속을 막는 경우가 있어, 그때는 이 저장본을 씁니다.
"""
import logging
import time
from datetime import datetime

import numpy as np
import pandas as pd

from .settings import DATA

log = logging.getLogger("alarm")
CNN_URL = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata/{start}"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"}


def _path(name):
    DATA.mkdir(exist_ok=True)
    return DATA / f"{name.replace('^', '_')}.csv"


def _load(name):
    p = _path(name)
    if not p.exists():
        return None
    return pd.read_csv(p, parse_dates=["date"], index_col="date")["value"].astype(float)


def fetch_close(ticker: str, save: bool = False) -> tuple[pd.Series, str]:
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
            if save:
                s.rename("value").to_csv(_path(ticker), index_label="date")
            return s, "Yahoo Finance"
        except Exception as e:
            last_err = e
            time.sleep(2 * (attempt + 1))
    cached = _load(ticker)
    if cached is not None:
        return cached, f"저장본({cached.index[-1].date()}까지)"
    raise RuntimeError(f"{ticker} 데이터를 받지 못했습니다: {last_err}")


def fetch_fear_greed(save: bool = False) -> tuple[pd.Series | None, str]:
    old = _load("fear_greed")
    try:
        import requests
        r = requests.get(CNN_URL.format(start="2011-01-01"), headers=UA, timeout=20)
        r.raise_for_status()
        js = r.json()
        s = pd.Series({pd.to_datetime(d["x"], unit="ms").normalize(): float(d["y"])
                       for d in js["fear_and_greed_historical"]["data"]}).sort_index()
        now = js.get("fear_and_greed", {})
        if "score" in now and "timestamp" in now:
            s.loc[pd.to_datetime(now["timestamp"]).tz_localize(None).normalize()] = float(now["score"])
        s = s[~s.index.duplicated(keep="last")]
        if old is not None:
            s = pd.concat([old[~old.index.isin(s.index)], s])
        s = s.sort_index()
        if save:
            s.rename("value").to_csv(_path("fear_greed"), index_label="date")
        return s, "CNN"
    except Exception as e:
        if old is not None:
            return old, f"저장본({old.index[-1].date()}까지)"
        log.warning("공포탐욕지수 수신 실패: %s", e)
        return None, "대체값(50)"


def load_all(cfg, test: bool = False, save: bool = False):
    """{티커: 시리즈}, 공포탐욕 시리즈, 알림 문구 목록."""
    if test:
        syn = synthetic()
        return {t: syn[t] for t in syn if t != "fear_greed"}, syn["fear_greed"], ["가상 데이터(테스트)"]
    notes, closes = [], {}
    for t in list(cfg.indices.values()) + list(cfg.vol_index.values()):
        if t in closes:
            continue
        s, src = fetch_close(t)
        closes[t] = s
        if src != "Yahoo Finance":
            notes.append(f"{t}: {src}")
    fg, src = fetch_fear_greed(save=save)
    if src != "CNN":
        notes.append(f"공포탐욕지수: {src}")
    return closes, fg, notes


def synthetic(seed: int = 7) -> dict:
    """인터넷 없이 화면·로직을 확인하기 위한 가상 데이터 (실제 시세 아님)."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("1999-01-01", datetime.now().strftime("%Y-%m-%d"))
    n = len(idx)

    def walk(mu, sigma, p0):
        r = rng.normal(mu, sigma, n)
        for pos, size, length in [(0.06, -0.9, 250), (0.38, -0.6, 120), (0.78, -0.35, 25),
                                  (0.86, -0.4, 180), (0.96, -0.2, 20)]:
            a = int(pos * n)
            r[a:a + length] += size / length
        return pd.Series(p0 * np.exp(np.cumsum(r)), index=idx)

    out = {"SPY": walk(0.0004, 0.011, 120), "QQQ": walk(0.0006, 0.016, 90)}
    ret = out["SPY"].pct_change().fillna(0)
    vix = (15 + (-ret.rolling(20).sum().fillna(0)) * 120).clip(9, 80)
    out["^VIX"], out["^VXN"] = vix, vix * 1.2
    fg = (50 + ret.rolling(30).sum().fillna(0) * 400).clip(1, 99)
    out["fear_greed"] = fg[fg.index >= "2011-01-01"]
    return out
