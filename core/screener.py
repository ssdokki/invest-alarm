"""코스피 단기(2일 보유) 반등 스크리너 + 과거 검증.

조건 (전날 종가 기준, 모두 만족):
  1) 종가 > 200일 이동평균           — 장기 상승 추세 안의 눌림만
  2) RSI(2) < rsi_max(기본 10)        — 2일 기준 과매도
  3) 종가 < 5일 이동평균              — 단기 급락
  4) 20일 평균 거래대금 >= 기준(50억) — 거래가 충분한 종목만

가정한 매매:  다음 거래일 시가 매수 → 보유 2일째 종가 매도 (기본 hold_days=2)
             왕복 비용 cost_pct(기본 0.3%: 거래세·수수료·체결 차이) 차감

비교 기준:   같은 기간 같은 종목들에서 '아무 날이나' 같은 방식으로 샀을 때의 결과
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .settings import DATA

log = logging.getLogger("alarm")
UNIVERSE_CSV = DATA / "universe_kospi.csv"

DEFAULTS = {"hold_days": 2, "rsi_period": 2, "rsi_max": 10, "min_trading_value": 5_000_000_000,
            "cost_pct": 0.3, "years": 10, "max_list": 20}


def params(cfg) -> dict:
    p = dict(DEFAULTS)
    p.update(getattr(cfg, "screener", {}) or {})
    return p


def load_universe() -> tuple[pd.DataFrame, str]:
    """코스피200 구성종목(KRX). 실패하면 저장소의 대형주 목록."""
    try:
        from pykrx import stock
        codes = stock.get_index_portfolio_deposit_file("1028")
        if codes is None or len(codes) < 150:
            raise RuntimeError(f"구성종목 {0 if codes is None else len(codes)}개")
        names = [stock.get_market_ticker_name(c) for c in codes]
        return pd.DataFrame({"code": list(codes), "name": names}), "코스피200 (KRX)"
    except Exception as e:
        log.warning("KRX 코스피200 구성종목 조회 실패 → 저장소 목록 사용 (%s)", e)
        df = pd.read_csv(UNIVERSE_CSV, dtype=str)
        return df, f"코스피 주요 대형주 {len(df)}종목 (저장소 목록)"


def fetch_ohlcv(codes: list[str]) -> dict:
    import yfinance as yf
    out = {}
    tick = [f"{c}.KS" for c in codes]
    for i in range(0, len(tick), 50):
        chunk = tick[i:i + 50]
        try:
            df = yf.download(chunk, period="12y", interval="1d", auto_adjust=False, group_by="ticker",
                             threads=True, progress=False)
        except Exception as e:
            log.warning("가격 다운로드 실패(%s~): %s", chunk[0], e)
            continue
        for t in chunk:
            try:
                d = df[t][["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
                d.index = pd.to_datetime(d.index).tz_localize(None).normalize()
                if len(d) > 250:
                    out[t[:-3]] = d
            except Exception:
                pass
    return out


def synthetic(codes: list[str], seed: int = 3) -> dict:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end=pd.Timestamp.now().normalize(), periods=252 * 12)
    out = {}
    for c in codes:
        r = rng.normal(0.0003, 0.02, len(idx))
        close = 10000 * np.exp(np.cumsum(r))
        op = close * np.exp(rng.normal(0, 0.006, len(idx)))
        out[c] = pd.DataFrame({"Open": op, "High": np.maximum(op, close) * 1.01, "Low": np.minimum(op, close) * 0.99,
                               "Close": close, "Volume": rng.integers(2e5, 5e6, len(idx))}, index=idx)
    return out


def _rsi(close: pd.Series, n: int) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def frame(df: pd.DataFrame, p: dict) -> pd.DataFrame:
    c = df["Close"]
    f = pd.DataFrame(index=df.index)
    f["close"] = c
    f["rsi"] = _rsi(c, p["rsi_period"])
    f["ma5"] = c.rolling(5).mean()
    f["ma200"] = c.rolling(200).mean()
    f["tv20"] = (c * df["Volume"]).rolling(20).mean()
    f["chg1"] = c.pct_change()
    f["vs_ma5"] = c / f["ma5"] - 1
    f["vs_ma200"] = c / f["ma200"] - 1
    f["signal"] = (c > f["ma200"]) & (f["rsi"] < p["rsi_max"]) & (c < f["ma5"]) & (f["tv20"] >= p["min_trading_value"])
    h = p["hold_days"]
    f["entry"] = df["Open"].shift(-1)                      # 다음 거래일 시가
    f["exit"] = c.shift(-h)                                # 보유 h일째 종가
    f["ret"] = (f["exit"] / f["entry"] - 1) * 100 - p["cost_pct"]
    return f


def _stats(r: pd.Series) -> dict | None:
    r = r.dropna()
    if not len(r):
        return None
    return {"n": int(len(r)), "win": round(float((r > 0).mean() * 100), 1), "mean": round(float(r.mean()), 2),
            "median": round(float(r.median()), 2), "worst": round(float(r.min()), 1),
            "p10": round(float(r.quantile(0.1)), 2)}


def run(cfg, universe: pd.DataFrame, prices: dict) -> dict:
    p = params(cfg)
    names = dict(zip(universe["code"], universe["name"]))
    frames = {c: frame(df, p) for c, df in prices.items()}
    if not frames:
        return {"error": "가격 데이터를 받지 못했습니다.", "params": p}
    last_day = max(f.index[-1] for f in frames.values())
    start = last_day - pd.DateOffset(years=p["years"])

    sig_rets, base_rets, rows, per_year = [], [], [], {}
    for code, f in frames.items():
        h = f[f.index >= start]
        s = h.loc[h["signal"], "ret"].dropna()
        sig_rets.append(s)
        base_rets.append(h["ret"].dropna())
        for y, v in s.groupby(s.index.year):
            per_year.setdefault(int(y), []).append(v)
        if f.index[-1] == last_day and bool(f["signal"].iloc[-1]):
            x = f.iloc[-1]
            own = _stats(f.loc[f["signal"], "ret"])
            rows.append({"code": code, "name": names.get(code, code), "close": round(float(x["close"]), 0),
                         "chg1": round(float(x["chg1"] * 100), 2), "rsi": round(float(x["rsi"]), 1),
                         "vs_ma5": round(float(x["vs_ma5"] * 100), 2), "vs_ma200": round(float(x["vs_ma200"] * 100), 1),
                         "tv20": round(float(x["tv20"] / 1e8), 0),
                         "own_n": own["n"] if own else 0, "own_win": own["win"] if own else None,
                         "own_mean": own["mean"] if own else None})
    sig = pd.concat(sig_rets) if sig_rets else pd.Series(dtype=float)
    base = pd.concat(base_rets) if base_rets else pd.Series(dtype=float)
    recent_cut = last_day - pd.DateOffset(years=3)
    rows.sort(key=lambda r: r["rsi"])
    years = [{"year": y, **_stats(pd.concat(v))} for y, v in sorted(per_year.items())]
    out = {
        "asof": last_day.strftime("%Y-%m-%d"), "params": p, "universe_n": len(frames),
        "candidates": rows[: p["max_list"]], "candidates_total": len(rows),
        "signal": _stats(sig), "baseline": _stats(base),
        "signal_recent": _stats(sig[sig.index >= recent_cut]), "baseline_recent": _stats(base[base.index >= recent_cut]),
        "years": years, "period": f"{start:%Y-%m-%d} ~ {last_day:%Y-%m-%d}",
    }
    s, b = out["signal"], out["baseline"]
    if s and b:
        edge = s["mean"] - b["mean"]
        out["verdict"] = (
            "과거 검증상 비용 차감 후 평균 수익이 플러스이고 무작위 진입보다 나았습니다." if s["mean"] > 0 and edge > 0.1 else
            "비용 차감 후 평균 수익이 플러스이지만 무작위 진입과 큰 차이가 없습니다." if s["mean"] > 0 else
            "비용 차감 후 평균 수익이 마이너스입니다. 이 조건만으로 단타하는 것은 과거 기준으로 불리했습니다.")
        out["edge"] = round(edge, 2)
    return out
