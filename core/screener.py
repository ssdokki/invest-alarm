"""코스피 단기(2일 보유) 반등 규칙 — 떼어 둔 기간 검증(아웃오브샘플).

방법:
  1) 학습 기간(앞 7년)에서 규칙 후보 12개를 모두 돌려 보고, 비용 차감 평균 수익이 가장 좋은 규칙을 고른다.
  2) 고른 규칙을 한 번도 보지 않은 검증 기간(최근 3년)에 그대로 적용해 결과를 따로 잰다.
  3) 검증 기간에서도 비용 차감 후 플러스이고 무작위 진입보다 나아야 '통과'.
  학습 기간 성적만 좋고 검증 기간에 무너지면, 과거에 우연히 맞았던(과최적화) 규칙이라는 뜻이다.

후보 규칙의 기본 틀(아래는 기본 규칙):

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
    f["rsi5"] = _rsi(c, 5)
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


GRID = [
    {"rsi_max": r, "drop": d, "market": m}
    for r in (5, 10, 15) for d in (False, True) for m in (False, True)
]


def _label(g) -> str:
    t = [f"RSI(2) {g['rsi_max']} 미만"]
    if g["drop"]:
        t.append("당일 -2% 이상 하락")
    if g["market"]:
        t.append("코스피 200일선 위일 때만")
    return " · ".join(t)


def _signal(f: pd.DataFrame, g: dict, p: dict, mkt_up: pd.Series | None) -> pd.Series:
    s = (f["close"] > f["ma200"]) & (f["rsi"] < g["rsi_max"]) & (f["close"] < f["ma5"]) & (f["tv20"] >= p["min_trading_value"])
    if g["drop"]:
        s &= f["chg1"] <= -0.02
    if g["market"] and mkt_up is not None:
        s &= mkt_up.reindex(f.index, method="ffill").fillna(False).astype(bool)
    return s


def run(cfg, universe: pd.DataFrame, prices: dict, kospi: pd.Series | None = None) -> dict:
    p = params(cfg)
    names = dict(zip(universe["code"], universe["name"]))
    frames = {c: frame(df, p) for c, df in prices.items()}
    if not frames:
        return {"error": "가격 데이터를 받지 못했습니다.", "params": p}
    last_day = max(f.index[-1] for f in frames.values())
    start = last_day - pd.DateOffset(years=p["years"])
    split = last_day - pd.DateOffset(years=p.get("holdout_years", 3))
    mkt_up = (kospi > kospi.rolling(200).mean()) if kospi is not None and len(kospi) else None

    def collect(g):
        tr, te = [], []
        for f in frames.values():
            sig = _signal(f, g, p, mkt_up)
            r = f.loc[sig, "ret"].dropna()
            tr.append(r[(r.index >= start) & (r.index < split)])
            te.append(r[r.index >= split])
        return pd.concat(tr), pd.concat(te)

    rets = [f["ret"].dropna() for f in frames.values()]
    base_tr = pd.concat([r[(r.index >= start) & (r.index < split)] for r in rets])
    base_te = pd.concat([r[r.index >= split] for r in rets])
    grid = []
    for g in GRID:
        tr, te = collect(g)
        grid.append({"rule": _label(g), "g": g, "train": _stats(tr), "test": _stats(te)})
    ok = [x for x in grid if x["train"] and x["train"]["n"] >= p.get("min_trades", 200)]
    best = max(ok, key=lambda x: x["train"]["mean"]) if ok else None
    default = next(x for x in grid if x["g"] == {"rsi_max": 10, "drop": False, "market": False})

    out = {"asof": last_day.strftime("%Y-%m-%d"), "params": p, "universe_n": len(frames),
           "train_period": f"{start:%Y-%m-%d} ~ {split - pd.Timedelta(days=1):%Y-%m-%d}",
           "test_period": f"{split:%Y-%m-%d} ~ {last_day:%Y-%m-%d}",
           "baseline_train": _stats(base_tr), "baseline_test": _stats(base_te),
           "grid": [{k: v for k, v in x.items() if k != "g"} for x in grid],
           "default": {k: v for k, v in default.items() if k != "g"}, "market_filter": mkt_up is not None}
    if not best:
        out["verdict"] = "학습 기간에 거래가 충분한 규칙이 없어 검증하지 못했습니다."
        out["passed"] = False
        return out
    out["best"] = {k: v for k, v in best.items() if k != "g"}
    te, bte = best["test"], out["baseline_test"]
    passed = bool(te and te["mean"] > 0 and bte and te["mean"] - bte["mean"] > 0.1 and te["n"] >= 50)
    out["passed"] = passed
    if passed:
        out["verdict"] = "학습 기간에 고른 규칙이 떼어 둔 최근 기간에서도 비용 차감 후 플러스였고 무작위 진입보다 나았습니다."
    elif te and te["mean"] > 0:
        out["verdict"] = "떼어 둔 기간에서 플러스였지만 무작위 진입과 차이가 작거나 거래가 적어, 규칙의 힘이라고 보기 어렵습니다."
    else:
        out["verdict"] = "학습 기간에서 가장 좋았던 규칙도 떼어 둔 최근 기간에서는 비용 차감 후 마이너스였습니다. 단타 규칙으로 쓰기 어렵습니다."
    gap = (best["train"]["mean"] - te["mean"]) if te else None
    out["overfit_gap"] = round(gap, 2) if gap is not None else None

    rows = []
    for code, f in frames.items():
        if f.index[-1] != last_day:
            continue
        sig = _signal(f, best["g"], p, mkt_up)
        if bool(sig.iloc[-1]):
            x = f.iloc[-1]
            rows.append({"code": code, "name": names.get(code, code), "close": round(float(x["close"]), 0),
                         "chg1": round(float(x["chg1"] * 100), 2), "rsi": round(float(x["rsi"]), 1),
                         "vs_ma5": round(float(x["vs_ma5"] * 100), 2)})
    rows.sort(key=lambda r: r["rsi"])
    out["candidates"], out["candidates_total"] = rows[: p["max_list"]], len(rows)
    return out
