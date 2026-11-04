"""CNN 방식을 따라 만든 '자체 계산' 공포탐욕지수.

CNN Fear & Greed는 7개 지표를 각각 과거 대비 얼마나 치우쳤는지로 점수화해 평균냅니다.
이 중 무료 공개 데이터로 구할 수 있는 6개를 Yahoo 데이터로 근사합니다.

  구성요소          CNN 원래 지표                         여기서 쓰는 근사
  모멘텀            S&P500 vs 125일 이동평균              ^GSPC / 125일선 - 1
  가격 강도         NYSE 52주 신고가 vs 신저가 종목 수     ^GSPC의 52주 고저 범위 내 위치
  시장 폭           NYSE 상승·하락 거래량(McClellan)       동일가중(RSP) / 시총가중(SPY) 50일선 대비
  풋·콜 비율        CBOE 5일 풋콜 비율                     (무료 데이터 없음 → 제외)
  변동성            VIX vs 50일 이동평균                   -(VIX / 50일선 - 1)
  안전자산 수요     주식 vs 국채 20일 수익률 차            SPY 20일 수익률 - TLT 20일 수익률
  정크본드 수요     하이일드 vs 투자등급 금리 스프레드     HYG 20일 수익률 - LQD 20일 수익률

각 구성요소를 최근 1년(252거래일) 안에서의 백분위(0~100)로 바꾸고, 계산 가능한 것들의 평균을 냅니다.
"""
import numpy as np
import pandas as pd

PROXY_TICKERS = ["^GSPC", "SPY", "^VIX", "RSP", "TLT", "HYG", "LQD"]
WINDOW = 252
MIN_COMPONENTS = 3


def _pct_rank(s: pd.Series) -> pd.Series:
    return s.rolling(WINDOW, min_periods=WINDOW // 2).rank(pct=True) * 100


def components(closes: dict) -> pd.DataFrame:
    g = closes.get("^GSPC", closes.get("SPY"))
    idx = g.index
    al = lambda t: closes[t].reindex(idx, method="ffill") if t in closes else None
    spy, vix, rsp, tlt, hyg, lqd = (al(t) for t in ("SPY", "^VIX", "RSP", "TLT", "HYG", "LQD"))
    raw = {"모멘텀": g / g.rolling(125).mean() - 1}
    hi, lo = g.rolling(WINDOW).max(), g.rolling(WINDOW).min()
    raw["가격 강도"] = (g - lo) / (hi - lo)
    if rsp is not None and spy is not None:
        ratio = rsp / spy
        raw["시장 폭"] = ratio / ratio.rolling(50).mean() - 1
    if vix is not None:
        raw["변동성"] = -(vix / vix.rolling(50).mean() - 1)
    if spy is not None and tlt is not None:
        raw["안전자산 수요"] = spy.pct_change(20) - tlt.pct_change(20)
    if hyg is not None and lqd is not None:
        raw["정크본드 수요"] = hyg.pct_change(20) - lqd.pct_change(20)
    return pd.DataFrame({k: _pct_rank(v.replace([np.inf, -np.inf], np.nan)) for k, v in raw.items()})


def proxy_index(closes: dict) -> tuple[pd.Series, pd.DataFrame]:
    comp = components(closes)
    n = comp.notna().sum(axis=1)
    idx = comp.mean(axis=1).where(n >= MIN_COMPONENTS)
    return idx.dropna(), comp


def blend(cnn: pd.Series | None, proxy: pd.Series):
    """CNN 값이 있는 날은 CNN, 없는 날은 자체 계산값. (혼합 시리즈, CNN 날짜 집합)"""
    if cnn is None or not len(cnn):
        return proxy.copy(), set()
    out = proxy.copy()
    cnn = cnn.dropna()
    out = pd.concat([out[~out.index.isin(cnn.index)], cnn]).sort_index()
    return out, set(cnn.index)


def compare(cnn: pd.Series | None, proxy: pd.Series) -> dict | None:
    """CNN과 자체 계산값이 겹치는 기간의 비교 통계."""
    if cnn is None or not len(cnn):
        return None
    j = pd.concat([cnn.rename("cnn"), proxy.rename("proxy")], axis=1, join="inner").dropna()
    if len(j) < 20:
        return None
    zone = lambda v: np.select([v < 25, v < 45, v <= 55, v <= 75], [0, 1, 2, 3], 4)
    return {
        "start": j.index[0].strftime("%Y-%m-%d"), "end": j.index[-1].strftime("%Y-%m-%d"), "days": int(len(j)),
        "corr": round(float(j["cnn"].corr(j["proxy"])), 3),
        "mae": round(float((j["cnn"] - j["proxy"]).abs().mean()), 1),
        "same_zone": round(float((zone(j["cnn"].values) == zone(j["proxy"].values)).mean()), 3),
        "last_cnn": round(float(j["cnn"].iloc[-1]), 1), "last_proxy": round(float(j["proxy"].iloc[-1]), 1),
        "series": {"date": [d.strftime("%Y-%m-%d") for d in j.index],
                   "cnn": [round(float(v), 1) for v in j["cnn"]], "proxy": [round(float(v), 1) for v in j["proxy"]]},
    }
