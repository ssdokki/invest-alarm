"""계절·이벤트 패턴 — 매년 같은 이벤트(복날·추석 등) 전후의 주가 흐름을 모아 매수·매도 시점을 찾고,
떼어 둔 최근 몇 년으로 따로 검증한다.

방법 (종목 하나, 이벤트 하나마다):
  1) 매년 이벤트 날짜(기준일, D-day)를 계산한다. 주말·휴일이면 그다음 거래일.
  2) 기준일 기준 D-60 ~ D+40 거래일 구간에서, 같은 기간 코스피보다 얼마나 더(덜) 올랐는지(초과수익)를 연도별로 잰다.
  3) 학습 연도(앞쪽)만으로 '매수일 D-a → 매도일 D+b' 조합(5거래일 간격) 중 평균 초과수익이 가장 큰 것을 고른다.
     왕복 비용 cost_pct를 뺀다.
  4) 고른 매수·매도일을 한 번도 보지 않은 최근 holdout_years년에 그대로 적용해 결과를 잰다.
  5) 무작위 날짜 대조: 기준일을 매년 아무 날짜로 바꿔 같은 방식으로 '가장 좋은 조합'을 찾으면
     얼마나 나오는지 n_random번 반복한다. 실제 이벤트의 학습 성적이 이보다 크게 낫지 않으면
     조합을 많이 시험해서 우연히 생긴 결과일 가능성이 크다.
  판정 '통과': 검증 연도 평균 초과수익 > 0, 검증 연도 중 75% 이상 플러스, 무작위 대조 상위 10% 이내.
"""
import logging
import math
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd

log = logging.getLogger("alarm")
KST = timezone(timedelta(hours=9))

DEFAULTS = {"pre": 60, "post": 40, "step": 5, "min_hold": 5, "holdout_years": 4, "min_train_years": 6,
            "cost_pct": 0.3, "n_random": 300, "benchmark": "^KS11"}

DEFAULT_EVENTS = [
    {"season": "여름", "name": "복날 (초복 기준)", "anchor": "chobok",
     "why": "초복·중복·말복에 닭고기 소비가 늘어나는 시기",
     "stocks": {"하림": "136480.KS", "마니커": "027740.KS", "체리부로": "066360.KQ"}},
]

# ── 날짜 계산 ─────────────────────────────────────────────────────────────────


def _sun_longitude(jd: float) -> float:
    """태양의 겉보기 황경(도). Meeus 저정밀식 — 오차 약 0.01°(15분 이내)."""
    t = (jd - 2451545.0) / 36525
    l0 = 280.46646 + 36000.76983 * t + 0.0003032 * t * t
    m = math.radians(357.52911 + 35999.05029 * t - 0.0001537 * t * t)
    c = ((1.914602 - 0.004817 * t - 0.000014 * t * t) * math.sin(m)
         + (0.019993 - 0.000101 * t) * math.sin(2 * m) + 0.000289 * math.sin(3 * m))
    om = math.radians(125.04 - 1934.136 * t)
    return (l0 + c - 0.00569 - 0.00478 * math.sin(om)) % 360


def _meeus_term(year: int, lon: float) -> datetime:
    guess = datetime(year, 3, 20, tzinfo=timezone.utc) + timedelta(days=(lon / 360) * 365.2422)
    jd = guess.timestamp() / 86400 + 2440587.5
    for _ in range(50):
        diff = (lon - _sun_longitude(jd) + 180) % 360 - 180
        jd += diff * 365.2422 / 360
        if abs(diff) < 1e-7:
            break
    return datetime.fromtimestamp((jd - 2440587.5) * 86400, tz=timezone.utc)


def solar_term(year: int, lon: float) -> date:
    """태양 황경이 lon°가 되는 날(한국시간). 90=하지, 135=입추, 315=입춘.
    ephem이 있으면 그 해 하지 시각(정밀)을 기준으로 보정하고, 없으면 Meeus 근사식만 쓴다."""
    t = _meeus_term(year, lon)
    try:
        import ephem
        sol = ephem.next_summer_solstice(f"{year}/1/1").datetime().replace(tzinfo=timezone.utc)
        t = t + (sol - _meeus_term(year, 90))
    except Exception:
        pass
    return t.astimezone(KST).date()


def _next_gyeong(d: date, nth: int = 1) -> date:
    """d 이후(d 포함) nth번째 경일(庚日)."""
    k = (2 - d.toordinal()) % 10      # 경일: ordinal % 10 == 2 (2023~2025 복날로 확인)
    return d + timedelta(days=k + 10 * (nth - 1))


def boknal(year: int) -> dict:
    haji, ipchu = solar_term(year, 90), solar_term(year, 135)
    return {"chobok": _next_gyeong(haji, 3), "jungbok": _next_gyeong(haji, 4), "malbok": _next_gyeong(ipchu, 1)}


def _lunar(year: int, m: int, d: int) -> date | None:
    try:
        from korean_lunar_calendar import KoreanLunarCalendar
        c = KoreanLunarCalendar()
        if not c.setLunarDate(year, m, d, False):
            return None
        return date.fromisoformat(c.SolarIsoFormat())
    except Exception:
        return None


def event_date(anchor: str, year: int) -> date | None:
    """anchor: chobok/jungbok/malbok, chuseok, seol, solar:<황경>, 또는 'MM-DD'."""
    if anchor in ("chobok", "jungbok", "malbok"):
        return boknal(year)[anchor]
    if anchor == "chuseok":
        return _lunar(year, 8, 15)
    if anchor == "seol":
        return _lunar(year, 1, 1)
    if anchor.startswith("solar:"):
        return solar_term(year, float(anchor[6:]))
    m, d = (int(x) for x in anchor.split("-"))
    return date(year, m, d)


# ── 분석 ──────────────────────────────────────────────────────────────────────


def params(cfg) -> dict:
    p = dict(DEFAULTS)
    p.update((getattr(cfg, "seasonal", {}) or {}).get("params", {}))
    return p


def events_cfg(cfg) -> list:
    return (getattr(cfg, "seasonal", {}) or {}).get("events") or DEFAULT_EVENTS


def _paths(lr: np.ndarray, idx: pd.DatetimeIndex, anchors: list, pre: int, post: int):
    """연도별 [D-pre .. D+post] 로그 누적(기준: D-pre = 0). anchors: [(연도, 날짜)]."""
    rows, used = [], []
    for y, d in anchors:
        a = idx.searchsorted(pd.Timestamp(d))
        if a - pre < 0 or a + post >= len(idx):
            continue
        seg = lr[a - pre: a + post + 1]
        if np.isnan(seg).any():
            continue
        rows.append(seg - seg[0])
        used.append((y, idx[a]))
    return (np.array(rows) if rows else np.empty((0, pre + post + 1))), used


def _best(P: np.ndarray, offs: np.ndarray, p: dict):
    """P: 연도×오프셋 누적로그. (평균 최대 (매수i, 매도j), 평균, 전체 표)"""
    sel = np.arange(0, len(offs), p["step"])
    best = (None, None, -1e9)
    for ii, i in enumerate(sel):
        for j in sel[ii + 1:]:
            if offs[j] - offs[i] < p["min_hold"]:
                continue
            m = (P[:, j] - P[:, i]).mean()
            if m > best[2]:
                best = (i, j, m)
    return best


def _pct(x):
    return round(float((math.exp(x) - 1) * 100), 2)


def _stats(v: np.ndarray, cost: float):
    if not len(v):
        return None
    r = (np.exp(v) - 1) * 100 - cost
    return {"n": int(len(r)), "mean": round(float(r.mean()), 2), "median": round(float(np.median(r)), 2),
            "win": round(float((r > 0).mean() * 100), 0), "wins": int((r > 0).sum()),
            "worst": round(float(r.min()), 2), "best": round(float(r.max()), 2)}


def analyze(close: pd.Series, bench: pd.Series, ev: dict, p: dict, today: date, rng) -> dict:
    close = close.sort_index().dropna()
    bench = bench.sort_index().reindex(close.index, method="ffill")
    idx = close.index
    lr_ex = np.log(close.to_numpy()) - np.log(bench.to_numpy())    # 코스피 대비
    lr_raw = np.log(close.to_numpy())
    pre, post = p["pre"], p["post"]
    offs = np.arange(-pre, post + 1)
    years = range(idx[0].year, today.year + 1)
    anchors = [(y, d) for y in years if (d := event_date(ev["anchor"], y)) is not None]
    P, used = _paths(lr_ex, idx, anchors, pre, post)
    R, _ = _paths(lr_raw, idx, anchors, pre, post)
    n = len(used)
    out = {"years": [u[0] for u in used], "anchor_days": [u[1].strftime("%Y-%m-%d") for u in used],
           "first": idx[0].strftime("%Y-%m-%d"), "n_years": n}
    ho = p["holdout_years"]
    if n - ho < p["min_train_years"]:
        out["verdict"] = f"이벤트 자료가 {n}년뿐이라 학습 {p['min_train_years']}년 + 검증 {ho}년을 나눌 수 없습니다."
        out["passed"] = False
        out["grade"] = "자료 부족"
        return out
    tr, te = slice(0, n - ho), slice(n - ho, n)
    i, j, m = _best(P[tr], offs, p)
    cost = p["cost_pct"]
    ex_tr, ex_te = P[tr, j] - P[tr, i], P[te, j] - P[te, i]
    raw_tr, raw_te = R[tr, j] - R[tr, i], R[te, j] - R[te, i]

    # 무작위 날짜 대조: 매년 아무 날짜를 기준일로 삼아 같은 방식으로 최적 조합을 찾았을 때의 학습 성적
    span_years = [u[0] for u in used][tr]
    rand_best = []
    for _ in range(p["n_random"]):
        fake = [(y, date(y, 1, 1) + timedelta(days=int(rng.integers(0, 365)))) for y in span_years]
        Q, _u = _paths(lr_ex, idx, fake, pre, post)
        if len(Q) >= p["min_train_years"]:
            rand_best.append(_best(Q, offs, p)[2])
    rand_best = np.array(rand_best)
    p_rand = float((rand_best >= m).mean()) if len(rand_best) else None

    st_te = _stats(ex_te, cost)
    passed = bool(st_te and st_te["mean"] > 0 and st_te["win"] >= 75 and p_rand is not None and p_rand <= 0.1)
    if passed:
        grade, verdict = "통과", "학습 연도에서 찾은 매수·매도일이 떼어 둔 최근 연도에서도 코스피보다 나았고, 무작위 날짜로는 쉽게 나오지 않는 성적입니다."
    elif st_te and st_te["mean"] > 0:
        grade = "약함"
        why = []
        if st_te["win"] < 75:
            why.append(f"검증 {st_te['n']}년 중 코스피보다 나았던 해가 {st_te['wins']}년뿐입니다")
        if p_rand is None or p_rand > 0.1:
            why.append(f"아무 날짜를 기준일로 잡아도 {round((p_rand or 0) * 100)}% 확률로 이만한 학습 성적이 나옵니다")
        verdict = "최근 연도 평균은 코스피보다 나았지만 패턴이라고 믿기 어렵습니다. " + " ".join(w + "." for w in why)
    else:
        grade, verdict = "미통과", "학습 연도에서 가장 좋았던 매수·매도일도 떼어 둔 최근 연도에서는 코스피보다 못했습니다."

    # 다가오는 일정: 올해 매도일이 아직 안 지났으면 올해, 지났으면 내년
    def plan(y):
        d = event_date(ev["anchor"], y)
        if d is None:
            return None
        bd = lambda k: (pd.Timestamp(d) + pd.offsets.BDay(int(k))).date()  # 주말만 고려한 근사
        return {"year": y, "event": d.isoformat(), "buy": bd(offs[i]).isoformat(), "sell": bd(offs[j]).isoformat()}
    nxt = plan(today.year)
    if nxt and date.fromisoformat(nxt["sell"]) < today:
        nxt = plan(today.year + 1)
    if nxt:
        b, s = date.fromisoformat(nxt["buy"]), date.fromisoformat(nxt["sell"])
        if today < b:
            nxt["phase"], nxt["dday"] = "매수일 전", (b - today).days
        elif today <= s:
            nxt["phase"], nxt["dday"] = "보유 구간", (s - today).days
    out.update({
        "grade": grade, "passed": passed, "verdict": verdict,
        "buy_off": int(offs[i]), "sell_off": int(offs[j]),
        "train_years": span_years, "test_years": [u[0] for u in used][te],
        "train": _stats(ex_tr, cost), "test": st_te,
        "train_raw": _stats(raw_tr, cost), "test_raw": _stats(raw_te, cost),
        "per_year": [{"year": used[k][0], "set": "학습" if k < n - ho else "검증",
                      "excess": round(float((math.exp(P[k, j] - P[k, i]) - 1) * 100 - cost), 2),
                      "raw": round(float((math.exp(R[k, j] - R[k, i]) - 1) * 100 - cost), 2)} for k in range(n)],
        "rand_p": round(p_rand, 3) if p_rand is not None else None,
        "rand_median": _pct(float(np.median(rand_best))) if len(rand_best) else None,
        "train_best_raw": _pct(m),
        "path": {"off": offs.tolist(),
                 "train": [round((math.exp(v) - 1) * 100, 2) for v in P[tr].mean(axis=0)],
                 "test": [round((math.exp(v) - 1) * 100, 2) for v in P[te].mean(axis=0)],
                 "raw": [round((math.exp(v) - 1) * 100, 2) for v in R.mean(axis=0)]},
        "next": nxt,
    })
    return out


def run(cfg, closes: dict, crema: dict | None = None, today: date | None = None) -> dict:
    """closes: {티커: 종가 시리즈} (benchmark 포함). crema: {티커: 현재 신호 요약}."""
    p = params(cfg)
    today = today or datetime.now(KST).date()
    bench = closes.get(p["benchmark"])
    if bench is None:
        return {"error": f"비교 지수 {p['benchmark']} 데이터가 없습니다.", "params": p}
    rng = np.random.default_rng(42)
    events = []
    for ev in events_cfg(cfg):
        dates = {}
        for y in (today.year - 1, today.year, today.year + 1):
            d = event_date(ev["anchor"], y)
            dates[y] = d.isoformat() if d else None
        if ev["anchor"] in ("chobok", "jungbok", "malbok"):
            dates["boknal"] = {y: {k: v.isoformat() for k, v in boknal(y).items()} for y in (today.year, today.year + 1)}
        items = []
        for name, t in ev["stocks"].items():
            s = closes.get(t)
            if s is None or len(s) < 300:
                items.append({"name": name, "ticker": t, "error": "가격 데이터를 받지 못했습니다."})
                continue
            try:
                r = analyze(s, bench, ev, p, today, rng)
            except Exception as e:
                log.warning("계절 분석 실패 %s: %s", t, e)
                r = {"error": str(e)}
            r.update({"name": name, "ticker": t, "now": (crema or {}).get(t)})
            items.append(r)
        events.append({k: ev.get(k) for k in ("season", "name", "anchor", "why")} | {"dates": dates, "items": items})
    return {"asof": today.isoformat(), "params": p, "events": events}
