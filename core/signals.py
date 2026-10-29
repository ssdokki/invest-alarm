"""지표 계산과 매수·매도 신호 판정.

- RSI는 와일더(Wilder) 방식 14기간
- 주간 봉은 금요일 종가 기준 (W-FRI). 영상의 Yahoo 차트는 주 시작일(월요일)로 표시하므로
  같은 주라도 날짜 표기가 4일 다를 수 있습니다.
- 매도 알람은 '확정된' 주간 봉으로만 판단합니다.
"""
from calendar import monthrange
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ET = ZoneInfo("America/New_York")
N = 14


def wilder(close: pd.Series, n: int = N):
    """(rsi, 평균상승, 평균하락) 시리즈."""
    v = close.to_numpy(dtype=float)
    d = np.diff(v, prepend=np.nan)
    g = np.where(d > 0, d, 0.0)
    l = np.where(d < 0, -d, 0.0)
    ag = np.full(len(v), np.nan)
    al = np.full(len(v), np.nan)
    if len(v) > n:
        ag[n] = g[1:n + 1].mean()
        al[n] = l[1:n + 1].mean()
        for i in range(n + 1, len(v)):
            ag[i] = (ag[i - 1] * (n - 1) + g[i]) / n
            al[i] = (al[i - 1] * (n - 1) + l[i]) / n
    with np.errstate(divide="ignore", invalid="ignore"):
        rsi = np.where(al == 0, 100.0, 100 - 100 / (1 + ag / al))
    rsi[np.isnan(ag)] = np.nan
    mk = lambda a: pd.Series(a, index=close.index)
    return mk(rsi), mk(ag), mk(al)


def next_close_for_rsi(ag: float, al: float, close: float, target: float, n: int = N):
    """다음 봉 종가가 얼마가 되면 RSI가 target에 도달하는지 역산."""
    if any(pd.isna(x) for x in (ag, al, close)) or al == 0:
        return None
    rs = target / (100 - target)
    cur = ag / al
    if cur > rs:   # 내려가야 도달
        down = (n - 1) * ag / rs - (n - 1) * al
        return close - down if down > 0 else None
    up = (n - 1) * (rs * al - ag)
    return close + up if up > 0 else None


def lin_score(x, lo, hi):
    """lo면 0점, hi면 100점, 그 사이 직선, 범위 밖은 0~100으로 자름."""
    return ((x - lo) / (hi - lo) * 100).clip(0, 100)


def us_market_state(now_utc: datetime | None = None):
    """(미국 동부 현재시각, 정규장 개장 여부)."""
    now = (now_utc or datetime.now(tz=ZoneInfo("UTC"))).astimezone(ET)
    open_ = now.weekday() < 5 and time(9, 30) <= now.time() < time(16, 0)
    return now, open_


def week_complete(label: pd.Timestamp, now_et: datetime) -> bool:
    """금요일 라벨 주봉이 마감됐는지 (미국 동부 금요일 16:15 이후)."""
    end = datetime.combine(label.date(), time(16, 15), tzinfo=ET)
    return now_et >= end


def compute(name, close, vol, fg, rules, cfg, now_utc=None) -> dict:
    now_et, market_open = us_market_state(now_utc)
    close = close.sort_index()
    idx = close.index

    # ── 주간 봉 (마지막 봉은 진행 중일 수 있음)
    weekly = close.resample("W-FRI").last().dropna()
    w_rsi, w_ag, w_al = wilder(weekly)
    w = pd.DataFrame({"close": weekly, "rsi": w_rsi, "ag": w_ag, "al": w_al})
    w["complete"] = [week_complete(t, now_et) for t in w.index]
    wc = w[w["complete"]]

    # ── 일별 지표
    d = pd.DataFrame({"close": close})
    d["ytd_high"] = close.groupby(idx.year).cummax()
    d["ytd_dd"] = close / d["ytd_high"] - 1
    d["ma200"] = close.rolling(200).mean()
    d["dist200"] = close / d["ma200"] - 1
    d["rsi_d"] = wilder(close)[0]
    # 그날까지의 주간 RSI (이번 주는 그날 종가를 주간 종가로 간주)
    lab = pd.Series(idx, index=idx).dt.to_period("W-FRI").dt.end_time.dt.normalize()
    prev = lab - pd.Timedelta(days=7)
    pg, pl, pc = (w[c].reindex(prev.values).to_numpy() for c in ("ag", "al", "close"))
    ch = close.to_numpy() - pc
    g2 = (pg * (N - 1) + np.where(ch > 0, ch, 0)) / N
    l2 = (pl * (N - 1) + np.where(ch < 0, -ch, 0)) / N
    with np.errstate(divide="ignore", invalid="ignore"):
        d["rsi_w"] = np.where(l2 == 0, 100.0, 100 - 100 / (1 + g2 / l2))
    d.loc[np.isnan(g2), "rsi_w"] = np.nan

    vol = vol.sort_index()
    d["vix"] = vol.reindex(idx, method="ffill")
    d["vix_normal"] = vol.rolling(cfg.vix_normal_days).mean().reindex(idx, method="ffill")
    d["vix_dev"] = d["vix"] / d["vix_normal"] - 1
    if fg is not None and len(fg):
        d["fg"] = fg.sort_index().reindex(idx, method="ffill", tolerance=pd.Timedelta(days=7))
    else:
        d["fg"] = np.nan
    d["fg_fallback"] = d["fg"].isna()
    d["fg"] = d["fg"].fillna(cfg.fg_fallback)

    cols = ["ytd_dd", "rsi_w", "rsi_d", "vix_dev", "dist200", "fg"]
    for i, c in enumerate(cols):
        d[f"s_{c}"] = lin_score(d[c], rules["lo"][i], rules["hi"][i])
    wts = np.array(rules["weights"], dtype=float)
    sc = d[[f"s_{c}" for c in cols]].to_numpy()
    d["score"] = (sc * wts).sum(axis=1) / wts.sum()
    d.loc[np.isnan(sc).any(axis=1), "score"] = np.nan

    # ── 매도 알람 상태머신 (확정 주봉)
    events, alarm_start = [], None
    streak = 0
    hist_alarm = []
    for t, row in wc.iterrows():
        r = row["rsi"]
        if pd.isna(r):
            continue
        streak = streak + 1 if r >= rules["alarm_rsi"] else 0
        if alarm_start is not None:
            expiry = (alarm_start + pd.DateOffset(months=rules["alarm_months"])).normalize()
            if t > expiry:
                events.append(dict(type="ALARM_EXPIRED", date=t, rsi=r, start=alarm_start, index=name))
                alarm_start = None
            elif r <= rules["sell_rsi"]:
                events.append(dict(type="SELL", date=t, rsi=r, start=alarm_start, close=row["close"], index=name))
                hist_alarm.append((alarm_start, t))
                alarm_start = None
                continue
        if alarm_start is None and streak >= rules["alarm_weeks"]:
            alarm_start = t
            events.append(dict(type="ALARM_ON", date=t, rsi=r, start=t, index=name))
    if alarm_start is not None:
        hist_alarm.append((alarm_start, None))

    # ── 매수 신호: 매월 점검일
    th = rules["buy_threshold"].get(name, 60)
    first = d["score"].first_valid_index()
    today = now_et.date()
    if first is not None:
        y, m = first.year, first.month
        while (y, m) <= (today.year, today.month):
            cd = date(y, m, min(cfg.buy_check_day, monthrange(y, m)[1]))
            if cd <= today:
                sub = d.loc[:pd.Timestamp(cd), "score"].dropna()
                if len(sub) and sub.index[-1] >= pd.Timestamp(cd) - pd.Timedelta(days=10):
                    if sub.iloc[-1] >= th:
                        events.append(dict(type="BUY", date=pd.Timestamp(cd), score=float(sub.iloc[-1]),
                                           asof=sub.index[-1], close=float(d.loc[sub.index[-1], "close"]),
                                           index=name))
            m += 1
            if m == 13:
                y, m = y + 1, 1
    if cfg.buy_daily_alert:
        s = d["score"]
        cross = (s >= th) & (s.shift(1) < th)
        for t in s[cross].index:
            events.append(dict(type="BUY_CROSS", date=t, score=float(s[t]), close=float(d.loc[t, "close"]),
                               index=name))

    # ── 현재 스냅샷
    last = d.dropna(subset=["close"]).iloc[-1]
    lc = wc.iloc[-1] if len(wc) else None
    expiry = (alarm_start + pd.DateOffset(months=rules["alarm_months"])).normalize() if alarm_start is not None else None
    snap = dict(
        index=name, ticker=cfg.indices[name], vol_ticker=cfg.vol_index[name],
        asof=d.index[-1], market_open=market_open, now_et=now_et,
        close=last["close"], ytd_high=last["ytd_high"], ytd_dd=last["ytd_dd"],
        ma200=last["ma200"], dist200=last["dist200"], rsi_d=last["rsi_d"],
        rsi_w_prov=last["rsi_w"], vix=last["vix"], vix_normal=last["vix_normal"], vix_dev=last["vix_dev"],
        fg=last["fg"], fg_fallback=bool(last["fg_fallback"]),
        scores={c: float(last[f"s_{c}"]) for c in cols}, score=float(last["score"]),
        threshold=th, avg_score=rules["avg_score"],
        week_label=lc.name if lc is not None else None,
        rsi_w=float(lc["rsi"]) if lc is not None else np.nan,
        alarm_start=alarm_start, alarm_expiry=expiry,
    )
    sc_ = snap["score"]
    snap["buy_status"] = "매수 구간" if sc_ >= th else ("비쌈 (추가 매수 비추천)" if sc_ < rules["avg_score"] else "대기")
    if alarm_start is None:
        snap["sell_status"] = "알람 없음"
        snap["sell_price"] = None
        ap = next_close_for_rsi(lc["ag"], lc["al"], lc["close"], rules["alarm_rsi"]) if lc is not None else None
        # 한 주에 10% 넘게 올라야 닿는 가격은 의미가 없어 표시하지 않음
        snap["alarm_price"] = ap if ap is not None and ap <= lc["close"] * 1.10 else None
    else:
        snap["sell_status"] = "경계 (알람 켜짐)"
        snap["sell_price"] = next_close_for_rsi(lc["ag"], lc["al"], lc["close"], rules["sell_rsi"])
        snap["alarm_price"] = None
    snap["gap_to_sell"] = snap["rsi_w"] - rules["sell_rsi"]

    return dict(name=name, daily=d, weekly=w, weekly_complete=wc, events=events, snap=snap,
                alarm_history=hist_alarm)
