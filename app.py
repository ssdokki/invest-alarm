"""투자알람 웹 대시보드 (Streamlit).

로컬 실행:  streamlit run app.py
테스트(가상 데이터): APP_TEST=1 APP_PASSWORD=1234 streamlit run app.py
"""
import hmac
import os
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from core import data, signals
from core.settings import AXES, load, with_rules

st.set_page_config(page_title="투자알람", page_icon="📈", layout="wide", initial_sidebar_state="collapsed")

BUY, SELL, WARN, MUTED, BLUE = "#1a8f4c", "#d32f2f", "#e69500", "#8a94a6", "#2f6fb2"
TEST = os.environ.get("APP_TEST") == "1"

st.markdown("""
<style>
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] {visibility:hidden;}
.block-container {padding-top:1.2rem; padding-bottom:3rem; max-width:1180px;}
h1,h2,h3 {letter-spacing:-0.02em;}
.app-title {font-size:1.6rem; font-weight:800; margin:0;}
.app-sub {color:#8a94a6; font-size:.85rem; margin-top:2px;}
.pill {display:inline-block; padding:3px 10px; border-radius:999px; font-size:.78rem; font-weight:700;}
.p-buy {background:rgba(26,143,76,.14); color:#1a8f4c;}
.p-sell {background:rgba(211,47,47,.14); color:#d32f2f;}
.p-warn {background:rgba(230,149,0,.16); color:#b87400;}
.p-none {background:rgba(138,148,166,.16); color:#8a94a6;}
.mini {border:1px solid rgba(138,148,166,.25); border-radius:14px; padding:10px 14px;}
.mini .n {font-weight:700;} .mini .v {font-size:1.35rem; font-weight:800;}
.kv {display:flex; justify-content:space-between; padding:6px 0; border-bottom:1px solid rgba(138,148,166,.18); font-size:.92rem;}
.kv span:first-child {color:#8a94a6;}
.big-note {font-size:.95rem; margin:.2rem 0 .6rem;}
[data-testid="stMetricValue"] {font-size:1.45rem;}
.tiles {display:grid; grid-template-columns:repeat(6,1fr); gap:10px; margin:.4rem 0 .3rem;}
.tile {border:1px solid rgba(138,148,166,.25); border-radius:12px; padding:10px 12px;}
.tile .tl {font-size:.78rem; color:#8a94a6;} .tile .tl span {float:right;}
.tile .tv {font-size:1.25rem; font-weight:800; margin:2px 0 6px;}
.tile .bar {height:6px; border-radius:6px; background:rgba(138,148,166,.2); overflow:hidden;}
.tile .bar i {display:block; height:100%; border-radius:6px;}
.tile .ts {font-size:.75rem; color:#8a94a6; margin-top:4px;}
@media (max-width: 900px) { .tiles {grid-template-columns:repeat(3,1fr);} }
@media (max-width: 640px) { .app-title {font-size:1.3rem;} [data-testid="stMetricValue"] {font-size:1.15rem;} }
</style>""", unsafe_allow_html=True)


# ───────────── 비밀번호 ─────────────
def password_ok() -> bool:
    if st.session_state.get("auth"):
        return True
    secret = os.environ.get("APP_PASSWORD") if TEST else st.secrets.get("APP_PASSWORD", None)
    _, mid, _ = st.columns([1, 1.2, 1])
    with mid:
        st.markdown("<div style='height:12vh'></div>", unsafe_allow_html=True)
        st.markdown("<div class='app-title' style='text-align:center'>투자알람</div>"
                    "<div class='app-sub' style='text-align:center'>크레마지수 매수 · 주간 RSI 매도</div>",
                    unsafe_allow_html=True)
        if not secret:
            st.error("앱 비밀번호가 설정되지 않았습니다. Streamlit 설정의 Secrets에 APP_PASSWORD를 넣어 주세요.")
            return False
        with st.form("login", border=True):
            pw = st.text_input("비밀번호", type="password", placeholder="비밀번호 입력")
            go_ = st.form_submit_button("들어가기", use_container_width=True, type="primary")
        if go_:
            if hmac.compare_digest(pw.encode(), str(secret).encode()):
                st.session_state.auth = True
                st.rerun()
            st.error("비밀번호가 맞지 않습니다.")
    return False


if not password_ok():
    st.stop()


# ───────────── 데이터 ─────────────
@st.cache_data(ttl=1800, show_spinner="최신 데이터를 받아 계산하는 중…")
def load_market(test: bool):
    cfg = load()
    closes, fg, notes = data.load_all(cfg, test=test)
    return closes, fg, notes, datetime.now(ZoneInfo("Asia/Seoul"))


def compute_all(cfg, closes, fg):
    out = {}
    for name, t in cfg.indices.items():
        out[name] = signals.compute(name, closes[t], closes[cfg.vol_index[name]], fg, cfg.rules, cfg)
    return out


cfg = load()
try:
    closes, fg, notes, fetched = load_market(TEST)
except Exception as e:
    st.error(f"데이터를 받지 못했습니다: {e}")
    st.stop()
res = compute_all(cfg, closes, fg)


# ───────────── 공통 표시 함수 ─────────────
def pill(text):
    c = "p-buy" if text.startswith("매수") else "p-sell" if text.startswith("매도") else \
        "p-warn" if text.startswith(("경계", "비쌈")) else "p-none"
    return f"<span class='pill {c}'>{text}</span>"


def fnum(v, d=2):
    return "-" if v is None or pd.isna(v) else f"{v:,.{d}f}"


def fpct(v):
    return "-" if v is None or pd.isna(v) else f"{v * 100:+.1f}%"


def fdate(v):
    return "-" if v is None or pd.isna(v) else pd.Timestamp(v).strftime("%Y-%m-%d")


def gauge(value, title, steps, rng=(0, 100), mark=None):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value, number={"font": {"size": 40}, "valueformat": ".1f"},
        title={"text": title, "font": {"size": 14}},
        gauge={"axis": {"range": rng, "tickwidth": 1}, "bar": {"color": "rgba(40,40,40,.75)", "thickness": 0.25},
               "steps": steps, "borderwidth": 0,
               **({"threshold": {"line": {"color": "#1F4E78", "width": 3}, "thickness": 0.9, "value": mark}}
                  if mark is not None else {})}))
    fig.update_layout(height=210, margin=dict(l=20, r=20, t=40, b=0))
    return fig



# ───────────── 알림 팝업 ─────────────
POP_DAYS = int(getattr(cfg, "popup_days", 30))
NEW_DAYS = int(getattr(cfg, "popup_new_days", 7))
POP_KIND = {"SELL": ("매도 신호", SELL), "BUY": ("매수 신호", BUY), "BUY_CROSS": ("매수 신호", BUY),
            "ALARM_ON": ("매도 알람 켜짐", WARN), "ALARM_EXPIRED": ("매도 알람 해제", MUTED)}
POP_ORDER = {"SELL": 0, "BUY": 1, "BUY_CROSS": 1, "ALARM_ON": 2, "ALARM_EXPIRED": 3}


def recent_events():
    today = pd.Timestamp(datetime.now(ZoneInfo("Asia/Seoul")).date())
    out = []
    for name, r in res.items():
        for e in r["events"]:
            age = (today - pd.Timestamp(e["date"]).normalize()).days
            if 0 <= age <= POP_DAYS:
                out.append((age, name, e))
    out.sort(key=lambda x: (POP_ORDER.get(x[2]["type"], 9), x[0]))
    return out


def event_line(name, e):
    t = e["type"]
    d = pd.Timestamp(e["date"]).strftime("%Y-%m-%d")
    if t in ("BUY", "BUY_CROSS"):
        return f"{d} 점검 · 크레마지수 {e['score']:.1f}점 → 모아둔 현금으로 매수 검토"
    if t == "SELL":
        return (f"{d} 주 · 주간 RSI {e['rsi']:.1f}로 매도선 도달 (알람 시작 "
                f"{pd.Timestamp(e['start']):%Y-%m-%d}) → 매도, 레버리지는 전량")
    if t == "ALARM_ON":
        return f"{d} 주 · 주간 RSI {e['rsi']:.1f} 과열 지속 → 아직 팔지 않고 경계"
    return f"{pd.Timestamp(e['start']):%Y-%m-%d}에 켜진 알람이 기한 안에 매도선에 닿지 않아 해제"


@st.dialog("알림", width="large")
def show_popup():
    items = recent_events()
    if items:
        st.markdown(f"**최근 {POP_DAYS}일 신호 {len(items)}건**")
        for age, name, e in items:
            label, color = POP_KIND[e["type"]]
            new = "<span class='pill p-sell' style='margin-left:6px'>NEW</span>" if age <= NEW_DAYS else ""
            st.markdown(
                f"<div style='border-left:5px solid {color};background:rgba(138,148,166,.08);"
                f"border-radius:8px;padding:10px 14px;margin:6px 0'>"
                f"<b style='color:{color}'>{name} {label}</b>{new}"
                f"<div style='font-size:.9rem;margin-top:3px'>{event_line(name, e)}</div></div>",
                unsafe_allow_html=True)
    else:
        st.success(f"최근 {POP_DAYS}일 동안 새 신호가 없습니다.")
    st.markdown("**현재 상태**")
    for name, r in res.items():
        s = r["snap"]
        extra = ""
        if s["alarm_start"] is not None and s.get("sell_price"):
            extra = f" · 이번 주 종가 {s['sell_price']:,.2f} 아래 마감 시 매도"
        st.markdown(f"<div class='kv'><span>{name}</span><span>크레마지수 {s['score']:.1f} {pill(s['buy_status'])}"
                    f" &nbsp; 주간 RSI {s['rsi_w']:.1f} {pill(s['sell_status'])}{extra}</span></div>",
                    unsafe_allow_html=True)
    st.write("")
    if st.button("확인", type="primary", use_container_width=True):
        st.rerun()


if not st.session_state.get("popup_done"):
    st.session_state.popup_done = True
    show_popup()

# ───────────── 상단 ─────────────
top_l, top_r = st.columns([3, 2])
with top_l:
    asof = max(r["snap"]["asof"] for r in res.values())
    st.markdown(f"<div class='app-title'>투자알람</div><div class='app-sub'>데이터 기준 {fdate(asof)} · "
                f"갱신 {fetched:%m-%d %H:%M} (KST){' · 장중 잠정값 포함' if any(r['snap']['market_open'] for r in res.values()) else ''}"
                f"</div>", unsafe_allow_html=True)
with top_r:
    b1, b2 = st.columns(2)
    n_new = sum(1 for a, _, _ in recent_events() if a <= NEW_DAYS)
    if b1.button(f"알림 {n_new}" if n_new else "알림", use_container_width=True,
                 type="primary" if n_new else "secondary"):
        show_popup()
    if b2.button("새로고침", use_container_width=True):
        load_market.clear()
        st.rerun()
for n in notes:
    st.caption(f"참고: {n}")

mc = st.columns(len(res))
for col, (name, r) in zip(mc, res.items()):
    s = r["snap"]
    with col:
        st.markdown(f"<div class='mini'><div class='n'>{name} <span class='app-sub'>{s['ticker']}</span></div>"
                    f"<div style='display:flex;gap:18px;align-items:baseline;margin-top:4px'>"
                    f"<div><div class='app-sub'>크레마지수</div><div class='v'>{s['score']:.1f}</div>{pill(s['buy_status'])}</div>"
                    f"<div><div class='app-sub'>주간 RSI</div><div class='v'>{s['rsi_w']:.1f}</div>{pill(s['sell_status'])}</div>"
                    f"</div></div>", unsafe_allow_html=True)

st.write("")
names = list(res.keys())
try:
    sel = st.segmented_control("지수", names, default=names[-1], label_visibility="collapsed") or names[-1]
except Exception:
    sel = st.radio("지수", names, horizontal=True, label_visibility="collapsed")
R = res[sel]
S = R["snap"]
rules = cfg.rules

# ───────────── 핵심 카드 2개 ─────────────
c1, c2 = st.columns(2)
with c1:
    with st.container(border=True):
        st.markdown(f"**매수 — 크레마지수** &nbsp; {pill(S['buy_status'])}", unsafe_allow_html=True)
        th = S["threshold"]
        st.plotly_chart(gauge(S["score"], "0 비쌈 · 100 쌈", [
            {"range": [0, rules["avg_score"]], "color": "rgba(211,47,47,.18)"},
            {"range": [rules["avg_score"], th], "color": "rgba(230,149,0,.18)"},
            {"range": [th, 100], "color": "rgba(26,143,76,.22)"}], mark=th),
            use_container_width=True, config={"displayModeBar": False})
        gap = th - S["score"]
        msg = (f"기준점수 {th:.0f}점 <b>도달</b> — 월 점검일에 모아둔 현금으로 매수 검토" if gap <= 0 else
               f"기준점수 {th:.0f}점까지 <b>{gap:.1f}점</b> 남음")
        st.markdown(f"<p class='big-note'>{msg}</p>", unsafe_allow_html=True)
        st.markdown(f"<div class='kv'><span>30년 평균</span><span>{rules['avg_score']:.0f}점</span></div>"
                    f"<div class='kv'><span>매수 점검일</span><span>매월 {cfg.buy_check_day}일</span></div>",
                    unsafe_allow_html=True)
with c2:
    with st.container(border=True):
        st.markdown(f"**매도 — 주간 RSI** &nbsp; {pill(S['sell_status'])}", unsafe_allow_html=True)
        st.plotly_chart(gauge(S["rsi_w"], f"확정 {fdate(S['week_label'])} 주 · 잠정 {S['rsi_w_prov']:.1f}", [
            {"range": [0, rules["sell_rsi"]], "color": "rgba(211,47,47,.18)"},
            {"range": [rules["sell_rsi"], rules["alarm_rsi"]], "color": "rgba(138,148,166,.15)"},
            {"range": [rules["alarm_rsi"], 100], "color": "rgba(230,149,0,.25)"}], mark=rules["sell_rsi"]),
            use_container_width=True, config={"displayModeBar": False})
        if S["alarm_start"] is not None:
            line = (f"이번 주 종가 <b>{fnum(S['sell_price'])}</b> 아래 마감 시 매도 "
                    f"({fpct(S['sell_price'] / S['close'] - 1)})" if S["sell_price"] else "매도선 계산 불가")
            st.markdown(f"<p class='big-note'>{line}</p>", unsafe_allow_html=True)
            st.markdown(f"<div class='kv'><span>알람 시작</span><span>{fdate(S['alarm_start'])}</span></div>"
                        f"<div class='kv'><span>알람 기한</span><span>{fdate(S['alarm_expiry'])}</span></div>"
                        f"<div class='kv'><span>매도선({rules['sell_rsi']:.0f})까지</span><span>{S['gap_to_sell']:+.1f}</span></div>",
                        unsafe_allow_html=True)
        else:
            ap = S.get("alarm_price")
            line = (f"이번 주 종가 {fnum(ap)} 이상이면 과열 구간 진입 ({fpct(ap / S['close'] - 1)})" if ap else
                    f"알람 없음 — 주간 RSI {rules['alarm_rsi']:.0f} 이상 {rules['alarm_weeks']}주 유지 시 알람")
            st.markdown(f"<p class='big-note'>{line}</p>", unsafe_allow_html=True)
            last_sell = [e for e in R["events"] if e["type"] == "SELL"]
            st.markdown(f"<div class='kv'><span>최근 매도 신호</span><span>{fdate(last_sell[-1]['date']) if last_sell else '-'}</span></div>"
                        f"<div class='kv'><span>알람 유효기간</span><span>{rules['alarm_months']}개월</span></div>",
                        unsafe_allow_html=True)

# ───────────── 지표 타일 ─────────────
raw = {"ytd_dd": fpct(S["ytd_dd"]), "rsi_w": f"{S['rsi_w_prov']:.1f}", "rsi_d": f"{S['rsi_d']:.1f}",
       "vix_dev": fpct(S["vix_dev"]), "dist200": fpct(S["dist200"]),
       "fg": f"{S['fg']:.0f}" + (" (대체)" if S["fg_fallback"] else "")}
tile_html = "".join(
    f"<div class='tile'><div class='tl'>{label} <span>{w:.0%}</span></div><div class='tv'>{raw[k]}</div>"
    f"<div class='bar'><i style='width:{S['scores'][k]:.0f}%;background:{BUY if S['scores'][k] >= 60 else WARN if S['scores'][k] >= 31 else SELL}'></i></div>"
    f"<div class='ts'>{S['scores'][k]:.0f}점</div></div>"
    for (k, label), w in zip(AXES, rules["weights"]))
st.markdown(f"<div class='tiles'>{tile_html}</div>", unsafe_allow_html=True)
st.caption(f"종가 {fnum(S['close'])} · 연중 고점 {fnum(S['ytd_high'])} · 200일선 {fnum(S['ma200'])} · "
           f"{S['vol_ticker']} {fnum(S['vix'])} (평소 {fnum(S['vix_normal'])}) · 타일 아래 숫자는 축별 점수")

# ───────────── 탭 ─────────────
tab_chart, tab_axes, tab_log, tab_sim, tab_help = st.tabs(["차트", "6개 축", "신호 기록", "규칙 시뮬레이션", "안내"])


def alarm_spans(r):
    return [(a, b if b is not None else r["weekly"].index[-1]) for a, b in r["alarm_history"]]


with tab_chart:
    w = R["weekly"].copy()
    d = R["daily"]
    w["ma200"] = d["ma200"].resample("W-FRI").last().reindex(w.index)
    w["score"] = d["score"].resample("W-FRI").last().reindex(w.index)
    w = w[w.index >= "1999-01-01"]
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.5, 0.25, 0.25],
                        subplot_titles=("", "크레마지수", "주간 RSI"))
    fig.add_trace(go.Scatter(x=w.index, y=w["close"], name="종가", line=dict(width=1.6)), 1, 1)
    fig.add_trace(go.Scatter(x=w.index, y=w["ma200"], name="200일선", line=dict(width=1, dash="dot", color=MUTED)), 1, 1)
    for typ, color, sym, label in [("BUY", BUY, "circle", "매수 신호"), ("ALARM_ON", WARN, "triangle-up", "매도 알람"),
                                   ("SELL", SELL, "diamond", "매도 신호")]:
        ev = [e for e in R["events"] if e["type"] == typ]
        if ev:
            xs = [pd.Timestamp(e["date"]).to_period("W-FRI").end_time.normalize() for e in ev]
            ys = [w["close"].get(x, np.nan) for x in xs]
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers", name=label,
                                     marker=dict(color=color, size=9 if typ != "BUY" else 7, symbol=sym,
                                                 line=dict(width=1, color="white"))), 1, 1)
    fig.add_trace(go.Scatter(x=w.index, y=w["score"], name="크레마지수", line=dict(width=1.3, color=BUY),
                             showlegend=False), 2, 1)
    fig.add_hline(y=S["threshold"], line=dict(color=BLUE, dash="dash", width=1), row=2, col=1)
    fig.add_hline(y=rules["avg_score"], line=dict(color=MUTED, dash="dot", width=1), row=2, col=1)
    fig.add_trace(go.Scatter(x=w.index, y=w["rsi"], name="주간 RSI", line=dict(width=1.3, color="#5b6ee1"),
                             showlegend=False), 3, 1)
    fig.add_hline(y=rules["alarm_rsi"], line=dict(color=SELL, dash="dash", width=1), row=3, col=1)
    fig.add_hline(y=rules["sell_rsi"], line=dict(color=WARN, dash="dash", width=1), row=3, col=1)
    for a, b in alarm_spans(R):
        for row in (1, 3):
            fig.add_vrect(x0=a, x1=b, fillcolor=WARN, opacity=0.10, line_width=0, row=row, col=1)
    end = w.index[-1]
    x0 = end - pd.DateOffset(years=5)
    fig.update_xaxes(range=[x0, end + pd.Timedelta(days=14)])
    vis = w.loc[x0:, ["close", "ma200"]].stack()
    fig.update_yaxes(range=[vis.min() * 0.92, vis.max() * 1.05], row=1, col=1)
    fig.update_xaxes(rangeselector=dict(buttons=[
        dict(count=1, label="1년", step="year", stepmode="backward"),
        dict(count=5, label="5년", step="year", stepmode="backward"),
        dict(count=10, label="10년", step="year", stepmode="backward"),
        dict(step="all", label="전체")], x=0, y=1.07, yanchor="bottom"), row=1, col=1)
    fig.update_yaxes(range=[0, 100], row=2, col=1)
    fig.update_yaxes(range=[0, 100], row=3, col=1)
    fig.update_layout(height=760, hovermode="x unified", margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(orientation="h", y=1.07, yanchor="bottom", x=1, xanchor="right"))
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")
    st.caption("주황 음영 = 매도 알람이 켜져 있던 기간 · 파란 점선 = 매수 기준점수 · 회색 점선 = 30년 평균 · 주간 봉은 금요일 종가 기준")

with tab_axes:
    labels = [f"{lbl} ({w:.0%})" for (k, lbl), w in zip(AXES, rules["weights"])]
    vals = [S["scores"][k] for k, _ in AXES]
    colors = [BUY if v >= 60 else WARN if v >= 31 else SELL for v in vals]
    f2 = go.Figure(go.Bar(x=vals, y=labels, orientation="h", marker_color=colors,
                          text=[f"{v:.0f}점" for v in vals], textposition="outside"))
    f2.update_layout(height=320, xaxis=dict(range=[0, 110], title="축별 점수 (100 = 매우 쌈)"),
                     yaxis=dict(autorange="reversed"), margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(f2, use_container_width=True, config={"displayModeBar": False})
    tbl = pd.DataFrame({
        "축": [lbl for _, lbl in AXES],
        "비중": [f"{w:.0%}" for w in rules["weights"]],
        "현재 값": [raw[k] for k, _ in AXES],
        "0점 기준": [fpct(v) if k in ("ytd_dd", "vix_dev", "dist200") else f"{v:g}" for (k, _), v in zip(AXES, rules["lo"])],
        "100점 기준": [fpct(v) if k in ("ytd_dd", "vix_dev", "dist200") else f"{v:g}" for (k, _), v in zip(AXES, rules["hi"])],
        "점수": [f"{v:.1f}" for v in vals],
    })
    st.dataframe(tbl, hide_index=True, use_container_width=True)
    st.caption(f"크레마지수 = 축별 점수의 가중평균 = {S['score']:.1f}점")

KIND = {"BUY": "매수 신호", "BUY_CROSS": "매수(돌파)", "ALARM_ON": "매도 알람 켜짐", "SELL": "매도 신호",
        "ALARM_EXPIRED": "알람 해제"}


def event_table(events, w):
    rows = []
    last = w["close"].iloc[-1]
    for e in sorted(events, key=lambda e: e["date"], reverse=True):
        wl = pd.Timestamp(e["date"]).to_period("W-FRI").end_time.normalize()
        px = e.get("close", w["close"].get(wl, np.nan))
        after = ""
        if e["type"] == "SELL":
            fut = w.loc[wl:wl + pd.DateOffset(years=2), "close"]
            after = f"이후 최대 {fut.min() / px - 1:+.1%}" if len(fut) > 1 else ""
        elif e["type"] in ("BUY", "BUY_CROSS"):
            after = f"현재까지 {last / px - 1:+.1%}"
        detail = (f"크레마지수 {e['score']:.1f}" if "score" in e else f"주간 RSI {e['rsi']:.1f}")
        rows.append({"날짜": fdate(e["date"]), "신호": KIND[e["type"]], "내용": detail,
                     "가격": fnum(px), "이후 흐름": after})
    return pd.DataFrame(rows)


with tab_log:
    kinds = st.multiselect("보기", list(KIND.values())[:5], default=["매수 신호", "매도 알람 켜짐", "매도 신호"])
    df = event_table(R["events"], R["weekly"])
    df = df[df["신호"].isin(kinds)] if len(df) else df
    st.dataframe(df, hide_index=True, use_container_width=True, height=420)
    st.caption("매도 신호의 '이후 흐름' = 판 뒤 2년 안 최저가까지의 하락폭 · 매수 신호는 그때 가격 대비 현재 수익률")

with tab_sim:
    st.markdown("규칙 숫자를 바꿔 보면 과거 신호가 어떻게 달라지는지 바로 확인할 수 있습니다. "
                "**여기서 바꾼 값은 저장되지 않으며**, 실제 알람 규칙은 저장소의 settings.json에서 바꿉니다.")
    a1, a2, a3, a4 = st.columns(4)
    sim_th = a1.slider("매수 기준점수", 40, 95, int(S["threshold"]), 5)
    sim_alarm = a2.slider("알람 RSI", 65, 80, int(rules["alarm_rsi"]))
    sim_sell = a3.slider("매도 RSI", 40, 60, int(rules["sell_rsi"]))
    sim_months = a4.slider("알람 유효기간(개월)", 6, 24, int(rules["alarm_months"]))
    bt = dict(rules["buy_threshold"]); bt[sel] = sim_th
    c2_ = with_rules(cfg, buy_threshold=bt, alarm_rsi=sim_alarm, sell_rsi=sim_sell, alarm_months=sim_months)
    r2 = signals.compute(sel, closes[cfg.indices[sel]], closes[cfg.vol_index[sel]], fg, c2_.rules, c2_)
    ev2 = r2["events"]
    sells = event_table([e for e in ev2 if e["type"] == "SELL"], r2["weekly"])
    buys = [e for e in ev2 if e["type"] == "BUY"]
    yrs = (r2["daily"]["score"].dropna().index[-1] - r2["daily"]["score"].dropna().index[0]).days / 365.25
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("매수 신호 (월 점검)", f"{len(buys)}회")
    k2.metric("매수 신호 빈도", f"{yrs / max(len(buys), 1):.1f}년에 1회" if buys else "-")
    k3.metric("매도 신호", f"{len(sells)}회")
    if len(sells):
        drops = sells["이후 흐름"].str.extract(r"([+-][\d.]+)%")[0].astype(float)
        k4.metric("매도 후 추가하락 중간값", f"{drops.median():.1f}%")
    st.dataframe(event_table(ev2, r2["weekly"]), hide_index=True, use_container_width=True, height=360)

with tab_help:
    st.markdown(f"""
**매수 (크레마지수)** — 6개 지표를 0~100점으로 바꿔 가중평균합니다. 높을수록 쌉니다.
매월 {cfg.buy_check_day}일 기준 점수가 기준점수 이상이면 모아둔 현금으로 매수합니다. 점수가 낮다고 팔지는 않습니다.

**매도 (주간 RSI)** — 주간 RSI가 {rules['alarm_rsi']:.0f} 이상을 {rules['alarm_weeks']}주 유지하면 알람이 켜집니다.
알람 후 {rules['alarm_months']}개월 안에 {rules['sell_rsi']:.0f} 이하로 내려오면 그 주에 매도(레버리지는 전량), 못 내려오면 알람 해제입니다.

**알림 팝업** — 앱을 열면 최근 {POP_DAYS}일 안의 신호와 현재 상태를 팝업으로 보여 줍니다({NEW_DAYS}일 이내는 NEW 표시). 오른쪽 위 '알림' 버튼으로 다시 열 수 있습니다.

**참고** — 주간 봉은 금요일 종가 기준입니다(Yahoo 차트는 월요일 날짜로 표시). 장중에는 이번 주 값이 '잠정'이며 알람은 확정값으로만 판단합니다.
데이터: Yahoo Finance, CNN 공포탐욕지수. 테스프레소 영상 규칙을 자동 계산한 도구이며 투자 권유가 아닙니다.
""")
    if TEST:
        st.warning("테스트 모드: 가상 데이터로 표시 중입니다.")
