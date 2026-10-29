"""이메일 알림 (네이버 SMTP) + 알람 중복 방지 상태 파일."""
import json
import logging
import smtplib
import ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formataddr
from pathlib import Path

import pandas as pd

import os

from .settings import DATA

log = logging.getLogger("alarm")
STATE = DATA / "state.json"

TYPE_KR = {
    "BUY": "매수 신호",
    "BUY_CROSS": "매수 신호 (기준점수 돌파)",
    "ALARM_ON": "매도 알람 켜짐",
    "SELL": "매도 신호",
    "ALARM_EXPIRED": "매도 알람 해제",
}
PRIORITY = {"SELL": 0, "BUY": 1, "BUY_CROSS": 1, "ALARM_ON": 2, "ALARM_EXPIRED": 3}


def event_key(e) -> str:
    return f"{e['type']}|{e['index']}|{pd.Timestamp(e['date']).date()}"


def load_state() -> dict:
    if STATE.exists():
        try:
            return json.loads(STATE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def save_state(st: dict) -> None:
    STATE.write_text(json.dumps(st, ensure_ascii=False, indent=2), encoding="utf-8")


def new_events(all_events, st, catchup_days, today):
    """아직 안 보낸 이벤트. 첫 실행이면 최근 catchup_days 안의 것만 보내고 나머지는 보낸 것으로 기록."""
    sent = set(st.get("sent", []))
    first_run = not st.get("initialized")
    cutoff = pd.Timestamp(today) - pd.Timedelta(days=catchup_days)
    out = []
    for e in all_events:
        k = event_key(e)
        if k in sent:
            continue
        if first_run and pd.Timestamp(e["date"]) < cutoff:
            sent.add(k)
            continue
        out.append(e)
    st["sent"] = sorted(sent)
    st["initialized"] = True
    out.sort(key=lambda e: (PRIORITY.get(e["type"], 9), e["date"]))
    return out


def mark_sent(st, events):
    s = set(st.get("sent", []))
    s.update(event_key(e) for e in events)
    st["sent"] = sorted(s)


# ───────────── 메일 본문 ─────────────
def _f(x, fmt="{:,.2f}"):
    try:
        if x is None or pd.isna(x):
            return "-"
        return fmt.format(x)
    except Exception:
        return str(x)


def _pct(x):
    return _f(x * 100 if x is not None and not pd.isna(x) else None, "{:+.1f}%")


def _d(x):
    return "-" if x is None or pd.isna(x) else pd.Timestamp(x).strftime("%Y-%m-%d")


def describe_event(e, today) -> str:
    late = (pd.Timestamp(today) - pd.Timestamp(e["date"])).days > 7
    tag = "[지난 알람] " if late else ""
    t = e["type"]
    if t in ("BUY", "BUY_CROSS"):
        return (f"{tag}<b>{e['index']} {TYPE_KR[t]}</b> — 점검일 {_d(e['date'])}, "
                f"크레마지수 {_f(e['score'], '{:.1f}')}점, 종가 {_f(e.get('close'))}. 모아둔 현금으로 매수 검토.")
    if t == "ALARM_ON":
        return (f"{tag}<b>{e['index']} {TYPE_KR[t]}</b> — {_d(e['date'])} 주 주간 RSI {_f(e['rsi'], '{:.1f}')}. "
                f"아직 팔지 않고 경계. 이후 주간 RSI가 매도선 이하로 내려오면 매도.")
    if t == "SELL":
        return (f"{tag}<b style='color:#c00'>{e['index']} {TYPE_KR[t]}</b> — {_d(e['date'])} 주 주간 RSI "
                f"{_f(e['rsi'], '{:.1f}')}로 매도선 도달 (알람 시작 {_d(e['start'])}). 종가 {_f(e.get('close'))}. "
                f"레버리지는 전량 매도.")
    if t == "ALARM_EXPIRED":
        return (f"{tag}<b>{e['index']} {TYPE_KR[t]}</b> — {_d(e['start'])}에 켜진 알람이 기한 안에 "
                f"매도선에 닿지 않아 해제됨.")
    return f"{tag}{e}"


def status_table(snaps, rules) -> str:
    th = "style='border:1px solid #ccc;padding:4px 8px;background:#1F4E78;color:#fff'"
    td = "style='border:1px solid #ccc;padding:4px 8px;text-align:right'"
    tl = "style='border:1px solid #ccc;padding:4px 8px'"
    names = [s["index"] for s in snaps]
    rows = [
        ("기준일(데이터)", [_d(s["asof"]) + (" (장중 잠정)" if s["market_open"] else "") for s in snaps]),
        ("종가", [_f(s["close"]) for s in snaps]),
        ("크레마지수", [_f(s["score"], "{:.1f}") for s in snaps]),
        ("매수 기준점수", [_f(s["threshold"], "{:.0f}") for s in snaps]),
        ("매수 판정", [s["buy_status"] for s in snaps]),
        ("연중낙폭", [_pct(s["ytd_dd"]) for s in snaps]),
        ("200일선 거리", [_pct(s["dist200"]) for s in snaps]),
        ("일간 RSI", [_f(s["rsi_d"], "{:.1f}") for s in snaps]),
        ("VIX 이격도", [_pct(s["vix_dev"]) for s in snaps]),
        ("공포탐욕지수", [_f(s["fg"], "{:.0f}") + (" (대체값)" if s["fg_fallback"] else "") for s in snaps]),
        ("주간 RSI (확정)", [f"{_f(s['rsi_w'], '{:.1f}')} ({_d(s['week_label'])} 주)" for s in snaps]),
        ("주간 RSI (이번 주 잠정)", [_f(s["rsi_w_prov"], "{:.1f}") for s in snaps]),
        ("매도 판정", [s["sell_status"] for s in snaps]),
        ("알람 시작 / 기한", [f"{_d(s['alarm_start'])} / {_d(s['alarm_expiry'])}" for s in snaps]),
        (f"매도선({rules['sell_rsi']:.0f})까지", [_f(s["gap_to_sell"], "{:+.1f}") for s in snaps]),
        ("이번 주 매도 가격선", [
            (f"종가 {_f(s['sell_price'])} 아래 마감 시 매도 ({_pct(s['sell_price'] / s['close'] - 1)})"
             if s["sell_price"] else "-") for s in snaps]),
        (f"이번 주 알람({rules['alarm_rsi']:.0f}) 가격선", [
            (f"종가 {_f(s['alarm_price'])} 이상 ({_pct(s['alarm_price'] / s['close'] - 1)})"
             if s.get("alarm_price") else "-") for s in snaps]),
    ]
    h = f"<table style='border-collapse:collapse;font-size:13px'><tr><th {th}>항목</th>" + \
        "".join(f"<th {th}>{n}</th>" for n in names) + "</tr>"
    for label, vals in rows:
        h += f"<tr><td {tl}>{label}</td>" + "".join(f"<td {td}>{v}</td>" for v in vals) + "</tr>"
    return h + "</table>"


def build_mail(events, snaps, rules, today, weekly=False, notes=()):
    if events:
        top = events[0]
        more = f" 외 {len(events) - 1}건" if len(events) > 1 else ""
        subject = f"[투자알람] {top['index']} {TYPE_KR[top['type']]}{more} ({today})"
    else:
        subject = f"[투자알람] 주간 현황 ({today})"
    body = "<div style='font-family:Malgun Gothic,sans-serif'>"
    if events:
        body += "<h3>새 알람</h3><ul>" + "".join(f"<li>{describe_event(e, today)}</li>" for e in events) + "</ul>"
    body += f"<h3>{'주간 현황' if weekly or not events else '현재 현황'}</h3>" + status_table(snaps, rules)
    if notes:
        body += "<p style='color:#a60'>" + "<br>".join(notes) + "</p>"
    body += ("<p style='color:#888;font-size:12px'>테스프레소 크레마지수·매도알람 규칙을 자동 계산한 결과입니다. "
             "투자 권유가 아니며 판단과 책임은 본인에게 있습니다. 데이터: Yahoo Finance, CNN.</p></div>")
    return subject, body


def send_mail(cfg, subject, html, dry_run=False) -> bool:
    """Gmail(SMTP SSL 465)로 발송. 비밀번호는 환경변수 GMAIL_APP_PASSWORD (GitHub Secrets)."""
    url = os.environ.get("APP_URL", "").strip()
    if url:
        html = html.replace("<div style='font-family:Malgun Gothic,sans-serif'>",
                            "<div style='font-family:Malgun Gothic,sans-serif'>"
                            f"<p><a href='{url}' style='background:#1F4E78;color:#fff;padding:8px 14px;"
                            "border-radius:8px;text-decoration:none'>대시보드 열기</a></p>", 1)
    if dry_run:
        out = DATA / "last_mail.html"
        out.write_text(f"<h2>{subject}</h2>{html}", encoding="utf-8")
        log.info("메일 미발송(테스트) → %s", out)
        return True
    pw = os.environ.get("GMAIL_APP_PASSWORD", "").replace(" ", "")
    if not pw:
        log.error("GMAIL_APP_PASSWORD가 설정되지 않았습니다 (GitHub Secrets 확인).")
        return False
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = formataddr(("투자알람", cfg.mail_from))
    msg["To"] = cfg.mail_to
    msg.attach(MIMEText(html, "html", "utf-8"))
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=ssl.create_default_context(), timeout=30) as s:
            s.login(cfg.mail_from, pw)
            s.send_message(msg)
        log.info("메일 발송: %s", subject)
        return True
    except Exception as e:
        log.error("메일 발송 실패: %s", e)
        return False
