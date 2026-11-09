"""정적 대시보드 생성: 데이터 수집 → 신호 계산 → 결과를 비밀번호로 암호화 → site/index.html

  SITE_PASSWORD=비밀번호 python build_site.py            # 실제 데이터
  SITE_PASSWORD=1234 python build_site.py --test         # 가상 데이터

결과 데이터는 AES-GCM(키: PBKDF2-SHA256)으로 암호화되어 페이지에 들어가며,
브라우저에서 비밀번호를 입력해야만 풀립니다.
"""
import argparse
import base64
import copy
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from core import data, screener, seasonal, signals
from core.settings import AXES, load

ROOT = Path(__file__).resolve().parent
ITER = 250_000
# 고정 salt: 비밀번호가 같으면 매일 새로 만든 페이지에서도 같은 키가 나와 "이 기기에서 기억"이 유지됨
SALT = b"invest-alarm/ssdokki/v1"


def clean(v):
    if v is None:
        return None
    if isinstance(v, (pd.Timestamp, datetime)):
        return pd.Timestamp(v).strftime("%Y-%m-%d")
    if isinstance(v, (float, np.floating)):
        return None if np.isnan(v) else round(float(v), 4)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def index_payload(name, r):
    d, w = r["daily"], r["weekly"]
    wk = pd.DataFrame({
        "close": w["close"], "rsi": w["rsi"],
        "ma200": d["ma200"].resample("W-FRI").last().reindex(w.index),
        "score": d["score"].resample("W-FRI").last().reindex(w.index),
    })
    wk = wk[wk.index >= "1999-01-01"]
    events = []
    for e in sorted(r["events"], key=lambda e: e["date"]):
        wl = pd.Timestamp(e["date"]).to_period("W-FRI").end_time.normalize()
        events.append({
            "type": e["type"], "date": clean(e["date"]), "week": clean(wl),
            "score": clean(e.get("score")), "rsi": clean(e.get("rsi")),
            "start": clean(e.get("start")),
            "price": clean(e.get("close", w["close"].get(wl, np.nan))),
        })
    spans = [[clean(a), clean(b if b is not None else w.index[-1])] for a, b in r["alarm_history"]]
    s = r["snap"]
    snap = {k: clean(v) for k, v in s.items() if k not in ("scores", "now_et")}
    snap["scores"] = {k: clean(v) for k, v in s["scores"].items()}
    return {
        "name": name, "snap": snap, "events": events, "spans": spans,
        "w": {"date": [t.strftime("%Y-%m-%d") for t in wk.index],
              **{c: [clean(v) for v in wk[c]] for c in wk.columns}},
    }


def encrypt(obj, password: str) -> dict:
    salt, iv = SALT, os.urandom(12)
    key = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=ITER).derive(password.encode())
    ct = AESGCM(key).encrypt(iv, json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode(), None)
    b64 = lambda b: base64.b64encode(b).decode()
    return {"v": 1, "iter": ITER, "salt": b64(salt), "iv": b64(iv), "ct": b64(ct)}


def _crema_now(name, ticker, close, fg, cfg):
    """계절 종목에도 지금 쓰는 크레마지수·주간RSI 로직을 그대로 적용 (한국 종목: 맞춤 보정)."""
    c = copy.deepcopy(cfg)
    c.indices, c.modes, c.vol_index = {name: ticker}, {name: "relative"}, {}
    s = signals.compute(name, close, None, fg, c.rules, c)["snap"]
    return {k: clean(s[k]) for k in ("asof", "close", "score", "avg_score", "threshold", "buy_status",
                                     "rsi_w", "sell_status", "dist200", "ytd_dd")}


def build_seasonal(cfg, closes, fg, test):
    evs = seasonal.events_cfg(cfg)
    tick = {t: n for ev in evs for n, t in ev["stocks"].items()}
    px = dict(closes)
    bm = seasonal.params(cfg)["benchmark"]
    if test:
        rng = np.random.default_rng(5)
        k = closes[bm]
        for i, t in enumerate(tick):
            r = rng.normal(0.0002, 0.02, len(k)) + np.log(k).diff().fillna(0).to_numpy()
            px[t] = pd.Series(10000 * np.exp(np.cumsum(r)), index=k.index)[k.index >= "2011-01-01"]
    else:
        if bm not in px:
            px[bm] = data.fetch_close(bm)[0]
        for t in tick:
            try:
                px[t] = data.fetch_close(t)[0]
            except Exception as e:
                print(f"계절 패턴: {t} 수신 실패 ({e})")
    crema = {}
    for t, n in tick.items():
        if t in px:
            try:
                crema[t] = _crema_now(n, t, px[t], fg, cfg)
            except Exception as e:
                print(f"계절 패턴: {n} 크레마지수 계산 실패 ({e})")
    out = seasonal.run(cfg, px, crema)
    for ev in out.get("events", []):
        for it in ev["items"]:
            if it.get("grade") and it.get("train"):
                print(f"계절 패턴 [{ev['name']}] {it['name']}: 매수 D{it['buy_off']:+d} → 매도 D{it['sell_off']:+d} | "
                      f"학습 {it['train_years'][0]}~{it['train_years'][-1]} 코스피 대비 {it['train']['mean']:+.2f}% "
                      f"| 검증 {it['test_years'][0]}~{it['test_years'][-1]} {it['test']['mean']:+.2f}% "
                      f"({it['test']['wins']}/{it['test']['n']}년) | 무작위 대조 {it['rand_p']} → {it['grade']}")
            else:
                print(f"계절 패턴 [{ev['name']}] {it['name']}: {it.get('grade') or ''} {it.get('verdict') or it.get('error')}")
    return out


def main(test=False, out="site"):
    pw = os.environ.get("SITE_PASSWORD", "").strip()
    if not pw:
        print("SITE_PASSWORD가 없습니다 (GitHub Secrets에 등록하세요).", file=sys.stderr)
        return 1
    cfg = load()
    closes, fg, notes, fginfo = data.load_all(cfg, test=test, save=not test)
    idx = []
    for name, t in cfg.indices.items():
        r = signals.compute(name, closes[t], closes.get(cfg.vol_index.get(name)), fg, cfg.rules, cfg)
        ip = index_payload(name, r)
        asof = r["snap"]["asof"]
        cnn_recent = any(abs((asof - d).days) <= 3 for d in fginfo["cnn_dates"]) if fginfo["cnn_dates"] else False
        ip["snap"]["fg_src"] = "CNN" if cnn_recent else "자체 계산"
        idx.append(ip)
        s = r["snap"]
        print(f"{name}: 크레마지수 {s['score']:.1f} ({s['buy_status']}) / 주간RSI {s['rsi_w']:.1f} ({s['sell_status']})")
    payload = {
        "generated": datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d %H:%M"),
        "notes": notes, "test": test,
        "rules": {k: cfg.rules[k] for k in ("weights", "lo", "hi", "avg_score", "alarm_rsi",
                                             "alarm_weeks", "alarm_months", "sell_rsi")},
        "axes": [[k, label] for k, label in AXES],
        "buy_check_day": cfg.buy_check_day,
        "default_view": getattr(cfg, "default_view", None),
        "popup_days": int(getattr(cfg, "popup_days", 30)),
        "popup_new_days": int(getattr(cfg, "popup_new_days", 7)),
        "indices": idx,
        "fg": {"status": fginfo["cnn_status"], "compare": fginfo["compare"],
               "proxy_last": clean(fginfo["proxy"].iloc[-1]) if len(fginfo["proxy"]) else None,
               "components": {k: clean(v) for k, v in fginfo["components"].iloc[-1].items()}},
    }
    cmp_ = fginfo["compare"]
    if cmp_:
        print(f"공포탐욕 비교({cmp_['start']}~{cmp_['end']}, {cmp_['days']}일): 상관계수 {cmp_['corr']}, "
              f"평균 차이 {cmp_['mae']}점, 같은 구간 {cmp_['same_zone']:.0%}, 최근 CNN {cmp_['last_cnn']} / 자체 {cmp_['last_proxy']}")
    else:
        print("공포탐욕 비교: CNN 데이터 없음")
    print("공포탐욕 상태:", fginfo["cnn_status"], "/ 자체 계산 최근값:", payload["fg"]["proxy_last"])
    try:
        if test:
            uni, uni_src = pd.read_csv(screener.UNIVERSE_CSV, dtype=str), "테스트(가상 데이터)"
            prices = screener.synthetic(list(uni["code"][:40]))
        else:
            uni, uni_src = screener.load_universe()
            prices = screener.fetch_ohlcv(list(uni["code"]))
        scr = screener.run(cfg, uni, prices, closes.get("^KS11"))
        scr["universe_src"] = uni_src
        if scr.get("best"):
            b, bt = scr["best"], scr["baseline_test"]
            print(f"단기 규칙 검증({scr['universe_n']}종목) 선택: {b['rule']} | 학습 {scr['train_period']} 평균 {b['train']['mean']}% "
                  f"({b['train']['n']}회) | 검증 {scr['test_period']} 평균 {b['test']['mean'] if b['test'] else '-'}% "
                  f"({b['test']['n'] if b['test'] else 0}회, 무작위 {bt['mean'] if bt else '-'}%) → {'통과' if scr['passed'] else '미통과'}")
    except Exception as e:
        print("단기 스크리너 실패:", e)
        scr = {"error": str(e)}
    payload["screener"] = scr
    try:
        payload["seasonal"] = build_seasonal(cfg, closes, fg, test)
    except Exception as e:
        print("계절 패턴 실패:", e)
        payload["seasonal"] = {"error": str(e)}
    enc = encrypt(payload, pw)
    tpl = (ROOT / "site_template.html").read_text(encoding="utf-8")
    chart_js = (ROOT / "vendor" / "chart.umd.js").read_text(encoding="utf-8")
    html = tpl.replace("/*__CHARTJS__*/", chart_js).replace("__PAYLOAD__", json.dumps(enc))
    o = ROOT / out
    o.mkdir(exist_ok=True)
    (o / "index.html").write_text(html, encoding="utf-8")
    (o / ".nojekyll").write_text("", encoding="utf-8")
    print(f"생성: {o / 'index.html'} ({len(html) // 1024} KB)")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--test", action="store_true")
    p.add_argument("--out", default="site")
    a = p.parse_args()
    sys.exit(main(a.test, a.out))
