"""정적 대시보드 생성: 데이터 수집 → 신호 계산 → 결과를 비밀번호로 암호화 → site/index.html

  SITE_PASSWORD=비밀번호 python build_site.py            # 실제 데이터
  SITE_PASSWORD=1234 python build_site.py --test         # 가상 데이터

결과 데이터는 AES-GCM(키: PBKDF2-SHA256)으로 암호화되어 페이지에 들어가며,
브라우저에서 비밀번호를 입력해야만 풀립니다.
"""
import argparse
import base64
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

from core import data, signals
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


def main(test=False, out="site"):
    pw = os.environ.get("SITE_PASSWORD", "").strip()
    if not pw:
        print("SITE_PASSWORD가 없습니다 (GitHub Secrets에 등록하세요).", file=sys.stderr)
        return 1
    cfg = load()
    closes, fg, notes = data.load_all(cfg, test=test)
    idx = []
    for name, t in cfg.indices.items():
        r = signals.compute(name, closes[t], closes[cfg.vol_index[name]], fg, cfg.rules, cfg)
        idx.append(index_payload(name, r))
        s = r["snap"]
        print(f"{name}: 크레마지수 {s['score']:.1f} ({s['buy_status']}) / 주간RSI {s['rsi_w']:.1f} ({s['sell_status']})")
    payload = {
        "generated": datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d %H:%M"),
        "notes": notes, "test": test,
        "rules": {k: cfg.rules[k] for k in ("weights", "lo", "hi", "avg_score", "alarm_rsi",
                                             "alarm_weeks", "alarm_months", "sell_rsi")},
        "axes": [[k, label] for k, label in AXES],
        "buy_check_day": cfg.buy_check_day,
        "popup_days": int(getattr(cfg, "popup_days", 30)),
        "popup_new_days": int(getattr(cfg, "popup_new_days", 7)),
        "indices": idx,
    }
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
