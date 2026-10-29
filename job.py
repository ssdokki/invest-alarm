"""매일 클라우드(GitHub Actions)에서 돌며 새 신호가 있으면 메일을 보냅니다.

  python job.py            # 실제 실행
  python job.py --test     # 가상 데이터, 메일은 data/last_mail.html로만 저장
"""
import argparse
import logging
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

from core import data, notify, signals
from core.settings import load

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("alarm")


def main(test=False, no_mail=False):
    cfg = load()
    closes, fg, notes = data.load_all(cfg, test=test, save=not test)
    results = {}
    for name, t in cfg.indices.items():
        try:
            results[name] = signals.compute(name, closes[t], closes[cfg.vol_index[name]], fg, cfg.rules, cfg)
            s = results[name]["snap"]
            log.info("%s 크레마지수 %.1f(%s) / 주간RSI 확정 %.1f / %s",
                     name, s["score"], s["buy_status"], s["rsi_w"], s["sell_status"])
        except Exception as e:
            log.exception("%s 계산 실패", name)
            notes.append(f"{name} 계산 실패: {e}")
    if not results:
        return 1

    today = datetime.now(ZoneInfo("Asia/Seoul")).date()
    st = {} if test else notify.load_state()
    events = [e for r in results.values() for e in r["events"]]
    fresh = notify.new_events(events, st, cfg.catchup_days, today)
    week = max((r["snap"]["week_label"] for r in results.values() if r["snap"]["week_label"] is not None), default=None)
    week_s = week.strftime("%Y-%m-%d") if week is not None else None
    weekly_due = bool(cfg.weekly_report and week_s and st.get("last_report_week") != week_s)

    if fresh or weekly_due:
        subj, html = notify.build_mail(fresh, [r["snap"] for r in results.values()], cfg.rules, today,
                                       weekly=weekly_due, notes=notes)
        if notify.send_mail(cfg, subj, html, dry_run=test or no_mail):
            notify.mark_sent(st, fresh)
            if weekly_due:
                st["last_report_week"] = week_s
        else:
            return 1  # 실패하면 상태를 저장하지 않아 다음 날 다시 시도
    else:
        log.info("새 알람 없음")
    st["last_run"] = datetime.now(ZoneInfo("Asia/Seoul")).isoformat(timespec="seconds")
    if not test:
        notify.save_state(st)
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--test", action="store_true")
    p.add_argument("--no-mail", action="store_true")
    a = p.parse_args()
    sys.exit(main(a.test, a.no_mail))
