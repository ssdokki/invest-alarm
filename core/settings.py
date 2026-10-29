"""settings.json 읽기. 규칙(비중·기준점수·72/50 등)은 이 파일 하나에서 관리합니다."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
AXES = [  # (키, 표시 이름)
    ("ytd_dd", "연중낙폭"), ("rsi_w", "주간 RSI"), ("rsi_d", "일간 RSI"),
    ("vix_dev", "VIX 이격도"), ("dist200", "200일선 거리"), ("fg", "공포탐욕지수"),
]


def load(path: Path | None = None) -> SimpleNamespace:
    js = json.loads((path or ROOT / "settings.json").read_text(encoding="utf-8"))
    cfg = SimpleNamespace(**js)
    for name in cfg.indices:
        cfg.vol_index.setdefault(name, "^VIX")
    return cfg


def with_rules(cfg: SimpleNamespace, **changes) -> SimpleNamespace:
    """화면에서 바꾼 값으로 규칙만 바꾼 사본 (원본은 그대로)."""
    c = copy.deepcopy(cfg)
    for k, v in changes.items():
        if k == "buy_threshold":
            c.rules["buy_threshold"] = dict(v)
        else:
            c.rules[k] = v
    return c
