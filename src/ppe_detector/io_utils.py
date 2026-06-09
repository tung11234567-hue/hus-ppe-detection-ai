from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import List

from .rules import Detection, PersonPPEStatus, summary_counts


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_result_json(path: str | Path, detections: List[Detection], statuses: List[PersonPPEStatus]) -> None:
    payload = {
        "summary": summary_counts(statuses),
        "detections": [asdict(d) for d in detections],
        "persons": [
            {
                "person": asdict(s.person),
                "helmet_ok": s.helmet_ok,
                "vest_ok": s.vest_ok,
                "violations": s.violations,
                "matched_helmet": asdict(s.matched_helmet) if s.matched_helmet else None,
                "matched_vest": asdict(s.matched_vest) if s.matched_vest else None,
            }
            for s in statuses
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
