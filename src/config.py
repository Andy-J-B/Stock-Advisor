from __future__ import annotations

import json

from src.database import init_db, Setting


def load_settings() -> dict:
    init_db()
    result = {}
    for s in Setting.select():
        if s.key == "risk_allocation":
            result[s.key] = json.loads(s.value)
        elif s.key in ("conservative", "moderate", "aggressive"):
            continue
        else:
            result[s.key] = s.value
    return result


def update_allocation(conservative: int, moderate: int, aggressive: int):
    init_db()
    Setting.get_or_create(key="risk_allocation", defaults={"value": "{}"})
    Setting.update(value=json.dumps({
        "conservative": conservative,
        "moderate": moderate,
        "aggressive": aggressive,
    })).where(Setting.key == "risk_allocation").execute()
