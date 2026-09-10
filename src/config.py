from __future__ import annotations

import json

from src.database import init_db, Setting

# Default scoring weights for the conviction score
DEFAULT_BRIEF_WEIGHTS = {
    "sentiment": 0.25,
    "technical": 0.20,
    "ml_pred": 0.30,
    "analyst": 0.25,
}


def load_settings() -> dict:
    init_db()
    result = {}
    for s in Setting.select():
        if s.key == "risk_allocation":
            result[s.key] = json.loads(s.value)
        elif s.key == "brief_weights":
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


def load_brief_weights() -> dict[str, float]:
    """Load conviction score weights, falling back to defaults."""
    settings = load_settings()
    user = settings.get("brief_weights", {})
    weights = dict(DEFAULT_BRIEF_WEIGHTS)
    weights.update(user)
    total = sum(weights.values())
    if total > 0:
        weights = {k: v / total for k, v in weights.items()}
    return weights


def update_brief_weights(weights: dict[str, float]):
    """Persist custom brief scoring weights."""
    init_db()
    Setting.get_or_create(key="brief_weights", defaults={"value": "{}"})
    Setting.update(value=json.dumps(weights)).where(Setting.key == "brief_weights").execute()
