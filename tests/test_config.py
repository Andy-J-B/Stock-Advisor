"""Tests for configuration operations."""

import pytest
from src.database import init_db, db, Setting
from src import config


@pytest.fixture(autouse=True)
def fresh_db():
    if not db.is_closed():
        db.close()
    original = db.database
    db.init(":memory:")
    db.connect()
    init_db()
    yield
    db.drop_tables([Setting])
    db.close()
    db.init(original)


def test_load_settings_empty():
    s = config.load_settings()
    assert s == {}


def test_update_and_load_allocation():
    config.update_allocation(30, 50, 20)
    s = config.load_settings()
    alloc = s["risk_allocation"]
    assert alloc["conservative"] == 30
    assert alloc["moderate"] == 50
    assert alloc["aggressive"] == 20


def test_update_allocation_overwrites():
    config.update_allocation(10, 20, 70)
    config.update_allocation(50, 30, 20)
    s = config.load_settings()
    alloc = s["risk_allocation"]
    assert alloc["conservative"] == 50
    assert alloc["moderate"] == 30
    assert alloc["aggressive"] == 20


def test_load_brief_weights_defaults():
    w = config.load_brief_weights()
    assert w["sentiment"] == 0.25
    assert w["technical"] == 0.20
    assert w["ml_pred"] == 0.30
    assert w["analyst"] == 0.25
    assert abs(sum(w.values()) - 1.0) < 1e-9


def test_load_brief_weights_partial_custom():
    config.update_brief_weights({"sentiment": 0.5})
    w = config.load_brief_weights()
    # Custom value merged with defaults, then normalized to sum 1.0
    # sentiment 0.5 + defaults 0.2+0.3+0.25 = 1.25 → normalized
    assert abs(w["sentiment"] - 0.5 / 1.25) < 1e-9
    assert abs(sum(w.values()) - 1.0) < 1e-9


def test_load_brief_weights_normalizes():
    config.update_brief_weights({
        "sentiment": 0.5, "technical": 0.5,
    })
    w = config.load_brief_weights()
    # Merged with defaults then normalized to sum 1.0
    assert w["sentiment"] == w["technical"]
    assert abs(sum(w.values()) - 1.0) < 1e-9
    assert abs(w["sentiment"] - 0.5 / 1.55) < 1e-9
