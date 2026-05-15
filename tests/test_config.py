"""Tests for configuration operations."""

import pytest
from src.database import init_db, db, Setting
from src import config


@pytest.fixture(autouse=True)
def fresh_db():
    if not db.is_closed():
        db.close()
    db.init(":memory:")
    db.connect()
    init_db()
    yield
    db.drop_tables([Setting])
    db.close()


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
