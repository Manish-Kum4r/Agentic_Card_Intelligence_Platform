"""Tests for the synthetic data generator + privacy properties of the data."""
from __future__ import annotations

import re

import numpy as np

PAN = re.compile(r"(?<!\d)\d{13,19}(?!\d)")


def test_shapes(small_data):
    assert len(small_data["cardholders"]) == 400
    assert len(small_data["engagement"]) == 400 * 12
    assert len(small_data["transactions"]) == 4000
    assert len(small_data["campaigns"]) == 10
    assert set(small_data["ab_test"]["arm"]) == {"control", "treatment"}


def test_engagement_score_in_range(small_data):
    s = small_data["engagement"]["engagement_score"]
    assert s.between(0, 100).all()
    assert s.std() > 5  # real variation, not a constant


def test_decline_label_defined(small_data):
    ev = small_data["engagement"]
    ev = ev[ev["engagement_decline"] >= 0]
    rate = ev["engagement_decline"].mean()
    assert 0.05 < rate < 0.60, f"implausible decline rate {rate}"


def test_no_pan_or_emails_in_data(small_data):
    for name, df in small_data.items():
        blob = df.to_csv(index=False)
        assert not PAN.search(blob), f"PAN-like sequence in {name}"
        assert "@" not in blob, f"email address in {name}"


def test_transactions_coherent(small_data):
    t = small_data["transactions"]
    assert (t["amount"] > 0).all()
    assert (t["rewards_earned"] >= 0).all()
    assert t["txn_date"].str.match(r"\d{4}-\d{2}-\d{2}").all()
    assert t["cardholder_id"].isin(small_data["cardholders"]["cardholder_id"]).all()


def test_reproducible():
    from src.data_generation import generate_all
    import tempfile
    with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
        a = generate_all(d1, seed=99, n_cardholders=50, months=6, n_transactions=200)
        b = generate_all(d2, seed=99, n_cardholders=50, months=6, n_transactions=200)
    assert a["transactions"].equals(b["transactions"])
