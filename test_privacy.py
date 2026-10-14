"""Tests for the privacy / responsible-AI data checks."""
from __future__ import annotations

import pandas as pd

from src.evaluation.privacy import PAN_REGEX, mask_pan, run_privacy_scan


def test_pan_regex_catches_standalone_pan():
    assert PAN_REGEX.search("card 4111111111111111 leaked")
    assert PAN_REGEX.search("x 6011111111111117 y")


def test_pan_regex_ignores_numbers_that_are_not_pans():
    # float mantissas / decimals / scientific notation are not card numbers
    assert not PAN_REGEX.search("0.1943818821313555")
    assert not PAN_REGEX.search("2.2992200543470176e-07")
    assert not PAN_REGEX.search("8200000000002.5")
    assert not PAN_REGEX.search("short 123 and 12345")


def test_mask_pan():
    out = mask_pan("pan 4111111111111111 end")
    assert "4111111111111111" not in out
    assert out == "pan 4111********1111 end"
    # runs shorter than 13 digits are not PAN-like and pass through unchanged
    assert mask_pan("id 123456789") == "id 123456789"


def test_privacy_scan_clean_on_synthetic_data(small_data, tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    for name, df in small_data.items():
        df.to_csv(d / f"{name}.csv", index=False)
    out = tmp_path / "out"
    report = run_privacy_scan(d, out, out)
    assert report["files_scanned"] == 5
    assert report["scan_clean"], report["pan_or_email_findings"]
    assert report["masking_utility_works"]


def test_privacy_scan_detects_planted_pan(tmp_path):
    """Positive control: the scanner MUST catch a planted PAN-like value."""
    d = tmp_path / "data"
    d.mkdir()
    pd.DataFrame({"note": ["all good", "card 4111111111111111 leaked here"]}
                 ).to_csv(d / "leak.csv", index=False)
    out = tmp_path / "out"
    report = run_privacy_scan(d, out, out)
    assert not report["scan_clean"]
    assert any("4111111111111111" in f["pan_like"]
               for f in report["pan_or_email_findings"])


def test_privacy_scan_ignores_float_mantissas(tmp_path):
    """Negative control: model artifacts contain long float mantissas — these
    must NOT be flagged as PANs."""
    d = tmp_path / "data"
    d.mkdir()
    pd.DataFrame({"importance": [0.1943818821313555, 0.05],
                  "p_value": [2.2992200543470176e-07, 0.5]}
                 ).to_csv(d / "model.csv", index=False)
    out = tmp_path / "out"
    report = run_privacy_scan(d, out, out)
    assert report["scan_clean"], report["pan_or_email_findings"]
