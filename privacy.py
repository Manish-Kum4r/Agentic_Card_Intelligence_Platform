"""Privacy & responsible-AI data tests.

- Scans every generated CSV (data/ + outputs/) for PAN-like digit sequences
  (13–19 consecutive digits) and email addresses.
- Verifies the masking utility works.
- Verifies the dataset contains only synthetic identifiers.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

# Context-aware PAN detector: a standalone run of 13-19 digits that is NOT
# embedded in a decimal / scientific-notation number (float mantissas like
# 2.2992200543470176e-07 and long decimals are not card numbers).
PAN_REGEX = re.compile(r"(?<![\d.])\d{13,19}(?![\d.])")
EMAIL_REGEX = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")


def scan_text(text: str) -> dict:
    return {
        "pan_like_sequences": PAN_REGEX.findall(text),
        "emails": EMAIL_REGEX.findall(text),
    }


def mask_pan(text: str) -> str:
    """Mask any 13–19 digit sequence: keep first 4 + last 4 only if >8 digits,
    else fully mask."""
    def _m(m):
        s = m.group(0)
        return s[:4] + "*" * (len(s) - 8) + s[-4:] if len(s) > 8 else "*" * len(s)
    return PAN_REGEX.sub(_m, text)


def run_privacy_scan(data_dir, outputs_dir, out_dir) -> dict:
    findings = []
    n_files = 0
    for base in (Path(data_dir), Path(outputs_dir)):
        for csv in sorted(base.rglob("*.csv")):
            n_files += 1
            df = pd.read_csv(csv, dtype=str)
            blob = df.to_csv(index=False)
            scan = scan_text(blob)
            if scan["pan_like_sequences"] or scan["emails"]:
                findings.append({"file": str(csv),
                                 "pan_like": scan["pan_like_sequences"][:5],
                                 "emails": scan["emails"][:5]})
    # masking utility self-check (positive control)
    sample = "card 4111111111111111 and 6011111111111117 leaked"
    masked = mask_pan(sample)
    mask_ok = ("4111" + "*" * 8 + "1111" in masked
               and "6011" + "*" * 8 + "1117" in masked
               and "4111111111111111" not in masked
               and "6011111111111117" not in masked)

    report = {
        "files_scanned": n_files,
        "pan_or_email_findings": findings,
        "scan_clean": not findings,
        "masking_utility_works": mask_ok,
        "policy": ("Analytics datasets must never contain full card numbers "
                   "(PAN) or email addresses; synthetic token IDs only (KB-11)."),
    }
    out = Path(out_dir) / "evaluation"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "privacy_scan.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report
