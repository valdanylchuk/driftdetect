#!/usr/bin/env python3
"""
Test the actual API with drift detection.

Usage:
    python examples/lendingclub/test_run.py
"""

import gzip
import logging
from pathlib import Path

from fastapi.testclient import TestClient

# Enable checkdrift logging, silence httpx
logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)

from examples.lendingclub.api import app

DATA_DIR = Path(__file__).parent / "data"


def load_samples(path: Path) -> list[dict]:
    """Load CSV as list of dicts."""
    samples = []
    with gzip.open(path, "rt") as f:
        header = f.readline().strip().split(",")
        for line in f:
            values = line.strip().split(",")
            try:
                samples.append({col: float(val) for col, val in zip(header, values)})
            except ValueError:
                continue
    return samples


def main():
    samples = load_samples(DATA_DIR / "lendingclub_2017_sample.csv.gz")[:200]

    print(f"Sending {len(samples)} requests to /triage...")
    print("Drift logs will appear below:\n")

    with TestClient(app) as client:
        for sample in samples:
            client.post("/triage", json=sample)

    print("\nDone.")


if __name__ == "__main__":
    main()
