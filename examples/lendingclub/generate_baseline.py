#!/usr/bin/env python3
"""
Generate baseline.json from a CSV file.

Usage:
    python generate_baseline.py data/lendingclub_2015_sample.csv.gz

Output:
    baseline.json with reference distributions for drift detection.
"""

import gzip
import json
import sys
from pathlib import Path


def load_csv(path: str | Path) -> dict[str, list[float]]:
    """Load CSV into dict of column -> values."""
    path = Path(path)

    opener = gzip.open if path.suffix == ".gz" else open

    with opener(path, "rt") as f:
        header = f.readline().strip().split(",")
        data = {col: [] for col in header}

        for line in f:
            values = line.strip().split(",")
            for col, val in zip(header, values):
                try:
                    data[col].append(float(val))
                except ValueError:
                    pass

    return data


def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_baseline.py <input.csv[.gz]> [output.json]")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("baseline.json")

    print(f"Loading {input_path}...")
    data = load_csv(input_path)

    n_samples = len(next(iter(data.values())))

    baseline = {
        "description": f"Reference distribution from {input_path.name}",
        "source": str(input_path),
        "n_samples": n_samples,
        "distributions": data,
    }

    with open(output_path, "w") as f:
        json.dump(baseline, f, indent=2)

    print(f"Written {output_path} ({n_samples} samples, {len(data)} features)")


if __name__ == "__main__":
    main()
