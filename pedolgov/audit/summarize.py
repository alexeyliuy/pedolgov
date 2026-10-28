from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pedolgov.audit import metrics


def summarize(paths: list[Path]) -> dict:
    if not paths:
        raise ValueError("at least one query file is required")
    runs = []
    for path in paths:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if not rows:
            raise ValueError(f"empty query file: {path}")
        truth = np.array([r["ground_truth"] for r in rows], dtype=np.int64)
        pred = np.array([r["predicted_tier"] for r in rows], dtype=np.int64)
        runs.append(
            {
                "file": str(path),
                "n": len(rows),
                "ordinal_accuracy": float(np.mean(truth == pred)),
                "macro_f1": float(metrics.macro_f1(truth, pred)),
                "coverage": float(
                    np.mean([r["ground_truth"] in r["prediction_set"] for r in rows])
                ),
                "set_size": float(np.mean([len(r["prediction_set"]) for r in rows])),
                "human_review_rate": float(np.mean([r["human_review"] for r in rows])),
            }
        )
    keys = ("ordinal_accuracy", "macro_f1", "coverage", "set_size", "human_review_rate")
    return {
        "runs": runs,
        "aggregate": {
            k: {
                "mean": float(np.mean([r[k] for r in runs])),
                "std": float(np.std([r[k] for r in runs], ddof=1)) if len(runs) > 1 else 0.0,
            }
            for k in keys
        },
    }
