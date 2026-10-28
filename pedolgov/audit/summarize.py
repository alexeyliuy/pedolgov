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
        risk = np.array([r["risk"] for r in rows], dtype=np.float64)
        probs = np.array([r["probabilities"] for r in rows], dtype=np.float64)
        binary = (truth >= 3).astype(np.int64)
        if len(np.unique(binary)) < 2:
            raise ValueError(f"AUROC requires high-risk and lower-risk records: {path}")
        runs.append(
            {
                "file": str(path),
                "n": len(rows),
                "ordinal_accuracy": float(np.mean(truth == pred)),
                "macro_f1": float(metrics.macro_f1(truth, pred)),
                "risk_auroc": float(metrics.auroc(risk, binary)),
                "quadratic_kappa": float(metrics.weighted_kappa(truth, pred)),
                "ordinal_mae": float(metrics.ordinal_mae(truth, pred)),
                "ece": float(metrics.expected_calibration_error(probs, truth)),
                "coverage": float(
                    np.mean([r["ground_truth"] in r["prediction_set"] for r in rows])
                ),
                "set_size": float(np.mean([len(r["prediction_set"]) for r in rows])),
                "human_review_rate": float(np.mean([r["human_review"] for r in rows])),
                "subgroups": {
                    field: {
                        group: {
                            "n": len(group_rows),
                            "coverage": float(
                                np.mean(
                                    [r["ground_truth"] in r["prediction_set"] for r in group_rows]
                                )
                            ),
                            "set_size": float(
                                np.mean([len(r["prediction_set"]) for r in group_rows])
                            ),
                        }
                        for group in sorted({str(r[field]) for r in rows})
                        for group_rows in [[r for r in rows if str(r[field]) == group]]
                    }
                    for field in ("age_group", "therapeutic_class")
                },
            }
        )
    keys = (
        "ordinal_accuracy",
        "macro_f1",
        "risk_auroc",
        "quadratic_kappa",
        "ordinal_mae",
        "ece",
        "coverage",
        "set_size",
        "human_review_rate",
    )
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
