from __future__ import annotations

import json
from pathlib import Path

from pedolgov.audit.bench import PedOLBench
from pedolgov.audit.evaluate import fit_pipeline
from pedolgov.manifest.config import Config, to_builtins
from pedolgov.records.seed import set_seed


def reanalyze(config: Config, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for seed in config.seeds:
        set_seed(seed)
        bench = PedOLBench.build(config.bench, seed)
        pipeline = fit_pipeline(bench, config, seed)
        rows = []
        for index in bench.splits["test"]:
            item = bench.items[index]
            result = pipeline.clear(item.sources, item.query)
            rows.append(
                {
                    "index": int(index),
                    "query_id": item.query.query_id,
                    "drug": item.query.drug,
                    "indication": item.query.indication,
                    "age_group": item.query.age_group.label,
                    "therapeutic_class": item.therapeutic_class,
                    "ground_truth": int(item.ground_truth),
                    "predicted_tier": int(result.tier),
                    "risk": float(result.risk),
                    "probabilities": result.probabilities.tolist(),
                    "prediction_set": list(result.decision.prediction_set),
                    "recommendation": result.decision.recommendation,
                    "human_review": result.decision.human_review,
                    "hard_flags": list(result.decision.hard_flags),
                    "soft_flags": list(result.decision.soft_flags),
                    "evidence": [
                        {
                            "source": item.source,
                            "grade": item.grade.label,
                            "statement": item.statement,
                            "citation": item.citation,
                        }
                        for item in result.summary.items
                    ],
                }
            )
        record = {
            "dataset": config.bench.path,
            "seed": seed,
            "config": to_builtins(config),
            "splits": {k: [int(i) for i in v] for k, v in bench.splits.items()},
            "weights": pipeline.scorer.weights.tolist(),
            "thresholds": list(map(float, pipeline.scorer.thresholds)),
            "conformal_quantile": float(pipeline.conformal.qhat),
        }
        (output / f"seed_{seed}.record.json").write_text(
            json.dumps(record, indent=2, allow_nan=False) + "\n"
        )
        with (output / f"seed_{seed}.queries.jsonl").open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row, allow_nan=False) + "\n")
