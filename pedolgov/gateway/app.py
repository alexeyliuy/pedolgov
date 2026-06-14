from __future__ import annotations

import numpy as np
from cyclopts import App

from pedolgov.audit.bench import PedOLBench
from pedolgov.audit.evaluate import (
    ablation_table,
    evaluate_conformal,
    evaluate_end_to_end,
    evaluate_governance,
    evaluate_risk,
    fit_pipeline,
)
from pedolgov.clearance.pipeline import GovernedPipeline
from pedolgov.manifest.config import Config, load_config
from pedolgov.records.logging import get_logger
from pedolgov.records.seed import set_seed

app = App(name="pedolgov", help="Pediatric off-label prescribing governance")
_log = get_logger("gateway")


def _prepare(config: str, overrides: list[str] | None) -> tuple[Config, PedOLBench]:
    cfg = load_config(config, overrides)
    set_seed(cfg.seeds[0])
    bench = PedOLBench.build(cfg.bench, cfg.seeds[0])
    return cfg, bench


def _fit(config: str, overrides: list[str] | None) -> tuple[Config, PedOLBench, GovernedPipeline]:
    cfg, bench = _prepare(config, overrides)
    pipeline = fit_pipeline(bench, cfg)
    return cfg, bench, pipeline


@app.command
def bench(config: str, overrides: list[str] | None = None) -> None:
    cfg, data = _prepare(config, overrides)
    tiers = np.bincount([int(i.ground_truth) for i in data.items], minlength=5)
    _log.info("bench items=%d distribution=%s", len(data.items), tiers.tolist())


@app.command
def synthesize(config: str, index: int = 0, overrides: list[str] | None = None) -> None:
    cfg, data, pipeline = _fit(config, overrides)
    item = data.subset("test")[index]
    result = pipeline.clear(item.sources, item.query)
    _log.info("query=%s", item.query.describe())
    _log.info("evidence=%s", pipeline.engine.narrative(result.summary))


@app.command
def stratify(config: str, overrides: list[str] | None = None) -> None:
    cfg, data, pipeline = _fit(config, overrides)
    _log.info("weights=%s", np.round(pipeline.scorer.weights, 3).tolist())
    _log.info("thresholds=%s", np.round(pipeline.scorer.thresholds, 3).tolist())
    scores = evaluate_risk(pipeline, data)
    _log.info("risk=%s", {k: round(v, 3) for k, v in scores.items()})


@app.command
def calibrate(config: str, overrides: list[str] | None = None) -> None:
    cfg, data, pipeline = _fit(config, overrides)
    scores = evaluate_conformal(pipeline, data)
    _log.info("conformal=%s", {k: round(v, 3) for k, v in scores.items()})


@app.command
def clear(config: str, index: int = 0, overrides: list[str] | None = None) -> None:
    cfg, data, pipeline = _fit(config, overrides)
    item = data.subset("test")[index]
    result = pipeline.clear(item.sources, item.query)
    _log.info("query=%s", item.query.describe())
    _log.info(
        "tier=%s recommendation=%s human_review=%s set=%s",
        result.tier.label,
        result.decision.recommendation,
        result.decision.human_review,
        result.decision.prediction_set,
    )


@app.command
def audit(config: str, overrides: list[str] | None = None) -> None:
    cfg, data, pipeline = _fit(config, overrides)
    _log.info("risk=%s", {k: round(v, 3) for k, v in evaluate_risk(pipeline, data).items()})
    _log.info(
        "governance=%s", {k: round(v, 3) for k, v in evaluate_governance(pipeline, data).items()}
    )
    _log.info(
        "end_to_end=%s", {k: round(v, 3) for k, v in evaluate_end_to_end(pipeline, data).items()}
    )
    table = ablation_table(cfg, cfg.seeds[0])
    for name, row in table.items():
        _log.info("ablation %s=%s", name, {k: round(v, 3) for k, v in row.items()})
