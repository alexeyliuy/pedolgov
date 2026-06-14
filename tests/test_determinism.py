from __future__ import annotations

from pedolgov.audit.bench import PedOLBench
from pedolgov.audit.evaluate import evaluate_risk, fit_pipeline
from pedolgov.broker.backend import ExtractionPlan, OfflineReasoner
from pedolgov.manifest.config import SignalConfig, load_config

_PLAN = ExtractionPlan(
    grounded=True, use_grading=True, use_kg=True, use_multi_source=True, signal=SignalConfig()
)


def test_offline_reasoner_is_deterministic() -> None:
    cfg = load_config("configs/experiment/_smoke.yaml")
    bench = PedOLBench.build(cfg.bench, cfg.seeds[0])
    item = bench.items[0]
    reasoner = OfflineReasoner()
    first = reasoner.reason(item.sources, item.query, _PLAN)
    second = reasoner.reason(item.sources, item.query, _PLAN)
    assert first == second


def test_pipeline_metrics_are_reproducible() -> None:
    cfg = load_config("configs/experiment/_smoke.yaml")
    a = evaluate_risk(
        fit_pipeline(PedOLBench.build(cfg.bench, cfg.seeds[0]), cfg),
        PedOLBench.build(cfg.bench, cfg.seeds[0]),
    )
    b = evaluate_risk(
        fit_pipeline(PedOLBench.build(cfg.bench, cfg.seeds[0]), cfg),
        PedOLBench.build(cfg.bench, cfg.seeds[0]),
    )
    assert a == b
