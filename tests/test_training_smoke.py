from __future__ import annotations

import numpy as np

from pedolgov.audit.bench import PedOLBench
from pedolgov.audit.evaluate import extract_features, fit_pipeline
from pedolgov.broker.backend import OfflineReasoner
from pedolgov.lanes.fusion import EvidenceSynthesisEngine
from pedolgov.manifest.config import load_config
from pedolgov.valuation.assignment import as_thresholds
from pedolgov.valuation.fitting import fit_weights


def test_two_step_fit_reduces_loss() -> None:
    cfg = load_config("configs/experiment/_smoke.yaml")
    bench = PedOLBench.build(cfg.bench, cfg.seeds[0])
    engine = EvidenceSynthesisEngine(config=cfg, backbone=OfflineReasoner())
    features, labels = extract_features(bench.subset("train"), engine, cfg.ablation)
    thresholds = as_thresholds(cfg.risk.thresholds)
    _, _, history = fit_weights(features, labels, thresholds, cfg.fit, cfg.risk.weight_cap)
    assert len(history) == cfg.fit.steps
    assert history[-1] < history[0]


def test_clear_two_queries_end_to_end() -> None:
    cfg = load_config("configs/experiment/_smoke.yaml")
    bench = PedOLBench.build(cfg.bench, cfg.seeds[0])
    pipeline = fit_pipeline(bench, cfg)
    for item in bench.subset("test")[:2]:
        result = pipeline.clear(item.sources, item.query)
        assert result.decision.recommendation in {"release", "warn", "defer"}
        assert 0 <= int(result.tier) <= 4
        assert np.isclose(result.probabilities.sum(), 1.0, atol=1e-6)
