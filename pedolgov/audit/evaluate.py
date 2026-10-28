from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from pedolgov.audit import metrics
from pedolgov.audit.bench import BenchItem, PedOLBench
from pedolgov.bond.conformal import (
    SplitConformal,
    average_set_size,
    empirical_coverage,
)
from pedolgov.broker.backend import build_backbone
from pedolgov.clearance.governance import GovernanceModule
from pedolgov.clearance.pipeline import GovernedPipeline
from pedolgov.inspection.hard import HardGuardrails
from pedolgov.inspection.soft import SoftGuardrails
from pedolgov.lanes.fusion import EvidenceSynthesisEngine
from pedolgov.manifest.config import Ablation, Config
from pedolgov.records.ledger import Ledger
from pedolgov.valuation.assignment import as_thresholds
from pedolgov.valuation.factors import factor_vector
from pedolgov.valuation.fitting import calibrate_thresholds, fit_weights
from pedolgov.valuation.scoring import RiskScorer, composite_risk

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


def extract_features(
    items: list[BenchItem], engine: EvidenceSynthesisEngine, ablation: Ablation
) -> tuple[FloatArray, IntArray]:
    rows = []
    labels = []
    for item in items:
        summary = engine.synthesize(item.sources, item.query, ablation)
        rows.append(factor_vector(summary))
        labels.append(int(item.ground_truth))
    return np.asarray(rows, dtype=np.float64), np.asarray(labels, dtype=np.int64)


def fit_pipeline(bench: PedOLBench, config: Config, seed: int | None = None) -> GovernedPipeline:
    backbone = build_backbone(
        config.inference.backbone,
        config.inference.max_input_tokens,
        config.inference.max_output_tokens,
        seed if seed is not None else config.seeds[0],
    )
    engine = EvidenceSynthesisEngine(config=config, backbone=backbone)
    train = bench.subset("train")
    f_train, y_train = extract_features(train, engine, config.ablation)
    thresholds0 = as_thresholds(config.risk.thresholds)
    weights, scale, _ = fit_weights(
        f_train, y_train, thresholds0, config.fit, config.risk.weight_cap
    )
    risk_train = np.array([composite_risk(row, weights) for row in f_train])
    thresholds = calibrate_thresholds(risk_train, y_train, config.fit.kappa_grid)
    scorer = RiskScorer(weights, thresholds, scale)

    calib = bench.subset("calibration")
    f_cal, y_cal = extract_features(calib, engine, config.ablation)
    probs_cal = np.array([scorer.probabilities(row) for row in f_cal])
    conformal = SplitConformal(
        alpha=config.conformal.alpha, ood_percentile=config.conformal.ood_percentile
    ).fit(probs_cal, y_cal)

    return GovernedPipeline(
        config=config,
        engine=engine,
        scorer=scorer,
        conformal=conformal,
        hard=HardGuardrails(),
        soft=SoftGuardrails(config.governance.large_set_threshold),
        governance=GovernanceModule(config.governance, config.conformal.set_size_target),
        ledger=Ledger(),
    )


@dataclass(frozen=True)
class TestRun:
    y_true: IntArray
    y_pred: IntArray
    risks: FloatArray
    probs: FloatArray
    sets: list[set[int]]


def run_test(pipeline: GovernedPipeline, items: list[BenchItem]) -> TestRun:
    y_true, y_pred, risks, probs, sets = [], [], [], [], []
    for item in items:
        result = pipeline.clear(item.sources, item.query)
        y_true.append(int(item.ground_truth))
        y_pred.append(int(result.tier))
        risks.append(result.risk)
        probs.append(result.probabilities)
        sets.append(set(result.decision.prediction_set))
    return TestRun(
        y_true=np.asarray(y_true, dtype=np.int64),
        y_pred=np.asarray(y_pred, dtype=np.int64),
        risks=np.asarray(risks, dtype=np.float64),
        probs=np.asarray(probs, dtype=np.float64),
        sets=sets,
    )


def evaluate_risk(pipeline: GovernedPipeline, bench: PedOLBench) -> dict[str, float]:
    run = run_test(pipeline, bench.subset("test"))
    binary = (run.y_true >= 3).astype(np.int64)
    return {
        "macro_f1": metrics.macro_f1(run.y_true, run.y_pred),
        "auroc": metrics.auroc(run.risks, binary),
        "risk_label_spearman_rho": metrics.spearman_rho(run.risks, run.y_true.astype(np.float64)),
        "ece": metrics.expected_calibration_error(run.probs, run.y_true),
        "coverage": empirical_coverage(run.sets, run.y_true),
        "set_size": average_set_size(run.sets),
        "kappa": metrics.weighted_kappa(run.y_true, run.y_pred),
        "ordinal_mae": metrics.ordinal_mae(run.y_true, run.y_pred),
        "accuracy": metrics.ordinal_accuracy(run.y_true, run.y_pred),
    }


def evaluate_conformal(pipeline: GovernedPipeline, bench: PedOLBench) -> dict[str, float]:
    run = run_test(pipeline, bench.subset("test"))
    return {
        "coverage": empirical_coverage(run.sets, run.y_true),
        "set_size": average_set_size(run.sets),
        "alpha": pipeline.config.conformal.alpha,
    }


def evaluate_governance(pipeline: GovernedPipeline, bench: PedOLBench) -> dict[str, float]:
    items = bench.subset("test")
    review = 0
    for item in items:
        result = pipeline.clear(item.sources, item.query)
        review += int(result.decision.human_review)
    n = max(1, len(items))
    return {
        "hitl_rate": review / n,
    }


def evaluate_end_to_end(pipeline: GovernedPipeline, bench: PedOLBench) -> dict[str, float]:
    items = bench.subset("test")
    correct = 0
    safe = 0
    covered = 0
    for item in items:
        result = pipeline.clear(item.sources, item.query)
        correct += int(int(result.tier) == int(item.ground_truth))
        safe += int(len(result.decision.hard_flags) == 0)
        covered += int(result.decision.recommendation != "defer")
    n = max(1, len(items))
    return {
        "risk_classification_accuracy": correct / n,
        "no_hard_flags_rate": safe / n,
        "response_rate": covered / n,
    }
