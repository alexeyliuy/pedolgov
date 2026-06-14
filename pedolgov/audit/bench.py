from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from pedolgov.broker.backend import ExtractionPlan, OfflineReasoner
from pedolgov.lanes.contracts import (
    Abstract,
    Contingency,
    KgNeighborhood,
    LabelSections,
    RawSources,
)
from pedolgov.manifest.config import BenchConfig, SignalConfig
from pedolgov.manifest.declaration import AgeGroup, Query, RiskTier
from pedolgov.valuation.assignment import as_thresholds, assign_tier
from pedolgov.valuation.factors import factor_vector
from pedolgov.valuation.scoring import composite_risk

FloatArray = npt.NDArray[np.float64]

TRUE_WEIGHTS: FloatArray = np.array(
    [0.22, 0.18, 0.14, 0.16, 0.08, 0.07, 0.08, 0.07], dtype=np.float64
)
TRUE_THRESHOLDS: FloatArray = np.array([0.20, 0.38, 0.58, 0.78], dtype=np.float64)

THERAPEUTIC_CLASSES = (
    "analgesics",
    "antibiotics",
    "cardiovascular",
    "anticonvulsants",
    "oncology",
    "psychiatric",
    "antihistamines",
    "corticosteroids",
    "antiemetics",
    "bronchodilators",
    "antivirals",
    "antifungals",
    "immunosuppressants",
    "gastrointestinal",
    "dermatologic",
)
_AGE_P = np.array([0.123, 0.247, 0.386, 0.244])
_CATEGORIES = ("indication", "age", "dose", "route")
_DESIGNS = ("rct", "observational", "case", "expert")
_MAX_INTERACTIONS = 40
_FLIP_PROB = 0.12
_GROUNDED_PLAN = ExtractionPlan(
    grounded=True, use_grading=True, use_kg=True, use_multi_source=True, signal=SignalConfig()
)


@dataclass(frozen=True)
class BenchItem:
    query: Query
    therapeutic_class: str
    sources: RawSources
    ground_truth: RiskTier


def _contingency(strength: float, rng: np.random.Generator) -> Contingency:
    rr = 1.0 + 6.0 * strength
    total_drug = 200
    total_other = 9800
    p0 = 0.01
    cc = int(round(p0 * total_other))
    a = int(round(min(1.0, p0 * rr) * total_drug)) + int(rng.integers(0, 3))
    return Contingency(a=a, b=total_drug - a, c=cc, d=total_other - cc)


def _sources(idx: int, drug: str, indication: str, rng: np.random.Generator) -> RawSources:
    strength = float(rng.beta(2.0, 2.0))
    has_contra = bool(rng.random() < 0.25)
    label_gap = bool(rng.random() < 0.4)
    pediatric_use = (
        "insufficient data in pediatric patients" if label_gap else "established in children"
    )
    label = LabelSections(
        drug=drug,
        indications=(indication,),
        contraindications=("hypersensitivity",) if has_contra else (),
        pediatric_use=pediatric_use,
        dose_min=0.5,
        dose_max=5.0,
        off_label_category=str(rng.choice(_CATEGORIES, p=[0.4, 0.25, 0.2, 0.15])),
    )
    n_abs = int(rng.integers(0, 6))
    abstracts = tuple(
        Abstract(
            design=str(rng.choice(_DESIGNS, p=[0.2, 0.35, 0.3, 0.15])),
            pediatric=bool(rng.random() < 0.5),
            pmid=f"{idx:05d}{j}",
        )
        for j in range(n_abs)
    )
    kg = KgNeighborhood(
        interaction_count=int(rng.integers(0, _MAX_INTERACTIONS)),
        max_interaction_count=_MAX_INTERACTIONS,
        narrow_therapeutic_index=bool(rng.random() < 0.3),
        on_kids_list=bool(rng.random() < 0.3),
        contraindicated_with_context=has_contra,
    )
    return RawSources(
        label=label, contingency=_contingency(strength, rng), abstracts=abstracts, kg=kg
    )


def synthesize_corpus(cfg: BenchConfig, seed: int) -> list[BenchItem]:
    rng = np.random.default_rng(seed)
    reasoner = OfflineReasoner()
    thresholds = as_thresholds(
        (
            float(TRUE_THRESHOLDS[0]),
            float(TRUE_THRESHOLDS[1]),
            float(TRUE_THRESHOLDS[2]),
            float(TRUE_THRESHOLDS[3]),
        )
    )
    items: list[BenchItem] = []
    for idx in range(cfg.n_items):
        klass = THERAPEUTIC_CLASSES[idx % len(THERAPEUTIC_CLASSES)]
        drug = f"{klass}_{idx % cfg.n_drugs}"
        indication = f"indication_{idx % cfg.n_indications}"
        age = AgeGroup(int(rng.choice(4, p=_AGE_P)))
        query = Query(drug=drug, indication=indication, age_group=age)
        sources = _sources(idx, drug, indication, rng)
        summary = reasoner.reason(sources, query, _GROUNDED_PLAN)
        risk = composite_risk(factor_vector(summary), TRUE_WEIGHTS)
        tier = int(assign_tier(risk, thresholds))
        if rng.random() < _FLIP_PROB:
            tier = int(np.clip(tier + rng.choice([-1, 1]), 0, 4))
        items.append(
            BenchItem(
                query=query, therapeutic_class=klass, sources=sources, ground_truth=RiskTier(tier)
            )
        )
    return items


@dataclass
class PedOLBench:
    items: list[BenchItem]
    splits: dict[str, list[int]]

    @classmethod
    def build(cls, cfg: BenchConfig, seed: int) -> PedOLBench:
        items = synthesize_corpus(cfg, seed)
        rng = np.random.default_rng(seed + 1)
        order = rng.permutation(len(items))
        n_train = int(cfg.train_frac * len(items))
        n_cal = int(cfg.calibration_frac * len(items))
        splits = {
            "train": list(order[:n_train]),
            "calibration": list(order[n_train : n_train + n_cal]),
            "test": list(order[n_train + n_cal :]),
        }
        return cls(items=items, splits=splits)

    def subset(self, name: str) -> list[BenchItem]:
        return [self.items[i] for i in self.splits[name]]
