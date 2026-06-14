from __future__ import annotations

import numpy as np
from conftest import make_summary

from pedolgov.bond.conformal import SplitConformal
from pedolgov.inspection.hard import HardGuardrails
from pedolgov.inspection.soft import SoftGuardrails
from pedolgov.manifest.declaration import AgeGroup, Query


def _query(drug: str = "analgesics_1") -> Query:
    return Query(drug=drug, indication="indication_0", age_group=AgeGroup.CHILD)


def test_hard_guardrails_fire_and_stay_silent() -> None:
    hard = HardGuardrails()
    clean = make_summary()
    assert hard.evaluate(clean, _query()) == []

    flagged = make_summary(
        drug="other_drug",
        has_contraindication=True,
        hallucinated_claims=1,
        dose_min=0.0,
        dose_max=0.0,
        narrow_ti=True,
        interaction_count=3,
    )
    triggered = set(hard.evaluate(flagged, _query()))
    assert triggered == {
        "drug_name",
        "contraindication",
        "citation_traceability",
        "dose_range",
        "interaction",
    }


def test_soft_guardrails() -> None:
    soft = SoftGuardrails(large_set_threshold=4)
    conformal = SplitConformal(alpha=0.1)
    conformal._calibration_scores = np.array([0.1, 0.2, 0.3])
    conformal.qhat = 0.5

    silent = soft.evaluate(
        make_summary(), {0, 1}, conformal, np.array([0.9, 0.04, 0.03, 0.02, 0.01])
    )
    assert silent == []

    disagree = make_summary(source_risk={"label": 0.9, "faers": 0.1})
    flags = soft.evaluate(disagree, {0, 1, 2, 3}, conformal, np.array([0.2, 0.2, 0.2, 0.2, 0.2]))
    assert "large_prediction_set" in flags
    assert "source_disagreement" in flags
