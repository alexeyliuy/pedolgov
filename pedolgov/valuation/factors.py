from __future__ import annotations

import numpy as np
import numpy.typing as npt

from pedolgov.lanes.contracts import EvidenceSummary
from pedolgov.manifest.declaration import AgeGroup

FloatArray = npt.NDArray[np.float64]

N_FACTORS = 8

_CATEGORY_SEVERITY = {"indication": 1.0, "age": 0.75, "dose": 0.5, "route": 0.25}
_AGE_VULNERABILITY = {
    AgeGroup.NEONATE: 1.0,
    AgeGroup.INFANT: 0.7,
    AgeGroup.CHILD: 0.4,
    AgeGroup.ADOLESCENT: 0.2,
}
_EXPECTED_STRONG_EVIDENCE = 3.0


def f1_disproportionality(summary: EvidenceSummary) -> float:
    return float(np.clip(summary.disproportionality_consensus, 0.0, 1.0))


def f2_category_severity(summary: EvidenceSummary) -> float:
    return _CATEGORY_SEVERITY.get(summary.off_label_category, 0.5)


def f3_evidence_scarcity(summary: EvidenceSummary) -> float:
    strong = summary.level_a_count + 0.5 * summary.level_b_count
    return float(np.clip(1.0 - strong / _EXPECTED_STRONG_EVIDENCE, 0.0, 1.0))


def f4_age_vulnerability(age: AgeGroup) -> float:
    return _AGE_VULNERABILITY[age]


def f5_interaction_burden(summary: EvidenceSummary) -> float:
    if summary.max_interaction_count <= 0:
        return 0.0
    return float(np.clip(summary.interaction_count / summary.max_interaction_count, 0.0, 1.0))


def f6_therapeutic_index(summary: EvidenceSummary) -> float:
    return 1.0 if summary.narrow_ti else 0.2


def f7_kids_list(summary: EvidenceSummary) -> float:
    return 1.0 if summary.on_kids_list else 0.0


def f8_label_gap(summary: EvidenceSummary) -> float:
    return 1.0 if summary.label_gap else 0.0


def factor_vector(summary: EvidenceSummary) -> FloatArray:
    return np.array(
        [
            f1_disproportionality(summary),
            f2_category_severity(summary),
            f3_evidence_scarcity(summary),
            f4_age_vulnerability(summary.age_group),
            f5_interaction_burden(summary),
            f6_therapeutic_index(summary),
            f7_kids_list(summary),
            f8_label_gap(summary),
        ],
        dtype=np.float64,
    )
