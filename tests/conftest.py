from __future__ import annotations

from typing import Any

from pedolgov.lanes.contracts import EvidenceGrade, EvidenceItem, EvidenceSummary
from pedolgov.manifest.declaration import AgeGroup


def make_summary(**overrides: Any) -> EvidenceSummary:
    base: dict[str, Any] = {
        "query_id": "abcd1234",
        "drug": "analgesics_1",
        "age_group": AgeGroup.CHILD,
        "disproportionality_consensus": 0.5,
        "off_label_category": "indication",
        "level_a_count": 1,
        "level_b_count": 1,
        "interaction_count": 4,
        "max_interaction_count": 40,
        "narrow_ti": False,
        "on_kids_list": False,
        "label_gap": False,
        "has_contraindication": False,
        "dose_min": 0.5,
        "dose_max": 5.0,
        "source_risk": {"label": 0.2, "faers": 0.5, "literature": 0.3},
        "items": (
            EvidenceItem("DailyMed", EvidenceGrade.C, "label", "DailyMed:analgesics_1"),
            EvidenceItem("FAERS", EvidenceGrade.B, "signal", "FAERS:abcd1234"),
        ),
        "grounded": True,
        "hallucinated_claims": 0,
    }
    base.update(overrides)
    return EvidenceSummary(**base)
