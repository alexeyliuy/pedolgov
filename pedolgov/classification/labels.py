from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pedolgov.manifest.declaration import RiskTier


@dataclass(frozen=True)
class LabelEvidence:
    kids: Literal["avoid", "caution", "none"]
    pediatric_indication_for_age: bool
    emlc_listed: bool
    disproportionality_signal: bool


def classify_reference(evidence: LabelEvidence) -> tuple[RiskTier, str]:
    if evidence.kids not in {"avoid", "caution", "none"}:
        raise ValueError("kids must be avoid, caution or none")
    if any(
        type(value) is not bool
        for value in (
            evidence.pediatric_indication_for_age,
            evidence.emlc_listed,
            evidence.disproportionality_signal,
        )
    ):
        raise ValueError("all reference indicators must be explicitly known")
    if evidence.kids == "avoid":
        if evidence.pediatric_indication_for_age:
            return RiskTier.HIGH, "kids_avoid_with_indication"
        return RiskTier.CRITICAL, "kids_avoid_without_indication"
    if evidence.kids == "caution":
        return RiskTier.MODERATE, "kids_caution"
    if evidence.emlc_listed:
        if evidence.pediatric_indication_for_age:
            return RiskTier.MINIMAL, "emlc_on_label_age"
        return RiskTier.LOW, "emlc_off_label_age"
    if evidence.disproportionality_signal:
        return RiskTier.MODERATE, "disproportionality_signal"
    return RiskTier.LOW, "remaining"
