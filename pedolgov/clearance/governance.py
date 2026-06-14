from __future__ import annotations

from dataclasses import dataclass

from pedolgov.manifest.config import GovernanceConfig
from pedolgov.manifest.declaration import AgeGroup, RiskTier


@dataclass(frozen=True)
class Decision:
    tier: RiskTier
    prediction_set: tuple[int, ...]
    hard_flags: tuple[str, ...]
    soft_flags: tuple[str, ...]
    human_review: bool
    recommendation: str
    rationale: str


def requires_human_review(
    tier: RiskTier,
    prediction_set: set[int],
    hard_flags: list[str],
    age: AgeGroup,
    set_size_target: int,
    cfg: GovernanceConfig,
) -> bool:
    if cfg.hitl_on_critical and tier == RiskTier.CRITICAL:
        return True
    if cfg.hitl_on_large_set and len(prediction_set) > set_size_target:
        return True
    if cfg.hitl_on_hard_guardrail and hard_flags:
        return True
    return bool(cfg.hitl_on_neonate and age == AgeGroup.NEONATE)


class GovernanceModule:
    def __init__(self, cfg: GovernanceConfig, set_size_target: int) -> None:
        self.cfg = cfg
        self.set_size_target = set_size_target

    def govern(
        self,
        tier: RiskTier,
        prediction_set: set[int],
        hard_flags: list[str],
        soft_flags: list[str],
        age: AgeGroup,
    ) -> Decision:
        review = requires_human_review(
            tier, prediction_set, hard_flags, age, self.set_size_target, self.cfg
        )
        if review:
            recommendation = "defer"
            rationale = "mandatory clinical review"
        elif soft_flags or tier >= RiskTier.HIGH:
            recommendation = "warn"
            rationale = "elevated risk or residual uncertainty"
        else:
            recommendation = "release"
            rationale = "calibrated low-risk recommendation"
        return Decision(
            tier=tier,
            prediction_set=tuple(sorted(prediction_set)),
            hard_flags=tuple(hard_flags),
            soft_flags=tuple(soft_flags),
            human_review=review,
            recommendation=recommendation,
            rationale=rationale,
        )
