from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from pedolgov.bond.conformal import SplitConformal
from pedolgov.clearance.governance import Decision, GovernanceModule
from pedolgov.inspection.hard import HardGuardrails
from pedolgov.inspection.soft import SoftGuardrails
from pedolgov.lanes.contracts import EvidenceSummary, RawSources
from pedolgov.lanes.fusion import EvidenceSynthesisEngine
from pedolgov.manifest.config import Config
from pedolgov.manifest.declaration import Query, RiskTier
from pedolgov.records.ledger import Ledger
from pedolgov.valuation.assignment import assign_tier
from pedolgov.valuation.factors import factor_vector
from pedolgov.valuation.scoring import RiskScorer

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class ClearanceResult:
    summary: EvidenceSummary
    probabilities: FloatArray
    risk: float
    tier: RiskTier
    decision: Decision


@dataclass
class GovernedPipeline:
    config: Config
    engine: EvidenceSynthesisEngine
    scorer: RiskScorer
    conformal: SplitConformal
    hard: HardGuardrails
    soft: SoftGuardrails
    governance: GovernanceModule
    ledger: Ledger

    def clear(self, sources: RawSources, query: Query) -> ClearanceResult:
        ablation = self.config.ablation
        summary = self.engine.synthesize(sources, query, ablation)
        features = factor_vector(summary)
        probs = self.scorer.probabilities(features)
        risk = self.scorer.risk(features)
        tier = assign_tier(risk, self.scorer.thresholds)

        pred_set = self.conformal.predict_set(probs) if ablation.use_conformal else {int(tier)}

        if ablation.use_governance:
            hard_flags = self.hard.evaluate(summary, query)
            soft_flags = self.soft.evaluate(summary, pred_set, self.conformal, probs)
            decision = self.governance.govern(
                tier, pred_set, hard_flags, soft_flags, query.age_group
            )
        else:
            decision = Decision(
                tier=tier,
                prediction_set=tuple(sorted(pred_set)),
                hard_flags=(),
                soft_flags=(),
                human_review=False,
                recommendation="release",
                rationale="governance disabled",
            )

        self.ledger.record(
            query.query_id,
            "clearance",
            {
                "tier": tier.label,
                "recommendation": decision.recommendation,
                "prediction_set": sorted(pred_set),
                "citations": list(summary.citations),
                "human_review": decision.human_review,
            },
        )
        return ClearanceResult(
            summary=summary, probabilities=probs, risk=risk, tier=tier, decision=decision
        )
