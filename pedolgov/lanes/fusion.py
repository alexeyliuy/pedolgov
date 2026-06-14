from __future__ import annotations

from dataclasses import dataclass

from pedolgov.broker.backend import Backbone, ExtractionPlan
from pedolgov.lanes.contracts import EvidenceSummary, RawSources
from pedolgov.manifest.config import Ablation, Config
from pedolgov.manifest.declaration import Query


@dataclass
class EvidenceSynthesisEngine:
    config: Config
    backbone: Backbone

    def synthesize(self, sources: RawSources, query: Query, ablation: Ablation) -> EvidenceSummary:
        plan = ExtractionPlan(
            grounded=ablation.use_rag,
            use_grading=ablation.use_grading,
            use_kg=ablation.use_kg,
            use_multi_source=ablation.use_multi_source,
            signal=self.config.signal,
        )
        return self.backbone.reason(sources, query, plan)

    def narrative(self, summary: EvidenceSummary) -> str:
        return self.backbone.summarize(summary, self.config.inference.temperature_summary)
