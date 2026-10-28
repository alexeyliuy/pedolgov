from __future__ import annotations

from dataclasses import dataclass, field, replace

from pedolgov.broker.backend import Backbone, ExtractionPlan
from pedolgov.lanes.contracts import EvidenceSummary, RawSources
from pedolgov.manifest.config import Ablation, Config
from pedolgov.manifest.declaration import Query


@dataclass
class EvidenceSynthesisEngine:
    config: Config
    backbone: Backbone
    retriever: object | None = field(default=None, repr=False)

    def synthesize(self, sources: RawSources, query: Query, ablation: Ablation) -> EvidenceSummary:
        if ablation.use_rag:
            from pedolgov.lanes.retrieval import ContrieverRetriever

            if self.retriever is None:
                self.retriever = ContrieverRetriever(self.config.retrieval)
            label = replace(
                sources.label,
                passages=self.retriever.retrieve(query.describe(), sources.label.passages),
            )
            sources = replace(sources, label=label)
        plan = ExtractionPlan(
            grounded=ablation.use_rag,
            use_grading=ablation.use_grading,
            use_kg=ablation.use_kg,
            use_multi_source=ablation.use_multi_source,
            signal=self.config.signal,
            temperature=self.config.inference.temperature_extract,
        )
        return self.backbone.reason(sources, query, plan)

    def narrative(self, summary: EvidenceSummary) -> str:
        return self.backbone.summarize(summary, self.config.inference.temperature_summary)
