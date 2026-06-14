from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Protocol

from pedolgov.classification.hierarchy import (
    grade_label_statement,
    grade_pharmacovigilance,
)
from pedolgov.lanes.contracts import (
    Abstract,
    EvidenceItem,
    EvidenceSummary,
    RawSources,
)
from pedolgov.lanes.knowledge_lane import parse_knowledge
from pedolgov.lanes.label_lane import parse_label
from pedolgov.lanes.literature_lane import grade_literature, literature_levels
from pedolgov.lanes.signal_lane import consensus_signal
from pedolgov.manifest.config import SignalConfig
from pedolgov.manifest.declaration import Query


@dataclass(frozen=True)
class ExtractionPlan:
    grounded: bool
    use_grading: bool
    use_kg: bool
    use_multi_source: bool
    signal: SignalConfig


class Backbone(Protocol):
    name: str

    def reason(
        self, sources: RawSources, query: Query, plan: ExtractionPlan
    ) -> EvidenceSummary: ...

    def summarize(self, summary: EvidenceSummary, temperature: float) -> str: ...


def _unit_hash(token: str) -> float:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "little") / 0xFFFFFFFF


class OfflineReasoner:
    name = "offline"

    def reason(self, sources: RawSources, query: Query, plan: ExtractionPlan) -> EvidenceSummary:
        label = parse_label(sources.label)
        verdict = consensus_signal(sources.contingency, plan.signal)
        kg = parse_knowledge(sources.kg, plan.use_kg)
        abstracts = sources.abstracts if plan.use_multi_source else ()

        if plan.grounded:
            consensus = verdict.consensus_fraction
            level_a, level_b = literature_levels(abstracts, plan.use_grading)
            hallucinated = 0
        else:
            consensus = 0.2 + 0.6 * _unit_hash(query.query_id)
            level_a, level_b = 0, 0
            hallucinated = 2 + (1 if sources.contingency.a < 5 else 0)

        items = self._items(
            sources, query, verdict.is_signal, abstracts, label.has_contraindication
        )
        source_risk = {
            "label": 0.7 if label.label_gap else (0.5 if label.has_contraindication else 0.2),
            "faers": consensus,
            "literature": 1.0 / (1.0 + level_a + 0.5 * level_b),
        }
        return EvidenceSummary(
            query_id=query.query_id,
            drug=label.drug,
            age_group=query.age_group,
            disproportionality_consensus=consensus,
            off_label_category=label.off_label_category,
            level_a_count=level_a,
            level_b_count=level_b,
            interaction_count=kg.interaction_count,
            max_interaction_count=kg.max_interaction_count,
            narrow_ti=kg.narrow_ti,
            on_kids_list=kg.on_kids_list,
            label_gap=label.label_gap,
            has_contraindication=label.has_contraindication,
            dose_min=label.dose_min,
            dose_max=label.dose_max,
            source_risk=source_risk,
            items=items,
            grounded=plan.grounded,
            hallucinated_claims=hallucinated,
        )

    def summarize(self, summary: EvidenceSummary, temperature: float) -> str:
        head = f"{summary.drug}: consensus={summary.disproportionality_consensus:.2f}"
        tail = f"levelA={summary.level_a_count} levelB={summary.level_b_count}"
        return f"{head} {tail} grounded={summary.grounded}"

    def _items(
        self,
        sources: RawSources,
        query: Query,
        consensus_positive: bool,
        abstracts: tuple[Abstract, ...],
        has_contraindication: bool,
    ) -> tuple[EvidenceItem, ...]:
        items: list[EvidenceItem] = []
        items.append(
            EvidenceItem(
                source="DailyMed",
                grade=grade_label_statement(
                    has_contraindication or len(sources.label.pediatric_use) > 0
                ),
                statement=f"pediatric use: {sources.label.pediatric_use[:64]}",
                citation=f"DailyMed:{sources.label.drug}",
            )
        )
        items.append(
            EvidenceItem(
                source="FAERS",
                grade=grade_pharmacovigilance(consensus_positive),
                statement="disproportionality consensus over PRR/ROR/IC/EBGM",
                citation=f"FAERS:{query.query_id}",
            )
        )
        for abstract, grade in zip(abstracts, grade_literature(list(abstracts)), strict=False):
            items.append(
                EvidenceItem(
                    source="PubMed",
                    grade=grade,
                    statement=f"{abstract.design} study (pediatric={abstract.pediatric})",
                    citation=f"PubMed:{abstract.pmid}",
                )
            )
        return tuple(items)


class VllmBackbone:
    name = "vllm"

    def __init__(self, model: str, max_input_tokens: int, max_output_tokens: int) -> None:
        try:
            import vllm
        except ImportError as exc:
            raise RuntimeError("vllm is not installed; install the 'serve' extra") from exc
        self._llm = vllm.LLM(model=model)
        self.max_input_tokens = max_input_tokens
        self.max_output_tokens = max_output_tokens

    def reason(self, sources: RawSources, query: Query, plan: ExtractionPlan) -> EvidenceSummary:
        raise NotImplementedError(
            "structured decoding adapter for vllm is configured per deployment"
        )

    def summarize(self, summary: EvidenceSummary, temperature: float) -> str:
        raise NotImplementedError(
            "structured decoding adapter for vllm is configured per deployment"
        )


class ApiBackbone:
    name = "api"

    def __init__(self, provider: str, model: str) -> None:
        if provider not in {"openai", "anthropic"}:
            raise ValueError(f"unknown provider {provider!r}")
        try:
            __import__(provider)
        except ImportError as exc:
            raise RuntimeError(f"{provider} is not installed; install the 'api' extra") from exc
        self.provider = provider
        self.model = model

    def reason(self, sources: RawSources, query: Query, plan: ExtractionPlan) -> EvidenceSummary:
        raise NotImplementedError("hosted-API adapter is configured per deployment")

    def summarize(self, summary: EvidenceSummary, temperature: float) -> str:
        raise NotImplementedError("hosted-API adapter is configured per deployment")


def build_backbone(
    name: str, max_input_tokens: int = 4096, max_output_tokens: int = 2048
) -> Backbone:
    if name == "offline":
        return OfflineReasoner()
    if name in {"llama_70b", "qwen2_72b", "mixtral_8x7b", "vllm"}:
        return VllmBackbone(name, max_input_tokens, max_output_tokens)
    if name.startswith("api:"):
        _, provider, model = name.split(":", 2)
        return ApiBackbone(provider, model)
    raise ValueError(f"unknown backbone {name!r}")
