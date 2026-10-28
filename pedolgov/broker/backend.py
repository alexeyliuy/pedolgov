from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, replace
from typing import Protocol

from pedolgov.classification.hierarchy import (
    grade_label_statement,
    grade_pharmacovigilance,
)
from pedolgov.lanes.contracts import (
    Abstract,
    EvidenceGrade,
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
    temperature: float = 0.1


class Backbone(Protocol):
    name: str

    def reason(
        self, sources: RawSources, query: Query, plan: ExtractionPlan
    ) -> EvidenceSummary: ...

    def summarize(self, summary: EvidenceSummary, temperature: float) -> str: ...


class EvidenceProcessor:
    name = "evidence_processor"

    def reason(self, sources: RawSources, query: Query, plan: ExtractionPlan) -> EvidenceSummary:
        label = parse_label(sources.label)
        verdict = consensus_signal(sources.contingency, plan.signal)
        kg = parse_knowledge(sources.kg, plan.use_kg)
        abstracts = sources.abstracts if plan.use_multi_source else ()

        consensus = verdict.consensus_fraction
        level_a, level_b = literature_levels(abstracts, plan.use_grading)
        hallucinated = 0

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
                citation=f"DailyMed:{sources.label.set_id or sources.label.drug}",
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


class ModelBackbone:
    def _generate(self, system: str, prompt: str, temperature: float) -> str:
        raise NotImplementedError

    def reason(self, sources: RawSources, query: Query, plan: ExtractionPlan) -> EvidenceSummary:
        if not plan.grounded:
            raise ValueError("This backend requires source evidence")
        base = EvidenceProcessor().reason(sources, query, plan)
        context = {
            "query": {
                "drug": query.drug,
                "indication": query.indication,
                "age_group": query.age_group.label,
                "context": query.context,
            },
            "label": {
                "drug": sources.label.drug,
                "set_id": sources.label.set_id,
                "passages": sources.label.passages,
            },
            "faers_contingency": asdict(sources.contingency),
            "literature": [asdict(a) for a in sources.abstracts] if plan.use_multi_source else [],
            "knowledge": asdict(sources.kg) if plan.use_kg else None,
        }
        allowed = {
            f"DailyMed:{sources.label.set_id or sources.label.drug}",
            f"FAERS:{query.query_id}",
        }
        allowed.update(f"PubMed:{a.pmid}" for a in sources.abstracts)
        if plan.use_kg:
            if sources.kg.drugbank_name:
                allowed.add(f"DrugBank:{sources.kg.drugbank_name}")
            if sources.kg.triples:
                allowed.add("PrimeKG:" + query.drug)
        instruction = (
            "Return only a JSON object with an items array. Each item must contain "
            "source, grade (A, B, C, D or E), statement and citation. Use only the supplied "
            "evidence, never invent citations, and use citation identifiers exactly as supplied: "
            + ", ".join(sorted(allowed))
            + ". Evidence grading: A pediatric randomized trial; "
            "B observational study or consensus pharmacovigilance signal; C case report or "
            "regulatory label; D expert opinion or adult extrapolation; E no direct evidence. "
            "Context: " + json.dumps(context, ensure_ascii=False)
        )
        raw = self._generate(
            "You are a pediatric drug-safety analyst. Ground every claim in the supplied evidence.",
            instruction,
            plan.temperature,
        )
        try:
            payload = json.loads(raw.strip().removeprefix("```json").removesuffix("```").strip())
            entries = payload["items"]
            if not isinstance(entries, list) or not entries:
                raise ValueError("empty evidence items")
            items = tuple(
                EvidenceItem(
                    source=str(item["source"]),
                    grade=EvidenceGrade[str(item["grade"]).upper()],
                    statement=str(item["statement"]),
                    citation=str(item["citation"]),
                )
                for item in entries
            )
            if any(not item.statement.strip() or item.citation not in allowed for item in items):
                raise ValueError("missing statement or unrecognized citation")
        except (ValueError, TypeError, KeyError) as exc:
            raise ValueError("Model response is not valid grounded evidence JSON") from exc
        return replace(
            base,
            items=items,
            level_a_count=sum(item.grade == EvidenceGrade.A for item in items),
            level_b_count=sum(item.grade == EvidenceGrade.B for item in items),
        )

    def summarize(self, summary: EvidenceSummary, temperature: float) -> str:
        context = [
            {
                "source": i.source,
                "grade": i.grade.label,
                "statement": i.statement,
                "citation": i.citation,
            }
            for i in summary.items
        ]
        return self._generate(
            "You are a pediatric drug-safety analyst.",
            "Summarize only the supplied evidence, include citations, and state uncertainty: "
            + json.dumps(context, ensure_ascii=False),
            temperature,
        )


class VllmBackbone(ModelBackbone):
    name = "vllm"

    def __init__(self, model: str, max_input_tokens: int, max_output_tokens: int) -> None:
        try:
            import vllm
        except ImportError as exc:
            raise RuntimeError("vllm is not installed; install the 'serve' extra") from exc
        self._vllm = vllm
        self._llm = vllm.LLM(
            model=model,
            max_model_len=max_input_tokens + max_output_tokens,
            tensor_parallel_size=int(os.environ.get("PEDOLGOV_TENSOR_PARALLEL_SIZE", "1")),
        )
        self.max_output_tokens = max_output_tokens

    def _generate(self, system: str, prompt: str, temperature: float) -> str:
        params = self._vllm.SamplingParams(
            temperature=temperature, max_tokens=self.max_output_tokens
        )
        result = self._llm.chat(
            [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            sampling_params=params,
            use_tqdm=False,
        )
        return result[0].outputs[0].text


class ApiBackbone(ModelBackbone):
    name = "api"

    def __init__(self, provider: str, model: str, max_output_tokens: int = 2048) -> None:
        if provider not in {"openai", "anthropic"}:
            raise ValueError(f"unknown provider {provider!r}")
        try:
            module = __import__(provider)
        except ImportError as exc:
            raise RuntimeError(f"{provider} is not installed; install the 'api' extra") from exc
        self.provider = provider
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.client = module.OpenAI() if provider == "openai" else module.Anthropic()

    def _generate(self, system: str, prompt: str, temperature: float) -> str:
        if self.provider == "openai":
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=temperature,
                max_tokens=self.max_output_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
            )
            text = response.choices[0].message.content
            if not text:
                raise ValueError("model returned empty content")
            return text
        response = self.client.messages.create(
            model=self.model,
            system=system,
            temperature=temperature,
            max_tokens=self.max_output_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(block.text for block in response.content if block.type == "text")
        if not text:
            raise ValueError("model returned empty content")
        return text


def build_backbone(
    name: str, max_input_tokens: int = 4096, max_output_tokens: int = 2048
) -> Backbone:
    models = {
        "llama_70b": "meta-llama/Meta-Llama-3.1-70B-Instruct",
        "qwen2_72b": "Qwen/Qwen2-72B-Instruct",
        "mixtral_8x7b": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    }
    if name in models:
        return VllmBackbone(models[name], max_input_tokens, max_output_tokens)
    if name.startswith("vllm:"):
        return VllmBackbone(name[5:], max_input_tokens, max_output_tokens)
    if name.startswith("api:"):
        _, provider, model = name.split(":", 2)
        return ApiBackbone(provider, model, max_output_tokens)
    raise ValueError("select a supported vllm model or api:provider:model")
