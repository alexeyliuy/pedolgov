from __future__ import annotations

from dataclasses import dataclass

from pedolgov.lanes.contracts import KgNeighborhood


@dataclass(frozen=True)
class KnowledgeReadout:
    interaction_count: int
    max_interaction_count: int
    narrow_ti: bool
    on_kids_list: bool
    contraindicated_with_context: bool


def parse_knowledge(kg: KgNeighborhood, use_kg: bool) -> KnowledgeReadout:
    if not use_kg:
        return KnowledgeReadout(
            interaction_count=0,
            max_interaction_count=kg.max_interaction_count,
            narrow_ti=False,
            on_kids_list=kg.on_kids_list,
            contraindicated_with_context=False,
        )
    return KnowledgeReadout(
        interaction_count=kg.interaction_count,
        max_interaction_count=kg.max_interaction_count,
        narrow_ti=kg.narrow_therapeutic_index,
        on_kids_list=kg.on_kids_list,
        contraindicated_with_context=kg.contraindicated_with_context,
    )
