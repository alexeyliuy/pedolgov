from __future__ import annotations

from dataclasses import dataclass

from pedolgov.lanes.contracts import EvidenceItem
from pedolgov.manifest.declaration import Query

_SYSTEM = (
    "You are a pediatric drug-safety analyst assessing the risk of an off-label prescription. "
    "Ground every statement in the retrieved evidence and flag any item without a citation."
)


@dataclass(frozen=True)
class PromptParts:
    system: str
    context: str
    instruction: str
    directive: str

    def render(self) -> str:
        return "\n\n".join([self.system, self.context, self.instruction, self.directive])


def build_prompt(query: Query, items: tuple[EvidenceItem, ...], instruction: str) -> PromptParts:
    lines = [
        f"[{item.source} | {item.grade.label} | {item.citation}] {item.statement}" for item in items
    ]
    context = "Retrieved evidence:\n" + ("\n".join(lines) if lines else "none")
    directive = "Think step by step about the evidence before stating a calibrated risk category."
    return PromptParts(
        system=_SYSTEM,
        context=f"Query: {query.describe()}\n{context}",
        instruction=instruction,
        directive=directive,
    )
