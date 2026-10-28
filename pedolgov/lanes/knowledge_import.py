from __future__ import annotations

import csv
import xml.etree.ElementTree as ET
from pathlib import Path


def knowledge_neighborhood(
    drugbank_xml: Path,
    primekg_csv: Path,
    drug: str,
    context: list[str],
    kids: list[str],
    narrow_index: list[str],
) -> dict:
    target = drug.strip().casefold()
    if not target:
        raise ValueError("drug is required")
    max_count = 0
    matches = []
    namespace = {"d": "http://www.drugbank.ca"}
    for _, element in ET.iterparse(drugbank_xml, events=("end",)):
        if element.tag != "{http://www.drugbank.ca}drug":
            continue
        name = element.findtext("d:name", default="", namespaces=namespace)
        if not name:
            element.clear()
            continue
        synonyms = [s.text or "" for s in element.findall("./d:synonyms/d:synonym", namespace)]
        interactions = [
            x.findtext("d:name", default="", namespaces=namespace)
            for x in element.findall("./d:drug-interactions/d:drug-interaction", namespace)
        ]
        max_count = max(max_count, len(interactions))
        if target in {x.strip().casefold() for x in [name, *synonyms]}:
            matches.append((name, interactions))
        element.clear()
    if len(matches) != 1:
        raise ValueError(f"DrugBank contains {len(matches)} matching entries for this drug")
    name, interactions = matches[0]
    with primekg_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"x_type", "x_name", "relation", "y_type", "y_name"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("PrimeKG CSV must contain x_type, x_name, relation, y_type and y_name")
        triples = []
        for row in reader:
            if (
                row["x_type"].casefold() == "drug" and row["x_name"].strip().casefold() == target
            ) or (
                row["y_type"].casefold() == "drug" and row["y_name"].strip().casefold() == target
            ):
                triples.append(f"{row['x_name']} | {row['relation']} | {row['y_name']}")
    context_names = {name.strip().casefold() for name in context}
    return {
        "interaction_count": len(interactions),
        "max_interaction_count": max_count,
        "narrow_therapeutic_index": target in {x.strip().casefold() for x in narrow_index},
        "on_kids_list": target in {x.strip().casefold() for x in kids},
        "contraindicated_with_context": bool(context_names & {x.casefold() for x in interactions}),
        "triples": triples,
        "interactions": interactions,
        "drugbank_name": name,
    }
