from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from cyclopts import App

from pedolgov.audit.bench import PedOLBench
from pedolgov.audit.evaluate import (
    evaluate_conformal,
    evaluate_end_to_end,
    evaluate_governance,
    evaluate_risk,
    fit_pipeline,
)
from pedolgov.clearance.pipeline import GovernedPipeline
from pedolgov.manifest.config import Config, load_config
from pedolgov.records.logging import get_logger
from pedolgov.records.seed import set_seed

app = App(name="pedolgov", help="Pediatric off-label prescribing governance")
_log = get_logger("gateway")


def _prepare(config: str, overrides: list[str] | None) -> tuple[Config, PedOLBench]:
    cfg = load_config(config, overrides)
    set_seed(cfg.seeds[0])
    bench = PedOLBench.build(cfg.bench, cfg.seeds[0])
    return cfg, bench


def _fit(config: str, overrides: list[str] | None) -> tuple[Config, PedOLBench, GovernedPipeline]:
    cfg, bench = _prepare(config, overrides)
    pipeline = fit_pipeline(bench, cfg)
    return cfg, bench, pipeline


@app.command
def bench(config: str, overrides: list[str] | None = None) -> None:
    cfg, data = _prepare(config, overrides)
    tiers = np.bincount([int(i.ground_truth) for i in data.items], minlength=5)
    _log.info("bench items=%d distribution=%s", len(data.items), tiers.tolist())


@app.command
def synthesize(config: str, index: int = 0, overrides: list[str] | None = None) -> None:
    cfg, data, pipeline = _fit(config, overrides)
    item = data.subset("test")[index]
    result = pipeline.clear(item.sources, item.query)
    _log.info("query=%s", item.query.describe())
    _log.info("evidence=%s", pipeline.engine.narrative(result.summary))


@app.command
def stratify(config: str, overrides: list[str] | None = None) -> None:
    cfg, data, pipeline = _fit(config, overrides)
    _log.info("weights=%s", np.round(pipeline.scorer.weights, 3).tolist())
    _log.info("thresholds=%s", np.round(pipeline.scorer.thresholds, 3).tolist())
    scores = evaluate_risk(pipeline, data)
    _log.info("risk=%s", {k: round(v, 3) for k, v in scores.items()})


@app.command
def calibrate(config: str, overrides: list[str] | None = None) -> None:
    cfg, data, pipeline = _fit(config, overrides)
    scores = evaluate_conformal(pipeline, data)
    _log.info("conformal=%s", {k: round(v, 3) for k, v in scores.items()})


@app.command
def clear(config: str, index: int = 0, overrides: list[str] | None = None) -> None:
    cfg, data, pipeline = _fit(config, overrides)
    item = data.subset("test")[index]
    result = pipeline.clear(item.sources, item.query)
    _log.info("query=%s", item.query.describe())
    _log.info(
        "tier=%s recommendation=%s human_review=%s set=%s",
        result.tier.label,
        result.decision.recommendation,
        result.decision.human_review,
        result.decision.prediction_set,
    )


@app.command
def audit(config: str, overrides: list[str] | None = None) -> None:
    cfg, data, pipeline = _fit(config, overrides)
    _log.info("risk=%s", {k: round(v, 3) for k, v in evaluate_risk(pipeline, data).items()})
    _log.info(
        "governance=%s", {k: round(v, 3) for k, v in evaluate_governance(pipeline, data).items()}
    )
    _log.info(
        "end_to_end=%s", {k: round(v, 3) for k, v in evaluate_end_to_end(pipeline, data).items()}
    )


@app.command
def reanalyze(config: str, output: str = "results", overrides: list[str] | None = None) -> None:
    from pedolgov.audit.reanalysis import reanalyze as run_reanalysis

    run_reanalysis(load_config(config, overrides), Path(output))
    _log.info("Reanalysis complete; input benchmark: %s", load_config(config, overrides).bench.path)


@app.command
def summarize(inputs: list[str], output: str = "summary.json") -> None:
    from pedolgov.audit.summarize import summarize as summarize_runs

    Path(output).write_text(json.dumps(summarize_runs([Path(p) for p in inputs]), indent=2) + "\n")


@app.command
def retrieve(drug: str, indication: str, output: str = "sources") -> None:
    from pedolgov.lanes.sources import (
        dailymed_search,
        dailymed_spl,
        pubmed_abstracts,
        pubmed_search,
    )

    target = Path(output)
    target.mkdir(parents=True, exist_ok=True)
    labels = dailymed_search(drug)
    (target / "dailymed_search.json").write_text(json.dumps(labels, indent=2) + "\n")
    for item in labels.get("data", []):
        identifier = item["setid"]
        (target / f"dailymed_{identifier}.xml").write_bytes(dailymed_spl(identifier))
    pmids = pubmed_search(drug, indication)
    if pmids:
        (target / "pubmed.xml").write_bytes(pubmed_abstracts(pmids))
    _log.info("Retrieved %d labels and %d PubMed records", len(labels.get("data", [])), len(pmids))


@app.command
def faers_table(records: str, synonyms: str, drug: str, event: str, output: str) -> None:
    from dataclasses import asdict

    from pedolgov.lanes.faers import contingency_table, pediatric_reports

    content = json.loads(Path(records).read_text())
    rows = content["results"] if isinstance(content, dict) else content
    mappings = json.loads(Path(synonyms).read_text())
    table = contingency_table(pediatric_reports(rows, mappings), drug, event)
    Path(output).write_text(json.dumps(asdict(table), indent=2) + "\n")


@app.command
def fit_signal_prior(counts: str, output: str) -> None:
    from pedolgov.lanes.signal_lane import fit_gamma_mixture

    rows = json.loads(Path(counts).read_text())
    observed = np.asarray([row["observed"] for row in rows], dtype=float)
    expected = np.asarray([row["expected"] for row in rows], dtype=float)
    prior = fit_gamma_mixture(observed, expected)
    Path(output).write_text(json.dumps({"eb_prior": prior}, indent=2) + "\n")


@app.command
def prepare_sources(manifest: str, output: str) -> None:
    from pedolgov.lanes.sources import dailymed_sections, parse_pubmed

    spec_path = Path(manifest)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    base = spec_path.parent
    sections = dailymed_sections((base / spec["label_xml"]).read_bytes())
    passages = [text for group in sections.values() for text in group]
    if not passages:
        raise ValueError("the label contains no target sections")
    label = {
        "drug": spec["drug"],
        "indications": sections["indications"],
        "contraindications": sections["contraindications"],
        "pediatric_use": "\n".join(sections["pediatric_use"]),
        "passages": passages,
        "dose_min": spec["dose_min"],
        "dose_max": spec["dose_max"],
        "off_label_category": spec["off_label_category"],
        "set_id": spec["set_id"],
    }
    result = {
        "label": label,
        "abstracts": parse_pubmed((base / spec["pubmed_xml"]).read_bytes()),
        "contingency": json.loads((base / spec["contingency_json"]).read_text()),
        "kg": json.loads((base / spec["kg_json"]).read_text()),
    }
    Path(output).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


@app.command
def prepare_knowledge(drugbank: str, primekg: str, curation: str, drug: str, output: str) -> None:
    from pedolgov.lanes.knowledge_import import knowledge_neighborhood

    curated = json.loads(Path(curation).read_text())
    result = knowledge_neighborhood(
        Path(drugbank),
        Path(primekg),
        drug,
        curated["context"],
        curated["kids"],
        curated["narrow_index"],
    )
    Path(output).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
