# Deviations

Each entry links a manuscript anchor and a justification.

## D1 — LLM backbones run through an offline reasoner by default

Methods §Model selection / §Inference configuration specify frozen 70B-class open-source backbones
served with vLLM on 4× A100 80GB. Those weights and that hardware are not part of a code release and
cannot run in CI. The backbone is therefore a port (`pedolgov.broker.backend.Backbone`) whose default
implementation (`OfflineReasoner`) is deterministic and seedable, reproducing the structured extraction,
grading, and risk-probability behaviour the pipeline consumes. Guarded adapters (`VllmBackbone`,
`ApiBackbone`) target the real backbones and raise a clear error when their optional dependency is
absent. All reported numbers in the manuscript come from the real backbones; the offline reasoner
exists so the framework is runnable and testable end to end.

## D2 — No distributed / mixed-precision training

R4 asks for DDP/FSDP and AMP readiness. This framework performs no neural-network training: the only
fitted objects are eight scalar risk-factor weights (ordinal cross-entropy) and four scalar thresholds
(weighted κ), plus the conformal quantile. These are solved on a single process with deterministic
numpy/torch and need no data or model parallelism. The `RiskHead` is a small differentiable module so a
gradient-flow test is still meaningful, but DDP/AMP would add machinery with no scientific effect.

## D3 — Retrieval encoder is a deterministic hash encoder by default

Methods §Drug label retrieval specify a Contriever dense encoder with a FAISS index. The default
`HashEncoder` is a deterministic feature-hashing projection so retrieval is reproducible offline; the
guarded `FaissIndex` and a Contriever adapter are selected when the `retrieval` extra is installed. The
512-token / 64-overlap chunking and top-k = 10 retrieval contract are identical across both paths.

## D5 — Evidence-extraction F1 is a recovery proxy on the synthetic benchmark

Table 1 reports macro-F1 over four extraction categories against manually annotated DailyMed sections.
The shipped benchmark is synthetic and has no human extraction annotation, so `evaluate_extraction`
measures a binary signal-recovery proxy: whether a configuration recovers the grounded
disproportionality verdict produced from the raw sources. This preserves the qualitative finding that
retrieval grounding is the dominant contributor (the no-RAG configuration collapses the proxy), without
claiming the manuscript's exact category-level numbers.

## D4 — External data sources are accessed through fixtures in tests

The four lanes target live public sources (DailyMed, openFDA FAERS, PubMed E-utilities, DrugBank/PrimeKG).
Tests and the smoke configuration drive the lanes from the synthetic PedOL-Bench generator
(`pedolgov.audit.bench`) so no network access is required; the source adapters keep the real-source
contract for production use.
