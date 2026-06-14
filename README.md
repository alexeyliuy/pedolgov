# PedOLGov — Standard Operating Procedure

| Controlled document | |
|---|---|
| Document no. | SOP-PEDOLGOV-001 |
| Title | Evidence synthesis and risk-stratified governance for pediatric off-label prescribing |
| Revision | A (0.1.0) |
| Effective date | 2025-01-01 |
| Process owner | PedOLGov maintainers |
| Classification | Decision-support software — advisory use only |

---

## 1. Purpose

This procedure describes how to install, fit, and operate PedOLGov, a modular large language model
framework that synthesises evidence from four public drug-information sources and returns a
conformally-calibrated ordinal risk category (Minimal, Low, Moderate, High, Critical) with layered
guardrail governance for a drug–indication–age-group query. The software supports a pharmacist or
prescribing clinician; it does not prescribe and does not replace clinical judgement.

## 2. Scope

The procedure applies to the `pedolgov` package and its command-line gateway. It covers the four
retrieval lanes (DailyMed labels, FAERS signals, PubMed literature, DrugBank/PrimeKG knowledge),
evidence grading, eight-factor risk scoring, split-conformal calibration, hard and soft guardrails,
and the human-in-the-loop trigger. The default backbone is a deterministic offline reasoner so the
pipeline runs without GPUs or network access; guarded adapters target the frozen open-source backbones
used for the reported results (see `docs/deviations.md`).

## 3. Definitions

- **PedOLGov** — the governed pipeline: synthesise → stratify → calibrate → inspect → clear.
- **PedOL-Bench** — the evaluation benchmark of drug–indication–age-group triples with consensus risk
  labels derived from the KIDs List 2025, WHO EMLc, and BNF for Children.
- **Tier** — one of five ordinal risk categories.
- **Prediction set** — the conformal set of tiers covering the truth with probability at least 1 − α.
- **Hard guardrail** — a deterministic check that can route a query to mandatory review.
- **Soft guardrail** — a probabilistic flag for residual uncertainty.

## 4. Responsibilities

The framework issues `release`, `warn`, or `defer`. A `defer` recommendation, a Critical tier, a
prediction set larger than the configured target, a triggered hard guardrail, or a neonatal query each
require pharmacist or physician review before any prescribing action.

## 5. Materials and equipment

Reported results were produced with frozen open-source backbones served by vLLM on 4× NVIDIA A100 80GB
GPUs. The offline reasoner runs on a single CPU. Python 3.11 or newer is required.

## 6. Procedure

### 6.1 Install

```
pip install .
# or
conda env create -f environment.yml && conda activate pedolgov
# or
docker build -t pedolgov:0.1.0 . && docker run --rm pedolgov:0.1.0 --help
```

Optional extras: `pip install .[retrieval]` (FAISS + Contriever), `.[serve]` (vLLM backbones),
`.[api]` (hosted baselines), `.[dev]` (test and lint tooling).

### 6.2 Build the benchmark

```
python -m pedolgov.gateway bench configs/experiment/main.yaml
```

### 6.3 Fit the risk model and calibrate

```
python -m pedolgov.gateway stratify configs/experiment/main.yaml
python -m pedolgov.gateway calibrate configs/experiment/main.yaml
```

`stratify` fits the eight factor weights by ordinal cross-entropy (each weight capped at 0.25, summing
to one), calibrates the four category thresholds by maximising the weighted Cohen κ, and reports
risk-stratification metrics. `calibrate` fits the split-conformal quantile at α = 0.10.

### 6.4 Clear a query

```
python -m pedolgov.gateway clear configs/experiment/main.yaml --index 0
```

The output states the tier, the prediction set, the governed recommendation, and whether human review
is mandatory.

### 6.5 Reproduce reported endpoints

Each row maps an endpoint to its command and the value reported for the Llama 3.1-70B backbone. The
offline default reproduces the pipeline behaviour; the exact reported numbers require the `serve` extra
and the corresponding backbone.

| Endpoint | Command | Reported value |
|---|---|---|
| Evidence extraction F1 | `gateway audit …` (extraction row) | 0.893 ± 0.018 |
| Risk AUROC (High+Critical vs rest) | `gateway stratify …` | 0.806 ± 0.019 |
| Spearman ρ vs FAERS pediatric ADR rate | `gateway stratify …` | 0.667 ± 0.030 |
| Conformal coverage (α = 0.10) | `gateway calibrate …` | 0.911 ± 0.018 |
| Average prediction-set size | `gateway calibrate …` | 2.41 ± 0.20 |
| Hallucination rate (unconstrained → governed) | `gateway audit …` (governance row) | 0.187 → 0.062 |
| End-to-end task accuracy | `gateway audit …` (end_to_end row) | 0.847 ± 0.018 |

## 7. Data sources

| Dataset | Size | Role | License |
|---|---|---|---|
| DailyMed SPLs | ~140K labels | drug label extraction | Public |
| FAERS (openFDA) | 20M+ reports | adverse-event signal detection | Public |
| DrugBank 6.0 | 11,891 drugs | drug properties, interactions | CC BY-NC 4.0 |
| PrimeKG | 17,080 diseases | knowledge-graph context | MIT |
| PubMed / MEDLINE | 37M+ citations | literature retrieval | Public |
| WHO EMLc 9th | 361 medicines | ground truth (low risk) | WHO Open Access |
| KIDs List 2025 | 60+ drugs | ground truth (high risk) | Public |
| BNF for Children | UK formulary | ground truth (dosing) | Public |

The shipped benchmark is synthetic and self-contained; the lanes keep the real-source contract for
production deployment.

## 8. Compute budget

| Backbone | Hardware | Latency | Throughput | Cost / 1K queries |
|---|---|---|---|---|
| Llama 3.1-70B | 4× A100 80GB (vLLM) | 892 ms/query | 67.3 q/min | $4.30 |
| Qwen2-72B | 4× A100 80GB (vLLM) | 918 ms/query | 65.4 q/min | $4.63 |
| Mixtral-8x7B | 4× A100 80GB (vLLM) | 524 ms/query | 114.5 q/min | $2.51 |
| Offline reasoner | 1× CPU | < 5 ms/query | — | negligible |

The governance layer adds about 34 ms/query; conformal prediction adds under 2 ms/query.

## 9. Quality records

```
make test     # pytest: disproportionality, grading, factors, monotonicity, fitting,
              # threshold calibration, conformal coverage, guardrails, governance,
              # determinism, metrics, gradient flow, end-to-end smoke, style guard
make lint     # ruff + black + isort
make type     # mypy --strict
```

## 10. Ethics

All datasets are publicly available. The work involves no human participants, no animal subjects, and
no identifiable patient data, and is therefore not human-subjects research. The framework is advisory:
high-risk and uncertain queries are routed to a human reviewer by design.

## 11. Revision history

| Revision | Date | Change |
|---|---|---|
| A | 2025-01-01 | Initial controlled release of the PedOLGov pipeline and PedOL-Bench harness. |
