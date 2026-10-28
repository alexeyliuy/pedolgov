# PedOLGov — Pediatric Off-Label Prescribing Governance

PedOLGov is a research framework for evidence synthesis and risk-stratified governance of off-label prescribing in children. It combines information from drug labels, adverse-event reports, scientific literature, and pharmacological knowledge resources to support assessment of drug–indication–age-group queries.

The framework is intended for research and decision support. It does not prescribe, replace clinical judgment, or constitute a validated clinical decision tool.

## Scope

The PedOLGov workflow brings together four components:

1. **Evidence synthesis:** retrieve and grade evidence from DailyMed, FAERS, PubMed, DrugBank, and PrimeKG.
2. **Risk stratification:** assess eight pharmacological, epidemiological, and regulatory factors and assign one of five ordered risk categories: Minimal, Low, Moderate, High, or Critical.
3. **Uncertainty assessment:** apply split conformal prediction to report a set of plausible risk categories.
4. **Governance:** apply checks and flag situations that require pharmacist or physician review.

The system uses the recommendations `release`, `warn`, and `defer`. High-risk, uncertain, guardrail-triggered, and neonatal cases require appropriate human review.

## Benchmark and evaluation

The associated study describes **PedOL-Bench**, a benchmark of drug–indication–age-group combinations using reference information from the KIDs List, WHO Essential Medicines List for Children, and British National Formulary for Children. The study evaluates evidence extraction, risk stratification, uncertainty calibration, governance, and computational efficiency across multiple model backbones.

The frozen benchmark and original per-query outputs are not included in this repository. The current implementation has not yet been validated to reproduce the study's reported numerical results.

## Data sources

| Resource | Role |
|---|---|
| DailyMed | Drug-label indications, contraindications, warnings, and pediatric-use information |
| FDA Adverse Event Reporting System (FAERS) | Adverse-event reporting and disproportionality analysis |
| PubMed / MEDLINE | Published clinical and pharmacological evidence |
| DrugBank | Drug properties and interactions |
| PrimeKG | Pharmacological knowledge-graph context |
| WHO Essential Medicines List for Children | Reference information on essential pediatric medicines |
| KIDs List | Reference information on potentially inappropriate pediatric medications |
| British National Formulary for Children | Pediatric prescribing and dosing reference |

Access to individual resources may be subject to their respective licenses and terms.

## Models and evaluation environment

The study describes open-source language model backbones including Llama 3.1-70B, Qwen2-72B, and Mixtral-8x7B, alongside commercial comparison models. The research evaluation reports use of vLLM with NVIDIA A100 GPUs for open-source models. Computational performance and costs depend on the model, hardware, data, and deployment configuration.

## Results reported in the study

The manuscript reports the following endpoints for the Llama 3.1-70B configuration. These values are presented for reference and have not been independently reproduced from the materials in this repository.

| Endpoint | Reported result |
|---|---|
| Evidence extraction F1 | 0.893 ± 0.018 |
| Risk AUROC (High and Critical vs. other categories) | 0.806 ± 0.019 |
| Spearman correlation with pediatric FAERS signals | 0.667 ± 0.030 |
| Conformal coverage at α = 0.10 | 0.911 ± 0.018 |
| Average prediction-set size | 2.41 ± 0.20 |
| Hallucination rate before and after governance | 0.187 → 0.062 |
| End-to-end task accuracy | 0.847 ± 0.018 |

## Computational performance reported in the study

| Backbone | Reported hardware | Latency | Throughput | Cost per 1,000 queries |
|---|---|---|---|---|
| Llama 3.1-70B | 4 × NVIDIA A100 80GB | 892 ms/query | 67.3 queries/min | $4.30 |
| Qwen2-72B | 4 × NVIDIA A100 80GB | 918 ms/query | 65.4 queries/min | $4.63 |
| Mixtral-8x7B | 4 × NVIDIA A100 80GB | 524 ms/query | 114.5 queries/min | $2.51 |

The manuscript reports an approximately 34 ms per query governance overhead and under 2 ms per query for conformal prediction. These are study-reported measurements, not guarantees for other deployments.

## Installation and quality checks

Python 3.11 or newer is required. Install the package with:

```bash
pip install .
```

The project includes automated checks for selected risk-scoring, evidence-grading, and governance functions:

```bash
make test
```

## Ethics and use

PedOLGov is intended to support research into pediatric medication safety. Source data and resulting outputs should be handled in accordance with their licenses and applicable privacy requirements. Recommendations should be reviewed by a qualified clinician or pharmacist before any prescribing decision.

## License

The project is available under the [MIT License](LICENSE). Licenses for external data and models apply separately.
