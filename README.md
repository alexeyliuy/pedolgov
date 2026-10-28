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
