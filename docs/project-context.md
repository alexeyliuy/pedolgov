# Project Context

PedOLGov (Pediatric Off-Label Governance) is a modular large language model framework that
synthesises evidence across four public drug-information sources and turns it into a
conformally-calibrated ordinal risk score with layered governance guardrails for off-label
prescribing in children. This document records the resolved build parameters used by the code
release.

| Field | Value | Confidence |
|---|---|---|
| package | `pedolgov` | HIGH |
| domain | pediatric clinical pharmacology — off-label prescribing governance (evidence synthesis, ordinal risk stratification, conformal calibration, guardrail governance) | HIGH |
| framework | Python 3.11+; numpy/scipy statistics core + a small torch risk head; retrieval via Contriever + FAISS behind a port (optional); LLM backbone behind a port with a deterministic offline reasoner | HIGH |
| venue | Scientific Reports | MEDIUM |
| compute_target | inference only: 4× NVIDIA A100 80GB via vLLM; per-query latency 127–918 ms across backbones; no neural-network training wall-clock | HIGH |

## Model selection (from Methods, "Model selection")

Open-source candidate backbones: Llama 3.1-70B-Instruct (primary), Qwen2-72B-Instruct (bilingual),
Mixtral-8x7B-Instruct-v0.1 (mixture-of-experts). Proprietary baselines: GPT-4, GPT-4o, Claude 3.5
Sonnet, o1-Preview. Additional open-source baselines: Llama 3.1-8B, Mistral-7B, BioMistral-7B,
Meditron-70B, Meerkat-7B, OpenMedLM (Yi-34B), Med42-v2-70B. No backbone is fine-tuned; all are frozen.

The released code calls every backbone through a single port (`pedolgov.broker.backend.Backbone`).
The default implementation is a deterministic, seedable offline reasoner so the pipeline runs and is
testable without GPUs or network access; guarded adapters for vLLM / Hugging Face / hosted APIs sit
behind the same port and raise honestly when their optional dependency is absent. This substitution
is logged in `deviations.md`.

## Datasets (Table 7; all publicly available)

| Dataset | Size | Role | License |
|---|---|---|---|
| DailyMed SPLs | ~140K labels | drug label extraction | Public |
| FAERS (openFDA) | 20M+ reports | adverse-event signal detection | Public |
| DrugBank 6.0 | 11,891 drugs | drug properties, DDIs | CC BY-NC 4.0 |
| PrimeKG | 17,080 diseases | knowledge-graph context | MIT |
| PubMed / MEDLINE | 37M+ citations | literature retrieval | Public |
| SIDER 4.1 | 1,430 drugs | drug–side-effect pairs | CC BY-NC-SA |
| OnSIDES | 51,460 labels | pre-extracted ADEs | Open source |
| WHO EMLc 9th | 361 medicines | ground truth (low risk) | WHO Open Access |
| KIDs List 2025 | 60+ drugs | ground truth (high risk) | Public |
| BNF for Children | UK formulary | ground truth (dosing) | Public |

Access points: openFDA `https://open.fda.gov/apis/drug/event/`; DailyMed `https://dailymed.nlm.nih.gov/`;
DrugBank `https://go.drugbank.com/` (academic CC BY-NC 4.0); PrimeKG
`https://zitniklab.hms.harvard.edu/projects/PrimeKG/`; OnSIDES `https://github.com/tatonetti-lab/onsides`;
WHO EMLc `https://www.who.int/groups/expert-committee-on-selection-and-use-of-essential-medicines`;
PubMed `https://pubmed.ncbi.nlm.nih.gov/`.

PedOL-Bench is the evaluation benchmark constructed from the authoritative ground-truth sources:
1,847 drug–indication–age-group triples (187 drugs, 94 indications, 15 therapeutic classes). Consensus
risk distribution: Minimal 18.2%, Low 22.8%, Moderate 28.4%, High 19.1%, Critical 11.5%. Age groups:
neonate 12.3%, infant 24.7%, child 38.6%, adolescent 24.4%. Split: train 40% / calibration 20% / test 40%.
Inter-rater agreement on a 200-triple subset: Cohen κ = 0.83 (95% CI 0.77–0.89).

## Hyperparameter reference (main configuration)

Retrieval: chunk size 512 tokens, overlap 64, Contriever dense encoder, FAISS index, top-k = 10.
Disproportionality: PRR ≥ 2 and χ² ≥ 4; ROR with 95% CI lower bound > 1; consensus when ≥ 3 of 4
methods (PRR, ROR, BCPNN IC, EBGM) agree. Inference: temperature 0.1 for extraction and risk,
0.3 for evidence-summary generation; max input context 4096 tokens; max generation 2048 tokens.
Risk scoring: J = 8 factors, weights sum to 1, no single weight exceeds 0.25. Category thresholds
τ1 < τ2 < τ3 < τ4 calibrated by maximising weighted Cohen κ (τ3 sits in the narrow 0.55–0.65 band).
Conformal: α = 0.10 (target ≥ 90% coverage), calibration sizes n ∈ {50, 100, 200, 500}, average
prediction-set-size target ≤ 3. Seeds: 42, 123, 456, 789, 1024. Statistics: 10,000-resample paired
bootstrap, Bonferroni correction, Cohen d for continuous and Cliff δ for ordinal comparisons.

The `_smoke` configuration shrinks every dimension and is marked for unit-test smoke only; it must
never be used for reporting.

## Scientific elements implemented

Equations: (1) evidence grade G(e) over levels A–E; (2) composite risk R(d,i,a) = Σ w_j f_j;
(3) category assignment C(R) over Minimal ≺ Low ≺ Moderate ≺ High ≺ Critical; (4) nonconformity
s = 1 − p̂(y∣x); (5) conformal quantile q̂; (6) prediction set 𝒞(x); (7) marginal coverage guarantee.

Eight risk factors: f1 disproportionality signal strength, f2 off-label category severity, f3 evidence
scarcity, f4 age-group vulnerability, f5 drug-interaction burden, f6 therapeutic index, f7 KIDs List
presence, f8 regulatory label gap.

Guardrails — hard: drug-name verification, contraindication check, citation traceability, dose-range
check, interaction check. Soft: large-prediction-set flag (∣𝒞∣ ≥ 4), evidence-source disagreement,
calibration-distribution distance (> 95th percentile of calibration scores). Human-in-the-loop trigger:
Critical risk, prediction-set size > 3, any hard-guardrail activation, or a neonatal query.

Reported tables reproduced by the evaluation harness: Table 1 (evidence extraction), Table 2 (ablation),
Table 3 (risk by therapeutic class), Table 4 (cross-dataset generalisation), Table 5 (computational
efficiency), Table 6 (end-to-end). Supplementary analyses: S1 (hyperparameter sensitivity), S2 (failure
mode taxonomy).

## Code availability statement (manuscript, verbatim — kept here, not in the README)

"Upon request and once published, the code implementing PedOLGov and the data that comprise the
PedOL-Bench will be available from the corresponding author. All items constituting the
PedOLGov/PedOL-Bench components utilized only publicly available tools and databases. There were no
proprietary or restricted tools."

## Ethics

All datasets are publicly available; there were no human participants, animal subjects, or identifiable
patient data, and the work is not human-subjects research, so no ethical approval was required.

## Citation

Yu P., Feng L., Qiu K., Zhu M. Governing Off-Label Prescribing in Children with Generative AI: A Large
Language Model Framework for Evidence Synthesis and Risk Stratification. Scientific Reports (under review).
