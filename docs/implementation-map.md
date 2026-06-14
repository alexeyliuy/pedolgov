# Implementation map

The source carries no inline comments or docstrings; this table is the single home for paper provenance.
Sections refer to the manuscript (Scientific Reports submission). Package root is `pedolgov/`.

## Query and configuration

| Paper anchor | Item | File | Symbol |
|---|---|---|---|
| Methods §Framework overview | query tuple (d, i, a, c); age group a ∈ {neonate, infant, child, adolescent} | `manifest/declaration.py` | `Query`, `AgeGroup` |
| Results; Methods §Risk category assignment | ordinal risk tiers Minimal ≺ Low ≺ Moderate ≺ High ≺ Critical | `manifest/declaration.py` | `RiskTier` |
| Methods §Inference / §Risk / §Conformal; SI S1 | full hyperparameter set | `manifest/config.py` | `Config` and nested `*Config` |

## Lanes — Evidence Synthesis Engine (Methods §Evidence Synthesis Engine)

| Paper anchor | Item | File | Symbol |
|---|---|---|---|
| §Drug label retrieval | DailyMed SPL parsing of sections 1, 4, 5, 8.4; chunk 512 / overlap 64; Contriever; FAISS; top-k 10 | `lanes/label_lane.py` | `LabelLane` |
| §Adverse event signal retrieval | FAERS age-stratified queries; PRR, ROR, BCPNN IC, EBGM; consensus ≥ 3/4 | `lanes/signal_lane.py` | `SignalLane`, `proportional_reporting_ratio`, `reporting_odds_ratio`, `bcpnn_information_component`, `empirical_bayes_geometric_mean`, `consensus_signal` |
| §Literature retrieval | PubMed boolean queries + pediatric MeSH; up to 50 abstracts; structured extraction | `lanes/literature_lane.py` | `LiteratureLane` |
| §Knowledge graph integration | DrugBank 6.0 + PrimeKG one-hop neighbourhood → NL serialisation | `lanes/knowledge_lane.py` | `KnowledgeLane` |
| Fig 1 retrieval strip; §Evidence grading | dense retriever port + offline hash encoder + guarded Contriever/FAISS; 512/64 chunker | `lanes/retrieval.py` | `Retriever`, `HashEncoder`, `Chunker`, `FaissIndex` |
| §Evidence Synthesis Engine | fuse four lanes into a graded evidence summary with per-source citations | `lanes/fusion.py` | `EvidenceSynthesisEngine`, `EvidenceItem`, `EvidenceSummary` |

## Classification — evidence grading

| Paper anchor | Item | File | Symbol |
|---|---|---|---|
| Eq (1) | G(e) maps each item to Level A–E (RCT / observational+pharmacovigilance ≥3 consensus / case report+label / expert+adult extrapolation / no direct evidence) | `classification/hierarchy.py` | `EvidenceGrade`, `grade_evidence`, `composite_grade` |

## Valuation — Risk Stratification Engine (Methods §Risk Stratification Engine)

| Paper anchor | Item | File | Symbol |
|---|---|---|---|
| §Multi-factor risk scoring; 8 factors f1–f8 | f1 disproportionality, f2 off-label category severity, f3 evidence scarcity, f4 age-group vulnerability, f5 interaction burden, f6 therapeutic index, f7 KIDs List presence, f8 regulatory label gap | `valuation/factors.py` | `RiskFactors`, `factor_vector`, `f1_disproportionality` … `f8_label_gap` |
| Eq (2) | R(d,i,a) = Σ_j w_j f_j; weights sum to 1, each ≤ 0.25; softmax risk head for class probabilities | `valuation/scoring.py` | `RiskScorer`, `RiskHead`, `composite_risk` |
| Eq (3) | C(R) thresholds τ1<τ2<τ3<τ4 → five tiers | `valuation/assignment.py` | `assign_tier`, `Thresholds` |
| §Multi-factor risk scoring (weights); §Risk category assignment (τ via weighted κ) | two-phase weight fit by ordinal cross-entropy with cap 0.25; threshold calibration by maximising weighted Cohen κ | `valuation/fitting.py` | `fit_weights`, `calibrate_thresholds`, `ordinal_cross_entropy` |

## Bond — conformal prediction (Methods §Conformal prediction for uncertainty quantification)

| Paper anchor | Item | File | Symbol |
|---|---|---|---|
| Eq (4) | nonconformity s_i = 1 − p̂(y_i ∣ x_i) | `bond/conformal.py` | `nonconformity_scores` |
| Eq (5) | q̂ = Quantile({s_i}; ⌈(n+1)(1−α)⌉ / n) | `bond/conformal.py` | `conformal_quantile` |
| Eq (6) | 𝒞(x) = {y : p̂(y∣x) ≥ 1 − q̂} | `bond/conformal.py` | `prediction_set` |
| Eq (7) | marginal coverage P[y ∈ 𝒞(x)] ≥ 1 − α; empirical coverage and set size | `bond/conformal.py` | `SplitConformal`, `empirical_coverage`, `average_set_size` |

## Inspection — guardrails (Methods §Soft guardrails, §Human-in-the-loop trigger; Results §Governance)

| Paper anchor | Item | File | Symbol |
|---|---|---|---|
| Fig 1 right governance rail (hard) | drug-name verification, contraindication check, citation traceability, dose-range check, interaction check | `inspection/hard.py` | `HardGuardrails`, individual `check_*` |
| §Soft guardrails (1)(2)(3) | large-set flag ∣𝒞∣ ≥ 4; evidence-source disagreement; calibration-distribution distance > 95th pct | `inspection/soft.py` | `SoftGuardrails` |

## Clearance — Governance Module (Methods §Framework overview; Results §End-to-end)

| Paper anchor | Item | File | Symbol |
|---|---|---|---|
| governance operator Γ → {release, warn, defer} | combine risk tier, prediction set, guardrails into a governed recommendation | `clearance/governance.py` | `GovernanceModule`, `Decision` |
| §Human-in-the-loop trigger | Critical tier OR set size > 3 OR hard-guardrail activation OR neonatal query → mandatory review | `clearance/governance.py` | `requires_human_review` |

## Clearance — assembled pipeline (Methods §Framework overview; Fig 4)

| Paper anchor | Item | File | Symbol |
|---|---|---|---|
| Fig 4 four-module pipeline | synthesize → stratify → calibrate → inspect → clear, with provenance ledger | `clearance/pipeline.py` | `GovernedPipeline`, `ClearanceResult` |

## Lane readouts — structured extraction helpers

| Paper anchor | Item | File | Symbol |
|---|---|---|---|
| §Drug label retrieval | parse SPL sections into structured fields | `lanes/label_lane.py` | `parse_label`, `LabelReadout` |
| §Literature retrieval; Eq (1) | grade abstracts and count Level A/B evidence | `lanes/literature_lane.py` | `grade_literature`, `literature_levels` |
| §Knowledge graph integration | extract interaction / therapeutic-index / KIDs fields | `lanes/knowledge_lane.py` | `parse_knowledge`, `KnowledgeReadout` |
| §Adverse event signal retrieval | consensus verdict object over the four estimators | `lanes/signal_lane.py` | `SignalVerdict` |

## Broker — LLM backbone and prompting (Methods §LLM backbone and prompt engineering)

| Paper anchor | Item | File | Symbol |
|---|---|---|---|
| §Model selection | backbone port; deterministic offline reasoner; guarded vLLM/HF/API adapters | `broker/backend.py` | `Backbone`, `OfflineReasoner`, `VllmBackbone`, `ApiBackbone` |
| §Prompt engineering | four-part prompt (system role, retrieved context, task instruction, reasoning directive); zero-shot chain-of-thought | `broker/prompts.py` | `build_prompt`, `PromptParts` |

## Records — provenance and runtime

| Paper anchor | Item | File | Symbol |
|---|---|---|---|
| R4 reproducibility | single seed utility stored with every artefact | `records/seed.py` | `set_seed` |
| R4 atomic checkpoints | tmp-file + os.replace atomic writer | `records/io.py` | `atomic_write_json`, `atomic_write_bytes` |
| §Evidence trail / auditability | per-query evidence and decision ledger | `records/ledger.py` | `Ledger`, `LedgerEntry` |
| R4 logging | module logging configuration | `records/logging.py` | `get_logger`, `configure_logging` |

## Audit — evaluation protocol (Methods §Evaluation protocol; §Statistical analysis)

| Paper anchor | Item | File | Symbol |
|---|---|---|---|
| Tier 1–5 metrics | macro-F1, AUROC, Spearman ρ, ECE, coverage, set size, ordinal accuracy, weighted Cohen κ, ordinal MAE | `audit/metrics.py` | `macro_f1`, `auroc`, `spearman_rho`, `expected_calibration_error`, `coverage`, `weighted_kappa`, `ordinal_mae`, `ordinal_accuracy` |
| §Statistical analysis | 10,000-resample paired bootstrap CI, Bonferroni, Cohen d, Cliff δ | `audit/bootstrap.py` | `paired_bootstrap`, `bonferroni`, `cohens_d`, `cliffs_delta` |
| Tables 1–6; SI S1–S2 | tier evaluation loops over PedOL-Bench; weight fit + threshold calibration + conformal fit assembly | `audit/evaluate.py` | `fit_pipeline`, `extract_features`, `evaluate_extraction`, `evaluate_risk`, `evaluate_conformal`, `evaluate_governance`, `evaluate_end_to_end`, `ablation_table` |
| §PedOL-Bench construction | synthetic, structured benchmark generator with planted signal | `audit/bench.py` | `PedOLBench`, `synthesize_corpus`, `BenchItem` |

## Gateway — CLI (cyclopts)

| Paper anchor | Item | File | Symbol |
|---|---|---|---|
| end-to-end usage | verbs: synthesize, stratify, calibrate, clear, audit, bench | `gateway/app.py`, `gateway/__main__.py` | `app`, command functions |

## Configs

| File | Content |
|---|---|
| `configs/model/*.yaml` | backbone selection (llama_70b, qwen2_72b, mixtral_8x7b, offline) |
| `configs/data/*.yaml` | source toggles and bench construction parameters |
| `configs/train/*.yaml` | weight-fit + threshold-calibration + conformal settings |
| `configs/experiment/main.yaml` | primary result (Table 1/3/6 headline) |
| `configs/experiment/ablation_*.yaml` | Table 2 single-component and pairwise removals |
| `configs/experiment/supplementary_*.yaml` | SI S1 hyperparameter sweeps |
| `configs/experiment/_smoke.yaml` | pytest smoke only |
