# Repository plan

## Directory tree

```
pedolgov/
  manifest/        query schema + typed configuration (msgspec Structs + loader)
  lanes/           four retrieval channels, dense-retriever port, disproportionality, fusion
  classification/  evidence grading G(e)
  valuation/       eight-factor scoring, category assignment, weight fit, threshold calibration
  bond/            split conformal prediction
  inspection/      hard and soft guardrails
  clearance/       governance operator + human-in-the-loop trigger
  broker/          LLM backbone port + offline reasoner + guarded adapters + prompting
  records/         seeding, atomic IO, provenance ledger, logging
  audit/           metrics, bootstrap statistics, tier evaluations, benchmark generator
  gateway/         cyclopts CLI (synthesize / stratify / calibrate / clear / audit / bench)
configs/{model,data,train,experiment}/
tests/
docs/
scripts/
assets/
```

## Module responsibilities

- **manifest** — the declared query and the single typed configuration object. `config.py` loads YAML
  into `msgspec.Struct` models with validation; CLI overrides are applied as a flat dotted-key map.
- **lanes** — each lane returns evidence items with source citations. `signal_lane` owns the four
  disproportionality estimators and the consensus rule. `retrieval` provides the dense-retriever port,
  a deterministic hash encoder (default), a 512/64 chunker, and a guarded FAISS index.
- **classification** — turns raw evidence into graded items via the backbone-prompted grader, with a
  deterministic fallback grading rule used by the offline reasoner.
- **valuation** — computes the eight factors, the weighted composite, the five-way assignment, and the
  two calibration routines (weight fit by ordinal cross-entropy; thresholds by weighted κ).
- **bond** — split conformal calibration and prediction-set construction with coverage diagnostics.
- **inspection** — hard checks (deterministic, can abstain) and soft probabilistic flags.
- **clearance** — the governance operator that maps tier + set + flags to release / warn / defer and
  decides whether human review is mandatory.
- **broker** — the backbone port. Offline reasoner is deterministic and seedable; guarded adapters for
  vLLM, Hugging Face, and hosted APIs raise on missing optional dependencies.
- **records** — reproducibility primitives.
- **audit** — evaluation, statistics, and the synthetic PedOL-Bench generator.
- **gateway** — command-line entry points.

## Dependencies (pinned ranges)

Runtime: `numpy>=1.26,<3`, `scipy>=1.11`, `torch>=2.2`, `msgspec>=0.18`, `cyclopts>=3`,
`pyyaml>=6`. Optional extras: `faiss-cpu`, `sentence-transformers` (`retrieval` extra); `vllm`,
`transformers`, `accelerate` (`serve` extra); `openai`, `anthropic` (`api` extra). Dev:
`pytest>=8`, `ruff>=0.5`, `mypy>=1.10`, `black>=24`, `isort>=5`, `pre-commit>=3`.

## Test coverage matrix

| Test file | Kind | Targets |
|---|---|---|
| `tests/test_disproportionality.py` | metric correctness | PRR/ROR/BCPNN-IC/EBGM vs hand-computed contingency tables; consensus rule |
| `tests/test_grading.py` | classification correctness | G(e) level boundaries; composite grade aggregation |
| `tests/test_factors_shape.py` | shape | factor vector length, range [0,1], dtype |
| `tests/test_risk_monotone.py` | ordinal invariant | composite risk monotone non-decreasing in each factor; tier order preserved |
| `tests/test_fitting_overfit.py` | overfit single batch | weight fit recovers planted weights on a separable batch; loss decreases |
| `tests/test_threshold_kappa.py` | metric correctness | weighted-κ threshold search beats random thresholds |
| `tests/test_conformal_coverage.py` | coverage guarantee | empirical coverage ≥ 1 − α on exchangeable synthetic data across seeds |
| `tests/test_guardrails.py` | guardrail trigger | each hard/soft check fires on a crafted positive and stays silent on a negative |
| `tests/test_governance.py` | decision logic | HITL trigger truth table; release/warn/defer mapping |
| `tests/test_determinism.py` | determinism | offline reasoner + full pipeline byte-identical under fixed seed |
| `tests/test_metrics.py` | metric correctness | F1/AUROC/κ/ECE vs sklearn / closed form |
| `tests/test_risk_head_grad.py` | gradient flow | risk head produces finite gradients for all parameters |
| `tests/test_training_smoke.py` | end-to-end smoke | two fit steps on `_smoke.yaml`, loss decreases; clear two queries |
| `tests/test_style_guard.py` | style guard | no comments / docstrings in `pedolgov`; no forbidden phrases; no private path strings |
