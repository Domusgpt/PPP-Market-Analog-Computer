# HEMOC + PPP Project Guide

> This file is the primary context document for AI coding agents (Claude Code, Jules, Codex).
> It is also the CLAUDE.md auto-read file for Claude Code sessions.

## Project Identity

**PPP (Polytopal Projection Processing)** — A real-time 4D visualization and telemetry runtime.
Software simulation of a physical analog computer that computes via light interference through
stacked kirigami sheets, governed by 4D polytope geometry and the golden ratio. 38 development
sessions, working adapter layer, phase-locked stereoscopy, 8-layer sonic geometry telemetry.
Repo: this repo (`PPP-Market-Analog-Computer`).

**HEMOC (Hexagonal Emergent Moire Optical Cognition)** — The physics-based moire encoder that
uses the Phillips 8x8 E8->H4 projection matrix to encode high-dimensional data as interference
patterns. 15 experiments documented, correlation from 0.008 to 0.916, CNN decoder proven,
V4 Plastic Encoder validated as optimal architecture.
Repo: `Domusgpt/HEMOC-Stain-Glass-Flower` (sibling directory or clone via env-setup.sh).

The Phillips matrix was NOT designed with golden-ratio properties as constraints. It was built
by Paul Phillips for hyper-dimensional audio analysis and quaternion system needs. The golden-ratio
block structure (U_R = phi * U_L) is a DISCOVERED property of the projection geometry.

---

## Required Reading (In This Order)

Before writing ANY code, read these documents. The order matters — each builds on the previous.

### Understanding the system (what and why)

| # | File | What it tells you |
|---|------|-------------------|
| 1 | `WHAT_THIS_DOES.md` | What the physical optical device is and how it computes |
| 2 | `CODING_AGENT_PROMPT.md` | Mathematical architecture: 24-cell, Trinity Decomposition, all invariants |
| 3 | `docs/HEMOC_PPP_CROSS_REPO_DIGEST.md` | Cross-repo synthesis: full research arc, what's proven, what's missing |

### Understanding the evidence (what's proven)

| # | File (HEMOC repo) | What it tells you |
|---|------|-------------------|
| 4 | `docs/CRITICAL_FINDING_AND_ROADMAP.md` | Centroid-zero discovery: WHY MLPs fail, WHY CNNs work |
| 5 | `docs/PROJECT_STATUS.md` | 15 experiments with correlation progression 0.008 -> 0.916 |
| 6 | `FINAL_ANALYSIS_REPORT.md` | V4 Plastic Encoder benchmark: why plastic ratio (1.32^L) wins |

### Understanding the gaps (what needs doing)

| # | File (HEMOC repo) | What it tells you |
|---|------|-------------------|
| 7 | `docs/TESTING_REVIEW_AND_DEV_TRACK.md` | Testing audit: what has tests, what doesn't, exact specs for tests to write |
| 8 | `docs/MUSICGEOMETRY_GAP_ANALYSIS.md` | 5 gaps vs MusicGeometry framework, CPE paper alignment, Trinity Decomposition |
| 9 | `docs/benchmark_contract.md` | Schema governance: required fields, thresholds, validation rules |

### Understanding the infrastructure

| # | File | What it tells you |
|---|------|-------------------|
| 10 | PPP `DEV_TRACK.md` | 38 sessions of PPP development: adapters, telemetry, sonic geometry |
| 11 | PPP `docs/refactor-plan.md` | PPP core/adapter architecture: how dashboards plug in |
| 12 | HEMOC `docs/STRUCTURAL_TRAINING_VISION.md` | Music as training curriculum: why audio domain is the right starting point |

---

## The Recovery Plan: Phases B through F

Phase A (benchmark contract + schema validator) is COMPLETE. The remaining phases:

### Phase B — Protect the Working Pipeline (PRIORITY 1)

The pipeline that achieves 0.916 correlation has ZERO automated tests.

**What to test** (specs in `docs/TESTING_REVIEW_AND_DEV_TRACK.md` Priority 1):
- `tests/unit/test_hybrid_encoder.py` — Target: `HybridEncoder` in `demos/dual_decoder.py`
  - test_determinism: same angles -> identical pattern
  - test_all_angles_affect: each of 6 angles changes output
  - test_pattern_shape: output is 64x64x3
  - test_linear_vs_hyper: angles 0-2 are direct, 3-5 go through hypercube
  - test_injectivity: different angles -> different patterns (0 collisions)
  - test_angle_sensitivity: per-angle sensitivity measurement

- `tests/unit/test_cnn_decoder.py` — Target: `AngleRecoveryCNN` in `demos/option_e_scaled_cnn.py`
  - test_model_architecture: layer shapes, parameter count
  - test_forward_pass: 64x64x3 input -> 6 outputs
  - test_gradient_flow: no vanishing/exploding gradients

- `tests/integration/test_encode_decode_roundtrip.py`
  - test_known_angles_recovery: encode -> decode -> compare
  - test_correlation_threshold: avg correlation > 0.8

**Exit criterion**: `pytest` catches quality regressions on the working pipeline.

### Phase C — Operationalize Domain-Diverse Training

Code exists at `demos/domain_diverse_training.py`. It has NEVER BEEN RUN.

**What to do**:
- Requires GPU (see `docs/GPU_CLOUD_GUIDE.md` for cloud setup)
- Run single-domain baseline, domain-diverse training, few-shot calibration sweep
- Produce unified report artifact with side-by-side comparison

**Exit criterion**: Reproducible command produces comparative artifact proving transfer behavior.
**Depends on**: Phase B (tests must exist before training experiments).

### Phase D — Baselines + Ablations

**Baseline runners**:
- Direct-feature MLP: audio features -> predictions with NO visual encoding (the ceiling test)
- Random pattern control: random images instead of moire patterns (the floor test)
- Simple visual baseline: lower-complexity encoder

**Ablation runners**:
- Hypercube path on/off
- Golden-ratio constants substituted with alternatives (sqrt(2), e, learned params)
- Resolution sweep: 32x32, 64x64, 128x128

**Exit criterion**: One matrix report showing where HEMOC wins, ties, or loses.

### Phase E — Unify Python Core + Visual System

**Shared interchange schema**:
```json
{
  "input_state": {"type": "e8_vector", "dim": 8, "values": [...]},
  "pattern": {"format": "rgb", "width": 64, "height": 64, "data": "..."},
  "decode": {"angles": [...], "correlation": 0.0, "method": "cnn"},
  "metadata": {"seed": 42, "encoder": "v4_plastic", "timestamp": "..."}
}
```

**PPP integration**: Use MarketQuoteAdapter/HemocPythonBridge patterns from `docs/refactor-plan.md` as reference for adapter design.

**Visual system**: Port Phillips Matrix to TypeScript, implement 2-4-2 uniform mapping, fragment shader for 4-petal moire. Architecture spec is in HEMOC `docs/HEMOC_ARCHITECTURE_DETAILED.md` (on visual-init branch).

**Exit criterion**: Data produced by Python core consumed by visual system without patching.

### Phase F — Documentation Reconciliation

- Create `docs/STATUS_SINGLE_SOURCE.md` as the ONE canonical status document
- Mark conflicting docs as archival/superseded
- Use ontology blueprint (`docs/ONTOLOGY_BLUEPRINT_ICE_ECT_DRAFT.md`) for publication prep
- Three-table structure: Claims Table + Evidence Table + Reproducibility Table

**Exit criterion**: New contributor finds current truth in under 5 minutes.

---

## Quality Standards

1. **Evidence before claims**: No architecture claim without a benchmark artifact
2. **Deterministic seeds**: All experiments use fixed seeds, recorded in result metadata
3. **Schema compliance**: Every result file must pass `scripts/validate_results_schema.py`
4. **Regression protection**: No merge without `pytest tests/unit/ && pytest tests/integration/`
5. **Academic rigor**: Claims require evidence table entries; conjectures are labeled as such
6. **Reproducibility by default**: Fixed fixture packs, versioned result schema, env reports

---

## Known Pitfalls

**PPP repo**:
- `_SYNERGIZED_SYSTEM/backend/engine/pipeline.py:33` has a blocking import error:
  `from .rules.enforcer` should be `from .enforcer`. This breaks full pipeline import.
- PPP test commands: `npm run test:all` (runs phase-lock, adapter, bridge tests)
- Backend tests: `cd _SYNERGIZED_SYSTEM/backend && source .venv/bin/activate && pytest tests/ -v`

**HEMOC repo**:
- 21+ open PRs across branching chains. Always check `git branch -a` before creating branches.
- The working pipeline (0.916 result) lives in `demos/`, NOT in `src/`.
  - Encoder: `demos/dual_decoder.py::HybridEncoder`
  - Decoder: `demos/option_e_scaled_cnn.py::AngleRecoveryCNN`
- ALL existing automated tests target the OLD `OpticalKirigamiEncoder` from `main.py`,
  not the HybridEncoder that actually achieves the claimed results.
- PyTorch is NOT installed by default. Install separately when GPU work begins:
  `pip install torch` (CPU) or `pip install torch --index-url https://download.pytorch.org/whl/cu121` (CUDA)
- Cross-domain zero-shot transfer FAILS (correlation -0.027). But 100-sample calibration
  improves it to 0.22. Domain-specific training is essential.
- HEMOC `pyproject.toml` differs between branches — `main` lists torch as core dep,
  `hemoc-visual-system-init` does not.

---

## Key Research Findings (Summary)

| Finding | Status | Reference |
|---------|--------|-----------|
| Encoding is injective (0 collisions across 1000 samples) | Proven | PROJECT_STATUS.md Exp 1-6 |
| CNN decodes 5/6 angles from moire patterns | Proven | PROJECT_STATUS.md Exp 7 |
| HybridEncoder required (pure hypercube fails at 0.09) | Proven | PROJECT_STATUS.md Exp 8 |
| Audio pipeline achieves 0.916 correlation | Proven | PROJECT_STATUS.md Exp 13 |
| More training data helps (8K->15K: 0.894->0.916) | Proven | PROJECT_STATUS.md Exp 12-13 |
| Cross-domain zero-shot transfer fails | Proven | PROJECT_STATUS.md Exp 14-15 |
| V4 Plastic Encoder > Hybrid > V3.1 > Trinity | Proven | FINAL_ANALYSIS_REPORT.md |
| Centroid-zero kills MLP decoders | Proven | CRITICAL_FINDING_AND_ROADMAP.md |
| Domain-diverse training effectiveness | UNTESTED | demos/domain_diverse_training.py |
| Golden ratio vs alternatives | UNTESTED | No ablation exists |
| Baselines (direct MLP, random pattern) | UNTESTED | No comparison exists |

---

## Environment Setup

Run the bootstrap script (idempotent, safe to run multiple times):

```bash
bash scripts/env-setup.sh
```

Then activate the appropriate venv:
```bash
# For PPP Python work
source _SYNERGIZED_SYSTEM/backend/.venv/bin/activate

# For HEMOC Python work
source ../HEMOC-Stain-Glass-Flower/.venv/bin/activate
```

---

## Quick Command Reference

```bash
# --- PPP ---
# All PPP tests (Node + TypeScript)
npm run test:all

# PPP Python backend tests
cd _SYNERGIZED_SYSTEM/backend && source .venv/bin/activate && pytest tests/ -v --tb=short

# --- HEMOC ---
# Unit tests
cd ../HEMOC-Stain-Glass-Flower && source .venv/bin/activate && pytest tests/unit/ -q

# Integration tests
pytest tests/integration/ -v

# Validate result artifacts against benchmark contract
python scripts/validate_results_schema.py

# Run benchmark suite (when available)
python scripts/run_benchmark_suite.py

# Visual system dev server (if on visual-init branch)
cd hemoc-visual-system && npm run dev
```

---

## Branch Strategy

| Repo | Primary branch | Purpose |
|------|---------------|---------|
| PPP | `main` | Stable PPP runtime |
| HEMOC | `main` | Last merged stable |
| HEMOC | `hemoc-visual-system-init-*` | Recovery plan execution, visual system, benchmark contract |
| HEMOC | `claude/restore-linear-path-*` | Experiment progression, visual system scaffolding |
| HEMOC | `claude/review-testing-progress-*` | Testing audit, 15 experiments documented |

When continuing Recovery Plan work, branch from `hemoc-visual-system-init-4592349023335104422`
as it has the most complete state (benchmark contract, schema validator, recovery audit).
