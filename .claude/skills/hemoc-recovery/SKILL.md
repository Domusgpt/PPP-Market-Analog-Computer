---
name: hemoc-recovery
description: Track Recovery Plan phase progression (A through F) with concrete evidence for each phase. Use when checking what work remains, what's blocked, or deciding what to do next.
allowed-tools: Read, Grep, Glob, Bash(git *), Bash(cd * && git *), Bash(test *), Bash(wc *), Bash(ls *)
---

# HEMOC Recovery Plan Phase Tracker

Check concrete evidence for each Recovery Plan phase (A through F) and report status.

## Recovery Plan Overview

| Phase | Name | Exit Criterion |
|-------|------|----------------|
| A | Benchmark Contract | Schema validator + contract doc exist |
| B | Protect Working Pipeline | pytest catches regressions on 0.916 pipeline |
| C | Domain-Diverse Training | Reproducible command produces comparative artifact |
| D | Baselines + Ablations | Matrix report: HEMOC wins/ties/loses |
| E | Unify Python + Visual System | Python output consumed by visual system |
| F | Documentation Reconciliation | New contributor finds truth in <5 minutes |

## Phase Detection Logic

### Phase A — Benchmark Contract (DONE on hemoc-visual-system-init branch)

Check for existence:
```bash
cd /home/user/HEMOC-Stain-Glass-Flower
git show remotes/origin/hemoc-visual-system-init-4592349023335104422:docs/benchmark_contract.md &>/dev/null && echo "EXISTS" || echo "MISSING"
git show remotes/origin/hemoc-visual-system-init-4592349023335104422:scripts/validate_results_schema.py &>/dev/null && echo "EXISTS" || echo "MISSING"
```

**Status**: DONE if both files exist on `hemoc-visual-system-init-*`.

### Phase B — Protect Working Pipeline

Check for test files in HEMOC repo (any branch):
```bash
cd /home/user/HEMOC-Stain-Glass-Flower
# Required test files (from CLAUDE.md / TESTING_REVIEW_AND_DEV_TRACK.md):
test -f tests/unit/test_hybrid_encoder.py && echo "EXISTS" || echo "MISSING"
test -f tests/unit/test_cnn_decoder.py && echo "EXISTS" || echo "MISSING"
test -f tests/integration/test_encode_decode_roundtrip.py && echo "EXISTS" || echo "MISSING"
```

Also check if existing tests cover the WORKING pipeline:
```bash
grep -rl "HybridEncoder" tests/ 2>/dev/null | wc -l    # Should be > 0
grep -rl "AngleRecoveryCNN" tests/ 2>/dev/null | wc -l  # Should be > 0
grep -rl "OpticalKirigamiEncoder" tests/ 2>/dev/null | wc -l  # OLD encoder (doesn't count)
```

**Status**: DONE when all 3 test files exist AND `pytest tests/unit/ && pytest tests/integration/` passes.

### Phase C — Domain-Diverse Training

```bash
cd /home/user/HEMOC-Stain-Glass-Flower
# Code exists?
test -f demos/domain_diverse_training.py && echo "CODE EXISTS" || echo "CODE MISSING"
# Results exist?
ls results/domain_diverse_*.json 2>/dev/null | wc -l
```

**Status**: DONE when result artifacts exist with valid schema.
**Blocker**: Requires GPU. Phase B must pass first. See `docs/GPU_CLOUD_GUIDE.md`.

### Phase D — Baselines + Ablations

```bash
cd /home/user/HEMOC-Stain-Glass-Flower
# Baseline results
ls results/baseline_direct_mlp_*.json 2>/dev/null | wc -l
ls results/baseline_random_pattern_*.json 2>/dev/null | wc -l
# Ablation results
ls results/ablation_*.json 2>/dev/null | wc -l
```

Required baselines (from CLAUDE.md):
- Direct-feature MLP (ceiling test)
- Random pattern control (floor test)
- Simple visual baseline

Required ablations:
- Hypercube path on/off
- Golden-ratio vs alternatives (sqrt(2), e, learned)
- Resolution sweep (32x32, 64x64, 128x128)

**Status**: DONE when at least 2 baseline + 1 ablation result files exist with valid schema.

### Phase E — Unify Python + Visual System

```bash
cd /home/user/HEMOC-Stain-Glass-Flower
# Visual system scaffold
test -d hemoc-visual-system/src && echo "SRC EXISTS" || echo "SRC MISSING"
# Key TypeScript files
test -f hemoc-visual-system/src/PhillipsMatrix.ts && echo "MATRIX EXISTS" || echo "MATRIX MISSING"
test -f hemoc-visual-system/src/E8ToUniforms.ts && echo "UNIFORMS EXISTS" || echo "UNIFORMS MISSING"
# Line count
find hemoc-visual-system/src -name '*.ts' -exec cat {} + 2>/dev/null | wc -l
```

Also check for the shared interchange schema (from CLAUDE.md Phase E spec):
```bash
grep -r "input_state.*e8_vector" hemoc-visual-system/ 2>/dev/null | wc -l
```

**Status**: DONE when PhillipsMatrix.ts exists AND >500 lines of TypeScript AND Python output is consumed.

### Phase F — Documentation Reconciliation

```bash
cd /home/user/HEMOC-Stain-Glass-Flower
test -f docs/STATUS_SINGLE_SOURCE.md && echo "EXISTS" || echo "MISSING"
```

If exists, check for the three-table structure:
```bash
grep -c "Claims Table\|Evidence Table\|Reproducibility Table" docs/STATUS_SINGLE_SOURCE.md 2>/dev/null
```

**Status**: DONE when `STATUS_SINGLE_SOURCE.md` exists with all three tables.

## Output Format

```
## Recovery Plan Progress — [DATE]

| Phase | Status | Evidence | Blocker |
|-------|--------|----------|---------|
| A — Benchmark Contract | DONE | benchmark_contract.md + validator on visual-init branch | — |
| B — Pipeline Tests | TODO | 0/3 required test files exist | Write tests from TESTING_REVIEW specs |
| C — Domain-Diverse | BLOCKED | Code exists at demos/domain_diverse_training.py | GPU required; Phase B first |
| D — Baselines | TODO | 0 baseline results, 0 ablation results | Phase B first |
| E — Visual System | IN PROGRESS | hemoc-visual-system/ exists, N lines TS | Port PhillipsMatrix.ts |
| F — Doc Reconciliation | TODO | No STATUS_SINGLE_SOURCE.md | Phases B-D inform the status doc |

### Dependency Graph
A ✓ → B → C (GPU) → D → F
            ↘ E (parallel with C/D)

### Recommended Next Action
[The single highest-priority action based on the dependency graph]
```

## Reference: Phase Exit Criteria (from CLAUDE.md)

- **Phase B**: `pytest` catches quality regressions on the working pipeline
- **Phase C**: Reproducible command produces comparative artifact proving transfer behavior
- **Phase D**: One matrix report showing where HEMOC wins, ties, or loses
- **Phase E**: Data produced by Python core consumed by visual system without patching
- **Phase F**: New contributor finds current truth in under 5 minutes
