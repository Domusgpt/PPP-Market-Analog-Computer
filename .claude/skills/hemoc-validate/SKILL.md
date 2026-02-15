---
name: hemoc-validate
description: Validate result artifacts, schema compliance, and documentation consistency across the HEMOC+PPP project. Use when checking if results are properly formatted, docs are accurate, or before merging changes.
allowed-tools: Read, Grep, Glob, Bash(git *), Bash(cd * && git *), Bash(python *), Bash(source *), Bash(test *), Bash(wc *), Bash(ls *)
---

# HEMOC Validation Suite

Run validation checks across both repositories. Covers result schema, documentation
consistency, code-doc alignment, and quality standard compliance.

## Repository Paths

```
PPP_ROOT=/home/user/PPP-Market-Analog-Computer
HEMOC_ROOT=/home/user/HEMOC-Stain-Glass-Flower
```

## Validation Checks

### 1. Result Schema Validation

Find all JSON result files:
```bash
cd $HEMOC_ROOT && find results/ -name "*.json" -type f 2>/dev/null
```

If `scripts/validate_results_schema.py` exists (hemoc-visual-system-init branch):
```bash
python scripts/validate_results_schema.py 2>/dev/null
```

Manual checks per result file:
- Required: `experiment.seed`, `config.encoder`, `results.per_angle_correlation`
- Correlations in [-1, 1] range
- `per_angle_correlation` has exactly 6 values
- `passing_angles` matches count above threshold
- Valid ISO 8601 timestamp
- `environment` section with python version

### 2. Code-Doc Alignment

| Claim (from CLAUDE.md) | Verification |
|------------------------|-------------|
| HybridEncoder in `demos/dual_decoder.py` | `grep -n "class HybridEncoder" $HEMOC_ROOT/demos/dual_decoder.py` |
| AngleRecoveryCNN in `demos/option_e_scaled_cnn.py` | `grep -n "class AngleRecoveryCNN" $HEMOC_ROOT/demos/option_e_scaled_cnn.py` |
| Pipeline error at `pipeline.py:33` | `grep -n "from .rules.enforcer" $PPP_ROOT/_SYNERGIZED_SYSTEM/backend/engine/pipeline.py` |
| 38 dev sessions | `grep -c "^## Session" $PPP_ROOT/DEV_TRACK.md` |
| `src/` doesn't exist | `test -d $HEMOC_ROOT/src && echo "EXISTS" \|\| echo "CONFIRMED: no src/"` |
| Domain-diverse never run | `ls $HEMOC_ROOT/results/domain_diverse_*.json 2>/dev/null \| wc -l` |

### 3. Known Contradiction Verification

Check that 3 known contradictions (from CLAUDE.md) are still present:

```bash
# 1. MUSICGEOMETRY_GAP_ANALYSIS claims 0.73 transfer
grep -n "0.73" $HEMOC_ROOT/docs/MUSICGEOMETRY_GAP_ANALYSIS.md 2>/dev/null

# 2. PROJECT_STATUS lists src/ paths
grep -n "src/" $HEMOC_ROOT/docs/PROJECT_STATUS.md 2>/dev/null | head -5

# 3. FINAL_ANALYSIS_REPORT references src/core/plastic_encoder
grep -n "src/core/plastic_encoder" $HEMOC_ROOT/FINAL_ANALYSIS_REPORT.md 2>/dev/null
```

Flag if any known contradiction is FIXED (update CLAUDE.md).
Flag any NEW contradictions found.

### 4. Test Coverage Audit

```bash
cd $HEMOC_ROOT
echo "=== Test files ==="
find tests/ -name "test_*.py" -o -name "*_test.py" 2>/dev/null

echo "=== Working pipeline coverage ==="
echo "HybridEncoder: $(grep -rl 'HybridEncoder' tests/ 2>/dev/null | wc -l) files"
echo "AngleRecoveryCNN: $(grep -rl 'AngleRecoveryCNN' tests/ 2>/dev/null | wc -l) files"
echo "OLD OpticalKirigamiEncoder: $(grep -rl 'OpticalKirigamiEncoder' tests/ 2>/dev/null | wc -l) files"
```

### 5. Quality Standards Checklist (from CLAUDE.md)

| Standard | Check |
|----------|-------|
| Evidence before claims | All KEY_FINDINGS entries have references |
| Deterministic seeds | Result files contain `experiment.seed` |
| Schema compliance | All result JSONs pass validation |
| Regression protection | `pytest tests/unit/` passes |
| Academic rigor | Conjectures labeled, not stated as fact |
| Reproducibility | Env reports and seeds recorded |

### 6. Incremental Validation

For recent changes only:
```bash
git diff --name-only HEAD~1       # Since last commit
git diff --name-only main...HEAD  # Since branching from main
```

Only validate changed files and their doc references.

### 7. Branch-Specific File Check

Files only on `hemoc-visual-system-init-4592349023335104422`:
- `docs/benchmark_contract.md`
- `scripts/validate_results_schema.py`
- `docs/HEMOC_ARCHITECTURE_DETAILED.md`
- `docs/ONTOLOGY_BLUEPRINT_ICE_ECT_DRAFT.md`

If on a different branch, report as "branch-dependent" not "MISSING".

## Output Format

```
## Validation Report — [DATE]

### Result Schema
| File | Valid | Issues |
|------|-------|--------|
| results/... | YES/NO | ... |

### Code-Doc Alignment
| Claim | Verified | Notes |
|-------|----------|-------|
| HybridEncoder location | YES/NO | ... |
| Pipeline import error | YES/NO | still broken / fixed |
| Dev session count | YES/NO | expected 38, found N |

### Known Contradictions
| # | Description | Still Present? |
|---|-------------|---------------|
| 1 | Domain-diverse "0.73" | YES/NO |
| 2 | src/ paths in PROJECT_STATUS | YES/NO |
| 3 | plastic_encoder.py reference | YES/NO |

New contradictions: [none / list]

### Test Coverage
- Working pipeline tests: N files
- OLD encoder tests: N files
- Phase B files present: N/3

### Quality Standards
| Standard | Status |
|----------|--------|
| Evidence before claims | PASS/FAIL |
| Deterministic seeds | PASS/FAIL |
| Schema compliance | PASS/FAIL |
| Regression protection | PASS/FAIL |

### Overall: PASS / WARN / FAIL
```
