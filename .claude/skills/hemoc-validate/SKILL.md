---
name: hemoc-validate
description: Validate result artifacts, schema compliance, and documentation consistency across the HEMOC+PPP project. Use when checking if results are properly formatted, docs are accurate, or before merging changes.
allowed-tools: Read, Grep, Glob, Bash(git *), Bash(cd * && git *), Bash(python *), Bash(source *)
---

# HEMOC Validation Suite

Run validation checks across both repositories. This covers result schema, documentation consistency, and code-doc alignment.

## Validation Checks

### 1. Result Schema Validation

Find all JSON result files and validate against the benchmark contract:

```bash
cd /home/user/HEMOC-Stain-Glass-Flower
# Find all result files
find results/ -name "*.json" -type f 2>/dev/null

# If validate_results_schema.py exists (check hemoc-visual-system-init branch):
python scripts/validate_results_schema.py 2>/dev/null
```

For each result file, verify:
- Has required fields: `experiment.seed`, `config.encoder`, `results.per_angle_correlation`
- Correlation values are in [-1, 1] range
- `per_angle_correlation` has exactly 6 values
- `passing_angles` count matches angles above threshold
- Timestamp is valid ISO 8601

### 2. Code-Doc Alignment

Check that claims in docs match actual code:

| Claim | Where to verify |
|-------|----------------|
| "HybridEncoder in demos/dual_decoder.py" | `grep -n "class HybridEncoder" demos/dual_decoder.py` |
| "AngleRecoveryCNN in demos/option_e_scaled_cnn.py" | `grep -n "class AngleRecoveryCNN" demos/option_e_scaled_cnn.py` |
| "0 collisions across 1000 samples" | Check if test exists in `tests/` |
| "0.916 correlation" | Verify in result artifacts |
| "V4 Plastic Encoder winner" | Check `FINAL_ANALYSIS_REPORT.md` numbers |

### 3. Known Contradiction Check

Verify the known contradictions documented in `CLAUDE.md` section "Known Doc Contradictions":
- MUSICGEOMETRY_GAP_ANALYSIS domain-diverse claim (should still say "done, 0.73" — it's wrong but documented)
- PROJECT_STATUS src/ paths (should still reference non-existent paths — wrong but documented)
- FINAL_ANALYSIS_REPORT plastic_encoder location (wrong but documented)

Flag if any NEW contradictions are found that aren't already documented.

### 4. Test Coverage Check

```bash
# HEMOC: What tests exist?
find /home/user/HEMOC-Stain-Glass-Flower/tests/ -name "test_*.py" -o -name "*_test.py" 2>/dev/null

# Do they test the WORKING pipeline or the OLD encoder?
grep -l "HybridEncoder\|AngleRecoveryCNN" /home/user/HEMOC-Stain-Glass-Flower/tests/**/*.py 2>/dev/null
grep -l "OpticalKirigamiEncoder" /home/user/HEMOC-Stain-Glass-Flower/tests/**/*.py 2>/dev/null
```

### 5. PPP Pipeline Check

```bash
# Check the known import error
grep -n "from .rules.enforcer" /home/user/PPP-Market-Analog-Computer/_SYNERGIZED_SYSTEM/backend/engine/pipeline.py
# Should still be broken (from .rules.enforcer instead of from .enforcer)
```

### 6. Output Format

```
## Validation Report

### Result Schema
| File | Valid | Issues |
|------|-------|--------|
| ... | YES/NO | ... |

### Code-Doc Alignment
| Claim | Verified | Notes |
|-------|----------|-------|
| ... | YES/NO | ... |

### Known Contradictions
All previously documented contradictions: [still present / fixed]
New contradictions found: [none / list]

### Test Coverage
- Tests targeting working pipeline (HybridEncoder/AngleRecoveryCNN): N
- Tests targeting old encoder (OpticalKirigamiEncoder): N
- Gap: [description]

### PPP Pipeline
- Import error status: [still broken / fixed]

### Overall: PASS / WARN / FAIL
```
