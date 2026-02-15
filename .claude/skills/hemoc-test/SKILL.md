---
name: hemoc-test
description: Run test suites across HEMOC and PPP repositories. Use when running tests, checking for regressions, or verifying changes before commit.
argument-hint: "[all|ppp|hemoc|ppp-python|ppp-node|hemoc-python|hemoc-integration]"
disable-model-invocation: true
allowed-tools: Bash(source * && pytest *), Bash(source * && python *), Bash(npm test*), Bash(npm run test*), Bash(cd * && *), Bash(test *), Bash(wc *), Read, Grep
---

# HEMOC + PPP Test Runner

Run test suites based on `$ARGUMENTS`. Default is `all` if no argument given.

## Repository Paths

```
PPP_ROOT=/home/user/PPP-Market-Analog-Computer
PPP_BACKEND=$PPP_ROOT/_SYNERGIZED_SYSTEM/backend
HEMOC_ROOT=/home/user/HEMOC-Stain-Glass-Flower
```

## Pre-Test Checks

Before running any suite, verify the venv is healthy:
```bash
test -f $PPP_BACKEND/.venv/bin/pytest || echo "PPP venv missing pytest — run: bash scripts/env-setup.sh"
test -f $HEMOC_ROOT/.venv/bin/pytest || echo "HEMOC venv missing pytest — run: bash scripts/env-setup.sh"
```

## Test Suites

### `ppp-python` — PPP Python Backend (20 test files)

```bash
cd $PPP_BACKEND && source .venv/bin/activate
pytest tests/ -v --tb=short
```

Key tests: `test_h4_geometry.py`, `test_phillips_matrix.py`, `test_quaternion.py`,
`test_moire_physics.py`, `test_kirigami_sheet.py`, `test_enforcer.py`

### `ppp-node` — PPP Node Frontend

```bash
cd $PPP_ROOT && npm run test:all
```

Runs: `node --test tests/*.test.js` + phase-lock + adapter + bridge TypeScript tests.

### `hemoc-python` — HEMOC Python Unit Tests

```bash
cd $HEMOC_ROOT && source .venv/bin/activate
pytest tests/unit/ -v --tb=short
```

Note: Some tests require PyTorch. Tests importing torch show as WARNINGS if not installed.

### `hemoc-integration` — HEMOC Integration Tests

```bash
cd $HEMOC_ROOT && source .venv/bin/activate
pytest tests/integration/ -v --tb=short 2>/dev/null || echo "No integration tests found"
```

### `ppp` — All PPP (Python + Node)
### `hemoc` — All HEMOC (Unit + Integration)
### `all` — Everything (all suites in order)

## Coverage Support

If coverage requested, add flags:
```bash
# HEMOC (matches CI: pytest tests/unit/ -v --cov=src --cov-report=xml)
pytest tests/unit/ -v --cov=src --cov-report=term-missing

# PPP
pytest tests/ -v --cov=engine --cov-report=term-missing
```

## Phase B Tracking

After running tests, check Recovery Plan Phase B status:

```bash
cd $HEMOC_ROOT
echo "--- Phase B Test File Check ---"
test -f tests/unit/test_hybrid_encoder.py && echo "test_hybrid_encoder.py: EXISTS" || echo "test_hybrid_encoder.py: MISSING"
test -f tests/unit/test_cnn_decoder.py && echo "test_cnn_decoder.py: EXISTS" || echo "test_cnn_decoder.py: MISSING"
test -f tests/integration/test_encode_decode_roundtrip.py && echo "test_encode_decode_roundtrip.py: EXISTS" || echo "test_encode_decode_roundtrip.py: MISSING"

echo "--- Pipeline Coverage ---"
echo "Tests targeting HybridEncoder: $(grep -rl 'HybridEncoder' tests/ 2>/dev/null | wc -l) files"
echo "Tests targeting AngleRecoveryCNN: $(grep -rl 'AngleRecoveryCNN' tests/ 2>/dev/null | wc -l) files"
echo "Tests targeting OLD OpticalKirigamiEncoder: $(grep -rl 'OpticalKirigamiEncoder' tests/ 2>/dev/null | wc -l) files"
```

Flag if tests only target the OLD encoder and not the WORKING pipeline.

## Output Format

```
## Test Results — [DATE]

| Suite | Passed | Failed | Warnings | Duration |
|-------|--------|--------|----------|----------|
| PPP Python | ... | ... | ... | ...s |
| PPP Node | ... | ... | ... | ...s |
| HEMOC Python | ... | ... | ... | ...s |
| HEMOC Integration | ... | ... | ... | ...s |
| **Total** | ... | ... | ... | ...s |

### Failures (if any)
[Details with file:line references]

### Phase B Status
- test_hybrid_encoder.py: [EXISTS/MISSING]
- test_cnn_decoder.py: [EXISTS/MISSING]
- test_encode_decode_roundtrip.py: [EXISTS/MISSING]
- Tests targeting working pipeline: N files
- Tests targeting OLD encoder: N files

### Verdict: [PASS | FAIL | WARN]
```
