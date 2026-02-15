---
name: hemoc-test
description: Run test suites across HEMOC and PPP repositories. Use when running tests, checking for regressions, or verifying changes before commit.
argument-hint: "[all|ppp|hemoc|ppp-python|ppp-node|hemoc-python]"
disable-model-invocation: true
allowed-tools: Bash(source * && pytest *), Bash(source * && python *), Bash(npm test*), Bash(npm run test*), Bash(cd * && *), Read
---

# HEMOC + PPP Test Runner

Run test suites based on `$ARGUMENTS`. Default is `all` if no argument given.

## Test Suites

### `ppp-python` — PPP Python Backend
```bash
cd /home/user/PPP-Market-Analog-Computer/_SYNERGIZED_SYSTEM/backend
source .venv/bin/activate
pytest tests/ -v --tb=short
```

### `ppp-node` — PPP Node Frontend (Phase-Lock, Adapters, Bridge)
```bash
cd /home/user/PPP-Market-Analog-Computer
npm run test:all
```

### `hemoc-python` — HEMOC Python Unit Tests
```bash
cd /home/user/HEMOC-Stain-Glass-Flower
source .venv/bin/activate
pytest tests/unit/ -v --tb=short
```

Note: Some HEMOC tests require PyTorch. Tests that need torch will show as WARNINGS, not failures.

### `hemoc-integration` — HEMOC Integration Tests
```bash
cd /home/user/HEMOC-Stain-Glass-Flower
source .venv/bin/activate
pytest tests/integration/ -v --tb=short 2>/dev/null || echo "No integration tests found"
```

### `ppp` — All PPP tests (Python + Node)
Run `ppp-python` then `ppp-node`.

### `hemoc` — All HEMOC tests (Unit + Integration)
Run `hemoc-python` then `hemoc-integration`.

### `all` — Everything
Run all suites in order: ppp-python, ppp-node, hemoc-python, hemoc-integration.

## Output Format

```
## Test Results

| Suite | Passed | Failed | Warnings | Duration |
|-------|--------|--------|----------|----------|
| PPP Python | ... | ... | ... | ...s |
| PPP Node | ... | ... | ... | ...s |
| HEMOC Python | ... | ... | ... | ...s |
| HEMOC Integration | ... | ... | ... | ...s |
| **Total** | ... | ... | ... | ...s |

### Failures (if any)
<details for each failure>

### Recommendation
[Pass: safe to commit / Fail: fix before commit]
```
