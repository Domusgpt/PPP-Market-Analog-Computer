---
name: hemoc-dev
description: Development workflow helpers for linting, formatting, and pre-commit checks across the HEMOC+PPP project.
argument-hint: "[lint|format|check|pre-push]"
disable-model-invocation: true
allowed-tools: Read, Grep, Glob, Bash(source * && black *), Bash(source * && isort *), Bash(source * && flake8 *), Bash(source * && pytest *), Bash(git *), Bash(cd * && *), Bash(npm *)
---

# HEMOC + PPP Development Workflow

Run development checks based on `$ARGUMENTS`. Default is `check` if no argument given.

## Repository Paths

```
PPP_ROOT=/home/user/PPP-Market-Analog-Computer
PPP_BACKEND=$PPP_ROOT/_SYNERGIZED_SYSTEM/backend
HEMOC_ROOT=/home/user/HEMOC-Stain-Glass-Flower
```

## Modes

### `lint` — Check Code Style (No Modifications)

Run linters on changed files in both repos. Match the CI configuration.

**PPP Python** (from backend venv):
```bash
cd $PPP_BACKEND && source .venv/bin/activate
black --check engine/ tests/ 2>&1
isort --check-only engine/ tests/ 2>&1
flake8 engine/ --max-line-length=100 --ignore=E501,W503 2>&1
```

**HEMOC Python** (from HEMOC venv):
```bash
cd $HEMOC_ROOT && source .venv/bin/activate
black --check demos/ tests/ main.py 2>&1
isort --check-only demos/ tests/ 2>&1
flake8 demos/ --max-line-length=100 --ignore=E501,W503 2>&1
```

Note: HEMOC CI uses `|| true` on lint checks (soft-fail). Report issues but don't treat as blocking.

### `format` — Auto-Fix Formatting

Run formatters to fix style issues:
```bash
# PPP
cd $PPP_BACKEND && source .venv/bin/activate
black engine/ tests/
isort engine/ tests/

# HEMOC
cd $HEMOC_ROOT && source .venv/bin/activate
black demos/ tests/ main.py
isort demos/ tests/
```

Report which files were changed.

### `check` — Pre-Commit Verification

Run a comprehensive pre-commit check:

1. **Lint check** (run `lint` mode)
2. **Staged file audit**:
   ```bash
   git diff --name-only --cached
   ```
   Flag if any of these are staged:
   - `.env`, `.env.*` — credentials
   - `*.pem`, `*.key`, `*.p12` — certificates
   - Files matching `*secret*`, `*credential*`, `*password*`
   - Files > 5MB (large binaries)
   - `__pycache__/`, `.pyc` files
   - `.venv/`, `node_modules/`

3. **Smoke test** (quick — just import checks):
   ```bash
   # PPP
   source $PPP_BACKEND/.venv/bin/activate && python -c "import numpy; import scipy" 2>&1

   # HEMOC
   source $HEMOC_ROOT/.venv/bin/activate && python -c "import numpy; import scipy" 2>&1
   ```

4. **Report**: PASS (safe to commit) or FAIL (fix issues first)

### `pre-push` — Full Pre-Push Validation

Everything from `check` plus full test suites:

```bash
# PPP Python tests
cd $PPP_BACKEND && source .venv/bin/activate && pytest tests/ -q --tb=line

# PPP Node tests
cd $PPP_ROOT && npm run test:all

# HEMOC Python tests
cd $HEMOC_ROOT && source .venv/bin/activate && pytest tests/unit/ -q --tb=line
```

Also verify:
- Working tree is clean (`git status --short` shows nothing unexpected)
- Branch is up-to-date with remote (`git status -sb` shows no behind count)

## Output Format

```
## Dev Check: [lint|format|check|pre-push]

### Lint Results
| Repo | Tool | Status | Issues |
|------|------|--------|--------|
| PPP | black | PASS/FAIL | N files need formatting |
| PPP | isort | PASS/FAIL | N import order issues |
| PPP | flake8 | PASS/FAIL | N warnings |
| HEMOC | black | PASS/FAIL | N files need formatting |
| HEMOC | isort | PASS/FAIL | N import order issues |
| HEMOC | flake8 | PASS/FAIL | N warnings |

### Staged File Audit
[Clean / N issues found]

### Tests (pre-push only)
| Suite | Status | Details |
|-------|--------|---------|
| PPP Python | PASS/FAIL | X passed, Y failed |
| PPP Node | PASS/FAIL | X passed |
| HEMOC Python | PASS/FAIL | X passed, Y warnings |

### CI Compatibility
These changes would [PASS/FAIL] the HEMOC CI workflow.
These changes would [PASS/FAIL] the PPP CI workflow.

### Verdict: [SAFE TO COMMIT / FIX ISSUES FIRST]
```

## HEMOC CI Reference

From `.github/workflows/ci.yml`:
- Python matrix: 3.8, 3.9, 3.10, 3.11
- Lint: `black --check`, `isort --check-only`, `flake8` (all soft-fail with `|| true`)
- Tests: `pytest tests/unit/ -v --cov=src`, `pytest tests/integration/ -v`
- Benchmark: `python tests/benchmarks/benchmark_encoder.py` (only if tests pass)
