---
name: hemoc-status
description: Check the health and status of the HEMOC+PPP cross-repo project. Use when the user wants a project overview, needs to know what's working, or before starting development work.
allowed-tools: Read, Grep, Glob, Bash(git *), Bash(cd * && git *), Bash(source * && pytest *), Bash(npm test*), Bash(npm run test*), Bash(python *)
---

# HEMOC + PPP Project Status Check

Run a comprehensive project health check across both repositories. Report findings in a structured table format.

## Steps

### 1. Repository State

Check both repos for branch, clean working tree, and remote sync:

```
PPP:   cd /home/user/PPP-Market-Analog-Computer && git branch --show-current && git status --short
HEMOC: cd /home/user/HEMOC-Stain-Glass-Flower && git branch --show-current && git status --short
```

### 2. Test Suite Health

Run smoke tests for both repos and report pass/fail:

**PPP Python** (from `_SYNERGIZED_SYSTEM/backend`):
```
source .venv/bin/activate && pytest tests/ -q --tb=line
```

**PPP Node**:
```
npm run test:all
```

**HEMOC Python** (from HEMOC root):
```
source .venv/bin/activate && pytest tests/unit/ -q --tb=line
```

### 3. Recovery Plan Phase Tracking

Check the current state of each Recovery Plan phase. Read `CLAUDE.md` for phase definitions.
For each phase (A through F), determine status:
- Phase A (Benchmark contract): Check if `scripts/validate_results_schema.py` exists on current HEMOC branch
- Phase B (Pipeline tests): Check if `tests/unit/test_hybrid_encoder.py` exists in HEMOC
- Phase C (Domain-diverse): Check if `demos/domain_diverse_training.py` has any result artifacts
- Phase D (Baselines): Check if baseline result files exist
- Phase E (Visual system): Check if `hemoc-visual-system/` has substantial code
- Phase F (Doc reconciliation): Check if `docs/STATUS_SINGLE_SOURCE.md` exists

### 4. HEMOC Branch & PR Overview

```
cd /home/user/HEMOC-Stain-Glass-Flower && git branch -a --format='%(refname:short)' | wc -l
```

Note which branches are most recently updated.

### 5. Key File Integrity

Verify critical files exist and are non-empty:
- PPP: `CLAUDE.md`, `WHAT_THIS_DOES.md`, `CODING_AGENT_PROMPT.md`, `scripts/env-setup.sh`
- HEMOC: `demos/dual_decoder.py` (HybridEncoder), `demos/option_e_scaled_cnn.py` (AngleRecoveryCNN)

### 6. Output Format

Present results as:

```
## Project Status Report

| Area | Status | Details |
|------|--------|---------|
| PPP branch | ... | ... |
| PPP tests (Python) | PASS/FAIL/SKIP | X passed, Y failed |
| PPP tests (Node) | PASS/FAIL/SKIP | X passed |
| HEMOC branch | ... | ... |
| HEMOC tests | PASS/FAIL/SKIP | X passed, Y warnings |
| Recovery Phase A | DONE/TODO | ... |
| Recovery Phase B | DONE/TODO | ... |
| Recovery Phase C | DONE/TODO | ... |
| Recovery Phase D | DONE/TODO | ... |
| Recovery Phase E | DONE/TODO | ... |
| Recovery Phase F | DONE/TODO | ... |
| HEMOC branches | N total | ... |
| Critical files | OK/MISSING | ... |

### Recommended Next Action
Based on the status, suggest the single highest-priority next step.
```
