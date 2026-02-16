---
name: hemoc-status
description: Check the health and status of the HEMOC+PPP cross-repo project. Use when the user wants a project overview, needs to know what's working, or before starting development work.
allowed-tools: Read, Grep, Glob, Bash(git *), Bash(cd * && git *), Bash(source * && pytest *), Bash(npm test*), Bash(npm run test*), Bash(python *), Bash(wc *), Bash(gh *), Bash(test *)
---

# HEMOC + PPP Project Status Check

Run a comprehensive project health check across both repositories.

## Repository Paths

| Repo | Root | Python venv | Node modules |
|------|------|-------------|-------------|
| PPP | `/home/user/PPP-Market-Analog-Computer` | `_SYNERGIZED_SYSTEM/backend/.venv` | `node_modules/` |
| HEMOC | `/home/user/HEMOC-Stain-Glass-Flower` | `.venv` | `hemoc-visual-system/node_modules/` |

## Steps

### 1. Repository State

Check both repos for branch, clean working tree, and remote sync:
```bash
# PPP
cd /home/user/PPP-Market-Analog-Computer && git branch --show-current && git status --short

# HEMOC
cd /home/user/HEMOC-Stain-Glass-Flower && git branch --show-current && git status --short
```

### 2. Test Suite Health

Run smoke tests for both repos:

**PPP Python** (from backend):
```bash
cd /home/user/PPP-Market-Analog-Computer/_SYNERGIZED_SYSTEM/backend
source .venv/bin/activate && pytest tests/ -q --tb=line 2>&1
```

**PPP Node**:
```bash
cd /home/user/PPP-Market-Analog-Computer && npm run test:all 2>&1
```

**HEMOC Python**:
```bash
cd /home/user/HEMOC-Stain-Glass-Flower
source .venv/bin/activate && pytest tests/unit/ -q --tb=line 2>&1
```

### 3. Recovery Plan Phase Tracking

Check concrete evidence for each phase (A-F):

| Phase | Check | Command |
|-------|-------|---------|
| A (Benchmark contract) | Validator exists on visual-init branch | `git show remotes/origin/hemoc-visual-system-init-4592349023335104422:scripts/validate_results_schema.py &>/dev/null` |
| B (Pipeline tests) | 3 required test files exist | `test -f tests/unit/test_hybrid_encoder.py` in HEMOC |
| C (Domain-diverse) | Result artifacts exist | `ls results/domain_diverse_*.json 2>/dev/null` in HEMOC |
| D (Baselines) | Baseline + ablation results | `ls results/baseline_*.json 2>/dev/null` in HEMOC |
| E (Visual system) | TypeScript source exists | `test -d hemoc-visual-system/src` in HEMOC |
| F (Doc reconciliation) | STATUS_SINGLE_SOURCE.md exists | `test -f docs/STATUS_SINGLE_SOURCE.md` in HEMOC |

### 4. HEMOC Branch & PR Overview

```bash
cd /home/user/HEMOC-Stain-Glass-Flower
git branch -a --format='%(refname:short)' | wc -l
git for-each-ref --sort=-committerdate refs/remotes/origin/ \
  --format='%(committerdate:relative) %(refname:short)' | head -5
```

If `gh` CLI available:
```bash
gh pr list --state open --limit 5 --json number,title,headRefName 2>/dev/null
```

### 5. Dev Tooling Versions

Check key tools in HEMOC venv:
```bash
source /home/user/HEMOC-Stain-Glass-Flower/.venv/bin/activate
python --version
python -c "import torch; print(f'torch {torch.__version__}')" 2>/dev/null || echo "torch: not installed"
python -c "import numpy; print(f'numpy {numpy.__version__}')" 2>/dev/null
black --version 2>/dev/null | head -1 || echo "black: not installed"
```

System tools:
```bash
gh --version 2>&1 | head -1 || echo "gh: not installed"
docker --version 2>&1 || echo "docker: not installed"
jq --version 2>&1 || echo "jq: not installed"
```

### 6. Critical File Integrity

Verify key files exist and are non-empty:

**PPP**: `CLAUDE.md`, `WHAT_THIS_DOES.md`, `CODING_AGENT_PROMPT.md`, `scripts/env-setup.sh`, `docs/HEMOC_PPP_CROSS_REPO_DIGEST.md`

**HEMOC**: `demos/dual_decoder.py` (HybridEncoder), `demos/option_e_scaled_cnn.py` (AngleRecoveryCNN), `demos/domain_diverse_training.py` (Phase C code), `main.py` (OLD encoder)

## Output Format

```
## Project Status Report — [DATE]

### Repository State
| Repo | Branch | Clean | Remote Sync |
|------|--------|-------|-------------|
| PPP | ... | YES/NO | UP TO DATE/BEHIND |
| HEMOC | ... | YES/NO | UP TO DATE/BEHIND |

### Test Health
| Suite | Passed | Failed | Warnings |
|-------|--------|--------|----------|
| PPP Python | ... | ... | ... |
| PPP Node | ... | ... | ... |
| HEMOC Python | ... | ... | ... |

### Recovery Plan Progress
| Phase | Status | Evidence |
|-------|--------|----------|
| A — Benchmark Contract | DONE/TODO | ... |
| B — Pipeline Tests | DONE/TODO | ... |
| C — Domain-Diverse | DONE/TODO/BLOCKED | ... |
| D — Baselines | DONE/TODO | ... |
| E — Visual System | DONE/TODO/IN PROGRESS | ... |
| F — Doc Reconciliation | DONE/TODO | ... |

### Dev Tooling
| Tool | Version | Status |
|------|---------|--------|
| Python | ... | OK |
| Node | ... | OK |
| torch | ... | installed/missing |
| black | ... | installed/missing |
| gh CLI | ... | installed/missing |
| docker | ... | installed/missing |

### HEMOC Branches
- Total: N branches
- Most recent: [branch] ([date])
- Open PRs: N (if gh available)

### Critical Files: [All OK / N missing]

### Recommended Next Action
[Single highest-priority action based on Recovery Plan dependency graph:
A ✓ → B → C (GPU) → D → F, with E parallel to C/D]
```
