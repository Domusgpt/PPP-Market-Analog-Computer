---
name: hemoc-env
description: Verify or set up the HEMOC+PPP development environment. Use when encountering import errors, missing tools, or starting a fresh development session.
argument-hint: "[check|setup|diagnose <tool>]"
allowed-tools: Read, Grep, Glob, Bash(bash *), Bash(source *), Bash(pip *), Bash(python *), Bash(npm *), Bash(which *), Bash(command *)
---

# HEMOC + PPP Environment Manager

Verify, set up, or diagnose the cross-repo development environment based on `$ARGUMENTS`.
Default mode is `check` if no argument given.

## Repository Paths

| Repo | Root | Python venv | Node modules |
|------|------|-------------|-------------|
| PPP | `/home/user/PPP-Market-Analog-Computer` | `_SYNERGIZED_SYSTEM/backend/.venv` | `node_modules/` |
| HEMOC | `/home/user/HEMOC-Stain-Glass-Flower` | `.venv` | `hemoc-visual-system/node_modules/` |

## Modes

### `check` (default) — Verify Environment Health

Run `bash scripts/env-setup.sh` from the PPP root and present the summary.
Additionally check:

1. **Venv freshness**: Compare `pip freeze` output against `requirements.txt` / `pyproject.toml`
2. **Node staleness**: Check if `node_modules/.package-lock.json` is newer than `package.json`
3. **Git state**: Both repos clean? On expected branches?

### `setup` — Full Bootstrap

Run the env-setup script, then verify everything passes:
```bash
bash /home/user/PPP-Market-Analog-Computer/scripts/env-setup.sh
```

If issues arise:
- PPP `pip install -e ".[dev]"` fails → Known build backend issue, fallback to direct deps
- HEMOC `requirements.txt` missing torch → Expected, torch is optional until GPU work
- `hemoc-visual-system/` missing → Only exists on `hemoc-visual-system-init-*` branches

### `diagnose <tool>` — Deep-Dive on a Specific Tool

For the named tool, report:

| Tool | Check | Fix |
|------|-------|-----|
| `python` | Version, venv locations, sys.path | `python3 -m venv .venv && source .venv/bin/activate` |
| `torch` | Import test, CUDA available, version | `pip install torch` (CPU) or `pip install torch --index-url https://download.pytorch.org/whl/cu121` |
| `gh` | `gh auth status`, version | `https://cli.github.com/` → `gh auth login` |
| `docker` | `docker info`, compose plugin | Install Docker Desktop or `apt install docker.io` |
| `black` | In which venv, version | `pip install black` in active venv |
| `wandb` | Login status, version | `pip install wandb && wandb login` |
| `mkdocs` | Version, can build HEMOC docs | `pip install mkdocs mkdocs-material` |
| `node` | Version, npm version | Install via nvm or package manager |

**For PyTorch specifically**, also report which scripts need it:
- `demos/option_e_scaled_cnn.py` (AngleRecoveryCNN) — REQUIRES torch
- `demos/domain_diverse_training.py` — REQUIRES torch
- `demos/dual_decoder.py` (HybridEncoder) — numpy only, NO torch needed
- `deploy/run_scaling_job.sh` — REQUIRES torch + CUDA for distributed training

## Tool Tier Reference

| Tier | Tools | Consequence if Missing |
|------|-------|----------------------|
| REQUIRED | python3 >=3.10, node >=18, git | Script fails |
| RECOMMENDED | gh CLI, jq, black, isort, flake8 | Warnings; reduced CI compatibility |
| OPTIONAL | docker, firebase, wandb, mkdocs, soundfile | Info only; needed for specific tasks |
| DEFERRED | torch, gcloud, runpodctl, vastai | Install per task; see `docs/GPU_CLOUD_GUIDE.md` |

## Output Format

```
## Environment Report

| Component | Status | Version | Location |
|-----------|--------|---------|----------|
| Python | OK | 3.11.x | /usr/bin/python3 |
| PPP venv | OK/STALE/MISSING | — | backend/.venv |
| HEMOC venv | OK/STALE/MISSING | — | ../HEMOC/.venv |
| Node | OK | 20.x | /usr/bin/node |
| gh CLI | OK/MISSING | 2.x | /usr/bin/gh |
| Docker | OK/MISSING | 24.x | /usr/bin/docker |
| torch | INSTALLED/MISSING | 2.x | in HEMOC venv |
| ... | ... | ... | ... |

### Issues Found
[List any problems with fix commands]

### Recommendation
[Next action to take]
```

## Known Pitfalls

- PPP `pyproject.toml` uses `setuptools.backends._legacy:_Backend` which doesn't exist.
  Env-setup.sh has a fallback for this. If you see "editable install failed", that's expected.
- HEMOC `pyproject.toml` on main lists torch as a core dep. On `hemoc-visual-system-init` it doesn't.
  Always check which branch you're on before interpreting dependency errors.
- `.venv/` is in `.gitignore`. Don't commit venvs.
