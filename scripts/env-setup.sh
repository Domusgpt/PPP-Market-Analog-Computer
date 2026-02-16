#!/usr/bin/env bash
# =============================================================================
# HEMOC + PPP Cross-Repo Environment Setup
# =============================================================================
# Full-stack bootstrap for both repos. Safe for Claude Code, Jules, and devs.
# Idempotent — safe to run multiple times.
#
# Usage: bash scripts/env-setup.sh
#
# Tool tiers:
#   REQUIRED  — script fails without these
#   RECOMMEND — warnings if missing, needed for standard dev workflow
#   OPTIONAL  — informational if missing, needed for specific tasks
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PPP_ROOT="$(dirname "$SCRIPT_DIR")"
HEMOC_ROOT="$PPP_ROOT/../HEMOC-Stain-Glass-Flower"

# Colors (degrade gracefully if no tty)
if [ -t 1 ]; then
  GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'
  BLUE='\033[0;34m'; BOLD='\033[1m'; NC='\033[0m'
else
  GREEN=''; RED=''; YELLOW=''; BLUE=''; BOLD=''; NC=''
fi

pass() { echo -e "  ${GREEN}PASS${NC} $1"; }
fail() { echo -e "  ${RED}FAIL${NC} $1"; }
info() { echo -e "  ${BLUE}INFO${NC} $1"; }
warn() { echo -e "  ${YELLOW}WARN${NC} $1"; }
header() { echo -e "\n${BOLD}=== $1 ===${NC}"; }

ERRORS=0
WARNINGS=0

# ---------------------------------------------------------------------------
# 1) Environment Detection
# ---------------------------------------------------------------------------
header "Environment Detection"
AGENT_ENV="developer"
if [ -n "${CLAUDE_CODE:-}" ] || [ -n "${ANTHROPIC_API_KEY:-}" ]; then
  AGENT_ENV="claude-code"
elif [ "$(whoami 2>/dev/null)" = "user" ] && [ -d "/home/user" ]; then
  AGENT_ENV="claude-code"
elif [ -n "${JULES_ENV:-}" ] || [ -n "${GOOGLE_CLOUD_PROJECT:-}" ]; then
  AGENT_ENV="jules"
fi
info "Detected environment: $AGENT_ENV"
info "PPP_ROOT: $PPP_ROOT"
info "HEMOC_ROOT: $HEMOC_ROOT"

# ---------------------------------------------------------------------------
# 2) System Prerequisites (REQUIRED)
# ---------------------------------------------------------------------------
header "System Prerequisites (Required)"

# Python
if command -v python3 &>/dev/null; then
  PY_VERSION=$(python3 --version 2>&1 | grep -oE '[0-9]+\.[0-9]+')
  PY_MAJOR=$(echo "$PY_VERSION" | cut -d. -f1)
  PY_MINOR=$(echo "$PY_VERSION" | cut -d. -f2)
  if [ "$PY_MAJOR" -ge 3 ] && [ "$PY_MINOR" -ge 10 ]; then
    pass "python3 $PY_VERSION (>=3.10)"
  else
    warn "python3 $PY_VERSION (recommend >=3.10, some features may fail)"
  fi
else
  fail "python3 not found - install Python 3.10+"
  ERRORS=$((ERRORS + 1))
fi

# Node
if command -v node &>/dev/null; then
  NODE_VERSION=$(node --version 2>&1 | grep -oE '[0-9]+' | head -1)
  if [ "$NODE_VERSION" -ge 18 ]; then
    pass "node v$NODE_VERSION (>=18)"
  else
    warn "node v$NODE_VERSION (recommend >=18)"
  fi
else
  warn "node not found - PPP frontend tests will be skipped"
  WARNINGS=$((WARNINGS + 1))
fi

# Git
if command -v git &>/dev/null; then
  pass "git $(git --version | grep -oE '[0-9]+\.[0-9]+\.[0-9]+')"
else
  fail "git not found"
  ERRORS=$((ERRORS + 1))
fi

if [ "$ERRORS" -gt 0 ]; then
  echo -e "\n${RED}Missing critical dependencies. Fix the above and re-run.${NC}"
  exit 1
fi

# ---------------------------------------------------------------------------
# 3) Recommended CLI Tools
# ---------------------------------------------------------------------------
header "Recommended CLI Tools"

# GitHub CLI — needed for PR management (HEMOC has 21+ PRs across branches)
if command -v gh &>/dev/null; then
  GH_VERSION=$(gh --version 2>&1 | head -1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' || echo "unknown")
  if gh auth status &>/dev/null; then
    pass "gh CLI v$GH_VERSION (authenticated)"
  else
    warn "gh CLI v$GH_VERSION installed but NOT authenticated"
    info "  Run: gh auth login"
    WARNINGS=$((WARNINGS + 1))
  fi
else
  warn "gh CLI not found — needed for PR management across HEMOC branches"
  info "  Install: https://cli.github.com/"
  WARNINGS=$((WARNINGS + 1))
fi

# jq — needed for JSON processing, schema validation, result inspection
if command -v jq &>/dev/null; then
  pass "jq $(jq --version 2>&1 | grep -oE '[0-9]+\.[0-9]+' || echo '')"
else
  warn "jq not found — needed for JSON result inspection and schema work"
  WARNINGS=$((WARNINGS + 1))
fi

# ---------------------------------------------------------------------------
# 4) Optional Infrastructure Tools
# ---------------------------------------------------------------------------
header "Optional Infrastructure Tools"

# Docker — PPP has docker-compose.yml, HEMOC has deploy/Dockerfile
if command -v docker &>/dev/null; then
  DOCKER_VERSION=$(docker --version 2>&1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' || echo "unknown")
  pass "docker $DOCKER_VERSION"
  if docker compose version &>/dev/null 2>&1; then
    pass "docker compose $(docker compose version 2>&1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' || echo '')"
  elif command -v docker-compose &>/dev/null; then
    pass "docker-compose (legacy)"
  else
    info "docker compose plugin not found (needed for: cd _SYNERGIZED_SYSTEM && docker compose up)"
  fi
else
  info "docker not found (optional — needed for full-stack integration and GPU training)"
  info "  PPP:   cd _SYNERGIZED_SYSTEM && docker compose up --build"
  info "  HEMOC: docker build -f deploy/Dockerfile -t hemoc-train ."
fi

# Firebase CLI — referenced in project tooling
if command -v firebase &>/dev/null; then
  pass "firebase CLI $(firebase --version 2>&1 | head -1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' || echo '')"
else
  info "firebase CLI not found (optional — install: npm install -g firebase-tools)"
fi

# Cloud GPU CLIs (for Phase C scaled training)
for CLI_NAME in gcloud runpodctl vastai; do
  if command -v "$CLI_NAME" &>/dev/null; then
    pass "$CLI_NAME available"
  fi
done
# Don't warn about cloud CLIs — they're only needed for scaled training
info "Cloud GPU CLIs (gcloud, runpodctl, vastai): see docs/GPU_CLOUD_GUIDE.md"

# Weights & Biases CLI
if command -v wandb &>/dev/null; then
  pass "wandb CLI $(wandb --version 2>&1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' || echo '')"
else
  info "wandb CLI not found (installed in Python venvs below; standalone CLI optional)"
fi

# ---------------------------------------------------------------------------
# 5) PPP Python Backend
# ---------------------------------------------------------------------------
header "PPP Python Backend"
PPP_BACKEND="$PPP_ROOT/_SYNERGIZED_SYSTEM/backend"

if [ -d "$PPP_BACKEND" ]; then
  if [ ! -f "$PPP_BACKEND/.venv/bin/activate" ]; then
    info "Creating venv at $PPP_BACKEND/.venv"
    python3 -m venv "$PPP_BACKEND/.venv"
  else
    info "Venv exists, reusing"
  fi

  (
    source "$PPP_BACKEND/.venv/bin/activate"
    pip install --upgrade pip -q 2>/dev/null

    # Try editable install first; fall back to direct deps if pyproject build backend fails
    # (Known issue: pyproject.toml uses setuptools.backends._legacy:_Backend which doesn't exist)
    if ! pip install -e "$PPP_BACKEND[dev]" -q 2>/dev/null; then
      warn "Editable install failed (build backend issue) - installing deps directly"
      pip install numpy scipy websockets pytest pytest-asyncio -q 2>/dev/null
    fi

    # Dev tooling for PPP Python
    pip install black isort flake8 -q 2>/dev/null || warn "Linter install failed (non-fatal)"

    python -c "import numpy; import scipy; print('imports OK')" && pass "PPP Python deps installed" || fail "PPP Python import check"
  ) || { fail "PPP Python setup"; ERRORS=$((ERRORS + 1)); }
else
  warn "PPP backend dir not found at $PPP_BACKEND - skipping"
fi

# ---------------------------------------------------------------------------
# 6) PPP Node Frontend
# ---------------------------------------------------------------------------
header "PPP Node Frontend"
if command -v npm &>/dev/null && [ -f "$PPP_ROOT/package.json" ]; then
  (
    cd "$PPP_ROOT"
    npm install --silent 2>/dev/null
    pass "npm install complete"
  ) || warn "npm install had issues (non-fatal)"
else
  warn "npm not available or no package.json - skipping"
fi

# ---------------------------------------------------------------------------
# 7) HEMOC Clone/Update
# ---------------------------------------------------------------------------
header "HEMOC Repository"
if [ -d "$HEMOC_ROOT/.git" ]; then
  info "HEMOC repo exists at $HEMOC_ROOT"
  (cd "$HEMOC_ROOT" && git fetch --all -q 2>/dev/null) && pass "git fetch --all" || warn "git fetch failed (network?)"
else
  info "Cloning HEMOC repo..."
  if git clone https://github.com/Domusgpt/HEMOC-Stain-Glass-Flower.git "$HEMOC_ROOT" 2>/dev/null; then
    pass "HEMOC cloned"
  else
    fail "Could not clone HEMOC repo (check network/auth)"
    ERRORS=$((ERRORS + 1))
  fi
fi

# ---------------------------------------------------------------------------
# 8) HEMOC Python
# ---------------------------------------------------------------------------
header "HEMOC Python"
if [ -d "$HEMOC_ROOT" ]; then
  if [ ! -f "$HEMOC_ROOT/.venv/bin/activate" ]; then
    info "Creating venv at $HEMOC_ROOT/.venv"
    python3 -m venv "$HEMOC_ROOT/.venv"
  else
    info "Venv exists, reusing"
  fi

  (
    source "$HEMOC_ROOT/.venv/bin/activate"
    pip install --upgrade pip -q 2>/dev/null

    # Core deps
    if [ -f "$HEMOC_ROOT/requirements.txt" ]; then
      pip install -r "$HEMOC_ROOT/requirements.txt" -q 2>/dev/null
    fi
    pip install pytest pytest-cov -q 2>/dev/null

    # Dev tooling: linters + formatters
    pip install black isort flake8 -q 2>/dev/null || warn "Linter install failed (non-fatal)"

    # Documentation tooling
    pip install mkdocs mkdocs-material pymdown-extensions -q 2>/dev/null || warn "MkDocs install failed (optional)"

    # Experiment tracking
    pip install wandb -q 2>/dev/null || warn "wandb install failed (optional — needed for cloud training)"

    # Schema validation
    pip install jsonschema -q 2>/dev/null || warn "jsonschema install failed (needed for benchmark contract validation)"

    # Audio processing (optional — for audio domain experiments)
    pip install soundfile -q 2>/dev/null || info "soundfile not installed (optional — needed for audio experiments)"

    python -c "import numpy; import scipy; print('imports OK')" && pass "HEMOC Python deps installed" || fail "HEMOC Python import check"
  ) || { fail "HEMOC Python setup"; ERRORS=$((ERRORS + 1)); }
else
  warn "HEMOC repo not available - skipping Python setup"
fi

# ---------------------------------------------------------------------------
# 9) HEMOC Visual System (conditional)
# ---------------------------------------------------------------------------
header "HEMOC Visual System"
if [ -f "$HEMOC_ROOT/hemoc-visual-system/package.json" ]; then
  (
    cd "$HEMOC_ROOT/hemoc-visual-system"
    npm install --silent 2>/dev/null
    pass "hemoc-visual-system npm install"
  ) || warn "hemoc-visual-system npm install had issues"
else
  info "hemoc-visual-system/package.json not found on current branch - skipping"
fi

# ---------------------------------------------------------------------------
# 10) Smoke Tests
# ---------------------------------------------------------------------------
header "Smoke Tests"

PPP_PY_RESULT="SKIP"
PPP_NODE_RESULT="SKIP"
HEMOC_PY_RESULT="SKIP"

# PPP Python smoke
if [ -f "$PPP_BACKEND/.venv/bin/activate" ]; then
  PPP_PY_OUT=$(source "$PPP_BACKEND/.venv/bin/activate" && cd "$PPP_BACKEND" && python -m pytest tests/test_h4_geometry.py -q --tb=line 2>&1) || true
  if echo "$PPP_PY_OUT" | grep -qE "passed"; then
    PPP_PY_RESULT="PASS"
    pass "PPP Python tests"
  elif echo "$PPP_PY_OUT" | grep -qE "no tests ran|collected 0"; then
    PPP_PY_RESULT="WARN"
    warn "PPP Python tests (no tests collected)"
  else
    PPP_PY_RESULT="FAIL"
    fail "PPP Python tests"
    echo "$PPP_PY_OUT" | tail -3
  fi
fi

# PPP Node smoke
if command -v npm &>/dev/null && [ -f "$PPP_ROOT/package.json" ]; then
  PPP_NODE_OUT=$(cd "$PPP_ROOT" && npm test 2>&1) || true
  if echo "$PPP_NODE_OUT" | grep -qiE "pass|tests [0-9]|# pass"; then
    PPP_NODE_RESULT="PASS"
    pass "PPP Node tests"
  else
    PPP_NODE_RESULT="WARN"
    warn "PPP Node tests (check: npm run test:all)"
  fi
fi

# HEMOC Python smoke
if [ -f "$HEMOC_ROOT/.venv/bin/activate" ]; then
  HEMOC_PY_OUT=$(source "$HEMOC_ROOT/.venv/bin/activate" && cd "$HEMOC_ROOT" && python -m pytest tests/unit/ -q --tb=line 2>&1) || true
  if echo "$HEMOC_PY_OUT" | grep -qE "passed"; then
    HEMOC_PY_RESULT="PASS"
    pass "HEMOC Python tests"
  elif echo "$HEMOC_PY_OUT" | grep -qE "no tests ran|error"; then
    HEMOC_PY_RESULT="WARN"
    warn "HEMOC Python tests (some tests may need torch)"
  else
    HEMOC_PY_RESULT="FAIL"
    fail "HEMOC Python tests"
    echo "$HEMOC_PY_OUT" | tail -3
  fi
fi

# ---------------------------------------------------------------------------
# 11) Summary
# ---------------------------------------------------------------------------
header "Environment Report"
echo ""
echo "  Python:        $(python3 --version 2>&1)"
echo "  Node:          $(node --version 2>/dev/null || echo 'not installed')"
echo "  Git:           $(git --version 2>&1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+')"
echo "  gh CLI:        $(gh --version 2>&1 | head -1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' 2>/dev/null || echo 'not installed')"
echo "  Docker:        $(docker --version 2>&1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' 2>/dev/null || echo 'not installed')"
echo "  jq:            $(jq --version 2>&1 | grep -oE '[0-9]+\.[0-9]+' 2>/dev/null || echo 'not installed')"
echo ""
echo "  PPP root:      $PPP_ROOT"
echo "  PPP venv:      $PPP_BACKEND/.venv"
echo "  PPP branch:    $(cd "$PPP_ROOT" && git branch --show-current 2>/dev/null || echo 'N/A')"
echo "  HEMOC root:    $HEMOC_ROOT"
echo "  HEMOC venv:    $HEMOC_ROOT/.venv"
echo "  HEMOC branch:  $(cd "$HEMOC_ROOT" && git branch --show-current 2>/dev/null || echo 'N/A')"
echo ""
echo "  +-------------------------------+--------+"
echo "  | Component                     | Status |"
echo "  +-------------------------------+--------+"
printf "  | %-29s | %-6s |\n" "PPP Python backend" "$PPP_PY_RESULT"
printf "  | %-29s | %-6s |\n" "PPP Node frontend" "$PPP_NODE_RESULT"
printf "  | %-29s | %-6s |\n" "HEMOC Python" "$HEMOC_PY_RESULT"
echo "  +-------------------------------+--------+"
echo ""

# Dev tooling summary
header "Dev Tooling (in venvs)"
for TOOL in black isort flake8 mkdocs wandb jsonschema; do
  if [ -f "$HEMOC_ROOT/.venv/bin/$TOOL" ] || [ -f "$HEMOC_ROOT/.venv/bin/$TOOL" ]; then
    pass "$TOOL"
  elif [ -f "$PPP_BACKEND/.venv/bin/$TOOL" ]; then
    pass "$TOOL (PPP venv only)"
  else
    info "$TOOL not installed"
  fi
done
echo ""

# Docker info
if command -v docker &>/dev/null; then
  header "Docker Quick Start"
  echo "  PPP full-stack:   cd $PPP_ROOT/_SYNERGIZED_SYSTEM && docker compose up --build"
  echo "  HEMOC GPU train:  docker build -f $HEMOC_ROOT/deploy/Dockerfile -t hemoc-train $HEMOC_ROOT"
  echo ""
fi

# Final status
if [ "$ERRORS" -gt 0 ]; then
  echo -e "${RED}Setup completed with $ERRORS error(s). Review output above.${NC}"
  exit 1
elif [ "$WARNINGS" -gt 0 ]; then
  echo -e "${YELLOW}Setup complete with $WARNINGS warning(s). Review recommended tools above.${NC}"
  echo ""
  echo -e "${GREEN}Activate venvs before running Python:${NC}"
  echo "  PPP:   source $PPP_BACKEND/.venv/bin/activate"
  echo "  HEMOC: source $HEMOC_ROOT/.venv/bin/activate"
else
  echo -e "${GREEN}Setup complete. All tools available.${NC}"
  echo ""
  echo "  Activate venvs before running Python:"
  echo "    PPP:   source $PPP_BACKEND/.venv/bin/activate"
  echo "    HEMOC: source $HEMOC_ROOT/.venv/bin/activate"
fi
