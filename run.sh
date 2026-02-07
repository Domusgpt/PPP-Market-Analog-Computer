#!/bin/bash
# PPP Synergized System — Single-command launcher
#
# Usage:
#   ./run.sh                  Start Python engine + open dashboard
#   ./run.sh --engine-only    Start Python engine only
#   ./run.sh --test           Run test suite
#   ./run.sh --help           Show this help
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKEND_DIR="$SCRIPT_DIR/_SYNERGIZED_SYSTEM/backend"
DASHBOARD="$SCRIPT_DIR/synergized-system.html"

RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m'

show_help() {
    echo ""
    echo "PPP Synergized System — Optical Kirigami Moiré Encoder"
    echo "======================================================="
    echo ""
    echo "Usage: ./run.sh [OPTION]"
    echo ""
    echo "Options:"
    echo "  (none)          Start the Python physics engine and open the dashboard"
    echo "  --engine-only   Start the Python WebSocket engine only (port 8765)"
    echo "  --test          Run the Python test suite"
    echo "  --install       Install Python dependencies"
    echo "  --help          Show this help message"
    echo ""
}

check_python() {
    if ! command -v python3 &>/dev/null; then
        echo -e "${RED}Error: python3 not found. Please install Python 3.10+.${NC}"
        exit 1
    fi
}

install_deps() {
    check_python
    echo -e "${CYAN}Installing Python dependencies...${NC}"
    cd "$BACKEND_DIR"
    pip install -e ".[dev]" 2>&1
    echo -e "${GREEN}Dependencies installed.${NC}"
}

run_tests() {
    check_python
    echo -e "${CYAN}Running test suite...${NC}"
    cd "$BACKEND_DIR"
    python3 -m pytest tests/ -v --tb=short
}

start_engine() {
    check_python
    echo -e "${CYAN}Starting physics engine on ws://localhost:8765...${NC}"
    cd "$BACKEND_DIR"
    python3 -m engine.websocket_server &
    ENGINE_PID=$!
    echo -e "${GREEN}Engine started (PID: $ENGINE_PID)${NC}"

    # Wait for WebSocket to be ready
    sleep 2

    if ! kill -0 "$ENGINE_PID" 2>/dev/null; then
        echo -e "${RED}Engine failed to start. Check Python dependencies.${NC}"
        echo "Try: ./run.sh --install"
        exit 1
    fi

    echo "$ENGINE_PID"
}

open_dashboard() {
    if [ -f "$DASHBOARD" ]; then
        echo -e "${CYAN}Opening dashboard: $DASHBOARD${NC}"
        if command -v open &>/dev/null; then
            open "$DASHBOARD"
        elif command -v xdg-open &>/dev/null; then
            xdg-open "$DASHBOARD"
        else
            echo -e "Open in browser: file://$DASHBOARD"
        fi
    else
        echo -e "${RED}Dashboard not found at $DASHBOARD${NC}"
    fi
}

# Parse arguments
case "${1:-}" in
    --help|-h)
        show_help
        exit 0
        ;;
    --test)
        run_tests
        exit $?
        ;;
    --install)
        install_deps
        exit 0
        ;;
    --engine-only)
        ENGINE_PID=$(start_engine)
        echo ""
        echo -e "${GREEN}Physics engine running.${NC}"
        echo "  WebSocket: ws://localhost:8765"
        echo "  PID: $ENGINE_PID"
        echo "  Press Ctrl+C to stop"
        echo ""
        trap "kill $ENGINE_PID 2>/dev/null; echo 'Engine stopped.'" EXIT
        wait "$ENGINE_PID"
        ;;
    "")
        ENGINE_PID=$(start_engine)
        open_dashboard
        echo ""
        echo -e "${GREEN}System running.${NC}"
        echo "  WebSocket: ws://localhost:8765"
        echo "  Dashboard: $DASHBOARD"
        echo "  Engine PID: $ENGINE_PID"
        echo "  Press Ctrl+C to stop"
        echo ""
        trap "kill $ENGINE_PID 2>/dev/null; echo 'Engine stopped.'" EXIT
        wait "$ENGINE_PID"
        ;;
    *)
        echo -e "${RED}Unknown option: $1${NC}"
        show_help
        exit 1
        ;;
esac
