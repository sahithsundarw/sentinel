#!/bin/bash
# Guardrail Arena — Demo Launcher (Mac/Linux)
# Double-click or: chmod +x demo.sh && ./demo.sh

set -e

echo ""
echo " ============================================"
echo "   Guardrail Arena — Starting Demo..."
echo " ============================================"
echo ""

# ── Check Python ──────────────────────────────────────────────────
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python3 not found. Please install Python 3.10+"
    exit 1
fi

PYTHON=$(command -v python3)
echo "Python: $($PYTHON --version)"

# ── Check if requirements installed ───────────────────────────────
$PYTHON -c "import fastapi, uvicorn, httpx" 2>/dev/null || {
    echo "Installing requirements..."
    pip install -r requirements.txt -q
}

# ── Kill any existing server on port 7860 ─────────────────────────
if command -v lsof &> /dev/null; then
    lsof -ti:7860 | xargs kill -9 2>/dev/null || true
fi

# ── Start server in background ────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Starting environment server on http://localhost:7860..."
$PYTHON -m uvicorn app.main:app --host 0.0.0.0 --port 7860 --log-level error &
SERVER_PID=$!

# ── Wait for server to be ready ───────────────────────────────────
echo "Waiting for server..."
READY=0
for i in {1..20}; do
    if curl -s http://localhost:7860/health 2>/dev/null | grep -q "healthy"; then
        READY=1
        break
    fi
    sleep 1
done

if [ $READY -eq 0 ]; then
    echo "ERROR: Server did not start within 20 seconds."
    kill $SERVER_PID 2>/dev/null || true
    exit 1
fi
echo "Server ready!"

# ── Open browser to demo runner ───────────────────────────────────
echo ""
echo "============================================"
echo "  GUARDRAIL ARENA DEMO READY"
echo "============================================"
echo ""
echo "  Demo Runner:  http://localhost:7860/demo_runner"
echo "  Local API:    http://localhost:7860"
echo "  Live Space:   https://varunventra-guardrail-arena.hf.space"
echo "  GitHub:       https://github.com/sahithsundarw/sentinel"
echo ""
echo "  Demo runner ready -- press SPACE to start the episode"
echo "  Press Ctrl+C to stop the server"
echo "============================================"
echo ""

if command -v open &> /dev/null; then
    open http://localhost:7860/demo_runner      # Mac
elif command -v xdg-open &> /dev/null; then
    xdg-open http://localhost:7860/demo_runner  # Linux
fi

# ── Keep running until Ctrl+C ─────────────────────────────────────
trap "kill $SERVER_PID 2>/dev/null; echo ''; echo 'Server stopped.'; exit 0" EXIT INT TERM
wait $SERVER_PID
