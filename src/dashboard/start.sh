#!/usr/bin/env sh
set -e

if [ -n "${GEMINI_API_KEY:-}" ]; then
  echo "[INFO] Starting Sapy chatbot backend on port ${CHATBOT_PORT:-3000}..."
  CHATBOT_PORT="${CHATBOT_PORT:-3000}" node src/chatbot/server.js &
  CHATBOT_PID=$!
  trap 'kill "$CHATBOT_PID" 2>/dev/null || true' EXIT INT TERM
else
  echo "[WARN] GEMINI_API_KEY not set; Sapy chatbot backend will not start."
fi

exec streamlit run src/dashboard/dashboard.py \
  --server.port "${PORT:-8501}" \
  --server.address 0.0.0.0 \
  --server.headless true
