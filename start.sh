#!/usr/bin/env bash
# Simple start script for local testing
export FLASK_SECRET_KEY=${FLASK_SECRET_KEY:-devkey}
export SKIP_INITIALIZE=${SKIP_INITIALIZE:-1}
PORT=${PORT:-8000}

# Prefer gunicorn if available
if command -v gunicorn >/dev/null 2>&1; then
  exec gunicorn web.app:app --bind 0.0.0.0:${PORT} --workers 1 --timeout 60
else
  echo "gunicorn not found, falling back to Flask dev server (not for production)"
  python -u web/app.py
fi
