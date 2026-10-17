#!/usr/bin/env bash
# Runs the mock Control API and the pipeline server without Docker.
#
#   scripts/run-local.sh           full pipeline (first run installs ~2 GB and takes a while)
#   scripts/run-local.sh --lite    Librosa only, installs in under a minute
#
# Needs Python 3.12 (python3.12 on PATH, or set PYTHON=/path/to/python3.12).
# Full mode also needs a C compiler for Madmom (macOS: xcode-select --install)
# and ffmpeg for mp3/m4a decoding (macOS: brew install ffmpeg).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODE="full"
[[ "${1:-}" == "--lite" ]] && MODE="lite"

PYTHON="${PYTHON:-python3.12}"
VENV="${VENV:-$ROOT/server/.venv}"
PORT="${PORT:-8000}"
MOCK_PORT="${MOCK_PORT:-8787}"

if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "Python 3.12 not found as '$PYTHON'. Install it, or run: PYTHON=/path/to/python3.12 $0" >&2
  exit 1
fi
if [[ "$("$PYTHON" -c 'import sys;print(f"{sys.version_info[0]}.{sys.version_info[1]}")')" != "3.12" ]]; then
  echo "'$PYTHON' is not Python 3.12. The pinned essentia-tensorflow build needs 3.12." >&2
  exit 1
fi

# --- install (once per mode; a stamp file records what's installed) -------
if [[ ! -x "$VENV/bin/python" ]]; then
  echo "Creating virtual environment in $VENV"
  "$PYTHON" -m venv "$VENV"
fi
PY="$VENV/bin/python"
STAMP="$VENV/.installed-$MODE"
REQS=("$ROOT/server/requirements-build.txt" "$ROOT/server/requirements.txt")
[[ "$MODE" == "lite" ]] && REQS=("$ROOT/server/requirements-lite.txt")

LAST_REQ="${REQS[${#REQS[@]}-1]}"
if [[ ! -f "$STAMP" || "$STAMP" -ot "$LAST_REQ" ]]; then
  echo "Installing $MODE requirements"
  "$PY" -m pip install --quiet --upgrade pip
  if [[ "$MODE" == "full" ]]; then
    "$PY" -m pip install --quiet -r "$ROOT/server/requirements-build.txt"
    "$PY" -m pip install --quiet --no-build-isolation -r "$ROOT/server/requirements.txt"
  else
    "$PY" -m pip install --quiet -r "$ROOT/server/requirements-lite.txt"
  fi
  touch "$STAMP"
fi

if [[ "$MODE" == "full" ]] && ! command -v ffmpeg >/dev/null 2>&1; then
  echo "Note: ffmpeg not found. WAV and FLAC work; mp3/m4a may fail to decode." >&2
fi

# --- run -----------------------------------------------------------------
pids=()
cleanup() { [[ ${#pids[@]} -gt 0 ]] && kill "${pids[@]}" 2>/dev/null; wait 2>/dev/null || true; }
trap cleanup EXIT INT TERM

"$PY" -u "$ROOT/mock/mock-control-api.py" --port "$MOCK_PORT" &
pids+=($!)

cd "$ROOT/server"
# CONTROL_API_URL comes from .env, defaulting to the mock on 127.0.0.1:8787
PIPELINE_MODE="$MODE" "$PY" -m uvicorn app.main:app --port "$PORT" &
pids+=($!)

echo
echo "Open http://localhost:$PORT   ($MODE mode, ctrl-c stops both servers)"
echo
wait
