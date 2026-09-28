#!/usr/bin/env bash
# Deploy the VCTR2.0 vectorization engine on the server.
#
#   ./deploy.sh                pull, install, restart, health-check
#   ./deploy.sh --test         also run the Python test suite (~7 min) before restarting
#   ./deploy.sh --no-rollback  keep the new version even if the health check fails
#
# Settings (environment variables, all optional):
#   PM2_NAME     pm2 process name            (default: vctr2-engine)
#   ENGINE_URL   where the engine answers    (default: http://127.0.0.1:8000)
#   BRANCH       branch to deploy            (default: current branch)
#
# If the restart or the health check fails, the previous commit is restored
# and restarted automatically, so the website keeps a working engine.

set -Eeuo pipefail

RUN_TESTS=0
ROLLBACK=1
for arg in "$@"; do
  case "$arg" in
    --test) RUN_TESTS=1 ;;
    --no-rollback) ROLLBACK=0 ;;
    -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
    *) echo "Unknown option: $arg" >&2; exit 2 ;;
  esac
done

PM2_NAME="${PM2_NAME:-vctr2-engine}"
ENGINE_URL="${ENGINE_URL:-http://127.0.0.1:8000}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENGINE_DIR="$ROOT/python-engine"
VENV_PY="$ENGINE_DIR/venv/bin/python"
TOTAL_STEPS=$(( 6 + RUN_TESTS ))

ok()   { printf '     \033[32m✓\033[0m %s\n' "$*"; }
warn() { printf '     \033[33m!\033[0m %s\n' "$*"; }
die()  { printf '\n\033[1;31m✗ %s\033[0m\n' "$*" >&2; exit 1; }

# --- progress display ------------------------------------------------------
# Numbered steps with timings, and a spinner with elapsed seconds for steps
# that are otherwise silent. When output is not a terminal (piped to a file)
# the spinner is replaced by plain start/finish lines.
DEPLOY_START=$SECONDS
STEP_NO=0
STEP_START=$SECONDS
IS_TTY=0; [ -t 1 ] && IS_TTY=1
LOG_FILE="$(mktemp -t vctr-deploy.XXXXXX.log)"
FRAMES=(⠋ ⠙ ⠹ ⠸ ⠼ ⠴ ⠦ ⠧ ⠇ ⠏)

elapsed() { local s=$(( SECONDS - $1 )); if [ "$s" -ge 60 ]; then printf '%dm%02ds' $((s / 60)) $((s % 60)); else printf '%ds' "$s"; fi; }

step() {
  if [ "$STEP_NO" -gt 0 ]; then printf '\033[2m     step took %s\033[0m\n' "$(elapsed "$STEP_START")"; fi
  STEP_NO=$((STEP_NO + 1))
  STEP_START=$SECONDS
  printf '\n\033[1;34m[%d/%d]\033[0m \033[1m%s\033[0m\n' "$STEP_NO" "$TOTAL_STEPS" "$*"
}

# spin "message" command args... : runs a quiet command with a live spinner;
# its output goes to $LOG_FILE and the last lines are shown if it fails.
spin() {
  local message="$1"; shift
  local started=$SECONDS out
  out="$(mktemp)"
  if [ "$IS_TTY" -eq 0 ]; then
    printf '     %s ...\n' "$message"
    if "$@" >"$out" 2>&1; then
      cat "$out" >>"$LOG_FILE"; rm -f "$out"
      printf '     %s - done (%s)\n' "$message" "$(elapsed "$started")"; return 0
    fi
    cat "$out" >>"$LOG_FILE"
    printf '     %s - FAILED (%s)\n' "$message" "$(elapsed "$started")"; tail -n 40 "$out" | sed 's/^/     | /'; rm -f "$out"; return 1
  fi
  "$@" >"$out" 2>&1 &
  local pid=$! i=0
  while kill -0 "$pid" 2>/dev/null; do
    printf '\r     \033[36m%s\033[0m %s \033[2m%s\033[0m\033[K' "${FRAMES[i++ % ${#FRAMES[@]}]}" "$message" "$(elapsed "$started")"
    sleep 0.15
  done
  if wait "$pid"; then
    cat "$out" >>"$LOG_FILE"; rm -f "$out"
    printf '\r     \033[32m✓\033[0m %s \033[2m(%s)\033[0m\033[K\n' "$message" "$(elapsed "$started")"
    return 0
  fi
  cat "$out" >>"$LOG_FILE"
  printf '\r     \033[31m✗\033[0m %s \033[2m(%s)\033[0m\033[K\n' "$message" "$(elapsed "$started")"
  tail -n 40 "$out" | sed 's/^/     | /'
  rm -f "$out"
  return 1
}

finish() {
  printf '\033[2m     step took %s\033[0m\n' "$(elapsed "$STEP_START")"
  printf '\n\033[1;32m✔ %s\033[0m \033[2m(total %s · log %s)\033[0m\n' "$*" "$(elapsed "$DEPLOY_START")" "$LOG_FILE"
}

cd "$ROOT"
BRANCH="${BRANCH:-$(git rev-parse --abbrev-ref HEAD)}"
printf '\033[1mDeploying VCTR2.0 engine\033[0m  (%s, branch %s)\n' "$ROOT" "$BRANCH"

# ---------------------------------------------------------------------------
step "Checking the working tree"
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  git status --short --untracked-files=no
  die "There are local changes on the server. Commit, stash or discard them first (deploys must match the repo)."
fi
PREV_SHA="$(git rev-parse HEAD)"
ok "Clean · current version $(git log -1 --format='%h %s')"

# ---------------------------------------------------------------------------
step "Pulling origin/$BRANCH"
spin "Fetching from GitHub" git fetch --prune origin
spin "Fast-forwarding" git pull --ff-only origin "$BRANCH"
NEW_SHA="$(git rev-parse HEAD)"
if [ "$NEW_SHA" = "$PREV_SHA" ]; then
  ok "Already up to date ($(git log -1 --format='%h'))"
else
  ok "Updated $(git rev-parse --short "$PREV_SHA") → $(git log -1 --format='%h %s')"
  git --no-pager log --oneline "$PREV_SHA..$NEW_SHA" | sed 's/^/       · /'
fi

# ---------------------------------------------------------------------------
step "Python environment"
if [ ! -x "$VENV_PY" ]; then
  PY=""
  for candidate in python3.12 python3.13; do
    if command -v "$candidate" >/dev/null 2>&1; then PY="$candidate"; break; fi
  done
  [ -n "$PY" ] || die "No venv and no python3.12/3.13 found. Install one: sudo apt install python3.12 python3.12-venv"
  spin "Creating venv with $PY" "$PY" -m venv "$ENGINE_DIR/venv"
fi
PY_VERSION="$("$VENV_PY" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
case "$PY_VERSION" in
  3.12|3.13) ok "venv Python $PY_VERSION" ;;
  *) die "venv uses Python $PY_VERSION; VTracer needs 3.12 or 3.13. Recreate python-engine/venv with python3.12." ;;
esac

REQ_CHANGED=1
if [ "$NEW_SHA" = "$PREV_SHA" ] || git diff --quiet "$PREV_SHA" "$NEW_SHA" -- python-engine/requirements.txt; then
  REQ_CHANGED=0
fi
if [ "$REQ_CHANGED" -eq 1 ] || ! "$VENV_PY" -c 'import fastapi, vtracer, cv2, numpy' >/dev/null 2>&1; then
  spin "Upgrading pip" "$VENV_PY" -m pip install --upgrade pip
  spin "Installing Python requirements" "$VENV_PY" -m pip install -r "$ENGINE_DIR/requirements.txt"
else
  ok "Requirements unchanged · skipping pip install"
fi

# ---------------------------------------------------------------------------
step "Potrace binary"
VENDOR_POTRACE="$ENGINE_DIR/vendor/potrace/potrace"
if [ -x "$VENDOR_POTRACE" ]; then
  ok "Linked: $(readlink -f "$VENDOR_POTRACE" 2>/dev/null || echo "$VENDOR_POTRACE")"
elif command -v potrace >/dev/null 2>&1; then
  ln -sf "$(command -v potrace)" "$VENDOR_POTRACE"
  ok "Linked system potrace → $VENDOR_POTRACE"
else
  warn "potrace is not installed: conversions will fall back to VTracer. Fix: sudo apt install -y potrace && ./deploy.sh"
fi

# ---------------------------------------------------------------------------
if [ "$RUN_TESTS" -eq 1 ]; then
  step "Running the engine test suite"
  spin "Running tests (several minutes)" bash -c "cd '$ENGINE_DIR' && '$VENV_PY' -m unittest tests.test_pipeline" \
    || die "Tests failed - nothing was restarted. Full output: $LOG_FILE"
fi

# ---------------------------------------------------------------------------
restart_engine() {
  if pm2 describe "$PM2_NAME" >/dev/null 2>&1; then
    pm2 restart "$PM2_NAME" --update-env
  else
    (cd "$ENGINE_DIR" && pm2 start "$VENV_PY" --name "$PM2_NAME" --cwd "$ENGINE_DIR" -- app.py)
  fi
}

engine_healthy() {
  local body
  body="$(curl -fsS -m 5 "$ENGINE_URL/health" 2>/dev/null)" || return 1
  printf '%s' "$body" | "$VENV_PY" -c '
import json, sys
data = json.load(sys.stdin)
engines = {e["name"]: e.get("available") for e in data.get("engines", [])}
sys.exit(0 if data.get("status") == "ok" and engines.get("vtracer") else 1)
'
}

wait_for_engine() {
  for _ in $(seq 1 30); do
    engine_healthy && return 0
    sleep 2
  done
  return 1
}

rollback() {
  if [ "$ROLLBACK" -eq 0 ] || [ "$NEW_SHA" = "$PREV_SHA" ]; then
    die "$1"
  fi
  warn "$1 - rolling back to $(git rev-parse --short "$PREV_SHA")"
  git reset --hard "$PREV_SHA" >/dev/null
  spin "Reinstalling previous requirements" "$VENV_PY" -m pip install -r "$ENGINE_DIR/requirements.txt" || true
  spin "Restarting previous version" restart_engine || true
  if spin "Waiting for the previous version to answer" wait_for_engine; then
    die "Deploy failed and was rolled back; the previous version is running again. Check: pm2 logs $PM2_NAME --lines 80"
  fi
  die "Deploy failed AND the rollback did not come up healthy. Check now: pm2 logs $PM2_NAME --lines 80"
}

step "Restarting $PM2_NAME"
spin "pm2 restart" restart_engine || rollback "pm2 could not restart the engine"

step "Health check"
spin "Waiting for the engine to answer ($ENGINE_URL/health)" wait_for_engine || rollback "Engine did not become healthy"
spin "Saving pm2 process list" pm2 save
POTRACE_OK="$(curl -fsS -m 5 "$ENGINE_URL/health" 2>/dev/null | "$VENV_PY" -c 'import json,sys; d=json.load(sys.stdin); print(any(e["name"]=="potrace" and e.get("available") for e in d.get("engines",[])))' 2>/dev/null || echo False)"
ok "Engine healthy"
if [ "$POTRACE_OK" = "True" ]; then
  ok "Potrace available"
else
  warn "Potrace NOT available - conversions are using the VTracer fallback"
fi

finish "Deployed $(git log -1 --format='%h %s')"
