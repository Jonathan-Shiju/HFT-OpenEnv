#!/usr/bin/env bash
#
# presumbission-validation.sh - OpenEnv submission validator for this Hft repo
#
# Checks the three things that matter before submission:
#   1. The Hugging Face Space is live and responds on /health
#   2. The Docker image builds from this repo layout
#   3. openenv validate passes in the repository root
#
# Usage:
#   ./presumbission-validation.sh <space_base_url> [repo_dir]
#
# Examples:
#   ./presumbission-validation.sh https://my-space.hf.space
#   ./presumbission-validation.sh https://my-space.hf.space /path/to/hft

set -uo pipefail

DOCKER_BUILD_TIMEOUT=600

if [ -t 1 ]; then
  RED='\033[0;31m'
  GREEN='\033[0;32m'
  YELLOW='\033[1;33m'
  BOLD='\033[1m'
  NC='\033[0m'
else
  RED=''
  GREEN=''
  YELLOW=''
  BOLD=''
  NC=''
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${2:-$SCRIPT_DIR}"
PING_URL="${1:-}"
PING_PATH="/health"

run_with_timeout() {
  local timeout_seconds="$1"
  shift

  if command -v timeout >/dev/null 2>&1; then
    timeout "$timeout_seconds" "$@"
    return $?
  fi

  "$@"
}

portable_mktemp() {
  if command -v mktemp >/dev/null 2>&1; then
    mktemp
  else
    local tmp_file="/tmp/validate-submission.$$.$RANDOM"
    : > "$tmp_file"
    printf '%s\n' "$tmp_file"
  fi
}

CLEANUP_FILES=()
cleanup() {
  if [ "${#CLEANUP_FILES[@]}" -gt 0 ]; then
    rm -f "${CLEANUP_FILES[@]}"
  fi
}
trap cleanup EXIT

log() {
  printf '[%s] %b\n' "$(date -u +%H:%M:%S)" "$*"
}

pass() {
  log "${GREEN}PASSED${NC} -- $1"
}

fail() {
  log "${RED}FAILED${NC} -- $1"
}

hint() {
  printf '  %sHint:%s %b\n' "$YELLOW" "$NC" "$1"
}

stop_at() {
  log "${BOLD}Stopping after $1.${NC}"
  exit 1
}

if [ -z "$PING_URL" ]; then
  printf '%sMissing required argument:%s space base URL\n' "$RED" "$NC" >&2
  printf 'Usage: %s <space_base_url> [repo_dir]\n' "$0" >&2
  exit 1
fi

if ! REPO_DIR="$(cd "$REPO_DIR" 2>/dev/null && pwd)"; then
  printf '%sInvalid repo directory:%s %s\n' "$RED" "$NC" "$REPO_DIR" >&2
  exit 1
fi

PING_URL="${PING_URL%/}"

if [ ! -f "$REPO_DIR/openenv.yaml" ]; then
  printf '%sThis folder does not look like an OpenEnv repo:%s %s\n' "$RED" "$NC" "$REPO_DIR" >&2
  exit 1
fi

printf '\n'
printf '%s========================================%s\n' "$BOLD" "$NC"
printf '%s  OpenEnv Submission Validator%s\n' "$BOLD" "$NC"
printf '%s========================================%s\n' "$BOLD" "$NC"
log "Repo:     $REPO_DIR"
log "Ping URL: $PING_URL"
printf '\n'

log "${BOLD}Step 1/3: Checking HF Space health${NC} (${PING_URL}${PING_PATH}) ..."

HEALTH_OUTPUT=$(portable_mktemp)
CLEANUP_FILES+=("$HEALTH_OUTPUT")

HTTP_CODE=$(curl -s -o "$HEALTH_OUTPUT" -w '%{http_code}' "${PING_URL}${PING_PATH}" || true)

if [ "$HTTP_CODE" = "200" ]; then
  pass "HF Space is live and responds to /health"
else
  fail "HF Space did not return 200 on /health (HTTP $HTTP_CODE)"
  if [ -s "$HEALTH_OUTPUT" ]; then
    log "  Response: $(cat "$HEALTH_OUTPUT")"
  fi
  hint "Make sure the Space is deployed, running, and exposes /health."
  stop_at "Step 1"
fi

log "${BOLD}Step 2/3: Building the Docker image${NC} ..."

if ! command -v docker >/dev/null 2>&1; then
  fail "docker is not installed or not on PATH"
  hint "Install Docker Desktop or make docker available in your shell."
  stop_at "Step 2"
fi

if [ -f "$REPO_DIR/Dockerfile" ]; then
  DOCKER_CONTEXT="$REPO_DIR"
  DOCKERFILE_PATH="$REPO_DIR/Dockerfile"
elif [ -f "$REPO_DIR/server/Dockerfile" ]; then
  DOCKER_CONTEXT="$REPO_DIR"
  DOCKERFILE_PATH="$REPO_DIR/server/Dockerfile"
else
  fail "No Dockerfile found in the repo root or server/"
  hint "This repo should contain either Dockerfile or server/Dockerfile."
  stop_at "Step 2"
fi

log "  Docker context: $DOCKER_CONTEXT"
log "  Dockerfile:   $DOCKERFILE_PATH"

BUILD_OK=false
BUILD_OUTPUT="$(run_with_timeout "$DOCKER_BUILD_TIMEOUT" docker build -f "$DOCKERFILE_PATH" "$DOCKER_CONTEXT" 2>&1)" && BUILD_OK=true

if [ "$BUILD_OK" = true ]; then
  pass "Docker build succeeded"
else
  fail "Docker build failed"
  printf '%s\n' "$BUILD_OUTPUT"
  hint "Fix the Dockerfile or project dependencies before submitting."
  stop_at "Step 2"
fi

log "${BOLD}Step 3/3: Running openenv validate${NC} ..."

if ! command -v openenv >/dev/null 2>&1; then
  fail "openenv is not installed or not on PATH"
  hint "Install openenv-core, then try again."
  stop_at "Step 3"
fi

VALIDATE_OK=false
VALIDATE_OUTPUT="$(cd "$REPO_DIR" && openenv validate 2>&1)" && VALIDATE_OK=true

if [ "$VALIDATE_OK" = true ]; then
  pass "openenv validate passed"
  if [ -n "$VALIDATE_OUTPUT" ]; then
    log "$VALIDATE_OUTPUT"
  fi
else
  fail "openenv validate failed"
  printf '%s\n' "$VALIDATE_OUTPUT"
  hint "Check openenv.yaml, app entrypoints, and the Dockerfile."
  stop_at "Step 3"
fi

printf '\n'
printf '%sAll validation steps passed.%s\n' "$GREEN" "$NC"
exit 0
