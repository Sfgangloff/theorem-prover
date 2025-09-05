#!/usr/bin/env bash
# pipeline.sh — Orchestrates the full RL Theorem Prover pipeline.
# Steps:
#   1) Instrumentation  -> python instrumenter.py
#   2) Data Recording   -> bash lean_env/dump_proof_steps.sh
#   3) Data Preparation -> bash rewrites.sh
#   4) Training         -> bash train.sh
#
# Usage:
#   bash pipeline.sh                      # run all steps (1-4)
#   bash pipeline.sh --steps 1,3          # run only steps 1 and 3
#   bash pipeline.sh --continue           # continue on step errors (default: stop on error)
#   bash pipeline.sh --log-dir logs       # write logs into custom dir
#   bash pipeline.sh --help
#
# Notes:
# - This script should be executed from the project root, alongside instrumenter.py, rewrites.sh, train.sh, and the lean_env/ folder.
# - It detects absolute paths internally, so it also works when called from elsewhere.

set -uo pipefail

# Default behavior: stop on error unless --continue is passed.
STOP_ON_ERROR=1
LOG_DIR="pipeline_logs"
SELECTED_STEPS=""

print_help() {
  cat << 'EOF'
pipeline.sh — RL Theorem Prover pipeline orchestrator

Options:
  --steps <list>    Comma-separated list from {1,2,3,4} to select specific steps.
                    1: Instrumentation
                    2: Data Recording
                    3: Data Preparation
                    4: Training
  --continue        Do not stop on the first error; attempt remaining steps.
  --log-dir <dir>   Directory to store logs (default: pipeline_logs).
  -h, --help        Show this help and exit.

Examples:
  bash pipeline.sh
  bash pipeline.sh --steps 1,3
  bash pipeline.sh --continue --log-dir out/logs
EOF
}

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --steps)
      SELECTED_STEPS="${2:-}"
      shift 2 ;;
    --continue)
      STOP_ON_ERROR=0
      shift ;;
    --log-dir)
      LOG_DIR="${2:-pipeline_logs}"
      shift 2 ;;
    -h|--help)
      print_help
      exit 0 ;;
    *)
      echo "Unknown option: $1"; echo; print_help; exit 2 ;;
  esac
done

# Resolve project root (directory containing this script if copied into repo root)
# If the script sits elsewhere, set PROJ_ROOT to current directory as fallback.
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJ_ROOT="$SCRIPT_DIR"

# If key files/folders are not here, try current working directory
if [[ ! -f "$PROJ_ROOT/instrumenter.py" ]] || [[ ! -d "$PROJ_ROOT/lean_env" ]]; then
  PROJ_ROOT="$(pwd)"
fi

# Verify expected layout
REQUIRED=(
  "$PROJ_ROOT/instrumenter.py"
  "$PROJ_ROOT/lean_env"
  "$PROJ_ROOT/lean_env/dump_proof_steps.sh"
  "$PROJ_ROOT/rewrites.sh"
  "$PROJ_ROOT/train.sh"
)
for f in "${REQUIRED[@]}"; do
  if [[ ! -e "$f" ]]; then
    echo "ERROR: Required path not found: $f"
    echo "Please run this script from the project root, or fix the paths."
    exit 3
  fi
done

mkdir -p "$LOG_DIR"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

# Utility to run a step with logging
run_step() {
  local num="$1"; shift
  local name="$1"; shift
  local cmd=("$@")
  local log_file="${LOG_DIR}/${TIMESTAMP}_step${num}_${name}.log"

  echo "------------------------------------------------------------"
  echo "STEP ${num}: ${name}"
  echo "Logging to: ${log_file}"
  echo "Command: ${cmd[*]}"
  echo "------------------------------------------------------------"

  set +e
  ("${cmd[@]}") &> >(tee "${log_file}")
  status=$?
  set -e

  if [[ $status -ne 0 ]]; then
    echo "STEP ${num} FAILED with status ${status}."
    if [[ $STOP_ON_ERROR -eq 1 ]]; then
      echo "Stopping due to failure (remove --continue to change behavior)."
      exit $status
    else
      echo "Continuing due to --continue flag."
    fi
  else
    echo "STEP ${num} SUCCEEDED."
  fi
  echo
}

# Set -e only after run_step is defined (we selectively override for step execution)
set -e

# Determine steps to run
ALL_STEPS=(1 2 3 4)
if [[ -z "$SELECTED_STEPS" ]]; then
  STEPS_TO_RUN=("${ALL_STEPS[@]}")
else
  IFS=',' read -r -a STEPS_TO_RUN <<< "$SELECTED_STEPS"
fi

# Execute selected steps
for s in "${STEPS_TO_RUN[@]}"; do
  case "$s" in
    1)
      run_step 1 "instrumentation" bash -lc "cd "$PROJ_ROOT" && python instrumenter.py"
      ;;
    2)
      run_step 2 "data_recording" bash -lc "cd "$PROJ_ROOT" && bash lean_env/dump_proof_steps.sh"
      ;;
    3)
      run_step 3 "data_preparation" bash -lc "cd "$PROJ_ROOT" && bash rewrites.sh"
      ;;
    4)
      run_step 4 "training" bash -lc "cd "$PROJ_ROOT" && bash train.sh"
      ;;
    *)
      echo "Unknown step id: $s (valid: 1,2,3,4)"
      if [[ $STOP_ON_ERROR -eq 1 ]]; then exit 4; fi
      ;;
  esac
done

echo "All requested steps completed."