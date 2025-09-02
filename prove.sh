#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="lean_env"
BIG_LOG="${PROJECT_ROOT}/data/goal_tactic_log.jsonl"
TEMPLATE="${PROJECT_ROOT}/LeanEnv/Example.lean"
OUTFILE="${PROJECT_ROOT}/LeanEnv/Example.proof.lean"
CKPT="runs/rl_bandit/policy.pt"

mkdir -p "${PROJECT_ROOT}/data"

python3 auto_prove.py \
  --ckpt "$CKPT" \
  --template "$TEMPLATE" \
  --out "$OUTFILE" \
  --log "$BIG_LOG" \
  --build "lake env lean LeanEnv/Example.proof.lean" \
  --project_root "$PROJECT_ROOT" \
  --decl id_apply \
  --topk 5 --max_steps 20 --verbose

# # merge the newest per-run auto log into the big log
# last_run_log=$(ls -t "${PROJECT_ROOT}"/data/auto-*.jsonl | head -n1)
# echo "Merging $last_run_log -> $BIG_LOG"
# cat "$last_run_log" >> "$BIG_LOG"

# # rebuild episodes + transitions
# python3 make_episodes.py \
#   --in "$BIG_LOG" \
#   --permissive \
#   --episodes "${PROJECT_ROOT}/data/episodes.json" \
#   --transitions "${PROJECT_ROOT}/data/transitions.jsonl"

# echo "Done."