#!/usr/bin/env bash
set -euo pipefail

python3 two_stages_auto_prove.py \
  --ckpt_template runs/template/template.pt \
  --ckpt_lemmas   runs/lemmas/lemmas.pt \
  --lemma_index   lean_env/data/lemma_index.json \
  --template      lean_env/LeanEnv/Example.lean \
  --out           lean_env/LeanEnv/Example.proof.lean \
  --build         "lake env lean LeanEnv/Example.proof.lean" \
  --project_root  lean_env \
  --decl          cast_add_rw \
  --topk_templates 5 \
  --topk_lemmas    10 \
  --max_steps      50 \
  --wait_s         0.5 \
  --verbose

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