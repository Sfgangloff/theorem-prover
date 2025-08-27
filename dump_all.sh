#!/bin/bash

# Path to your log file
LOG="goal_tactic_log.jsonl"

# Clear previous output
> "$LOG"

# Run Lean on all .lean files in your project
for file in lean_env/LeanEnv/instrumented_files/*.lean; do
  echo "Processing $file"
  lake env lean --run "$file"
done

echo "Done. Log saved to $LOG"