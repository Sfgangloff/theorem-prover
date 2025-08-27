#!/usr/bin/env bash
set -u  # keep -u, but don't use -e so one failure doesn't stop all
set -o pipefail

LOG="goal_tactic_log.jsonl"

# Optional: clear if asked
if [[ "${1-}" == "--clear" ]]; then
  : > "$LOG"
  echo "Cleared $LOG"
fi

# Ensure the log exists (and optionally start clean if you didn't pass --clear)
: > "$LOG"

# Build the package so `logStep` is compiled
lake build

ok=0
fail=0

# Find and process all instrumented Lean files (use -print0 for safety)
while IFS= read -r -d '' file; do
  echo "Processing $file"
  # Compile as a module (not --run); elaboration triggers tactics & logging.
  # No -o /dev/null — let Lean place artifacts under .lake/build.
  if lake env lean --root . "$file"; then
    ((ok++))
  else
    echo "⚠️  Compile errors in: $file (continuing)"
    ((fail++))
  fi
done < <(find LeanEnv/instrumented_files -type f -name '*.lean' -print0)

echo "Done. Log saved to $LOG"
echo "Summary: ${ok} succeeded, ${fail} failed"
exit 0