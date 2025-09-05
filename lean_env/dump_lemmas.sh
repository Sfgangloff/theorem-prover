#!/usr/bin/env bash
set -euo pipefail

# Resolve to the directory containing this script (which should be lean_env/)
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "$SCRIPT_DIR"

# Sanity checks
if [[ ! -f "lakefile.lean" && ! -f "lakefile.toml" ]]; then
  echo "Error: no lakefile in ${SCRIPT_DIR}. Are you in the Lean project root?" >&2
  exit 1
fi
if [[ ! -f "LeanEnv/RunLemmaDump.lean" ]]; then
  echo "Error: LeanEnv/RunLemmaDump.lean not found." >&2
  exit 1
fi

# Prepare output dir
mkdir -p data

# Warm dependencies/caches (cache get may be a no-op; that's fine)
lake update
lake exe cache get || true
lake build

# Run the dumper with unlimited heartbeats so it doesn't time out
lake env lean -DmaxHeartbeats=0 LeanEnv/RunLemmaDump.lean

echo "Wrote: $(pwd)/data/lemmas.jsonl"