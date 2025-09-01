# LeanEnv

This project is a Lean 4 environment for instrumenting proofs and generating data for training a reinforcement-learning-based theorem prover.

## Structure

- **LeanEnv/**
  - `ExtractionTactic.lean` — custom Lean tactic (`logStep`) to log goals before/after each tactic, tactic string, proof ID, positions, and metadata into JSONL.
  - `Basic.lean` — basic placeholder module (can be removed if unused).
- **instrumented_files/** — copies of mathlib files instrumented with `logStep` annotations.
- **goal_tactic_log.jsonl** — output log file (JSON Lines), each line = one tactic execution.
- **scripts/**
  - `dump_all.sh` — script to rebuild and run instrumentation over all instrumented files.
  - `make_episodes.py` — ETL script to transform logs into RL episodes and transitions.

## Usage

### 1. Build the environment

```bash
lake build
```

### 2. Run instrumentation

```bash
bash scripts/dump_all.sh
```

This compiles instrumented mathlib files, logging each tactic application into `goal_tactic_log.jsonl`.

### 3. Convert logs to RL data

```bash
python3 scripts/make_episodes.py \
  --in lean_env/goal_tactic_log.jsonl \
  --episodes lean_env/episodes.json \
  --transitions lean_env/transitions.jsonl
```

### 4. Outputs

- `episodes.json`: dictionary mapping `proof_id` → list of steps (`obs`, `action`, `next_obs`, `done`, `meta`).
- `transitions.jsonl`: flat list of transitions, one per line, for replay buffers.

## Notes

- JSON vs JSONL: `json` stores the whole structure as one object/array. `jsonl` stores one JSON object per line, making it easier to stream or grep.
- `done = true` when a proof finishes (no more goals or error).
- Ensure `mathlib` is properly built (`lake exe cache get! && lake build`) before running `dump_all.sh`.
