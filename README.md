# RL Theorem Prover Pipeline

This project builds a reinforcement learning (RL) pipeline around Lean4 theorem proving.  
We log tactics, extract training data, and prepare it for models.

---

## 1. Instrumentation

We define a custom Lean tactic `logStep` that records:
- goals before tactic
- the tactic itself
- goals after or error
- metadata (proof id, step index, timing, Lean version, etc.)

Logs are stored in `goal_tactic_log.jsonl`.

---

## 2. Dump All

We run all instrumented Lean files:

```bash
bash dump_all.sh
```

This appends all tactic logs to `goal_tactic_log.jsonl`.

---

## 3. ETL: Episodes & Transitions

Convert the raw JSONL log into episodes and transitions:

```bash
python3 scripts/make_episodes.py \
  --in $LOG_FILE \
  --episodes episodes.json \
  --transitions transitions.jsonl
```

- **episodes.json** groups steps by proof_id.  
- **transitions.jsonl** flattens into transitions with (obs, action, next_obs).

---

## 4. Observation Pairs

We build simplified `(input, action_raw)` pairs for training.

```bash
python3 scripts/make_obs_pairs.py \
  --infile transitions.jsonl \
  --outfile obs_pairs.jsonl
```

Output format:

```json
{
  "input": "GOAL: univ.WellFoundedOn r ↔ WellFounded r\nCONTEXT:\n  α : Type u_2\n  r : α → α → Prop",
  "proof_id": "LeanEnv/.../WellFoundedSet.lean::Set.wellFoundedOn_univ::L92",
  "meta": { "decl": "Set.wellFoundedOn_univ", "status": "ok", "step_idx": 0 },
  "action_raw": "(Tactic.simp ...)"
}
```

---

## 5. Canonicalization of Tactics

The raw `action_raw` strings are verbose Lean ASTs.  
We canonicalize them into compact Lean tactic strings.

**Command:**

```bash
python3 scripts/canonicalize_tactics.py \
  --infile obs_pairs.jsonl \
  --outfile canonicalized_pairs.jsonl
```

**Output Format:**

Each line is a JSON object with the same input, but `action_raw` replaced by a canonicalized `action`:

```json
{
  "input": "GOAL: univ.WellFoundedOn r ↔ WellFounded r\nCONTEXT:\n  α : Type u_2\n  r : α → α → Prop",
  "action": "simp [wellFoundedOn_iff]",
  "proof_id": "LeanEnv/.../WellFoundedSet.lean::Set.wellFoundedOn_univ::L92",
  "meta": { "decl": "Set.wellFoundedOn_univ", "status": "ok", "step_idx": 0 }
}
```

---

## 6. Sanity Checks

Run quick checks on the data:

- Ensure number of input lines equals number of output lines.
- Sample random entries and confirm:
  - `input` always contains both **GOAL** and **CONTEXT**.
  - `action` is non-empty after canonicalization.
  - Proof_ids group into consistent episodes.

---

## RL Bandit Baseline

We train a simple **policy network** (bag-of-words → linear classifier) to predict tactics from goals.  

### Training

```bash
python3 rl_bandit.py --mode train   --infile lean_env/canonicalized_pairs.jsonl   --outdir runs/rl_bandit
```

- Input = goal + context tokens.  
- Output = tactic (from canonicalized set).  
- Reward = `1` only when a step is terminal (`done == True` and `status == "ok"`).  
- Loss = cross-entropy weighted by reward (bandit-style).  
- Checkpoints saved under `runs/rl_bandit/policy.pt`.

### Prediction

```bash
python3 rl_bandit.py --mode predict   --outdir runs/rl_bandit   --text "GOAL: s.WellFoundedOn r ..."
```

or batch over a file:

```bash
python3 rl_bandit.py --mode predict   --infile lean_env/canonicalized_pairs.jsonl   --outdir runs/rl_bandit
```

---

## Next Steps

- TODO: test the RL + elaborate strategy. Reward proportionally to the decrease in goal "logical complexity", + additional reward inversely proportional to time (favour short proofs) + eq negative rewards.
- TODO: database of goal + context that can be used to recognize known knowledge.
- USE the world model to predict few steps and calculate reward on these steps. Train the policy using these rewards.

- Evaluate accuracy vs reward-weighted CE on held-out data.
- Add curriculum (longer proofs / harder goals later).
- Extend policy beyond bag-of-words (RNN / Transformer encoders).
- Later: integrate a **world model** to predict next goals, enabling lookahead RL.
