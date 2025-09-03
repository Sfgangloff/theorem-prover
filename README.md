# RL Theorem Prover Pipeline

This project builds a pipeline for proving theorems written in Lean and using RL (reinforcement learning), 
which is constituted of the following elements: 

1. **Instrumentation.** We extract proof steps data from mathlib using an ad-hoc tactic which we write 
in every mathlib file before running them. In particular, the tactic records every tactic chosen 
and the context (hypotheses and goals).
2. **Data recording.** We then run every file in mathlib in order to run the written tactics and 
record data.
3. **Data preparation.** We then transform these data to prepare them for learning.
4. **Training.** We then train some RL model in order to predict the next tactic from the context.
5. **Proving.** The prover then uses Lean to write the proof and collect the context after each tactic it chooses, and choose the next one.

# Structure of the project

The Lean part of the project is contained in the `lean_env` folder. 

# Running it 

## 1. Instrumentation

First, run the instrumenter using the following command: 

```bash
python instrumenter.py
```

This inserts the tactic `logStep` defined in the file `lean_env/LeanEnv/ExtractionTactic.Lean` at the beginning of every line of a tactic block in every .lean file in the mathlib library (contained in the `extern` folder).
The modified files are all written in `lean_env/instrumented_files`.

## 2. Data recording

In order to record the data, run the following command: 

```bash
bash lean_env/dump_all.sh
```

You can find the proof steps data in `lean_env/data/goal_tactic_log.jsonl`, and logs for instances of executing 
the `logStep` tactic which went wrong in `lean_env/data/goal_tactic_log.jsonl.bad`.

## 3. Data preparation

Then run the following in order to prepare the data for training: 

```bash
bash rewrites.sh
```

## 4. Training 

In order to train the model, use the following: 

```bash
bash train.sh
```

## 5. Proving

You can find an example of theorem with incomplete proof in `lean_env/LeanEnv/Example.lean`. When running 

```bash
bash prove.sh
```

the prover will create a file `Example.proof.lean` in the same folder and progressively write proof steps. 
These steps are recorded in `lean_env/data` in some `auto-*.jsonl` file, where `*` corresponds to the run id.

# Explanations

We describe these steps in more detail below.

