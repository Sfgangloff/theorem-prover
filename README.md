# RL Theorem Prover Pipeline

This project provides a pipeline for theorem proving in Lean using reinforcement learning (RL). The pipeline consists of several distinct stages, from instrumenting Lean proofs to training an RL agent that attempts to construct new proofs automatically.

---

## Overview of the Pipeline

1. **Instrumentation.**  
   Proof steps are extracted from Lean's mathlib using a custom tactic. This tactic is injected into tactic blocks across the library and records, for each step, the chosen tactic along with the context (hypotheses and goals).

2. **Data Recording.**  
   All instrumented Lean files are executed. Each occurrence of the `logStep` tactic records information about proof states and chosen tactics.

3. **Data Preparation.**  
   The raw logs are transformed into structured datasets suitable for machine learning models.

4. **Training.**  
   An RL model is trained to predict the next tactic given a proof context.

5. **Proving.**  
   Using the trained model, the prover attempts to complete proofs: Lean provides the proof state, the RL model selects tactics, and the cycle repeats until success or failure.

---

## Project Structure

- **`lean_env/`**  
  Contains the Lean environment, the instrumentation tactic, and supporting scripts.  
  Important files include:  
  - `LeanEnv/ExtractionTactic.lean`: defines the `logStep` tactic.  
  - `LeanEnv/Example.lean`: example Lean file with an incomplete proof for testing the prover.  
  - `instrumented_files/`: instrumented versions of mathlib files.  
  - `data/`: directory where logs and processed datasets are stored.

- **`extern/`**  
  Contains mathlib as an external dependency.

- **Scripts**  
  - `instrumenter.py`: Python script that adds the `logStep` instrumentation across Lean files.  
  - `dump_proof_steps.sh`: Executes Lean on instrumented files to record proof data.  
  - `rewrites.sh`: Prepares and rewrites collected data into a training-ready format.  
  - `train.sh`: Launches RL model training.  
  - `prove.sh`: Runs the trained prover on incomplete proofs.

---

## Running the Pipeline

### 1. Data pipeline

To extract data and train the RL agent, run the following: 

```bash
bash pipeline.sh --steps 1,3
```

The option `--steps` specifies which steps this script should run: 

1: Instrumentation
2: Data Recording
3: Data Preparation
4: Training

Logs for each steps are stored in `pipeline_logs/`.

The option `--continue` allows the script to continue even if a step fails. 

The option `--log-dir` defines another folder for the logs. 


### 2. Proving

Test the trained prover on incomplete proofs. For example, with `Example.lean`:

```bash
bash prove.sh
```

The prover creates a new file, `Example.proof.lean`, and progressively writes proof steps.  
All attempted proof traces are also logged in `lean_env/data/auto-*.jsonl`.

---

## Notes

- The pipeline is experimental and depends on Lean, mathlib, and the proper functioning of the `logStep` tactic.  
- RL model details (architecture, hyperparameters) can be customized in the training scripts.  
- Logs provide both successful and failed attempts, useful for debugging and dataset improvement.

---

## License

This project is released under the MIT License. Contributions are welcome.


<!-- # Next

1. To collect all the lemmas names: 

```bash
cd lean_env
lake env lean LeanEnv/LemmaDump.lean
```

The names are then all in `lean_env/data/lemmas.jsonl`

2. Build the index: 

```bash
python3 lemma_index.py build \
  --lemmas lean_env/data/lemmas.jsonl \
  --out    lean_env/data/lemma_index.json
```

3. Build the template dataset: 

```bash 
python3 build_template_dataset.py \
    --pairs data/canonicalized_pairs.jsonl \
    --index lean_env/data/lemma_index.json \
    --out   data/template_lemmas.jsonl
```

4. Train the prover: 

First step: 

```bash
python3 two_stage_prover.py --mode train_template \
  --infile data/template_lemmas.jsonl \
  --outdir runs/template
```

Second step: 

```bash
python3 two_stage_prover.py --mode train_lemmas \
  --infile data/template_lemmas.jsonl \
  --outdir runs/lemmas \
  --init_from_template runs/template/template.pt
```

5. Predict: 

```bash
python3 two_stage_prover.py --mode predict \
  --template_ckpt runs/template/template.pt \
  --lemma_ckpt runs/lemmas/lemmas.pt \
  --text "GOAL: ... CONTEXT: ..." \
  --topk 5
``` -->

