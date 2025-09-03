#!/usr/bin/env python3
"""
auto_prove.py
=============

Greedy Lean 4 auto-prover driven by a small learned policy and lightweight
heuristics. It edits a Lean proof file created from a template, running Lean
builds between edits and mining a per-run JSONL log produced by a custom Lean
tactic (`logStepAuto`) to decide whether a candidate tactic constitutes
progress.

High-level flow
---------------
1) **Inputs**
   - A PyTorch checkpoint with fields: {"model": state_dict, "vocab": dict, "act2id": dict}.
   - A Lean template file containing a marker line `-- @TACTICS@`.
   - A target output Lean file to write the evolving proof into.
   - A declaration name `--decl` (the theorem we are proving).
   - A build command (default: `lake env lean Main.lean`) and a project root.

2) **Search loop**
   - Render the output file by replacing the marker with the current tactic
     script. Tactic lines are emitted as `try logStepAuto <tactic>`.
     The import `import LeanEnv.ExtractionTactic` is inserted if missing.
   - Run the build with env variables injected:
       RUN_ID_OVERRIDE="<auto-uuid>"    # tags all log rows for this run
       RUN_JSON_PATH=".../auto-uuid.jsonl"
       RUN_SOURCE="auto-prove"          # just an extra tag; harmless
   - Read new JSONL rows for this run/decl and extract the active goal,
     context, and a coarse “progress score”.
   - If not closed, propose tactics from (a) the learned policy’s top-k
     predictions given the goal text and (b) cheap heuristics from the goal
     shape/context. Try each candidate in turn:
       - Accept it if Lean reports success AND
           * the progress score strictly improves, OR
           * it is a "productive" structural step (e.g. splitting ↔/∧,
             or introducing an implication) AND the new state differs,
             OR
           * it increases the number of usable hypotheses in context
             (e.g. `rcases` introduced `hp`, `hq`).
       - Otherwise, revert the output file to the last accepted script.

3) **On success**
   - The file is post-processed to strip `try logStepAuto …` wrappers into
     bare tactics, drop `skip` lines, and remove the extra logger import if it
     is no longer referenced. (Use `--keep_wrappers` to skip this cleanup.)

Scoring heuristic
-----------------
The progress score is a 4-tuple (lower is better):
  (# of '↔' anywhere in all active targets,
   # of '→' anywhere in all active targets,
   total length of pretty-printed targets,
   number of goals)

This rewards splitting equivalences/implications, shrinking text, and reducing
goals, while remaining robust to pretty-printer noise.

Template requirements
---------------------
Your template must contain a line with exactly:
    -- @TACTICS@
possibly indented. Example:

    open Classical
    theorem ex_and_comm (p q : Prop) : p ∧ q ↔ q ∧ p := by
      -- @TACTICS@

During search, this becomes:

    import LeanEnv.ExtractionTactic
    open Classical
    theorem ... := by
      try logStepAuto constructor
      try logStepAuto intro h
      ...

Lean side (logger) expectations
-------------------------------
The Lean tactic `logStepAuto` should:
  - Execute the wrapped tactic sequence,
  - Write one JSON line with fields including at least:
        run_id, decl, file, step_idx, goals_before, goals_after, status, pos
  - Respect env variables:
        RUN_ID_OVERRIDE : string (required)
        RUN_JSON_PATH   : output JSONL path (required)
Option A (“strict-gated”) logger is recommended: it **refuses** to run if the
env variables are missing; in that case `try logStepAuto …` is a no-op.

CLI synopsis
------------
    python3 auto_prove.py \
      --ckpt runs/rl_bandit/policy.pt \
      --template lean_env/LeanEnv/Example.lean \
      --out lean_env/LeanEnv/Example.proof.lean \
      --project_root lean_env \
      --build "lake env lean LeanEnv/Example.proof.lean" \
      --decl ex_and_comm \
      --topk 5 --max_steps 50 --wait_s 0.5 --verbose

Environment variables are injected automatically for each build. Per-run logs
go to `<project_root>/data/auto-<uuid>.jsonl` unless `--log` is used only to
choose the directory (the exact per-run filename is always auto-<uuid>.jsonl).

Dependencies
------------
- Python 3.9+
- PyTorch (CPU is fine)
- Lean 4 toolchain and your project build (`lake env …`)
- A working `logStepAuto` implementation in Lean (see above)
"""

import argparse, json, os, subprocess, sys, time, uuid, re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import torch
import torch.nn as nn

# =========================
# Policy scaffolding
# =========================

def tokenize(s: str):
    """Simple whitespace tokenizer used by the policy for goal text."""
    return s.replace("\n", " ").split()

class PolicyNet(nn.Module):
    """
    Tiny text->action classifier:
      EmbeddingBag(mean) over token ids -> Linear to |actions|.
    Softmax is applied externally when sampling top-k.
    """
    def __init__(self, vocab_size, num_actions, d_model=128):
        super().__init__()
        self.emb = nn.EmbeddingBag(vocab_size, d_model, mode="mean")
        self.fc = nn.Linear(d_model, num_actions)
    def forward(self, tokens, offsets):
        return self.fc(self.emb(tokens, offsets))

@dataclass
class Policy:
    """
    Convenience wrapper around PolicyNet with vocabulary and (action<->id) maps.
    Checkpoint format must contain:
      - "model": state_dict
      - "vocab": dict[str -> int]  (index 0 is treated as <unk>)
      - "act2id": dict[str -> int] (maps tactic strings to class ids)
    """
    model: PolicyNet
    vocab: dict
    act2id: dict
    id2act: dict
    @classmethod
    def load(cls, ckpt: Path):
        """Load model + vocab + action maps from a torch checkpoint file."""
        ckpt = torch.load(ckpt, map_location="cpu")
        vocab, act2id = ckpt["vocab"], ckpt["act2id"]
        model = PolicyNet(len(vocab), len(act2id))
        model.load_state_dict(ckpt["model"])
        model.eval()
        id2act = {v:k for k,v in act2id.items()}
        return cls(model, vocab, act2id, id2act)
    def topk(self, goal_text: str, k: int):
        """
        Return the top-k (action, probability) predictions for the given goal text.
        If the text is empty, return the first k actions with zero probabilities.
        """
        if not goal_text:
            return [(self.id2act[i], 0.0) for i in range(min(k, len(self.id2act)))]
        ids = [self.vocab.get(t, 0) for t in tokenize(goal_text)]
        tokens = torch.tensor(ids, dtype=torch.long)
        offsets = torch.tensor([0], dtype=torch.long)
        with torch.no_grad():
            probs = torch.softmax(self.model(tokens, offsets), dim=-1).squeeze(0)
            top = torch.topk(probs, k=min(k, probs.numel()))
        return [(self.id2act[i], float(p)) for i,p in zip(top.indices.tolist(), top.values.tolist())]

# =========================
# Subprocess / I/O
# =========================

def run_build(cmd_base: str, run_id: str, run_log: Path, cwd: Path) -> Tuple[int, str, str]:
    """
    Run a build, injecting the per-run logging env vars inline so both `lake`
    and `lean` see them no matter how shells are configured.

    Returns:
        (return_code, stdout_text, stderr_text)
    """
    full_cmd = f'RUN_ID_OVERRIDE="{run_id}" RUN_JSON_PATH="{run_log}" RUN_SOURCE="auto-prove" {cmd_base}'
    p = subprocess.Popen(["bash", "-lc", full_cmd],
                         cwd=str(cwd),
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE,
                         text=True)
    out, err = p.communicate()
    return p.returncode, out, err

def _decode_line(b: bytes):
    """
    Robust line decoder for JSONL: try UTF-8 first, then UTF-8-SIG, then latin-1,
    finally fall back to UTF-8 with replacement. Avoids hard crashes on rare bytes.
    """
    for enc in ("utf-8","utf-8-sig","latin-1"):
        try: return b.decode(enc)
        except UnicodeDecodeError: pass
    return b.decode("utf-8", errors="replace")

def read_jsonl(path: Path):
    """
    Generator over JSON objects from a .jsonl file. Skips blank lines and silently
    ignores malformed JSON lines (useful when a log may be partially written).
    """
    if not path.exists(): return
    with path.open("rb") as f:
        for raw in f:
            if raw.strip():
                s = _decode_line(raw.rstrip(b"\n"))
                try: yield json.loads(s)
                except json.JSONDecodeError: continue

# =========================
# Template rendering (no blank lines)
# =========================

# Marker line that gets replaced by the current tactic block.
TACTIC_MARKER   = "-- @TACTICS@"
# The strict-gated logger import; inserted automatically if missing.
REQUIRED_IMPORT = "import LeanEnv.ExtractionTactic"  # your strict-gated module

def render_block(tactics: List[str], indent: str) -> List[str]:
    """
    Render the tactic block as a list of lines, each aligned to `indent`.
    Tactics are wrapped as `try logStepAuto ...` so that in-editor runs (without
    env) are no-ops instead of errors when using a strict-gated logger.
    """
    if not tactics:
        return [f"{indent}try logStepAuto (skip)"]
    out = []
    for t in map(str.strip, tactics):
        out.append(f"{indent}try logStepAuto ({t})" if ";" in t else f"{indent}try logStepAuto {t}")
    return out

def write_from_template(template: Path, out: Path, tactics: List[str]) -> None:
    """
    Load the template, ensure the required import is present, find the single
    marker line `-- @TACTICS@`, and replace that *entire* line with the rendered
    tactic block (no blank lines added). The rest of the file is preserved.
    """
    src = template.read_text(encoding="utf-8")
    if REQUIRED_IMPORT not in src:
        src = REQUIRED_IMPORT + "\n" + src
    lines = src.splitlines()
    # find the marker line (alone)
    try:
        idx = next(i for i, ln in enumerate(lines) if ln.strip() == TACTIC_MARKER)
    except StopIteration:
        raise SystemExit(f"Template is missing marker {TACTIC_MARKER!r} on its own line")
    # indentation of the marker line
    marker_line = lines[idx]
    indent = marker_line[:len(marker_line) - len(marker_line.lstrip())]
    block_lines = render_block(tactics, indent)
    new_lines = lines[:idx] + block_lines + lines[idx+1:]
    out.write_text("\n".join(new_lines) + "\n", encoding="utf-8")

# =========================
# Log interpretation
# =========================

def current_goal_and_ctx(rec: dict) -> Tuple[str, List[Tuple[str,str]]]:
    """
    Pick the *active* goal and its context from a log record:
      - Prefer the first goal in `goals_after` if it exists,
      - otherwise the first in `goals_before`,
      - otherwise ('', []).
    Returns: (target_text, [(hyp_name, hyp_type), ...])
    """
    arr = rec.get("goals_after")
    if isinstance(arr, list) and arr:
        g = arr[0]
    else:
        arr = rec.get("goals_before", [])
        g = arr[0] if arr else {}
    target = g.get("target", "")
    ctx    = [(h.get("name","?"), h.get("type","")) for h in g.get("context", [])]
    return target, ctx

def targets_after(rec: dict) -> List[str]:
    """
    Return the list of *all* active target strings after the tactic step (or
    before if `goals_after` is missing).
    """
    arr = rec.get("goals_after")
    if not isinstance(arr, list): arr = rec.get("goals_before", [])
    return [g.get("target","") for g in arr]

def status_ok(rec: dict) -> bool:
    """
    True iff the step status equals 'ok'. (Also supports a record layout with
    nested `meta.status` for compatibility.)
    """
    return (rec.get("status") or rec.get("meta",{}).get("status")) == "ok"

def measure_targets(targets: List[str]) -> Tuple[int,int,int,int]:
    """
    Compute the progress score (lower is better):
        (# of '↔', # of '→', total text length, number of goals)
    Used lexicographically to compare states.
    """
    num_iff = sum(t.count("↔") for t in targets)
    num_imp = sum(t.count("→") for t in targets) + sum(t.count(" -> ") for t in targets)
    total_len = sum(len(t) for t in targets)
    n_goals = len(targets)
    return (num_iff, num_imp, total_len, n_goals)

def _read_rows_with_retry(run_log: Path, run_id: str, decl: str,
                          prev_n: int, wait_s: float, retries: int = 6, verbose: bool=False):
    """
    Read JSONL rows for this run with a small retry loop (the build may have
    exited but the OS has not flushed the write yet). Returns:
      rows_all : all rows for run_id
      rows     : rows for run_id AND this decl
      grew     : whether rows_all length increased beyond prev_n
    """
    rows_all = []
    for _ in range(retries):
        rows_all = [r for r in read_jsonl(run_log) if r.get("run_id")==run_id]
        if len(rows_all) > prev_n:
            break
        time.sleep(wait_s/3)
    grew = len(rows_all) > prev_n
    rows = [r for r in rows_all if r.get("decl")==decl]
    if not rows and rows_all and verbose:
        got = sorted({r.get("decl","?") for r in rows_all})
        print(f"[debug] saw decls in log: {got} ; expected: {decl!r}")
    return rows_all, rows, grew

# =========================
# Heuristics & acceptance
# =========================

def top_has_iff(goal: str) -> bool:
    """Cheap test: treat '↔' as top level when there is no '→' present."""
    return "↔" in goal and "→" not in goal

def top_has_and(goal: str) -> bool:
    """Cheap test: treat '∧' as top level when neither '→' nor '↔' is present."""
    return (" ∧ " in goal) and ("→" not in goal) and ("↔" not in goal)

def heuristic_candidates(goal: str, ctx: List[Tuple[str,str]]) -> List[str]:
    """
    Add structure- and context-driven candidates:
      - If↔/∧ at top: 'constructor'
      - If implication present: 'intro h'
      - If some hypothesis is a conjunction: 'rcases <h> with ⟨hp, hq⟩'
      - If 'hp'/'hq' are already in context: 'exact hp' / 'exact hq'
    The caller merges these with the policy’s top-k predictions.
    """
    add: List[str] = []
    if top_has_iff(goal) or top_has_and(goal):
        add.append("constructor")
    if "→" in goal or " -> " in goal:
        add.append("intro h")
    names = [n for (n,_) in ctx]
    types = {n:t for (n,t) in ctx}
    conj = next((n for n,t in types.items() if "∧" in t), None)
    if conj is not None:
        add.append(f"rcases {conj} with ⟨hp, hq⟩")
    if "hp" in names: add.append("exact hp")
    if "hq" in names: add.append("exact hq")
    return add

def productive_step(prev_goal: str, tactic: str) -> bool:
    """
    Returns True for tactics that are considered productive *on shape grounds*
    given the previous goal text:
      - 'intro …' when there is an implication,
      - 'constructor' when ↔ or ∧ is (heuristically) top-level,
      - any 'rcases …' (often increases usable hypotheses).
    """
    t = tactic.strip()
    if (("→" in prev_goal) or (" -> " in prev_goal)) and t.startswith("intro"):
        return True
    if top_has_iff(prev_goal) and t == "constructor":
        return True
    if top_has_and(prev_goal) and t == "constructor":
        return True
    if t.startswith("rcases "):
        return True
    return False

def context_gain(prev_ctx: List[Tuple[str,str]], new_ctx: List[Tuple[str,str]]) -> bool:
    """
    True if the number of named hypotheses strictly increased (e.g. after
    'rcases h with ⟨hp, hq⟩'). Used to accept steps that improve the *context*
    even when the target string hasn’t obviously shrunk.
    """
    prev_names = {n for (n,_) in prev_ctx}
    new_names  = {n for (n,_) in new_ctx}
    # accept if we strictly gained names (e.g. hp,hq after rcases)
    return len(new_names) > len(prev_names)

# =========================
# Greedy loop
# =========================

def greedy_prove(policy, template: Path, out: Path, base_log_dir: Path,
                 decl: str, build_cmd: str, project_root: Path,
                 topk: int, max_steps: int, wait_s: float, verbose: bool):
    """
    Greedy search that builds the proof one accepted tactic at a time.

    Args:
        policy:      Policy wrapper for top-k tactic suggestions.
        template:    Lean file with a `-- @TACTICS@` marker line.
        out:         Lean file that will be (re)written each iteration.
        base_log_dir:Directory to write `auto-<uuid>.jsonl` per-run log.
        decl:        Name of the declaration/theorem being proved.
        build_cmd:   Shell command to build (invoked under `project_root`).
        project_root:Working directory for the build command.
        topk:        Number of policy suggestions to consider per step.
        max_steps:   Hard cap on accepted steps.
        wait_s:      Base waiting time used for I/O retries.
        verbose:     Print detailed progress.

    Returns:
        (success: bool, accepted_script: List[str], run_log_path: Path)
    """
    run_id = f"auto-{uuid.uuid4().hex}"
    base_log_dir.mkdir(parents=True, exist_ok=True)
    run_log = base_log_dir / f"{run_id}.jsonl"

    if verbose:
        print(f"[run] per-run log = {run_log}")
        print(f"[run] build cmd   = {build_cmd}")
        print(f"[run] project cwd = {project_root}")

    tactics: List[str] = []
    n_rows = 0

    for step in range(1, max_steps+1):
        # --- Build current script (first build logs with skip) ---
        write_from_template(template, out, tactics)
        code, out_s, err_s = run_build(build_cmd, run_id, run_log, project_root)
        if verbose:
            print(f"[build step {step}] code={code}")
            if code != 0:
                print("---- STDOUT ----"); print(out_s)
                print("---- STDERR ----"); print(err_s)

        rows_all, rows, grew = _read_rows_with_retry(run_log, run_id, decl, prev_n=n_rows, wait_s=wait_s, verbose=verbose)
        if not grew:
            if verbose: print("[warn] No new log rows after build; stopping.")
            return False, tactics, run_log
        n_rows = len(rows_all)
        last = rows[-1] if rows else rows_all[-1]
        ok1 = status_ok(last)
        prev_targets = targets_after(last)
        prev_score   = measure_targets(prev_targets)
        prev_goal, prev_ctx = current_goal_and_ctx(last)

        if verbose:
            head = ("GOAL: " + prev_goal) if prev_goal else "(none)"
            print(f"[state] status={'ok' if ok1 else 'error'} ; score={prev_score} ; head = {head}")

        if not ok1:
            if verbose: print("[warn] Initial step status=error; stopping.")
            return False, tactics, run_log
        if prev_score[3] == 0:
            if verbose: print("[done] proof closed.")
            return True, tactics, run_log

        # --- Candidates (policy + heuristics from CURRENT goal/ctx) ---
        cand = policy.topk("GOAL: " + prev_goal, topk)
        for h in heuristic_candidates(prev_goal, prev_ctx):
            if all(h != a for a,_ in cand):
                cand.append((h, 0.0))

        if verbose:
            print("[candidates]")
            for i,(a,p) in enumerate(cand,1):
                print(f"  {i}. {a} (p={p:.3f})")

        # --- Try candidates; accept on score improv OR productive/context gain ---
        progressed = False
        for a,_ in cand:
            trial = tactics + [a]
            write_from_template(template, out, trial)
            code2, out2, err2 = run_build(build_cmd, run_id, run_log, project_root)
            if verbose and code2 != 0:
                print(f"  - TRY '{a}' → build code={code2} (ok if it still logged)")
            rows_all2, rows2, grew2 = _read_rows_with_retry(run_log, run_id, decl, prev_n=n_rows, wait_s=wait_s, verbose=verbose)
            if not grew2:
                if verbose: print(f"  - TRY '{a}' → no new log rows; skipping")
                write_from_template(template, out, tactics)
                continue

            n_rows = len(rows_all2)
            last2 = rows2[-1] if rows2 else rows_all2[-1]
            ok2 = status_ok(last2)
            new_targets = targets_after(last2)
            new_score   = measure_targets(new_targets)
            new_goal, new_ctx = current_goal_and_ctx(last2)

            if verbose:
                print(f"  - TRY '{a}' → status={'ok' if ok2 else 'error'} ; score {prev_score} -> {new_score}")

            gained_ctx = context_gain(prev_ctx, new_ctx)
            accept = ok2 and ( (new_score < prev_score)
                               or (productive_step(prev_goal, a) and (new_score != prev_score or gained_ctx))
                               or gained_ctx )

            if accept:
                tactics = trial
                progressed = True
                if new_score[3] == 0:
                    if verbose: print("[done] proof closed by accepted tactic.")
                    return True, tactics, run_log
                break
            else:
                # restore file to last accepted tactics
                write_from_template(template, out, tactics)

        if not progressed:
            if verbose: print("[stuck] no candidate improved the score; stopping.")
            return False, tactics, run_log

    if verbose: print("[budget] max_steps reached.")
    return False, tactics, run_log

# =========================
# Post-success cleanup
# =========================

# Regexes to strip wrappers like:
#   "  try logStepAuto (constructor)"  -> "  constructor"
#   "  try logStepAuto intro h"        -> "  intro h"
PAT_PAREN = re.compile(r'^(\s*)try\s+logStepAuto(?:Soft)?\s*\(\s*(.+)\s*\)\s*$')
PAT_PLAIN = re.compile(r'^(\s*)try\s+logStepAuto(?:Soft)?\s+(.+?)\s*$')
# Matches the logger import so it can be removed when unused.
IMPORT_LINE = re.compile(r'^\s*import\s+LeanEnv\.ExtractionTactic\s*$')

def cleanup_proof_file(out_path: Path, remove_import_if_unused: bool = True) -> None:
    """
    Post-process the generated Lean file to unwrap logging wrappers and polish:
      - Turn lines like
            '  try logStepAuto (constructor)'  -> '  constructor'
            '  try logStepAuto intro h'        -> '  intro h'
      - Drop 'skip' lines
      - Remove the logger import if the file no longer mentions 'logStepAuto'
      - Collapse accidental runs of 3+ blank lines to 2

    This runs only when the proof search succeeded, unless --keep_wrappers is set.
    """
    txt = out_path.read_text(encoding="utf-8")
    new_lines: List[str] = []
    for line in txt.splitlines():
        m = PAT_PAREN.match(line) or PAT_PLAIN.match(line)
        if m:
            indent, body = m.group(1), m.group(2).strip()
            if body == "skip":
                continue  # drop
            new_lines.append(f"{indent}{body}")
        else:
            new_lines.append(line)

    new_txt = "\n".join(new_lines) + "\n"

    # If no logStepAuto/Soft remains, remove import if present
    if remove_import_if_unused and ("logStepAuto" not in new_txt):
        new_txt = "\n".join(
            ln for ln in new_txt.splitlines()
            if not IMPORT_LINE.match(ln)
        ) + "\n"

    # Squash accidental multiple blank lines inside the tactic block
    new_txt = re.sub(r'\n{3,}', '\n\n', new_txt)

    out_path.write_text(new_txt, encoding="utf-8")

# =========================
# CLI
# =========================

def main():
    """
    Parse CLI arguments, run the greedy prover, optionally clean up the resulting
    file, and print a short result summary.

    Flags:
      --ckpt           Path to policy checkpoint (required).
      --template       Lean template with `-- @TACTICS@` marker (required).
      --out            Output Lean file to write/overwrite (required).
      --log            Only used to pick a directory for per-run JSONL logs.
                       The actual file is always `auto-<uuid>.jsonl`.
      --build          Build command. Example:
                         "lake env lean LeanEnv/Example.proof.lean"
      --project_root   Working directory to run the build in (default ".").
      --decl           Name of the theorem being proved (required).
      --topk           Number of policy suggestions to try per step (default 5).
      --max_steps      Max accepted steps before giving up (default 50).
      --wait_s         Base wait used for log read retries (default 0.5).
      --verbose        Print detailed progress logs.
      --keep_wrappers  Keep `try logStepAuto …` wrappers even on success.

    Exit status is always 0; success/failure is printed in the summary.
    """
    ap = argparse.ArgumentParser(description="Greedy auto-prover with strict-gated logStepAuto.")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--template", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--log", default=None, help="Only to pick a directory; per-run file is auto-<uuid>.jsonl")
    ap.add_argument("--build", default="lake env lean Main.lean")
    ap.add_argument("--project_root", default=".")
    ap.add_argument("--decl", required=True)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--max_steps", type=int, default=50)
    ap.add_argument("--wait_s", type=float, default=0.5)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--keep_wrappers", action="store_true",
                    help="Do not strip 'try logStepAuto …' lines on success.")
    a = ap.parse_args()

    policy = Policy.load(Path(a.ckpt))
    template = Path(a.template)
    out      = Path(a.out)
    project  = Path(a.project_root).resolve()
    base_dir = (Path(a.log).resolve().parent if a.log else (project/"data").resolve())

    ok, script, run_log = greedy_prove(policy, template, out, base_dir, a.decl,
                                       a.build, project, a.topk, a.max_steps, a.wait_s, a.verbose)

    # Post-success cleanup (unwrap tactics)
    if ok and not a.keep_wrappers:
        cleanup_proof_file(out)
        if a.verbose:
            print(f"[clean] stripped logStepAuto wrappers in {out}")

    print("\n=== RESULT ===")
    print("success:", ok)
    print("tactics:", script)
    print("per-run log:", run_log)

if __name__ == "__main__":
    main()