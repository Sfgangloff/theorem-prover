#!/usr/bin/env python3
import argparse, json, os, shlex, subprocess, sys, time, uuid
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import torch
import torch.nn as nn

def tokenize(s: str): return s.replace("\n", " ").split()

class PolicyNet(nn.Module):
    def __init__(self, vocab_size, num_actions, d_model=128):
        super().__init__()
        self.emb = nn.EmbeddingBag(vocab_size, d_model, mode="mean")
        self.fc = nn.Linear(d_model, num_actions)
    def forward(self, tokens, offsets):
        return self.fc(self.emb(tokens, offsets))

@dataclass
class Policy:
    model: PolicyNet
    vocab: dict
    act2id: dict
    id2act: dict
    @classmethod
    def load(cls, ckpt: Path):
        ckpt = torch.load(ckpt, map_location="cpu")
        vocab, act2id = ckpt["vocab"], ckpt["act2id"]
        model = PolicyNet(len(vocab), len(act2id))
        model.load_state_dict(ckpt["model"])
        model.eval()
        id2act = {v:k for k,v in act2id.items()}
        return cls(model, vocab, act2id, id2act)
    def topk(self, goal_text: str, k: int):
        if not goal_text:
            return [(self.id2act[i], 0.0) for i in range(min(k, len(self.id2act)))]
        ids = [self.vocab.get(t, 0) for t in tokenize(goal_text)]
        tokens = torch.tensor(ids, dtype=torch.long)
        offsets = torch.tensor([0], dtype=torch.long)
        with torch.no_grad():
            probs = torch.softmax(self.model(tokens, offsets), dim=-1).squeeze(0)
            top = torch.topk(probs, k=min(k, probs.numel()))
        return [(self.id2act[i], float(p)) for i,p in zip(top.indices.tolist(), top.values.tolist())]

def run_cmd(cmd: str, cwd: Optional[Path]=None, env=None):
    p = subprocess.Popen(shlex.split(cmd), cwd=str(cwd) if cwd else None,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    out, err = p.communicate()
    return p.returncode, out, err

def _decode_line(b: bytes):
    for enc in ("utf-8","utf-8-sig","latin-1"):
        try: return b.decode(enc)
        except UnicodeDecodeError: pass
    return b.decode("utf-8", errors="replace")

def read_jsonl(path: Path):
    if not path.exists(): return
    with path.open("rb") as f:
        for raw in f:
            if raw.strip():
                s = _decode_line(raw.rstrip(b"\n"))
                try: yield json.loads(s)
                except json.JSONDecodeError: continue

# ---- Templating / injection ----
TACTIC_MARKER = "-- @TACTICS@"
REQUIRED_IMPORT = "import LeanEnv.ExtractionTactic"  # your module that defines `logStepAuto`

def render_block(tactics: List[str]) -> str:
    lines = []
    if not tactics:
        lines.append("  logStepAuto (skip)")
    else:
        for t in tactics:
            t = t.strip()
            lines.append(f"  logStepAuto ({t})" if ";" in t else f"  logStepAuto {t}")
    return "\n".join(lines) + "\n"

def write_from_template(template: Path, out: Path, tactics: List[str]) -> None:
    """
    Ensures the output file imports the module that defines `logStepAuto`,
    and replaces the tactic marker with the rendered tactic block.
    """
    src = template.read_text(encoding="utf-8")
    if TACTIC_MARKER not in src:
        raise SystemExit(f"Template is missing marker {TACTIC_MARKER!r}")
    if REQUIRED_IMPORT not in src:
        src = REQUIRED_IMPORT + "\n" + src
    out.write_text(src.replace(TACTIC_MARKER, render_block(tactics)), encoding="utf-8")

# ---- Log interpretation helpers ----
def goal_text(rec: dict) -> str:
    gs = rec.get("goals_before", [])
    if not gs: return ""
    g0 = gs[0]
    target = g0.get("target","")
    ctx = "\n".join([f"  {h.get('name','?')} : {h.get('type','?')}" for h in g0.get("context",[])])
    return "GOAL: " + target + "\nCONTEXT:\n" + ctx

def goals_after(rec: dict) -> int:
    ga = rec.get("goals_after")
    return len(ga) if isinstance(ga, list) else 0

def status_ok(rec: dict) -> bool:
    return (rec.get("status") or rec.get("meta",{}).get("status")) == "ok"

def _read_rows_with_retry(run_log: Path, run_id: str, decl: str,
                          prev_n: int, wait_s: float, retries: int = 6):
    """
    Poll the per-run log until at least one new row for this run+decl appears,
    or retries are exhausted. Returns (rows, grew).
    """
    for _ in range(retries):
        rows = [r for r in read_jsonl(run_log) if r.get("run_id")==run_id and r.get("decl")==decl]
        if len(rows) > prev_n:
            return rows, True
        time.sleep(wait_s/3)
    return rows, (len(rows) > prev_n)

# ---- Main greedy loop ----
def greedy_prove(policy: Policy, template: Path, out: Path, base_log_dir: Path,
                 decl: str, build_cmd: str, project_root: Path,
                 topk: int, max_steps: int, wait_s: float, verbose: bool):
    run_id = f"auto-{uuid.uuid4().hex}"
    base_log_dir.mkdir(parents=True, exist_ok=True)
    run_log = base_log_dir / f"{run_id}.jsonl"

    env = os.environ.copy()
    env["RUN_ID_OVERRIDE"] = run_id
    env["RUN_JSON_PATH"]   = str(run_log)
    env["RUN_SOURCE"]      = "auto-prove"

    if verbose:
        print(f"[run] per-run log = {run_log}")
        print(f"[run] build cmd   = {build_cmd}")
        print(f"[run] project cwd = {project_root}")

    tactics: List[str] = []
    n_rows = 0  # number of rows seen so far

    for step in range(1, max_steps+1):
        # --- build current script (first build logs with skip) ---
        write_from_template(template, out, tactics)
        code, out_s, err_s = run_cmd(build_cmd, cwd=project_root, env=env)
        if verbose:
            print(f"[build step {step}] code={code}")
            if code != 0:
                print("---- STDOUT ----"); print(out_s)
                print("---- STDERR ----"); print(err_s)

        # Do not abort on code!=0; instead require a new row appeared
        rows, grew = _read_rows_with_retry(run_log, run_id, decl, prev_n=n_rows, wait_s=wait_s)
        if not grew:
            if verbose: print("[warn] No new log rows after build; stopping.")
            return False, tactics, run_log
        n_rows = len(rows)
        last = rows[-1]
        ng = goals_after(last)
        ok1 = status_ok(last)
        if verbose:
            gt = goal_text(last)
            head = gt.splitlines()[0] if gt else "(none)"
            print(f"[state] status={'ok' if ok1 else 'error'} ; goals_after={ng} ; head = {head}")

        # If last step errored, treat as no progress and stop (we haven't tried any tactic yet)
        if not ok1:
            if verbose: print("[warn] Initial log row has status=error; stopping.")
            return False, tactics, run_log

        if ng == 0:
            if verbose: print("[done] proof closed.")
            return True, tactics, run_log

        # --- propose candidates from policy ---
        cand = policy.topk(goal_text(last), topk)
        if verbose:
            print("[candidates]")
            for i,(a,p) in enumerate(cand,1):
                print(f"  {i}. {a} (p={p:.3f})")

        # --- try candidates; accept first that reduces goals with status ok ---
        progressed = False
        for a,_ in cand:
            trial = tactics + [a]
            write_from_template(template, out, trial)
            code2, out2, err2 = run_cmd(build_cmd, cwd=project_root, env=env)
            if verbose and code2 != 0:
                print(f"  - TRY '{a}' → build code={code2} (ok if it still logged)")
            rows2, grew2 = _read_rows_with_retry(run_log, run_id, decl, prev_n=n_rows, wait_s=wait_s)
            if not grew2:
                if verbose: print(f"  - TRY '{a}' → no new log rows; skipping")
                continue

            n_rows = len(rows2)
            last2 = rows2[-1]
            ok2 = status_ok(last2)
            ng2 = goals_after(last2)
            if verbose:
                print(f"  - TRY '{a}' → status={'ok' if ok2 else 'error'} ; goals_after={ng2}")

            # Only accept if Lean reported ok *and* goals decreased (or closed)
            if ok2 and ng2 < ng:
                tactics = trial
                progressed = True
                if ng2 == 0:
                    if verbose: print("[done] proof closed by accepted tactic.")
                    return True, tactics, run_log
                break

        if not progressed:
            if verbose: print("[stuck] no candidate reduced goals with status ok; stopping.")
            return False, tactics, run_log

    if verbose: print("[budget] max_steps reached.")
    return False, tactics, run_log

def main():
    ap = argparse.ArgumentParser(description="Greedy auto-prover with logStepAuto.")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--template", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--log", default=None, help="Only used to infer base log dir; actual per-run path is auto-<uuid>.jsonl")
    ap.add_argument("--build", default="lake env lean Main.lean")
    ap.add_argument("--project_root", default=".")
    ap.add_argument("--decl", required=True)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--max_steps", type=int, default=50)
    ap.add_argument("--wait_s", type=float, default=0.5)
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()


    policy = Policy.load(Path(a.ckpt))
    template = Path(a.template)
    out      = Path(a.out)
    project  = Path(a.project_root).resolve()
    base_dir = (Path(a.log).resolve().parent if a.log else (project/"data").resolve())

    ok, script, run_log = greedy_prove(policy, template, out, base_dir, a.decl,
                                       a.build, project, a.topk, a.max_steps, a.wait_s, a.verbose)

    print("\n=== RESULT ===")
    print("success:", ok)
    print("tactics:", script)
    print("per-run log:", run_log)

if __name__ == "__main__":
    main()



# #!/usr/bin/env python3
# """
# Greedy auto-prover using a learned tactic policy and Lean logging via `logStepAuto`.

# Requirements on the Lean side:
# - A module defines `logStepAuto` (and it’s imported by the template you pass in).
# - `logStepAuto` reads:
#     - RUN_ID_OVERRIDE  -> string to store in "run_id"
#     - RUN_JSON_PATH    -> path to write JSONL (one line per tactic)
# - The template file must have a marker line:    -- @TACTICS@
#   This script replaces it with a block of:
#       logStepAuto (skip)            # first build; logs initial goals
#       logStepAuto <tactic_1>
#       logStepAuto <tactic_2>
#       ...

# Usage example (Lake project lives in `lean_env`):
#   python3 auto_prove.py \
#     --ckpt runs/rl_bandit/policy.pt \
#     --template lean_env/LeanEnv/Example.lean \
#     --out lean_env/LeanEnv/Example.proof.lean \
#     --log lean_env/data/goal_tactic_log.jsonl \
#     --build "lake build" \
#     --project_root lean_env \
#     --decl id_apply \
#     --topk 5 --max_steps 20 --verbose
# """

# import argparse, json, math, os, shlex, subprocess, sys, time, uuid
# from dataclasses import dataclass
# from pathlib import Path
# from typing import List, Tuple, Optional

# # ------------------------------
# # Minimal policy (must match your trainer)
# # ------------------------------
# import torch
# import torch.nn as nn

# def tokenize(s: str) -> List[str]:
#     return s.replace("\n", " ").split()

# class PolicyNet(nn.Module):
#     def __init__(self, vocab_size, num_actions, d_model=128):
#         super().__init__()
#         self.emb = nn.EmbeddingBag(vocab_size, d_model, mode="mean")
#         self.fc = nn.Linear(d_model, num_actions)
#     def forward(self, tokens, offsets):
#         emb = self.emb(tokens, offsets)
#         return self.fc(emb)

# @dataclass
# class Policy:
#     model: PolicyNet
#     vocab: dict
#     act2id: dict
#     id2act: dict

#     @classmethod
#     def load(cls, ckpt_path: Path):
#         ckpt = torch.load(ckpt_path, map_location="cpu")
#         vocab, act2id = ckpt["vocab"], ckpt["act2id"]
#         model = PolicyNet(len(vocab), len(act2id))
#         model.load_state_dict(ckpt["model"])
#         model.eval()
#         id2act = {v:k for k,v in act2id.items()}
#         return cls(model=model, vocab=vocab, act2id=act2id, id2act=id2act)

#     def topk(self, goal_text: str, k: int) -> List[Tuple[str,float]]:
#         if not goal_text:
#             # fallback: first k actions by id
#             return [(self.id2act[i], 0.0) for i in range(min(k, len(self.id2act)))]
#         ids = [self.vocab.get(t, 0) for t in tokenize(goal_text)]
#         tokens = torch.tensor(ids, dtype=torch.long)
#         offsets = torch.tensor([0], dtype=torch.long)
#         with torch.no_grad():
#             logits = self.model(tokens, offsets)  # [1, A]
#             probs = torch.softmax(logits, dim=-1).squeeze(0)  # [A]
#             top = torch.topk(probs, k=min(k, probs.numel()))
#         return [(self.id2act[i], float(p)) for i,p in zip(top.indices.tolist(), top.values.tolist())]

# # ------------------------------
# # Shell + JSONL helpers
# # ------------------------------

# def run_cmd(cmd: str, cwd: Optional[Path] = None, env=None) -> Tuple[int, str, str]:
#     proc = subprocess.Popen(shlex.split(cmd),
#                             cwd=str(cwd) if cwd else None,
#                             stdout=subprocess.PIPE, stderr=subprocess.PIPE,
#                             text=True, env=env)
#     out, err = proc.communicate()
#     return proc.returncode, out, err

# def _decode_line(blob: bytes) -> Tuple[str, str]:
#     for enc in ("utf-8", "utf-8-sig", "latin-1"):
#         try:
#             return blob.decode(enc), "ok"
#         except UnicodeDecodeError:
#             pass
#     return blob.decode("utf-8", errors="replace"), "replaced"

# def read_jsonl(path: Path):
#     """Robust JSONL reader that tolerates encoding junk and skips malformed JSON."""
#     if not path.exists():
#         return
#     with path.open("rb") as f:
#         for raw in f:
#             if not raw.strip(): continue
#             text, _ = _decode_line(raw.rstrip(b"\n"))
#             try:
#                 yield json.loads(text)
#             except json.JSONDecodeError:
#                 continue

# # ------------------------------
# # Proof text + log filtering
# # ------------------------------

# TACTIC_MARKER = "-- @TACTICS@"

# def render_proof_block(tactics: List[str]) -> str:
#     """
#     Always log once to capture the initial state. Then log each committed tactic.
#     Uses logStepAuto (your new tactic).
#     """
#     lines = []
#     if not tactics:
#         lines.append("  logStepAuto (skip)")
#     else:
#         for t in tactics:
#             t = t.strip()
#             # Keep tacticSeq intact when it contains ';'
#             lines.append(f"  logStepAuto ({t})" if ";" in t else f"  logStepAuto {t}")
#     return "\n".join(lines) + "\n"

# def write_lean_from_template(template_path: Path, out_path: Path, tactics: List[str]) -> None:
#     src = template_path.read_text(encoding="utf-8")
#     if TACTIC_MARKER not in src:
#         raise SystemExit(f"Template is missing marker {TACTIC_MARKER!r}")
#     proof_text = render_proof_block(tactics)
#     dst = src.replace(TACTIC_MARKER, proof_text)
#     out_path.write_text(dst, encoding="utf-8")

# def filter_logs_for_run(log_path: Path, run_id: str, decl: Optional[str]) -> List[dict]:
#     """Since we write to a per-run file, filtering by run_id is redundant but kept."""
#     rows = []
#     for rec in read_jsonl(log_path):
#         if rec.get("run_id") != run_id:
#             continue
#         if decl and rec.get("decl") != decl:
#             continue
#         rows.append(rec)
#     # Preserve file order; do not sort unless step indices are guaranteed
#     return rows

# def goal_text_from_rec(rec: dict) -> str:
#     goals = rec.get("goals_before", [])
#     if not goals:
#         return ""
#     g0 = goals[0]
#     target = g0.get("target","")
#     ctx_lines = []
#     for h in g0.get("context", []):
#         nm = h.get("name","?")
#         ty = h.get("type","?")
#         ctx_lines.append(f"  {nm} : {ty}")
#     return "GOAL: " + target + "\nCONTEXT:\n" + "\n".join(ctx_lines)

# def count_goals_after(rec: dict) -> int:
#     ga = rec.get("goals_after", None)
#     if isinstance(ga, list):
#         return len(ga)
#     return 0

# # ------------------------------
# # Greedy search
# # ------------------------------

# def greedy_prove(
#     policy: Policy,
#     template: Path,
#     out_file: Path,
#     base_log_dir: Path,
#     decl: str,
#     build_cmd: str,
#     project_root: Path,
#     topk: int,
#     max_steps: int,
#     wait_s: float,
#     verbose: bool
# ) -> Tuple[bool, List[str]]:
#     """
#     At each step:
#       - build current script,
#       - read last log row,
#       - propose top-k tactics,
#       - try them; commit the first that reduces goals or closes the proof.
#     """
#     run_id = f"auto-{uuid.uuid4().hex}"
#     base_log_dir.mkdir(parents=True, exist_ok=True)
#     run_log = base_log_dir / f"{run_id}.jsonl"

#     # Env for Lean so `logStepAuto` writes to our per-run file.
#     env = os.environ.copy()
#     env["RUN_ID_OVERRIDE"] = run_id
#     env["RUN_JSON_PATH"] = str(run_log)

#     tactics: List[str] = []

#     for step in range(1, max_steps+1):
#         # 1) Build current script (first build logs initial state via logStepAuto (skip))
#         write_lean_from_template(template, out_file, tactics)
#         code, out, err = run_cmd(build_cmd, cwd=project_root, env=env)
#         if verbose:
#             print(f"[build step {step}] code={code}")
#             if code != 0:
#                 print(err.strip()[:500])
#         if code != 0:
#             return False, tactics

#         # 2) Read the latest state from THIS run's log
#         time.sleep(wait_s)
#         rows = filter_logs_for_run(run_log, run_id, decl=decl)
#         if not rows:
#             print("[warn] No log rows found yet for this run (unexpected).")
#             return False, tactics
#         last = rows[-1]
#         ng = count_goals_after(last)
#         if verbose:
#             head = goal_text_from_rec(last).splitlines()[0] if goal_text_from_rec(last) else "(no goal)"
#             print(f"[state] goals_after={ng} ; head = {head}")

#         # Already closed?
#         if ng == 0:
#             if verbose:
#                 print("[done] proof closed.")
#             return True, tactics

#         goal_text = goal_text_from_rec(last)
#         # 3) Ask policy for candidates
#         candidates = policy.topk(goal_text, topk)
#         if verbose:
#             print("[candidates]")
#             for i,(a,p) in enumerate(candidates,1):
#                 print(f"  {i}. {a} (p={p:.3f})")

#         # 4) Try candidates; keep the first that helps
#         progressed = False
#         for a, p in candidates:
#             trial = tactics + [a]
#             write_lean_from_template(template, out_file, trial)
#             code, out, err = run_cmd(build_cmd, cwd=project_root, env=env)
#             if code != 0:
#                 if verbose:
#                     print(f"  - TRY '{a}' → build failed; skipping")
#                 continue

#             time.sleep(wait_s/2)
#             rows2 = filter_logs_for_run(run_log, run_id, decl=decl)
#             if not rows2:
#                 if verbose:
#                     print(f"  - TRY '{a}' → no new logs; skipping")
#                 continue
#             last2 = rows2[-1]
#             ng2 = count_goals_after(last2)
#             if verbose:
#                 print(f"  - TRY '{a}' → goals_after: {ng2}")

#             if ng2 < ng:
#                 tactics = trial
#                 progressed = True
#                 if ng2 == 0:
#                     return True, tactics
#                 break

#         if not progressed:
#             if verbose:
#                 print("[stuck] no candidate reduced goals.")
#             return False, tactics

#     return False, tactics  # step budget hit

# # ------------------------------
# # CLI
# # ------------------------------

# def main():
#     ap = argparse.ArgumentParser(description="Greedy auto-prover using logStepAuto.")
#     ap.add_argument("--ckpt", required=True, help="Path to policy.pt")
#     ap.add_argument("--template", required=True, help="Lean template with marker '-- @TACTICS@'")
#     ap.add_argument("--out", required=True, help="Output .lean path where tactics are written")
#     ap.add_argument("--log", required=False, default=None,
#                     help="Path to your usual scrape log (only used to infer base log dir). "
#                          "We will write per-run logs next to this.")
#     ap.add_argument("--build", default="lake build", help='Build command (e.g., "lake build" or "lean --make <file>")')
#     ap.add_argument("--project_root", default=".", help="Directory where the build command runs (Lake project root)")
#     ap.add_argument("--decl", required=True, help="Declaration (theorem) name to filter the log")
#     ap.add_argument("--topk", type=int, default=5)
#     ap.add_argument("--max_steps", type=int, default=50)
#     ap.add_argument("--wait_s", type=float, default=0.25, help="Sleep between build and log read")
#     ap.add_argument("--verbose", action="store_true")
#     args = ap.parse_args()

#     ckpt = Path(args.ckpt)
#     template = Path(args.template)
#     out_file = Path(args.out)
#     project_root = Path(args.project_root).resolve()

#     # Base dir for per-run logs:
#     if args.log:
#         base_log_dir = Path(args.log).resolve().parent
#     else:
#         base_log_dir = (project_root / "data").resolve()

#     policy = Policy.load(ckpt)

#     ok, script = greedy_prove(
#         policy=policy,
#         template=template,
#         out_file=out_file,
#         base_log_dir=base_log_dir,
#         decl=args.decl,
#         build_cmd=args.build,
#         project_root=project_root,
#         topk=args.topk,
#         max_steps=args.max_steps,
#         wait_s=args.wait_s,
#         verbose=args.verbose,
#     )

#     print("\n=== RESULT ===")
#     print("success:", ok)
#     print("tactics:")
#     for t in script:
#         print("  -", t)
#     print(f"\nWritten proof to: {out_file}")
#     print(f"Per-run log at: { (base_log_dir / f'auto-<uuid>.jsonl').parent }  (actual file varies per run)")

# if __name__ == "__main__":
#     main()