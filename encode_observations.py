#!/usr/bin/env python3
import argparse, json, os, sys
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

# Defaults from .env if present
DEF_IN  = os.getenv("TRANSITIONS_OUT", "lean_env/data/transitions.jsonl")
DEF_OUT = os.getenv("PAIRS_OBS_OUT",  "lean_env/data/pairs_obs.jsonl")

SEP = "\n---\n"

def fmt_ctx_item(c: Dict[str, Any]) -> str:
    nm = c.get("name", "").strip()
    ty = c.get("type", "").strip()
    if nm and ty:
        return f"  {nm} : {ty}"
    if ty:
        return f"  _: {ty}"
    return "  _ : _"

def encode_single_goal(g: Dict[str, Any]) -> str:
    tgt = g.get("target", "").strip()
    ctx = g.get("context", []) or []
    lines = [f"GOAL: {tgt}", "CONTEXT:"]
    for c in ctx:
        lines.append(fmt_ctx_item(c))
    return "\n".join(lines)

def encode_obs(obs: Dict[str, Any]) -> str:
    """
    Turn obs['goals'] into a flat, readable text block.
    Multiple goals are separated by ---.
    """
    goals = obs.get("goals", [])
    if not goals:
        return "GOAL: <none>\nCONTEXT:\n  <empty>"
    blocks = [encode_single_goal(g) for g in goals]
    return SEP.join(blocks)

def iter_jsonl(path: str):
    with open(path, "r") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[ERROR] {path}:{line_no}: {e}", file=sys.stderr)

def main():
    ap = argparse.ArgumentParser(description="Encode observations from transitions.jsonl into pairs_obs.jsonl")
    ap.add_argument("--in",  dest="inp",  default=DEF_IN,  help="transitions.jsonl (default from TRANSITIONS_OUT)")
    ap.add_argument("--out", dest="out", default=DEF_OUT, help="pairs_obs.jsonl (default PAIRS_OBS_OUT)")
    ap.add_argument("--max", type=int, default=None, help="optionally cap number of lines for a quick sample")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    n_in, n_out = 0, 0
    with open(args.out, "w") as w:
        for tr in iter_jsonl(args.inp):
            n_in += 1
            obs = tr.get("obs", {})
            meta = tr.get("meta", {})
            input_text = encode_obs(obs)

            out_line = {
                "input": input_text,
                "proof_id": obs.get("proof_id") or tr.get("obs", {}).get("proof_id"),
                "meta": {
                    "decl": meta.get("decl"),
                    "status": meta.get("status"),
                    "step_idx": meta.get("step_idx"),
                    "time_ms": meta.get("time_ms"),
                    "versions": meta.get("versions"),
                    "run_id": meta.get("run_id"),
                    "_seq": meta.get("_seq"),
                },
                # Keep the raw action for step 2 (we’ll later replace this with a canonicalized tactic string)
                "action_raw": tr.get("action", "")
            }
            w.write(json.dumps(out_line, ensure_ascii=False) + "\n")
            n_out += 1
            if args.max is not None and n_out >= args.max:
                break

    print(f"[OK] wrote {n_out} pairs to {args.out} from {n_in} transitions")

if __name__ == "__main__":
    main()