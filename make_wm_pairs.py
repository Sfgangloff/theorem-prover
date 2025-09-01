#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
from dotenv import load_dotenv

def obs_to_text(obs: dict) -> str:
    """Format {goals:[{target,context:[]}]} to 'GOAL …\\nCONTEXT …' (first goal only)."""
    if not obs or not obs.get("goals"):
        return "GOAL: <none>\nCONTEXT:\n  <none>"
    g = obs["goals"][0]
    tgt = g.get("target","<none>")
    ctx = g.get("context",[])
    ctx_lines = []
    for h in ctx:
        nm, ty = h.get("name","_"), h.get("type","_")
        ctx_lines.append(f"  {nm} : {ty}")
    ctx_block = "\n".join(ctx_lines) if ctx_lines else "  <empty>"
    return f"GOAL: {tgt}\nCONTEXT:\n{ctx_block}"

def main():
    load_dotenv()
    in_path  = Path(os.getenv("TRANSITIONS_OUT", "data/transitions.jsonl"))
    out_path = Path(os.getenv("WM_PAIRS_FILE", "data/wm_pairs.jsonl"))

    n_in = n_out = 0
    with in_path.open() as fin, out_path.open("w") as fout:
        for line in fin:
            line = line.strip()
            if not line: 
                continue
            n_in += 1
            try:
                rec = json.loads(line)
            except Exception:
                continue

            # Require an observable next state for supervised dynamics
            nxt = rec.get("next_obs")
            if not nxt:
                continue

            # Build state text from `obs` (do not assume an `input` field exists)
            state_txt = obs_to_text(rec.get("obs", {}))
            # Prefer the raw s-expression action; fall back to canonicalized `action`
            action_txt = rec.get("action_raw") or rec.get("action") or ""
            # Skip lines with no action text
            if not action_txt:
                continue

            x = f"### State\n{state_txt}\n\n### Action\n{action_txt}"
            y = f"### NextState\n{obs_to_text(nxt)}"
            out = {
                "x": x,
                "y": y,
                "done": bool(rec.get("done", False)),
                "valid": (rec.get("meta", {}).get("status") == "ok"),
                "proof_id": rec.get("proof_id"),
                "meta": rec.get("meta", {})
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            n_out += 1

    print(f"Wrote {n_out} world-model pairs from {n_in} transitions → {out_path}", file=sys.stderr)

if __name__ == "__main__":
    main()