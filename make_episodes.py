#!/usr/bin/env python3
import argparse, json
from collections import defaultdict
from pathlib import Path
import os
from dotenv import load_dotenv

# Load .env variables
load_dotenv()

goal_log = os.getenv("GOAL_LOG")
episodes_out = os.getenv("EPISODES_OUT")
transitions_out = os.getenv("TRANSITIONS_OUT")

def parse_log(path):
    rows = []
    with open(path, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            d["_seq"] = i  # global order
            rows.append(d)
    return rows

def to_episodes(rows):
    by_proof = defaultdict(list)
    for d in rows:
        pid = d.get("proof_id")
        if pid is None:
            # skip orphan lines without proof_id
            continue
        by_proof[pid].append(d)
    # keep file order within each proof
    for pid in by_proof:
        by_proof[pid].sort(key=lambda x: x["_seq"])
    return by_proof

def step_to_transition(cur, nxt):
    obs = {
        "goals": cur.get("goals_before", []),
        "file": cur.get("file"),
        "pos": cur.get("pos"),
        "proof_id": cur.get("proof_id"),
    }
    action = cur.get("tactic", "")

    meta = {
        "status": cur.get("status"),
        "time_ms": cur.get("time_ms"),
        "step_idx": cur.get("step_idx"),
        "decl": cur.get("decl"),
        "versions": cur.get("versions", {}),
        "run_id": cur.get("run_id"),
        "_seq": cur.get("_seq"),
    }

    status = cur.get("status")
    goals_after = cur.get("goals_after", None)

    # candidate next_obs (only valid if not terminal)
    next_obs = None
    if nxt is not None:
        next_obs = {
            "goals": nxt.get("goals_before", []),
            "file": nxt.get("file"),
            "pos": nxt.get("pos"),
            "proof_id": nxt.get("proof_id"),
        }

    done = False
    if status != "ok":
        done = True
        next_obs = None
    elif isinstance(goals_after, list) and len(goals_after) == 0:
        done = True
        next_obs = None
    elif nxt is None:
        done = True
        next_obs = None

    return {
        "obs": obs,
        "action": action,
        "next_obs": next_obs,
        "done": done,
        "meta": meta,
        # "reward": None,  # optional slot
    }

def build_transitions(by_proof):
    episodes = {}
    transitions = []
    for pid, steps in by_proof.items():
        epi = []
        for i, cur in enumerate(steps):
            nxt = steps[i + 1] if i + 1 < len(steps) else None
            tr = step_to_transition(cur, nxt)
            epi.append(tr)
            transitions.append(tr)
        episodes[pid] = epi
    return episodes, transitions

def main():
    ap = argparse.ArgumentParser(description="Convert Lean goal/tactic logs to RL episodes + transitions.")
    ap.add_argument("--in", dest="inp", default=goal_log,
                    help="Input JSONL log (default: goal_tactic_log.jsonl)")
    ap.add_argument("--episodes", dest="episodes", default=episodes_out,
                    help="Output episodes JSON (default: episodes.json)")
    ap.add_argument("--transitions", dest="transitions", default=transitions_out,
                    help="Output transitions JSONL (default: transitions.jsonl)")
    args = ap.parse_args()

    inp = Path(args.inp)
    rows = parse_log(inp)
    by_proof = to_episodes(rows)
    episodes, transitions = build_transitions(by_proof)

    # write episodes.json (pretty for inspection)
    with open(args.episodes, "w") as f:
        json.dump(episodes, f)

    # write transitions.jsonl (one per line)
    with open(args.transitions, "w") as f:
        for tr in transitions:
            f.write(json.dumps(tr) + "\n")

    print(f"proofs: {len(episodes)}")
    print(f"steps: {sum(len(ep) for ep in episodes.values())}")
    print(f"wrote: {args.episodes}, {args.transitions}")

if __name__ == "__main__":
    main()