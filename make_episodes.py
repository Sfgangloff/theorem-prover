#!/usr/bin/env python3
import argparse, json, ast, re, sys
from collections import defaultdict
from pathlib import Path
import os
from dotenv import load_dotenv

# Load .env variables
load_dotenv()
goal_log = os.getenv("GOAL_LOG")
episodes_out = os.getenv("EPISODES_OUT")
transitions_out = os.getenv("TRANSITIONS_OUT")

# ---------- Robust JSONL + encoding handling ----------

_CLEAN_PATTERNS = [
    (r'(?<!["\w])NaN(?!["\w])', 'null'),
    (r'(?<!["\w])Infinity(?!["\w])', 'null'),
    (r'(?<!["\w])-Infinity(?!["\w])', 'null'),
]

def _minimal_cleanup(s: str) -> str:
    for pat, rep in _CLEAN_PATTERNS:
        s = re.sub(pat, rep, s)
    s = s.replace("\x00", "")  # strip NULs
    if s and s[0] == "\ufeff":  # BOM
        s = s.lstrip("\ufeff")
    return s

def _decode_line(blob: bytes, enc_priority=None):
    """
    Try to decode a binary line using a small cascade of encodings.
    Returns (text, status) where status is one of: 'ok', 'fallback', 'replaced'.
    """
    enc_priority = enc_priority or ["utf-8", "utf-8-sig", "latin-1"]
    for enc in enc_priority:
        try:
            return blob.decode(enc), "ok"
        except UnicodeDecodeError:
            continue
    # Last resort: replace undecodable sequences so we don't crash
    return blob.decode("utf-8", errors="replace"), "replaced"

def parse_log(path, *, permissive=False, bad_suffix=".bad", enc_priority=None):
    """
    Read a possibly dirty JSONL log into a list of dicts.
    - Binary read + per-line decoding cascade (utf-8, utf-8-sig, latin-1, then utf-8 replace)
    - Strict JSON first; minimal cleanup; optional literal_eval fallback if permissive
    - Bad/unsalvageable lines are written to <file>.bad
    Each parsed row is augmented with '_seq' preserving file order (0-based).
    """
    path = Path(path)
    bad_path = path.with_suffix(path.suffix + bad_suffix)
    rows, replaced_count = [], 0
    bad_f = None

    def _record_bad(i, raw_text, err_msg, col=None, pos=None):
        nonlocal bad_f
        if bad_f is None:
            bad_f = bad_path.open("w", encoding="utf-8")
        if pos is None:
            pos = 0
        ctx_start = max(0, pos - 80)
        ctx_end = min(len(raw_text), pos + 80)
        bad_f.write(f"LINE {i}: {err_msg}")
        if col is not None or pos is not None:
            bad_f.write(f" (col {col}, char {pos})")
        bad_f.write("\n")
        bad_f.write(raw_text[ctx_start:ctx_end] + "\n\n")

    with path.open("rb") as fb:
        for i, raw in enumerate(fb, 1):  # human line numbers
            if not raw.strip():
                continue
            text, status = _decode_line(raw.rstrip(b"\n"), enc_priority=enc_priority)
            if status == "replaced":
                replaced_count += 1
            # Try JSON strictly
            try:
                d = json.loads(text)
            except json.JSONDecodeError as e1:
                cleaned = _minimal_cleanup(text)
                if cleaned != text:
                    try:
                        d = json.loads(cleaned)
                    except json.JSONDecodeError as e2:
                        if not permissive:
                            ctx = cleaned[max(0, e2.pos-80): e2.pos+80]
                            raise ValueError(
                                f"Bad JSON on line {i}: {e2.msg} at col {e2.colno} (char {e2.pos})\n"
                                f"Context: …{ctx}…"
                            ) from e2
                        # permissive fallback
                        try:
                            d = ast.literal_eval(text)
                            if not isinstance(d, dict):
                                raise ValueError("literal_eval produced non-dict")
                        except Exception as e3:
                            _record_bad(i, text, f"unreadable after cleanup and literal_eval: {e3}")
                            continue
                else:
                    if not permissive:
                        ctx = text[max(0, e1.pos-80): e1.pos+80]
                        raise ValueError(
                            f"Bad JSON on line {i}: {e1.msg} at col {e1.colno} (char {e1.pos})\n"
                            f"Context: …{ctx}…"
                        ) from e1
                    try:
                        d = ast.literal_eval(text)
                        if not isinstance(d, dict):
                            raise ValueError("literal_eval produced non-dict")
                    except Exception as e3:
                        _record_bad(i, text, f"unreadable after literal_eval: {e3}")
                        continue
            d["_seq"] = i - 1
            rows.append(d)

    if bad_f is not None:
        bad_f.close()
        print(f"[warn] Some malformed lines skipped. See: {bad_path}", file=sys.stderr)
    if replaced_count:
        print(f"[warn] {replaced_count} line(s) contained undecodable bytes; characters were replaced.", file=sys.stderr)
    return rows

# ---------- Episode & transition construction (unchanged) ----------

def to_episodes(rows):
    by_proof = defaultdict(list)
    for d in rows:
        pid = d.get("proof_id")
        if pid is None:
            continue
        by_proof[pid].append(d)
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
        "obs": obs, "action": action, "next_obs": next_obs, "done": done, "meta": meta,
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

# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser(description="Convert Lean goal/tactic logs to RL episodes + transitions.")
    ap.add_argument("--in", dest="inp", default=goal_log,
                    help="Input JSONL log (default: $GOAL_LOG)")
    ap.add_argument("--episodes", dest="episodes", default=episodes_out or "episodes.json",
                    help="Output episodes JSON (default: $EPISODES_OUT or episodes.json)")
    ap.add_argument("--transitions", dest="transitions", default=transitions_out or "transitions.jsonl",
                    help="Output transitions JSONL (default: $TRANSITIONS_OUT or transitions.jsonl)")
    ap.add_argument("--permissive", action="store_true",
                    help="Skip bad lines; attempt cleanup and literal_eval fallback.")
    ap.add_argument("--encoding", nargs="+", default=None,
                    help="Override decoding priority (e.g., --encoding utf-8 latin-1). Default: utf-8, utf-8-sig, latin-1.")
    args = ap.parse_args()

    if not args.inp:
        ap.error("No input provided: set --in or GOAL_LOG")

    enc_priority = args.encoding or ["utf-8", "utf-8-sig", "latin-1"]

    rows = parse_log(Path(args.inp), permissive=args.permissive, enc_priority=enc_priority)
    by_proof = to_episodes(rows)
    episodes, transitions = build_transitions(by_proof)

    with open(args.episodes, "w", encoding="utf-8") as f:
        json.dump(episodes, f, ensure_ascii=False, indent=2)
    with open(args.transitions, "w", encoding="utf-8") as f:
        for tr in transitions:
            f.write(json.dumps(tr, ensure_ascii=False) + "\n")

    print(f"proofs: {len(episodes)}")
    print(f"steps: {sum(len(ep) for ep in episodes.values())}")
    print(f"wrote: {args.episodes}, {args.transitions}")

if __name__ == "__main__":
    main()
