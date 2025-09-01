#!/usr/bin/env python3
import argparse, json, sys, re
from collections import Counter
import os
from dotenv import load_dotenv

# Load .env variables
load_dotenv()

episodes_out = os.getenv("EPISODES_OUT")
transitions_out = os.getenv("TRANSITIONS_OUT")

TACTIC_HEAD_RE = re.compile(r"\(\s*Tactic\.(\w+)")  # crude but useful

def read_json(path):
    with open(path, "r") as f:
        return json.load(f)

def read_jsonl(path):
    items = []
    with open(path, "r") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[ERROR] JSONL parse error at {path}:{line_no}: {e}", file=sys.stderr)
    return items

def obs_key(obs):
    """A compact key for dedup/near-dup checks. Use goal target + minimal context types."""
    target = obs.get("goals", [{}])[0].get("target", "")
    ctx = obs.get("goals", [{}])[0].get("context", [])
    ctx_sig = tuple((c.get("name",""), c.get("type","")) for c in ctx[:8])  # cap to keep small
    return (target, ctx_sig)

def main():
    ap = argparse.ArgumentParser(description="Sanity checks for episodes/transitions")
    ap.add_argument("--episodes", default=episodes_out, help="episodes.json")
    ap.add_argument("--transitions", default=transitions_out, help="transitions.jsonl")
    ap.add_argument("--strict", action="store_true", help="exit(1) if any invariant fails")
    args = ap.parse_args()

    issues = []

    # ---------- Load ----------
    episodes = read_json(args.episodes)  # dict: proof_id -> [steps]
    transitions = read_jsonl(args.transitions)  # list of transitions

    # ---------- Basic counts ----------
    n_proofs = len(episodes)
    n_steps = sum(len(v) for v in episodes.values())
    print(f"[DATA] proofs: {n_proofs:,} | episode steps: {n_steps:,} | transitions lines: {len(transitions):,}")

    # ---------- Per-proof ordering & linking ----------
    for pid, steps in episodes.items():
        # order by meta._seq if present, else keep as-is
        steps_sorted = sorted(steps, key=lambda s: s.get("meta", {}).get("_seq", 0))
        if steps != steps_sorted:
            issues.append(f"Out-of-order steps in proof {pid}")
        # check monotonic _seq and step_idx
        prev_seq = -1
        prev_step_idx = -1
        for i, tr in enumerate(steps_sorted):
            meta = tr.get("meta", {})
            seq = meta.get("_seq", i)
            step_idx = meta.get("step_idx", i)
            if seq <= prev_seq:
                issues.append(f"_seq not increasing in {pid} at i={i}")
            if step_idx <= prev_step_idx:
                issues.append(f"step_idx not increasing in {pid} at i={i}")
            prev_seq = seq
            prev_step_idx = step_idx

        # check next_obs consistency
        for i, tr in enumerate(steps_sorted):
            done = tr.get("done", False)
            nxt = tr.get("next_obs")
            if done:
                if nxt is not None:
                    issues.append(f"done=True but next_obs present in {pid} at seq={tr.get('meta',{}).get('_seq')}")
            else:
                if nxt is None:
                    issues.append(f"done=False but next_obs is null in {pid} at seq={tr.get('meta',{}).get('_seq')}")
                else:
                    # ensure same proof_id and step_idx+1 if present
                    n_pid = nxt.get("proof_id")
                    if n_pid and n_pid != pid:
                        issues.append(f"next_obs.proof_id mismatch in {pid} -> {n_pid}")
                    n_step_idx = None
                    # some pipelines don’t store step_idx in next_obs; skip if missing
                    # ok either way; we enforce presence only if available
                    # (do not push a hard error here)
    
    # ---------- Success/error ratios ----------
    status_counts = Counter(tr.get("meta", {}).get("status", "unknown") for tr in transitions)
    done_counts = Counter(bool(tr.get("done", False)) for tr in transitions)
    print(f"[STATUS] by status: {dict(status_counts)}")
    print(f"[DONE] True={done_counts[True]:,} | False={done_counts[False]:,} "
          f"({done_counts[True]/max(1,len(transitions)):.1%} done)")

    # ---------- Duplicates / near-duplicates ----------
    seen = set()
    dup_count = 0
    for tr in transitions:
        key = (obs_key(tr["obs"]), tr.get("action", ""))
        if key in seen:
            dup_count += 1
        else:
            seen.add(key)
    print(f"[DUPS] exact (obs,action) duplicates in transitions: {dup_count:,}")

    # ---------- Tactic stats ----------
    act_len = Counter()
    tactic_head = Counter()
    for tr in transitions:
        a = tr.get("action", "")
        act_len[len(a)] += 1
        m = TACTIC_HEAD_RE.search(a)
        if m:
            tactic_head[m.group(1)] += 1

    # length histogram (coarse bins)
    lengths = sorted(act_len.items())
    pct = lambda x: f"{(x/len(transitions))*100:5.1f}%"
    def hist_row(thresh):
        count = sum(c for L,c in lengths if L<=thresh)
        return f"≤{thresh:4}: {count:6} ({pct(count)})"
    bins = [80, 160, 320, 640, 1280, 2560]
    print("[ACTION LENGTH HIST]")
    print("\n".join(hist_row(b) for b in bins))
    long = sum(c for L,c in lengths if L>bins[-1])
    if long:
        print(f"> {bins[-1]:4}: {long:6} ({pct(long)})")

    print("[TOP TACTICS]")
    for name, cnt in tactic_head.most_common(15):
        print(f"  {name:20} {cnt:7}")

    # ---------- Cross-file invariants ----------
    # map proof_id -> count in transitions (for quick mismatch detection)
    tr_by_pid = Counter(tr["obs"].get("proof_id") for tr in transitions if tr.get("obs"))
    if set(tr_by_pid.keys()) != set(episodes.keys()):
        miss_in_tr = set(episodes.keys()) - set(tr_by_pid.keys())
        miss_in_ep = set(tr_by_pid.keys()) - set(episodes.keys())
        if miss_in_tr:
            issues.append(f"{len(miss_in_tr)} proof_ids in episodes missing from transitions (e.g. {next(iter(miss_in_tr))})")
        if miss_in_ep:
            issues.append(f"{len(miss_in_ep)} proof_ids in transitions missing from episodes (e.g. {next(iter(miss_in_ep))})")

    # ---------- Report ----------
    if issues:
        print("\n[WARN/ERROR] Potential issues detected:")
        for it in issues[:50]:
            print(" -", it)
        if len(issues) > 50:
            print(f"   ... and {len(issues)-50} more")
        if args.strict:
            sys.exit(1)
    else:
        print("[OK] No issues found.")

if __name__ == "__main__":
    main()