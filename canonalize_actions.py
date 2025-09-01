#!/usr/bin/env python3
import argparse, json, os, re, sys
from dotenv import load_dotenv
from pathlib import Path

# ---- utilities ---------------------------------------------------------------

def find_blocks(s: str, key: str):
    """Yield (start,end) spans of substrings starting at 'key' with balanced parens."""
    out = []
    i = 0
    while True:
        i = s.find(key, i)
        if i == -1: break
        # expand to balanced parentheses from first '(' after key or from key itself
        j = s.find("(", i)
        if j == -1: break
        depth, k = 0, j
        while k < len(s):
            c = s[k]
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
                if depth == 0:
                    out.append((i, k+1))
                    break
            k += 1
        i = k+1
    return out

def backticked_idents(s: str):
    # matches `foo.bar_baz etc
    return re.findall(r"`([A-Za-z0-9_\.]+)", s)

def has_only_flag(s: str):
    # quick heuristic: a "only" string token appears in simp blocks
    return " [\"only\"]" in s or "[\"only\"" in s or " [\"only\" ]" in s

def canonicalize(action_raw: str) -> str:
    s = action_raw

    pieces = []

    # 1) simp blocks
    for a,b in find_blocks(s, "(Tactic.simp"):
        block = s[a:b]
        lemmas = backticked_idents(block)
        only = has_only_flag(block)
        if lemmas:
            pieces.append(f"simp{' only' if only else ''} [{', '.join(lemmas)}]")
        else:
            pieces.append(f"simp{' only' if only else ''}")

    # 2) rwSeq blocks
    for a,b in find_blocks(s, "(Tactic.rwSeq"):
        block = s[a:b]
        lemmas = backticked_idents(block)
        at_star = "locationWildcard" in block
        if lemmas:
            cmd = f"rw [{', '.join(lemmas)}]"
        else:
            cmd = "rw"
        if at_star:
            cmd += " at *"
        pieces.append(cmd)

    # 3) exact blocks
    for a,b in find_blocks(s, "(Tactic.exact"):
        block = s[a:b]
        # exact rfl?
        if re.search(r"`refl\b", block):
            pieces.append("rfl")  # normalize `exact refl` → `rfl`
        else:
            ids = backticked_idents(block)
            if ids:
                pieces.append(f"exact {ids[-1]}")
            else:
                pieces.append("exact _")

    # 4) intro
    for a,b in find_blocks(s, "(Tactic.intro"):
        block = s[a:b]
        names = backticked_idents(block)
        if names:
            pieces.append("intro " + " ".join(names))
        else:
            pieces.append("intro")

    # 5) constructor
    if "(Tactic.constructor" in s:
        pieces.append("constructor")

    # 6) refine
    for a,b in find_blocks(s, "(Tactic.refine"):
        # We can’t reconstruct the term reliably; keep a placeholder.
        pieces.append("refine _")

    # 7) change
    if "(Tactic.change" in s:
        pieces.append("change _")

    # 8) rcases / cases
    if "(Tactic.rcases" in s:
        pieces.append("rcases _")
    if "(Tactic.cases" in s:
        pieces.append("cases _")

    # 9) fallback: if we found nothing, just keep a compacted first line
    if not pieces:
        compact = " ".join(s.split())
        pieces = [compact[:120] + ("…" if len(compact) > 120 else "")]

    # Preserve tactic order by sorting pieces by first occurrence index
    order = []
    for p in pieces:
        # guess a key to locate; fall back to tactic keyword
        key = p.split()[0]
        idx = s.find(key)
        order.append((idx if idx >=0 else len(s), p))
    ordered = [p for _,p in sorted(order, key=lambda x:x[0])]

    # de-duplicate while respecting order
    seen, deduped = set(), []
    for p in ordered:
        if p not in seen:
            deduped.append(p); seen.add(p)

    return "; ".join(deduped)


# ---- env helpers ------------------------------------------------------------

def pick_path(cli_value: str | None, env_keys: list[str], kind: str) -> str:
    """
    Choose a path from CLI if provided, otherwise the first non-empty env var in env_keys.
    Exit with a helpful message if nothing is provided.
    """
    if cli_value:
        return cli_value
    for k in env_keys:
        v = os.environ.get(k)
        if v:
            return v
    keys_str = ", ".join(env_keys)
    print(f"[canonalize_actions] Missing {kind} path. Provide --{kind} or set one of: {keys_str}", file=sys.stderr)
    raise SystemExit(2)


# ---- main --------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile")
    ap.add_argument("--outfile")
    args = ap.parse_args()

    # Load .env so CANON_INFILE / CANON_OUTFILE can be used
    load_dotenv()

    # Prefer CLI args; else fall back to env
    in_path_str = pick_path(args.infile, ["CANON_INFILE"], "infile")
    out_path_str = pick_path(args.outfile, ["CANON_OUTFILE"], "outfile")

    in_path = Path(in_path_str)
    out_path = Path(out_path_str)
    n_in, n_out = 0, 0

    print(f"[canonalize_actions] infile={in_path} (source={'CLI' if args.infile else 'ENV'}), outfile={out_path} (source={'CLI' if args.outfile else 'ENV'})", file=sys.stderr)


    with in_path.open() as fin, out_path.open("w") as fout:
        for line in fin:
            line = line.strip()
            if not line: continue
            n_in += 1
            rec = json.loads(line)
            act_raw = rec.get("action_raw","")
            act = canonicalize(act_raw)
            out = {
                "input": rec["input"],
                "action": act,
                "proof_id": rec.get("proof_id"),
                "meta": rec.get("meta", {}),
                "done": rec.get("done", False)  # <— keep terminal flag
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            n_out += 1

    print(f"Canonicalized {n_out}/{n_in} records → {out_path}", file=sys.stderr)

if __name__ == "__main__":
    main()