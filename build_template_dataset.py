#!/usr/bin/env python3
"""
build_template_dataset.py
Parse action strings from your mathlib logs into (template, lemma list) using the lemma index.

Input  (JSONL): fields at least {"input": <goal+ctx>, "action": <tactic string>, ...}
Output (JSONL): {"input": str, "template": str, "lemmas": [str], "lemma_ids": [int]}

Usage:
  python3 build_template_dataset.py \
    --pairs data/canonicalized_pairs.jsonl \
    --index lean_env/data/lemma_index.json \
    --out   data/template_lemmas.jsonl
"""
import argparse, json, re
from pathlib import Path

import lemma_index as L  # must provide: load(path) -> {"name2id": {...}, "base2ids": {...}}

# ---------------------------------
# Regexes for common templates
# ---------------------------------

# apply NAME
RE_APPLY = re.compile(r'^\s*apply\s+([A-Za-z0-9_\.`"]+)\s*$', re.IGNORECASE)

# exact NAME  (we now try to extract NAME if it is a global lemma)
RE_EXACT = re.compile(r'^\s*exact\s+([A-Za-z0-9_\.`"]+)\s*$', re.IGNORECASE)

# rw [a, ←b, -c] (optionally: "... ] at ...")
RE_RW = re.compile(r'^\s*rw\b[^[]*\[([^\]]*)\][^$]*$', re.IGNORECASE)

# simp / simp_all with optional "only", bracket list, and "at ..."
# e.g. "simp", "simp?", "simp only [a, ←b] at h", "simp_all [foo] at *"
RE_SIMP = re.compile(r'^\s*simp(?:_all)?\??(?:\s+only)?(?:\s*\[([^\]]*)\])?(?:\s+at\b.*)?\s*$', re.IGNORECASE)

# ---------------------------------
# Helpers
# ---------------------------------

def split_list(s: str):
    """Split comma-separated items, strip and drop empties."""
    return [x.strip() for x in s.split(",") if x.strip()]

def normalize_name(nm: str) -> str:
    """Remove quotes/backticks; keep module-qualified names if present."""
    return nm.strip('`"').strip()

_NAME_LIKE = re.compile(r'^[A-Za-z_][A-Za-z0-9_\.]*$')

def cleanup_bracket_item(x: str) -> str | None:
    """
    Clean one item from [ ... ]:
      - remove direction '←' or '<-'
      - remove leading '-' (disabling)
      - strip quotes/backticks
      - keep only name-like tokens (A.z, foo_bar, etc.)
    Return None if it doesn't look like a global lemma.
    """
    x = x.strip()
    if not x:
        return None
    # strip direction markers
    if x.startswith("←"):
        x = x[1:].lstrip()
    if x.startswith("<-"):
        x = x[2:].lstrip()
    # strip disabling '-'
    while x.startswith("-"):
        x = x[1:].lstrip()
    x = normalize_name(x)
    # very conservative filter
    if not _NAME_LIKE.fullmatch(x):
        return None
    return x

def parse_bracket_names(content: str) -> list[str]:
    """Parse [a, ←b, -c] -> ['a','b'] (drop disabled/non-name-like items)."""
    out: list[str] = []
    for raw in split_list(content):
        nm = cleanup_bracket_item(raw)
        if nm:
            out.append(nm)
    return out

def map_names_to_ids(idx, names: list[str]) -> list[int]:
    """
    Map global names to global lemma IDs using the lemma index.
    Fall back to base name lookup when full name is missing/ambiguous.
    """
    ids: list[int] = []
    for nm in names:
        nm = normalize_name(nm)
        if nm in idx["name2id"]:
            ids.append(idx["name2id"][nm])
        else:
            base = nm.split(".")[-1]
            ids_for_base = idx["base2ids"].get(base, [])
            if ids_for_base:
                # pick the first deterministically; later you can disambiguate by module
                ids.append(ids_for_base[0])
    return ids

# ---------------------------------
# Action parser -> (template, lemma-names)
# ---------------------------------

def parse_action(action: str) -> tuple[str|None, list[str]]:
    a = action.strip()

    # Basic structural tactics (no lemmas)
    if a.startswith("constructor"):
        return "constructor", []
    if a.startswith("intro"):
        return "intro", []
    if a.startswith("rcases"):
        return "rcases", []

    # apply NAME
    m = RE_APPLY.match(a)
    if m:
        nm = normalize_name(m.group(1))
        return "apply", [nm]

    # exact NAME  (try to extract a lemma if NAME looks global; otherwise leave empty list)
    m = RE_EXACT.match(a)
    if m:
        nm = normalize_name(m.group(1))
        return "exact", [nm]  # will be mapped to ID only if global

    # rw [ ... ] (with optional 'at ...')
    m = RE_RW.match(a)
    if m:
        items = parse_bracket_names(m.group(1))
        return "rw", items

    # simp variants; if no bracket list, we output empty lemma list
    m = RE_SIMP.match(a)
    if m:
        items = []
        if m.group(1):
            items = parse_bracket_names(m.group(1))
        return "simp", items

    # default: unrecognized; drop row
    return None, []

# ---------------------------------
# Main
# ---------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True, help="JSONL with fields: input, action, ...")
    ap.add_argument("--index", required=True, help="lemma_index.json produced from dumper")
    ap.add_argument("--out",   required=True, help="Output JSONL with input/template/lemmas/lemma_ids")
    a = ap.parse_args()

    idx = L.load(Path(a.index))
    out = Path(a.out)

    n_all = n_kept = n_with_lemmas = n_with_lemma_ids = 0

    with open(a.pairs, "r", encoding="utf-8") as f_in, \
         open(out, "w", encoding="utf-8") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            n_all += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = rec.get("input", "")
            action = rec.get("action", "")
            if not isinstance(text, str) or not isinstance(action, str) or not text or not action:
                continue

            tmpl, names = parse_action(action)
            if tmpl is None:
                continue

            lemma_ids = map_names_to_ids(idx, names)
            if names:
                n_with_lemmas += 1
            if lemma_ids:
                n_with_lemma_ids += 1

            out_rec = {
                "input": text,
                "template": tmpl,
                "lemmas": names,
                "lemma_ids": lemma_ids,
            }
            f_out.write(json.dumps(out_rec) + "\n")
            n_kept += 1

    # Stats
    print(f"[build_template_dataset] read={n_all}  kept={n_kept}  "
          f"with_lemmas(text)={n_with_lemmas}  with_lemma_ids(mapped)={n_with_lemma_ids}")
    if n_with_lemma_ids == 0:
        print("[warn] No lemma IDs mapped. Check that your index covers names and that actions actually contain lemma names.")
        print("       Spot-check a few rows with 'simp [foo]' / 'rw [bar]' / 'apply baz' / 'exact qux'.")

if __name__ == "__main__":
    main()