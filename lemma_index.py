#!/usr/bin/env python3
"""
lemma_index.py
Build and query a compact lemma index with ids, names, basenames, modules, and BM25 data.

Build:
  python3 lemma_index.py build --lemmas lean_env/data/lemmas.jsonl --out lean_env/data/lemma_index.json

Lookup (by name/basename):
  python3 lemma_index.py lookup --index lean_env/data/lemma_index.json --name and_comm

Retrieve (BM25 demo):
  python3 lemma_index.py search --index lean_env/data/lemma_index.json --query "p ∧ q ↔ q ∧ p" --k 10
"""
import argparse, json, math
from pathlib import Path
from typing import List, Dict, Tuple

def basename(full: str) -> str:
    return full.split('.')[-1]

def tok(s: str) -> List[str]:
    s = s.replace("\n"," ").replace(";", " ").replace(",", " ")
    for sym in ["↔","→","¬","∧","∨","∀","∃","=","≠","≤","≥","⊆","⊂","∈","∉","∪","∩"]:
        s = s.replace(sym, f" {sym} ")
    return [t for t in s.lower().split() if t]

def build(lemmas_jsonl: Path, out_path: Path):
    docs = []
    with open(lemmas_jsonl, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip(): continue
            r = json.loads(line)
            full = r["name"]
            docs.append({
                "id": i,
                "name": full,
                "base": basename(full),
                "module": r.get("module",""),
                "stmt": r.get("stmt","")
            })
    # inverted maps
    name2id = {d["name"]: d["id"] for d in docs}
    base2ids: Dict[str, List[int]] = {}
    for d in docs:
        base2ids.setdefault(d["base"], []).append(d["id"])
    # BM25 bits
    N = len(docs)
    for d in docs:
        d["tokens"] = tok(d["stmt"] + " " + d["name"].replace(".", " "))
    df: Dict[str,int] = {}
    for d in docs:
        for w in set(d["tokens"]):
            df[w] = df.get(w, 0) + 1
    idf = {w: math.log((N - c + 0.5)/(c + 0.5) + 1.0) for w, c in df.items()}
    for d in docs:
        tf: Dict[str,int] = {}
        for w in d["tokens"]:
            tf[w] = tf.get(w, 0) + 1
        d["tf"] = tf
        d["len"] = len(d["tokens"])
        del d["tokens"]
    out = {"docs": docs, "name2id": name2id, "base2ids": base2ids, "idf": idf, "N": N}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f)
    print(f"Wrote {len(docs)} lemmas → {out_path}")

def load(index_path: Path):
    with open(index_path, "r", encoding="utf-8") as f:
        return json.load(f)

def bm25(idx, query: str, k1=1.5, b=0.75) -> List[Tuple[int,float]]:
    q = tok(query)
    docs = idx["docs"]; idf = idx["idf"]
    avgdl = sum(d["len"] for d in docs)/max(1,len(docs))
    out: List[Tuple[int,float]] = []
    for d in docs:
        s = 0.0
        for w in q:
            if w not in d["tf"] or w not in idf: continue
            tf = d["tf"][w]; dl = d["len"]
            s += idf[w] * (tf * (k1+1)) / (tf + k1*(1 - b + b*dl/avgdl))
        if s: out.append((d["id"], s))
    out.sort(key=lambda x: x[1], reverse=True)
    return out

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("--lemmas"); b.add_argument("--out")
    l = sub.add_parser("lookup"); l.add_argument("--index"); l.add_argument("--name")
    s = sub.add_parser("search"); s.add_argument("--index"); s.add_argument("--query"); s.add_argument("--k", type=int, default=10)
    a = ap.parse_args()
    if a.cmd == "build":
        build(Path(a.lemmas), Path(a.out))
    elif a.cmd == "lookup":
        idx = load(Path(a.index))
        nm = a.name
        ids = []
        if nm in idx["name2id"]:
            ids = [idx["name2id"][nm]]
        elif nm in idx["base2ids"]:
            ids = idx["base2ids"][nm]
        print([idx["docs"][i] for i in ids])
    else:
        idx = load(Path(a.index))
        res = bm25(idx, a.query)[:a.k]
        for i, s in res:
            d = idx["docs"][i]
            print(f"{s:8.3f} {d['name']}  [{d['module']}]")
if __name__ == "__main__":
    main()