#!/usr/bin/env python3
# two_stages_auto_prove.py
# - Template policy (single head)
# - Lemma policy (single head or multi-head checkpoints)
# - Greedy loop; logs via logStepAuto; post-success cleanup.

import argparse, json, os, subprocess, time, uuid, re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import torch
import torch.nn as nn

# ------------------------------
# Small model bricks
# ------------------------------

class EmbBagEncoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.emb = nn.EmbeddingBag(vocab_size, d_model, mode="mean")
    def forward(self, tokens, offsets):
        return self.emb(tokens, offsets)

class EmbBagClassifier(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_classes: int):
        super().__init__()
        self.encoder = EmbBagEncoder(vocab_size, d_model)
        self.head    = nn.Linear(d_model, num_classes)
    def forward(self, tokens, offsets):
        return self.head(self.encoder(tokens, offsets))

class EmbBagMultiHeadClassifier(nn.Module):
    """Shared encoder + one linear head per template."""
    def __init__(self, vocab_size: int, d_model: int, dims: List[int]):
        super().__init__()
        self.encoder = EmbBagEncoder(vocab_size, d_model)
        self.heads   = nn.ModuleList([nn.Linear(d_model, n) for n in dims])
    def forward_head(self, head_idx: int, tokens, offsets):
        h = self.encoder(tokens, offsets)
        return self.heads[head_idx](h)

# ------------------------------
# Robust state_dict loader
# ------------------------------

def _load_state_dict_compat(model: nn.Module, state: Dict[str, Any], *, map_to: str):
    """
    map_to: 'template' (EmbBagClassifier) or 'lemmas-single' (EmbBagClassifier) or
            'lemmas-multi' (EmbBagMultiHeadClassifier)
    Renames common prefixes:
      - 'emb.weight' or 'encoder.emb.weight' -> model.encoder.emb.weight
      - 'fc.*' or 'head.*' -> model.head.* (single) or model.heads.0.* (multi)
      - 'lemma_heads.N.*'  -> model.heads.N.*
      - 'heads.N.*'        -> model.heads.N.*
    """
    new = {}
    for k, v in state.items():
        nk = k

        # strip optional leading 'model.'
        if nk.startswith("model."):
            nk = nk[len("model."):]

        # encoder mapping
        if nk == "emb.weight":
            nk = "encoder.emb.weight"
        if nk.startswith("encoder.emb."):
            pass
        if nk.startswith("emb.") and not nk.startswith("encoder.emb."):
            nk = "encoder." + nk  # emb.weight -> encoder.emb.weight

        # heads mapping
        if map_to in ("template", "lemmas-single"):
            if nk.startswith("fc."):
                nk = "head." + nk[len("fc."):]
            if nk.startswith("head."):
                pass
            if nk.startswith("heads.0."):
                nk = "head." + nk[len("heads.0."):]
            if nk.startswith("lemma_heads.0."):
                nk = "head." + nk[len("lemma_heads.0."):]
        elif map_to == "lemmas-multi":
            if nk.startswith("lemma_heads."):
                nk = "heads." + nk[len("lemma_heads."):]
            if nk.startswith("fc."):
                nk = "heads.0." + nk[len("fc."):]
            if nk.startswith("head."):
                nk = "heads.0." + nk[len("head."):]
        new[nk] = v

    missing, unexpected = model.load_state_dict(new, strict=False)
    return missing, unexpected

# ------------------------------
# Vocab/policy wrappers
# ------------------------------

def _tok(s: str): return s.replace("\n", " ").split()

@dataclass
class TemplatePolicy:
    model: EmbBagClassifier
    vocab: Dict[str,int]
    id2template: Dict[int,str]  # class id -> template string

    @classmethod
    def load(cls, ckpt_path: Path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        vocab = ckpt["vocab"]
        # template mapping (support several names)
        if "id2template" in ckpt:
            if isinstance(ckpt["id2template"], dict):
                id2template = {int(k): v for k, v in ckpt["id2template"].items()}
            else:
                id2template = {i: s for i, s in enumerate(ckpt["id2template"])}
        elif "act2id" in ckpt:
            id2template = {v: k for k, v in ckpt["act2id"].items()}
        elif "template2id" in ckpt:
            id2template = {v: k for k, v in ckpt["template2id"].items()}
        else:
            raise KeyError("Template ckpt lacks class mapping (need id2template/act2id/template2id).")

        num_classes = 1 + max(id2template.keys()) if id2template else 0
        d_model = ckpt.get("d_model", 128)
        model = EmbBagClassifier(len(vocab), d_model, num_classes)
        state = ckpt["model"]
        _load_state_dict_compat(model, state, map_to="template")
        model.eval()
        return cls(model, vocab, id2template)

    def topk(self, text: str, k: int) -> List[Tuple[str,float]]:
        ids = [self.vocab.get(t, 0) for t in _tok(text)]
        if not ids:
            return [(self.id2template[i], 0.0) for i in range(min(k, len(self.id2template)))]
        tokens = torch.tensor(ids, dtype=torch.long)
        offsets = torch.tensor([0], dtype=torch.long)
        with torch.no_grad():
            probs = torch.softmax(self.model(tokens, offsets), dim=-1).squeeze(0)
            top = torch.topk(probs, k=min(k, probs.numel()))
        return [(self.id2template[i], float(p)) for i, p in zip(top.indices.tolist(), top.values.tolist())]

@dataclass
class LemmaPolicy:
    """Multi-head or single-head lemma classifier with id->name mapping."""
    model: nn.Module  # EmbBagMultiHeadClassifier or EmbBagClassifier
    vocab: Dict[str,int]
    template2id: Dict[str,int]  # which head to use; if single head, map everything to 0
    head_class_to_lemma_id: List[Dict[int,int]]  # local class -> global lemma id
    lemma_id2name: Dict[int,str]
    d_model: int

    @classmethod
    def _infer_multi_dims_from_state(cls, state: Dict[str, Any]) -> List[int]:
        """Build dims list where dims[hid] = out_features of head hid, from checkpoint keys."""
        head_out: Dict[int, int] = {}
        for k, v in state.items():
            if k.endswith(".weight"):
                m = re.match(r"(?:lemma_heads|heads)\.(\d+)\.weight$", k)
                if m:
                    hid = int(m.group(1))
                    head_out[hid] = int(v.shape[0])
        if not head_out:
            return []
        max_h = max(head_out.keys())
        dims = [1] * (max_h + 1)
        for hid, outdim in head_out.items():
            dims[hid] = outdim
        return dims

    @classmethod
    def _normalize_head_maps(cls, raw_maps: Any, dims: List[int]) -> List[Dict[int,int]]:
        """
        Accepts:
          - list of dicts (local->global) or list of lists,
          - dict {head_idx: dict_or_list}
          - 'lemma_id_rev_maps': global->local (will invert)
        Returns list indexed by head, mapping local->global.
        """
        def to_local2global(obj) -> Dict[int,int]:
            if isinstance(obj, dict):
                return {int(k): int(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return {i: int(v) for i, v in enumerate(obj)}
            else:
                return {}

        if raw_maps is None:
            # identity per head
            return [{i: i for i in range(d)} for d in dims]

        if isinstance(raw_maps, list):
            out: List[Dict[int,int]] = []
            for i, item in enumerate(raw_maps):
                m = to_local2global(item)
                if not m and i < len(dims):
                    m = {j: j for j in range(dims[i])}
                out.append(m)
            while len(out) < len(dims):
                out.append({j: j for j in range(dims[len(out)])})
            return out

        if isinstance(raw_maps, dict):
            max_h = len(dims)
            out = []
            for hid in range(max_h):
                item = raw_maps.get(str(hid), raw_maps.get(hid))
                m = to_local2global(item) if item is not None else {j: j for j in range(dims[hid])}
                out.append(m)
            return out

        return [{i: i for i in range(d)} for d in dims]

    @classmethod
    def load(cls, ckpt_path: Path, lemma_index_path: Optional[Path] = None):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        vocab = ckpt["vocab"]
        d_model = ckpt.get("d_model", 128)

        # template ids for heads
        if "template2id" in ckpt:
            template2id = ckpt["template2id"]
        elif "id2template" in ckpt:
            tmp = ckpt["id2template"]
            if isinstance(tmp, dict):
                template2id = {v: int(k) for k, v in tmp.items()}
            else:
                template2id = {s: i for i, s in enumerate(tmp)}
        else:
            template2id = {"_default": 0}

        # names: from ckpt or lemma_index.json
        lemma_id2name: Dict[int, str] = {}
        for key in ("id2name", "lemma_id2name"):
            if key in ckpt:
                v = ckpt[key]
                if isinstance(v, dict):
                    lemma_id2name = {int(k): str(vv) for k, vv in v.items()}
                elif isinstance(v, list):
                    lemma_id2name = {i: s for i, s in enumerate(v)}
                break
        if not lemma_id2name and lemma_index_path:
            with open(lemma_index_path, "r", encoding="utf-8") as f:
                idx = json.load(f)
            v = idx.get("id2name") or idx.get("lemma_id2name")
            if isinstance(v, dict):
                lemma_id2name = {int(k): str(vv) for k, vv in v.items()}
            elif isinstance(v, list):
                lemma_id2name = {i: s for i, s in enumerate(v)}

        # Build model: detect multi-head via keys
        state = ckpt["model"]
        keys = list(state.keys())
        is_multi = any(k.startswith(("lemma_heads.", "heads.")) for k in keys)

        if is_multi:
            dims = cls._infer_multi_dims_from_state(state)
            if not dims:
                dims_raw = ckpt.get("lemma_dims")
                if isinstance(dims_raw, dict):
                    max_h = max(int(k) for k in dims_raw) if dims_raw else -1
                    dims = [1] * (max_h + 1)
                    for k, v in dims_raw.items():
                        dims[int(k)] = int(v)
                elif isinstance(dims_raw, list):
                    dims = [int(x) for x in dims_raw]
                else:
                    raise RuntimeError("Cannot infer multi-head dims from checkpoint; missing shapes and 'lemma_dims'.")

            model = EmbBagMultiHeadClassifier(len(vocab), d_model, dims)
            _load_state_dict_compat(model, state, map_to="lemmas-multi")

            # head maps: prefer forward maps; if only reverse present, invert
            raw_maps = ckpt.get("lemma_id_maps")
            if raw_maps is None and "lemma_id_rev_maps" in ckpt:
                rev = ckpt["lemma_id_rev_maps"]
                if isinstance(rev, list):
                    raw_maps = []
                    for item in rev:
                        if isinstance(item, dict):
                            inv = {int(v): int(k) for k, v in item.items()}
                        elif isinstance(item, list):
                            inv = {int(v): i for i, v in enumerate(item)}
                        else:
                            inv = {}
                        raw_maps.append(inv)
                elif isinstance(rev, dict):
                    raw_maps = {}
                    for hk, item in rev.items():
                        if isinstance(item, dict):
                            inv = {int(v): int(k) for k, v in item.items()}
                        elif isinstance(item, list):
                            inv = {int(v): i for i, v in enumerate(item)}
                        else:
                            inv = {}
                        raw_maps[hk] = inv

            head_maps = cls._normalize_head_maps(raw_maps, dims)

        else:
            # single head
            num_classes = None
            for key in ("fc.weight", "head.weight", "heads.0.weight"):
                if key in state:
                    num_classes = int(state[key].shape[0])
                    break
            if num_classes is None:
                raise RuntimeError("Lemma checkpoint doesn't expose a recognizable classifier head.")
            model = EmbBagClassifier(len(vocab), d_model, num_classes)
            _load_state_dict_compat(model, state, map_to="lemmas-single")
            head_maps = [{i: i for i in range(num_classes)}]
            template2id = {k: 0 for k in template2id.keys()}

        model.eval()
        return cls(model, vocab, template2id, head_maps, lemma_id2name, d_model)

    def topk(self, text: str, template: str, k: int) -> List[Tuple[str,float]]:
        """
        Return lemma *names* (not ids) with probabilities for given template.
        Unknown lemma IDs (not found in id->name mapping) are SKIPPED
        to avoid emitting placeholders like 'lemma_2089'.
        """
        ids = [self.vocab.get(t, 0) for t in text.replace("\n"," ").split()]
        tokens = torch.tensor(ids if ids else [0], dtype=torch.long)
        offsets = torch.tensor([0], dtype=torch.long)

        head_idx = self.template2id.get(template, self.template2id.get("_default", 0))

        with torch.no_grad():
            if isinstance(self.model, EmbBagMultiHeadClassifier):
                head_idx = min(head_idx, len(self.head_class_to_lemma_id)-1)
                logits = self.model.forward_head(head_idx, tokens, offsets)
            else:
                head_idx = 0
                logits = self.model(tokens, offsets)
            probs = torch.softmax(logits, dim=-1).squeeze(0)
            top = torch.topk(probs, k=min(k, probs.numel()))
            local2global = self.head_class_to_lemma_id[head_idx] if head_idx < len(self.head_class_to_lemma_id) else {}
            lemmas: List[Tuple[str,float]] = []
            seen: set[str] = set()
            for cls_id, p in zip(top.indices.tolist(), top.values.tolist()):
                lemma_id = local2global.get(int(cls_id), int(cls_id))
                name = self.lemma_id2name.get(lemma_id)
                if not name:
                    continue  # skip unknown → prevents 'lemma_1234'
                if name not in seen:
                    lemmas.append((name, float(p)))
                    seen.add(name)
            return lemmas

# ------------------------------
# Build / JSONL helpers
# ------------------------------

def load_lemma_index_json(path: Optional[str | Path]):
    """
    Returns (id2name, name2id) dicts or ({}, {}) if path is None/missing.
    Supports files produced by your index builder:
      {
        "name2id": {"Nat.cast_add": 247, ...},
        "id2name": {"247": "Nat.cast_add", ...}
      }
    """
    if not path:
        return {}, {}
    p = Path(path)
    if not p.exists():
        return {}, {}
    with p.open("r", encoding="utf-8") as f:
        idx = json.load(f)
    if "id2name" in idx and "name2id" in idx:
        id2name = {int(k): v for k, v in idx["id2name"].items()}
        name2id = {k: int(v) for k, v in idx["name2id"].items()}
        return id2name, name2id
    if "name2id" in idx:
        name2id = {k: int(v) for k, v in idx["name2id"].items()}
        id2name = {v: k for k, v in name2id.items()}
        return id2name, name2id
    if "id2name" in idx:
        id2name = {int(k): v for k, v in idx["id2name"].items()}
        name2id = {v: k for k, v in id2name.items()}
        return id2name, name2id
    return {}, {}

def run_build(cmd_base: str, run_id: str, run_log: Path, cwd: Path) -> Tuple[int, str, str]:
    full_cmd = f'RUN_ID_OVERRIDE="{run_id}" RUN_JSON_PATH="{run_log}" RUN_SOURCE="auto-prove" {cmd_base}'
    p = subprocess.Popen(["bash", "-lc", full_cmd], cwd=str(cwd),
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
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

# ------------------------------
# Template rendering
# ------------------------------

TACTIC_MARKER   = "-- @TACTICS@"
REQUIRED_IMPORT = "import LeanEnv.ExtractionTactic"

def render_block(tactics: List[str], indent: str) -> List[str]:
    if not tactics:
        return [f"{indent}try logStepAuto (skip)"]
    out = []
    for t in map(str.strip, tactics):
        out.append(f"{indent}try logStepAuto ({t})" if ";" in t else f"{indent}try logStepAuto {t}")
    return out

def write_from_template(template: Path, out: Path, tactics: List[str]) -> None:
    src = template.read_text(encoding="utf-8")
    if REQUIRED_IMPORT not in src:
        src = REQUIRED_IMPORT + "\n" + src
    lines = src.splitlines()
    try:
        idx = next(i for i, ln in enumerate(lines) if ln.strip() == TACTIC_MARKER)
    except StopIteration:
        raise SystemExit(f"Template is missing marker {TACTIC_MARKER!r} on its own line")
    indent = lines[idx][:len(lines[idx]) - len(lines[idx].lstrip())]
    block_lines = render_block(tactics, indent)
    new_lines = lines[:idx] + block_lines + lines[idx+1:]
    out.write_text("\n".join(new_lines) + "\n", encoding="utf-8")

# ------------------------------
# Log interpretation & scoring
# ------------------------------

def current_goal_and_ctx(rec: dict) -> Tuple[str, List[Tuple[str,str]]]:
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
    arr = rec.get("goals_after")
    if not isinstance(arr, list): arr = rec.get("goals_before", [])
    return [g.get("target","") for g in arr]

def status_ok(rec: dict) -> bool:
    return (rec.get("status") or rec.get("meta",{}).get("status")) == "ok"

def measure_targets(targets: List[str]) -> Tuple[int,int,int,int]:
    num_iff = sum(t.count("↔") for t in targets)
    num_imp = sum(t.count("→") for t in targets) + sum(t.count(" -> ") for t in targets)
    total_len = sum(len(t) for t in targets)
    n_goals = len(targets)
    return (num_iff, num_imp, total_len, n_goals)

def _read_rows_with_retry(run_log: Path, run_id: str, decl: str,
                          prev_n: int, wait_s: float, retries: int = 6, verbose: bool=False):
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

# ------------------------------
# Heuristics
# ------------------------------

def top_has_iff(goal: str) -> bool:
    return "↔" in goal and "→" not in goal

def top_has_and(goal: str) -> bool:
    return (" ∧ " in goal) and ("→" not in goal) and ("↔" not in goal)

def heuristic_candidates(goal: str, ctx: List[Tuple[str,str]]) -> List[str]:
    add: List[str] = []
    # NEW: if equality and not a negation, try rfl as a cheap finisher
    if (" = " in goal) and ("≠" not in goal):
        add.append("rfl")
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
    prev_names = {n for (n,_) in prev_ctx}
    new_names  = {n for (n,_) in new_ctx}
    return len(new_names) > len(prev_names)

# ------------------------------
# Tactic synthesis from (template, lemmas)
# ------------------------------

def order_candidates(goal: str,
                     ctx: List[Tuple[str,str]],
                     policy_templates: List[str],
                     heuristic_list: List[str]) -> List[str]:
    """
    Put heuristics first; drop unsafe 'exact' on implication goals.
    De-duplicate while preserving order.
    """
    seen, out = set(), []
    forbid_exact = ("→" in goal) or (" -> " in goal)
    for a in heuristic_list + policy_templates:
        if a == "exact" and forbid_exact:
            continue
        if a not in seen:
            out.append(a); seen.add(a)
    return out

def needs_lemmas(template: str) -> bool:
    t = template.strip().lower()
    return t in {"exact", "apply", "rw", "simp", "simp_all"}

def build_tactic(template: str, lemmas: List[str]) -> Optional[str]:
    """Return a concrete tactic string, or None if we can't safely build it."""
    t = template.strip()
    low = t.lower()

    if low == "apply":
        return f"apply {lemmas[0]}" if lemmas else None

    if low in ("simp", "simp_all"):
        payload = ", ".join(lemmas[:3]) if lemmas else ""
        return f"simp [{payload}]" if payload else None

    if low == "rw":
        payload = ", ".join(lemmas[:3]) if lemmas else ""
        return f"rw [{payload}]" if payload else None

    if low == "exact":
        return f"exact {lemmas[0]}" if lemmas else None

    # templates that don’t require lemmas (constructor/intro/rcases/rfl/etc.)
    return t

# ------------------------------
# Greedy loop
# ------------------------------

def greedy_prove(tmpl_policy: TemplatePolicy,
                 lemma_policy: Optional[LemmaPolicy],
                 template: Path, out: Path, base_log_dir: Path,
                 decl: str, build_cmd: str, project_root: Path,
                 topk_templates: int, topk_lemmas: int, max_steps: int,
                 wait_s: float, verbose: bool):
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

        # ---- templates (policy) + heuristics
        cand_templates = [tpl for tpl,_ in tmpl_policy.topk("GOAL: " + prev_goal, topk_templates)]
        heurs = heuristic_candidates(prev_goal, prev_ctx)
        cand = order_candidates(prev_goal, prev_ctx, cand_templates, heurs)

        if verbose:
            print("[templates]")
            for i, tpl in enumerate(cand, 1):
                print(f"  {i}. {tpl}")

        # ---- Try candidates; for those that need lemmas, query lemma_policy
        progressed = False
        for tpl in cand:
            tactic: Optional[str] = None
            if needs_lemmas(tpl):
                if lemma_policy is not None:
                    lemmas = [name for name,_ in lemma_policy.topk("GOAL: " + prev_goal, tpl, topk_lemmas)]
                    tactic = build_tactic(tpl, lemmas)
                if tactic is None:
                    if verbose:
                        print(f"  - SKIP '{tpl}' (no lemma payload available)")
                    continue
            else:
                tactic = build_tactic(tpl, [])

            trial = tactics + [tactic]
            write_from_template(template, out, trial)
            code2, out2, err2 = run_build(build_cmd, run_id, run_log, project_root)
            if verbose and code2 != 0:
                print(f"  - TRY '{tactic}' → build code={code2} (ok if it still logged)")

            rows_all2, rows2, grew2 = _read_rows_with_retry(
                run_log, run_id, decl, prev_n=n_rows, wait_s=wait_s, verbose=verbose)
            if not grew2:
                if verbose: print(f"  - TRY '{tactic}' → no new log rows; skipping")
                write_from_template(template, out, tactics)
                continue

            n_rows = len(rows_all2)
            last2 = rows2[-1] if rows2 else rows_all2[-1]
            ok2 = status_ok(last2)
            new_targets = targets_after(last2)
            new_score   = measure_targets(new_targets)
            new_goal, new_ctx = current_goal_and_ctx(last2)

            if verbose:
                print(f"  - TRY '{tactic}' → status={'ok' if ok2 else 'error'} ; score {prev_score} -> {new_score}")

            gained_ctx = context_gain(prev_ctx, new_ctx)
            accept = ok2 and ( (new_score < prev_score)
                            or (productive_step(prev_goal, tactic) and (new_score != prev_score or gained_ctx))
                            or gained_ctx )

            if accept:
                tactics = trial
                progressed = True
                if new_score[3] == 0:
                    if verbose: print("[done] proof closed by accepted tactic.")
                    return True, tactics, run_log
                break
            else:
                write_from_template(template, out, tactics)

        if not progressed:
            if verbose: print("[stuck] no candidate improved the score; stopping.")
            return False, tactics, run_log

    if verbose: print("[budget] max_steps reached.")
    return False, tactics, run_log

# ------------------------------
# Cleanup
# ------------------------------

PAT_PAREN = re.compile(r'^(\s*)try\s+logStepAuto(?:Soft)?\s*\(\s*(.+)\s*\)\s*$')
PAT_PLAIN = re.compile(r'^(\s*)try\s+logStepAuto(?:Soft)?\s+(.+?)\s*$')
IMPORT_LINE = re.compile(r'^\s*import\s+LeanEnv\.ExtractionTactic\s*$')

def cleanup_proof_file(out_path: Path, remove_import_if_unused: bool = True) -> None:
    txt = out_path.read_text(encoding="utf-8")
    new_lines: List[str] = []
    for line in txt.splitlines():
        m = PAT_PAREN.match(line) or PAT_PLAIN.match(line)
        if m:
            indent, body = m.group(1), m.group(2).strip()
            if body == "skip":
                continue
            new_lines.append(f"{indent}{body}")
        else:
            new_lines.append(line)
    new_txt = "\n".join(new_lines) + "\n"
    if remove_import_if_unused and ("logStepAuto" not in new_txt):
        new_txt = "\n".join(ln for ln in new_txt.splitlines() if not IMPORT_LINE.match(ln)) + "\n"
    new_txt = re.sub(r'\n{3,}', '\n\n', new_txt)
    out_path.write_text(new_txt, encoding="utf-8")

# ------------------------------
# CLI
# ------------------------------

def main():
    ap = argparse.ArgumentParser(description="Two-stage auto-prover (template + lemma policies).")
    ap.add_argument("--ckpt_template", required=True, help="Template classifier checkpoint")
    ap.add_argument("--ckpt_lemmas",   default=None,     help="Lemma classifier checkpoint (multi-head or single)")
    ap.add_argument("--lemma_index",   default=None,     help="lemma_index.json to map lemma_id -> name (used if ckpt lacks names)")
    ap.add_argument("--template", required=True)
    ap.add_argument("--out",      required=True)
    ap.add_argument("--log", default=None, help="Only to pick a directory; per-run file is auto-<uuid>.jsonl")
    ap.add_argument("--build", default="lake env lean Main.lean")
    ap.add_argument("--project_root", default=".")
    ap.add_argument("--decl", required=True)
    ap.add_argument("--topk_templates", type=int, default=5)
    ap.add_argument("--topk_lemmas",    type=int, default=10)
    ap.add_argument("--max_steps", type=int, default=50)
    ap.add_argument("--wait_s",    type=float, default=0.5)
    ap.add_argument("--verbose",   action="store_true")
    ap.add_argument("--keep_wrappers", action="store_true")
    args = ap.parse_args()

    tmpl_policy  = TemplatePolicy.load(Path(args.ckpt_template))
    lemma_policy = None
    if args.ckpt_lemmas:
        lemma_policy = LemmaPolicy.load(Path(args.ckpt_lemmas),
                                        Path(args.lemma_index) if args.lemma_index else None)

    template = Path(args.template)
    out      = Path(args.out)
    project  = Path(args.project_root).resolve()
    base_dir = (Path(args.log).resolve().parent if args.log else (project/"data").resolve())

    ok, script, run_log = greedy_prove(
        tmpl_policy, lemma_policy, template, out, base_dir,
        args.decl, args.build, project,
        args.topk_templates, args.topk_lemmas, args.max_steps, args.wait_s, args.verbose
    )

    if ok and not args.keep_wrappers:
        cleanup_proof_file(out)
        if args.verbose:
            print(f"[clean] stripped logStepAuto wrappers in {out}")

    print("\n=== RESULT ===")
    print("success:", ok)
    print("tactics:", script)
    print("per-run log:", run_log)

if __name__ == "__main__":
    main()