#!/usr/bin/env python3
"""
two_stage_prover.py
-------------------
Two-stage learning pipeline for Lean tactic prediction:

Stage 1 (template classification)
  input (goal+context)  --> template ∈ {constructor, intro, rcases, rw, simp, exact, ...}

Stage 2 (lemma retrieval/classification, conditional on template)
  input + template      --> multi-label set of lemma IDs (only for templates that use lemmas)

Input JSONL (from your template builder):
  {
    "input": str,
    "template": str,
    "lemmas": [str],        # global names (may be empty)
    "lemma_ids": [int],     # global lemma IDs (aligned with "lemmas")
    "locals": [str]?        # (optional) local names -- ignored here
  }

Checkpoints:
  - Template:  outdir_template/template.pt
  - Lemmas:    outdir_lemmas/lemmas.pt

CLI:
  Train template model:
    python3 two_stage_prover.py --mode train_template --infile data/template_lemmas.jsonl --outdir runs/template

  Train lemma model (optionally init encoder from template ckpt):
    python3 two_stage_prover.py --mode train_lemmas --infile data/template_lemmas.jsonl \
        --outdir runs/lemmas --init_from_template runs/template/template.pt

  Predict template only:
    python3 two_stage_prover.py --mode predict_template --template_ckpt runs/template/template.pt \
        --text "GOAL: ... CONTEXT: ..."

  Predict lemmas (full pipeline):
    python3 two_stage_prover.py --mode predict --template_ckpt runs/template/template.pt \
        --lemma_ckpt runs/lemmas/lemmas.pt --text "GOAL: ... CONTEXT: ..." --topk 5

Notes:
  * Lemma heads exist only for templates seen with non-empty lemma_ids in training (e.g., 'simp','rw','apply').
  * Lemma prediction returns original global lemma IDs (the ones from your index).
"""

import os, json, argparse, random, sys
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Optional tqdm
try:
    from tqdm import tqdm
    _HAVE_TQDM = True
except Exception:
    _HAVE_TQDM = False


# =========================
# Utilities & tokenization
# =========================

def log(msg: str, *, flush=True):
    print(msg, flush=flush)

def tokenize(s: str) -> List[str]:
    # very simple whitespace tokenizer; keeps numbers/symbols
    return s.replace("\n", " ").split()

def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def build_vocab(samples: List[str]) -> Dict[str,int]:
    vocab = {"<unk>": 0}
    for s in samples:
        for t in tokenize(s):
            if t not in vocab:
                vocab[t] = len(vocab)
    return vocab


# =========================
# Datasets
# =========================

class TemplateDataset(Dataset):
    """For stage 1: template classification."""
    def __init__(self, records, vocab, template2id):
        self.records = records
        self.vocab = vocab
        self.template2id = template2id
    def __len__(self): return len(self.records)
    def __getitem__(self, idx):
        r = self.records[idx]
        text = r["input"]
        tmpl = r["template"]
        ids = [self.vocab.get(t, 0) for t in tokenize(text)]
        return torch.tensor(ids, dtype=torch.long), self.template2id[tmpl]

class LemmaDataset(Dataset):
    """For stage 2: lemma prediction (multi-label) conditional on template.
       We keep only records with non-empty lemma_ids and whose template has a head."""
    def __init__(self, records, vocab, template2id, lemma_id_maps):
        self.vocab = vocab
        self.template2id = template2id
        self.lemma_id_maps = lemma_id_maps  # template_id -> {globalLemmaId: localIndex}
        self.samples = []
        for r in records:
            gl_ids = r.get("lemma_ids", [])
            tmpl = r["template"]
            if not gl_ids:
                continue
            if tmpl not in template2id:
                continue
            tid = template2id[tmpl]
            if tid not in lemma_id_maps:
                continue
            local_map = lemma_id_maps[tid]
            # map to local indices, drop those not in map (shouldn't happen if built from same data)
            loc = [local_map[g] for g in gl_ids if g in local_map]
            if not loc:
                continue
            self.samples.append((r["input"], tid, sorted(set(loc))))
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        text, tid, loc_ids = self.samples[idx]
        ids = [self.vocab.get(t, 0) for t in tokenize(text)]
        return torch.tensor(ids, dtype=torch.long), tid, torch.tensor(loc_ids, dtype=torch.long)

def collate_template(batch):
    token_tensors, tmpl_ids = zip(*batch)
    lengths = torch.tensor([len(t) for t in token_tensors], dtype=torch.long)
    offsets = torch.zeros(len(token_tensors), dtype=torch.long)
    if len(lengths) > 1:
        offsets[1:] = torch.cumsum(lengths[:-1], dim=0)
    tokens = torch.cat(token_tensors, dim=0) if token_tensors else torch.tensor([], dtype=torch.long)
    return tokens, offsets, torch.tensor(tmpl_ids, dtype=torch.long)

def collate_lemmas_b1(batch):
    # Keep it simple and robust: batch_size == 1 (set via CLI default or override).
    assert len(batch) == 1, "Use --lemma_batch_size 1 (the collate expects singletons)"
    (tokens, tid, loc_ids) = batch[0]
    offsets = torch.tensor([0], dtype=torch.long)
    return tokens, offsets, tid, loc_ids


# =========================
# Models
# =========================

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model=128):
        super().__init__()
        self.emb = nn.EmbeddingBag(vocab_size, d_model, mode="mean")
    def forward(self, tokens, offsets):
        return self.emb(tokens, offsets)  # [B, d_model]

class TemplateModel(nn.Module):
    def __init__(self, vocab_size, num_templates, d_model=128):
        super().__init__()
        self.encoder = Encoder(vocab_size, d_model)
        self.head = nn.Linear(d_model, num_templates)
    def forward(self, tokens, offsets):
        h = self.encoder(tokens, offsets)
        return self.head(h)  # [B, num_templates]

class LemmaModel(nn.Module):
    """
    Shared encoder + per-template lemma heads.
    - lemma_heads: ModuleDict of Linear layers, keyed by string template_id.
      Each head outputs |lemma_vocab(template)| logits.
    """
    def __init__(self, vocab_size, lemma_dims_by_template: Dict[int,int], d_model=128):
        super().__init__()
        self.encoder = Encoder(vocab_size, d_model)
        self.lemma_heads = nn.ModuleDict()
        for tid, dim in lemma_dims_by_template.items():
            self.lemma_heads[str(tid)] = nn.Linear(d_model, dim)

    def has_head(self, tid: int) -> bool:
        return str(tid) in self.lemma_heads

    def forward_head(self, tokens, offsets, tid: int):
        h = self.encoder(tokens, offsets)           # [B, d_model]
        head = self.lemma_heads[str(tid)]           # Linear(d_model, dim_t)
        return head(h)                              # [B, dim_t]


# =========================
# Loading & preprocessing
# =========================

def load_template_records(path: Path) -> List[dict]:
    recs = []
    for r in read_jsonl(path):
        # Must have template and input
        if not r.get("template") or not r.get("input"):
            continue
        # Normalize lemma_ids to list[int]
        lis = r.get("lemma_ids") or []
        if not isinstance(lis, list):
            lis = []
        recs.append({"input": r["input"], "template": r["template"], "lemma_ids": lis})
    if not recs:
        raise SystemExit(f"No usable records in {path}")
    return recs

def build_label_spaces(recs: List[dict]):
    """Build template label space and per-template lemma local vocabularies."""
    templates = sorted({r["template"] for r in recs})
    template2id = {t:i for i,t in enumerate(templates)}
    id2template = {i:t for t,i in template2id.items()}

    # Per-template global lemma IDs encountered (only from samples with lemmas)
    lemma_globals_by_tid: Dict[int,set] = defaultdict(set)
    for r in recs:
        if r["lemma_ids"]:
            tid = template2id[r["template"]]
            lemma_globals_by_tid[tid].update(r["lemma_ids"])

    # Map per-template global lemma IDs --> local contiguous indices
    lemma_id_maps: Dict[int,Dict[int,int]] = {}
    lemma_id_rev_maps: Dict[int,Dict[int,int]] = {}
    lemma_dims_by_template: Dict[int,int] = {}
    for tid, s in lemma_globals_by_tid.items():
        ordered = sorted(s)  # arbitrary but stable
        local = {g:i for i,g in enumerate(ordered)}
        rev   = {i:g for i,g in enumerate(ordered)}
        lemma_id_maps[tid] = local
        lemma_id_rev_maps[tid] = rev
        lemma_dims_by_template[tid] = len(ordered)

    return template2id, id2template, lemma_id_maps, lemma_id_rev_maps, lemma_dims_by_template


# =========================
# Training loops
# =========================

def train_template(recs, outdir, d_model=128, epochs=5, lr=1e-3, seed=0,
                   batch_size=64, val_batch_size=128, verbose=False, use_bar=True):
    random.seed(seed); torch.manual_seed(seed)

    # Split
    random.shuffle(recs)
    n = max(1, int(0.8 * len(recs)))
    train_recs, val_recs = recs[:n], recs[n:]

    # Vocab & labels
    vocab = build_vocab([r["input"] for r in train_recs])
    template2id, id2template, *_ = build_label_spaces(recs)

    # Datasets/loaders
    train_ds = TemplateDataset(train_recs, vocab, template2id)
    val_ds   = TemplateDataset(val_recs,   vocab, template2id)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_template)
    val_loader   = DataLoader(val_ds, batch_size=val_batch_size, shuffle=False,
                              collate_fn=collate_template)

    # Model/opt
    model = TemplateModel(len(vocab), len(template2id), d_model=d_model)
    opt = optim.Adam(model.parameters(), lr=lr)

    # Train
    for ep in range(1, epochs+1):
        model.train()
        it = tqdm(train_loader, desc=f"Template ep{ep}", unit="batch") if (use_bar and _HAVE_TQDM) else train_loader
        tot_loss = 0.0; nb = 0
        for tokens, offsets, tmpl_ids in it:
            logits = model(tokens, offsets)  # [B, T]
            loss = nn.functional.cross_entropy(logits, tmpl_ids)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot_loss += loss.item(); nb += 1
        tr_loss = tot_loss / max(1, nb)

        # Val
        model.eval()
        correct = 0; total = 0
        with torch.no_grad():
            for tokens, offsets, tmpl_ids in val_loader:
                logits = model(tokens, offsets)
                pred = logits.argmax(-1)
                correct += (pred == tmpl_ids).sum().item()
                total   += tmpl_ids.numel()
        acc = correct / total if total else 0.0
        log(f"[Template] epoch {ep:02d} | train_loss={tr_loss:.4f} | val_acc={acc:.3f}")

    # Save
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "vocab": vocab,
        "template2id": template2id,
        "id2template": id2template,
        "model": model.state_dict(),
        "d_model": d_model,
    }
    path = outdir / "template.pt"
    torch.save(ckpt, path)
    log(f"[Template] saved to {path}")
    return model, vocab, template2id, id2template

def train_lemmas(recs, outdir, d_model=128, epochs=5, lr=1e-3, seed=0,
                 lemma_batch_size=1, init_from_template: Optional[Path]=None,
                 verbose=False, use_bar=True):
    """
    Train per-template lemma heads with BCE-with-logits.
    We iterate samples one-by-one (batch_size=1) to avoid variable head dims complexity.
    """
    random.seed(seed); torch.manual_seed(seed)

    # Vocab & label spaces from ALL records (stable mappings)
    vocab = build_vocab([r["input"] for r in recs])
    template2id, id2template, lemma_id_maps, lemma_id_rev_maps, lemma_dims = build_label_spaces(recs)

    # Keep only records that have lemma supervision and whose template got a head
    train_recs = recs[:]  # simple split; could stratify
    random.shuffle(train_recs)
    n = max(1, int(0.8 * len(train_recs)))
    tr, va = train_recs[:n], train_recs[n:]
    train_ds = LemmaDataset(tr, vocab, template2id, lemma_id_maps)
    val_ds   = LemmaDataset(va, vocab, template2id, lemma_id_maps)

    if len(train_ds) == 0:
        raise SystemExit("No lemma-supervised samples found (lemma_ids empty everywhere).")

    train_loader = DataLoader(train_ds, batch_size=lemma_batch_size, shuffle=True,
                              collate_fn=collate_lemmas_b1)
    val_loader   = DataLoader(val_ds, batch_size=1, shuffle=False,
                              collate_fn=collate_lemmas_b1)

    # Model
    model = LemmaModel(len(vocab), lemma_dims, d_model=d_model)

    # Optionally initialize encoder from a trained template model
    if init_from_template:
        tck = torch.load(init_from_template, map_location="cpu")
        try:
            model.encoder.load_state_dict(tck["model"], strict=False)  # will ignore unmatched keys
        except Exception:
            # try loading only encoder weights if nested
            if "encoder.emb.weight" in tck["model"]:
                sd = model.state_dict()
                for k,v in tck["model"].items():
                    if k.startswith("encoder."):
                        sd[k] = v
                model.load_state_dict(sd, strict=False)

    opt = optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()

    # Train
    for ep in range(1, epochs+1):
        model.train()
        it = tqdm(train_loader, desc=f"Lemmas ep{ep}", unit="ex") if (use_bar and _HAVE_TQDM) else train_loader
        tot_loss = 0.0; nb = 0
        for tokens, offsets, tid, loc_ids in it:
            if not model.has_head(int(tid)):
                continue
            logits = model.forward_head(tokens, offsets, int(tid))   # [1, dim_t]
            dim_t = logits.shape[-1]
            target = torch.zeros((1, dim_t), dtype=torch.float32)
            if loc_ids.numel() > 0:
                target[0, loc_ids] = 1.0
            loss = bce(logits, target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot_loss += loss.item(); nb += 1
        tr_loss = tot_loss / max(1, nb)

        # Validate: report Hit@1 / Hit@5 micro
        model.eval()
        hits1 = hits5 = 0; tot = 0
        with torch.no_grad():
            for tokens, offsets, tid, loc_ids in val_loader:
                if not model.has_head(int(tid)):
                    continue
                logits = model.forward_head(tokens, offsets, int(tid)).squeeze(0)  # [dim_t]
                scores = torch.sigmoid(logits)
                topk = min(5, scores.numel())
                top_idx = torch.topk(scores, k=topk).indices.tolist()
                gold = set(loc_ids.tolist())
                if scores.numel() > 0:
                    hits1 += int(top_idx[0] in gold)
                    hits5 += int(any(i in gold for i in top_idx[:topk]))
                    tot += 1
        h1 = hits1 / tot if tot else 0.0
        h5 = hits5 / tot if tot else 0.0
        log(f"[Lemmas] epoch {ep:02d} | train_loss={tr_loss:.4f} | val@1={h1:.3f} | val@5={h5:.3f}")

    # Save
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "vocab": vocab,
        "template2id": template2id,
        "id2template": id2template,
        "lemma_id_maps": lemma_id_maps,           # tid -> {globalId: localIdx}
        "lemma_id_rev_maps": lemma_id_rev_maps,   # tid -> {localIdx: globalId}
        "lemma_dims": {int(k):int(v) for k,v in lemma_dims.items()},
        "model": model.state_dict(),
        "d_model": d_model,
    }
    path = outdir / "lemmas.pt"
    torch.save(ckpt, path)
    log(f"[Lemmas] saved to {path}")
    return model, vocab, template2id, lemma_id_maps, lemma_id_rev_maps


# =========================
# Inference
# =========================

def load_template_ckpt(path: Path):
    ck = torch.load(path, map_location="cpu")
    model = TemplateModel(len(ck["vocab"]), len(ck["template2id"]), d_model=ck.get("d_model",128))
    model.load_state_dict(ck["model"]); model.eval()
    return model, ck["vocab"], ck["template2id"], ck["id2template"]

def load_lemma_ckpt(path: Path):
    ck = torch.load(path, map_location="cpu")
    model = LemmaModel(len(ck["vocab"]), ck["lemma_dims"], d_model=ck.get("d_model",128))
    model.load_state_dict(ck["model"]); model.eval()
    return model, ck["vocab"], ck["template2id"], ck["id2template"], ck["lemma_id_maps"], ck["lemma_id_rev_maps"]

def predict_template(text: str, model: TemplateModel, vocab, id2template):
    tokens = torch.tensor([vocab.get(t, 0) for t in tokenize(text)], dtype=torch.long)
    offsets = torch.tensor([0], dtype=torch.long)
    with torch.no_grad():
        logits = model(tokens, offsets)
        tid = logits.argmax(-1).item()
    return id2template[tid], tid

def predict_lemmas(text: str, template_tid: int, model: LemmaModel, vocab, lemma_id_rev_maps, topk=5) -> List[int]:
    if not model.has_head(int(template_tid)):
        return []
    tokens = torch.tensor([vocab.get(t, 0) for t in tokenize(text)], dtype=torch.long)
    offsets = torch.tensor([0], dtype=torch.long)
    with torch.no_grad():
        logits = model.forward_head(tokens, offsets, int(template_tid)).squeeze(0)  # [dim_t]
        scores = torch.sigmoid(logits)
        k = min(topk, scores.numel())
        top_local = torch.topk(scores, k=k).indices.tolist()
    # map local -> global lemma IDs
    local2global = lemma_id_rev_maps[int(template_tid)]
    return [local2global[i] for i in top_local]


# =========================
# CLI
# =========================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=["train_template","train_lemmas","predict_template","predict_lemmas","predict"])
    ap.add_argument("--infile", help="template dataset JSONL (input/template/lemma_ids)")
    ap.add_argument("--outdir", default="runs/out")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--val_batch_size", type=int, default=128)
    ap.add_argument("--lemma_batch_size", type=int, default=1, help="keep 1 unless you change collate")
    ap.add_argument("--no_bar", action="store_true")

    # Inference
    ap.add_argument("--text", help="Single text input (goal+context) for predict modes")
    ap.add_argument("--topk", type=int, default=5)

    # Checkpoints
    ap.add_argument("--template_ckpt", help="path to template.pt for predict or init")
    ap.add_argument("--lemma_ckpt", help="path to lemmas.pt for predict")
    ap.add_argument("--init_from_template", help="initialize lemma encoder from template ckpt (train_lemmas)")

    args = ap.parse_args()
    use_bar = (not args.no_bar) and _HAVE_TQDM

    if args.mode == "train_template":
        if not args.infile:
            raise SystemExit("--infile required")
        recs = load_template_records(Path(args.infile))
        train_template(recs, args.outdir, d_model=args.d_model, epochs=args.epochs, lr=args.lr,
                       seed=args.seed, batch_size=args.batch_size, val_batch_size=args.val_batch_size,
                       verbose=True, use_bar=use_bar)

    elif args.mode == "train_lemmas":
        if not args.infile:
            raise SystemExit("--infile required")
        recs = load_template_records(Path(args.infile))
        init_path = Path(args.init_from_template) if args.init_from_template else None
        train_lemmas(recs, args.outdir, d_model=args.d_model, epochs=args.epochs, lr=args.lr,
                     seed=args.seed, lemma_batch_size=args.lemma_batch_size,
                     init_from_template=init_path, verbose=True, use_bar=use_bar)

    elif args.mode == "predict_template":
        if not args.template_ckpt:
            raise SystemExit("--template_ckpt required")
        if not args.text:
            raise SystemExit("--text required")
        tm, tvocab, t2id, id2t = load_template_ckpt(Path(args.template_ckpt))
        tmpl, tid = predict_template(args.text, tm, tvocab, id2t)
        log(tmpl)

    elif args.mode == "predict_lemmas":
        if not args.template_ckpt or not args.lemma_ckpt:
            raise SystemExit("--template_ckpt and --lemma_ckpt required")
        if not args.text:
            raise SystemExit("--text required")
        tm, tvocab, t2id, id2t = load_template_ckpt(Path(args.template_ckpt))
        lm, lvocab, lt2id, lid2t, lmaps, lrev = load_lemma_ckpt(Path(args.lemma_ckpt))

        tmpl, tid = predict_template(args.text, tm, tvocab, id2t)
        global_ids = predict_lemmas(args.text, tid, lm, lvocab, lrev, topk=args.topk)
        log(json.dumps({"template": tmpl, "lemma_ids": global_ids}))

    elif args.mode == "predict":
        if not args.template_ckpt or not args.lemma_ckpt:
            raise SystemExit("--template_ckpt and --lemma_ckpt required")
        if not args.text:
            raise SystemExit("--text required")
        tm, tvocab, t2id, id2t = load_template_ckpt(Path(args.template_ckpt))
        lm, lvocab, lt2id, lid2t, lmaps, lrev = load_lemma_ckpt(Path(args.lemma_ckpt))

        tmpl, tid = predict_template(args.text, tm, tvocab, id2t)
        lemma_ids = predict_lemmas(args.text, tid, lm, lvocab, lrev, topk=args.topk)
        print(json.dumps({"template": tmpl, "lemma_ids": lemma_ids}))

if __name__ == "__main__":
    main()