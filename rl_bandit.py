#!/usr/bin/env python3
import os, json, argparse, random, sys
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Optional progress bar (falls back to simple prints if not installed)
try:
    from tqdm import tqdm
    _HAVE_TQDM = True
except Exception:
    _HAVE_TQDM = False


# ------------------------------
# Utilities
# ------------------------------

def log(msg: str, *, flush=True):
    print(msg, flush=flush)

def load_pairs(path, *, verbose=False, max_bad=20):
    """
    Expect JSONL with fields:
      - input: str
      - action: str
      - meta.status: "ok" or "error"
      - done: bool  (True when goal closed / terminal)
    Reward = 1.0 iff (done and status == ok), else 0.0
    Skips malformed lines with a warning (up to max_bad shown).
    """
    data = []
    bad = 0
    total = 0
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                bad += 1
                if bad <= max_bad:
                    log(f"[warn] bad JSON on line {i}: {e}")
                continue
            text = rec.get("input", "")
            act  = rec.get("action", "")
            meta = rec.get("meta", {}) or {}
            done = bool(rec.get("done", False))
            ok   = (meta.get("status") == "ok")
            reward = 1.0 if (done and ok) else 0.0
            # Filter out empty (text or action) rows
            if not isinstance(text, str) or not isinstance(act, str) or not text or not act:
                continue
            data.append((text, act, reward))
    if verbose:
        log(f"[load_pairs] total lines={total}, kept={len(data)}, skipped_bad={bad}")
    if len(data) == 0:
        raise SystemExit(f"No valid data found at {path}")
    return data

def tokenize(s):
    # very simple whitespace tokenizer; keeps numbers/symbols
    return s.replace("\n", " ").split()


# ------------------------------
# Dataset / Collate
# ------------------------------

class TextDataset(Dataset):
    def __init__(self, pairs, vocab, act2id):
        self.pairs = pairs
        self.vocab = vocab
        self.act2id = act2id
    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx):
        text, act, reward = self.pairs[idx]
        ids = [self.vocab.get(t, 0) for t in tokenize(text)]
        return torch.tensor(ids, dtype=torch.long), self.act2id[act], torch.tensor(reward, dtype=torch.float32)

def collate_batch(batch):
    """
    Concatenate variable-length token lists and build EmbeddingBag offsets.
    Returns:
      tokens: LongTensor [sum_len]
      offsets: LongTensor [B]
      actions: LongTensor [B]
      rewards: FloatTensor [B]
    """
    token_tensors, actions, rewards = zip(*batch)
    lengths = torch.tensor([len(t) for t in token_tensors], dtype=torch.long)
    offsets = torch.zeros(len(token_tensors), dtype=torch.long)
    if len(lengths) > 1:
        offsets[1:] = torch.cumsum(lengths[:-1], dim=0)
    tokens = torch.cat(token_tensors, dim=0) if token_tensors else torch.tensor([], dtype=torch.long)
    return tokens, offsets, torch.tensor(actions, dtype=torch.long), torch.stack(rewards)


# ------------------------------
# Model
# ------------------------------

class PolicyNet(nn.Module):
    def __init__(self, vocab_size, num_actions, d_model=128):
        super().__init__()
        self.emb = nn.EmbeddingBag(vocab_size, d_model, mode="mean")
        self.fc = nn.Linear(d_model, num_actions)

    def forward(self, tokens, offsets):
        emb = self.emb(tokens, offsets)   # [B, d_model]
        return self.fc(emb)               # [B, num_actions]


# ------------------------------
# Training / Evaluation
# ------------------------------

def describe_dataset(name, pairs, *, head=3):
    log(f"[{name}] size={len(pairs)}")
    if len(pairs) == 0:
        return
    rw = sum(1 for *_, r in pairs if r > 0)
    log(f"[{name}] positive-reward examples: {rw} ({rw/len(pairs):.1%})")
    # Show a few examples
    for i, (t,a,r) in enumerate(pairs[:head], 1):
        log(f"[{name}] sample {i}: reward={r} action={a!r} text[:80]={t[:80]!r}")

def describe_actions(name, pairs):
    cnt = Counter(a for _, a, _ in pairs)
    if not cnt:
        return
    log(f"[{name}] action distribution (top 10):")
    for act, c in cnt.most_common(10):
        log(f"  {act!r}: {c}")

def make_loader(ds, batch_size, shuffle, num_workers=0):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, collate_fn=collate_batch, drop_last=False)

def train_model(train_data, val_data, vocab, act2id, outdir,
                epochs=5, lr=1e-3, seed=0, verbose=False, log_every=10, use_bar=True,
                batch_size=32, val_batch_size=64):
    torch.manual_seed(seed)
    random.seed(seed)

    if verbose:
        describe_dataset("train", train_data)
        describe_dataset("val", val_data)
        describe_actions("train", train_data)
        log(f"[meta] vocab_size={len(vocab)}, num_actions={len(act2id)}, epochs={epochs}, lr={lr}")

    train_ds = TextDataset(train_data, vocab, act2id)
    val_ds   = TextDataset(val_data,   vocab, act2id)

    train_loader = make_loader(train_ds, batch_size, True)
    val_loader   = make_loader(val_ds,  val_batch_size, False)

    model = PolicyNet(len(vocab), len(act2id))
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs+1):
        # ---- Train ----
        model.train()
        running_loss = 0.0
        n_batches = 0

        iterator = train_loader
        if use_bar and _HAVE_TQDM:
            iterator = tqdm(train_loader, desc=f"Epoch {epoch} (train)", unit="batch")

        for b, (tokens, offsets, actions, rewards) in enumerate(iterator, 1):
            logits = model(tokens, offsets)                   # [B, A]
            ce = nn.functional.cross_entropy(logits, actions, reduction="none")  # [B]
            sum_rewards = rewards.sum()
            if sum_rewards.item() > 0:
                loss = (ce * rewards).sum() / (sum_rewards + 1e-8)
            else:
                loss = ce.mean()  # fallback when no terminal steps in batch

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

            if verbose and (b % log_every == 0):
                with torch.no_grad():
                    preds = logits.argmax(dim=-1)
                    acc = (preds == actions).float().mean().item()
                    log(f"  [epoch {epoch:02d} batch {b:04d}] "
                        f"loss={loss.item():.4f} acc={acc:.3f} reward_sum={sum_rewards.item():.1f}")

        avg_train_loss = running_loss / max(n_batches, 1)

        # ---- Validate ----
        model.eval()
        val_correct, val_total = 0, 0
        val_rw_ce_sum, val_rw_norm = 0.0, 0.0
        iterator = val_loader
        if use_bar and _HAVE_TQDM:
            iterator = tqdm(val_loader, desc=f"Epoch {epoch} (val)", unit="batch")
        with torch.no_grad():
            for tokens, offsets, actions, rewards in iterator:
                logits = model(tokens, offsets)
                preds = logits.argmax(dim=-1)
                val_correct += (preds == actions).sum().item()
                val_total   += actions.numel()

                ce = nn.functional.cross_entropy(logits, actions, reduction="none")
                val_rw_ce_sum += (ce * rewards).sum().item()
                val_rw_norm   += rewards.sum().item()

        val_acc = val_correct / val_total if val_total else 0.0
        val_rw_ce = (val_rw_ce_sum / val_rw_norm) if val_rw_norm > 0 else 0.0
        log(f"Epoch {epoch:02d} | train_loss={avg_train_loss:.4f} | val_acc={val_acc:.3f} | val_rwCE={val_rw_ce:.4f}")

        if verbose:
            # print a few calibrated predictions on tiny synthetic batch from val
            try:
                tokens, offsets, actions, rewards = next(iter(val_loader))
                with torch.no_grad():
                    logits = model(tokens, offsets)
                    preds = logits.argmax(dim=-1)
                k = min(5, actions.numel())
                log(f"  [val head] true[:{k}]={actions[:k].tolist()} pred[:{k}]={preds[:k].tolist()} reward[:{k}]={rewards[:k].tolist()}")
            except StopIteration:
                pass

    # save
    Path(outdir).mkdir(parents=True, exist_ok=True)
    ckpt_path = Path(outdir) / "policy.pt"
    torch.save({"model": model.state_dict(), "vocab": vocab, "act2id": act2id},
               ckpt_path)
    log(f"Saved model to {ckpt_path}")
    return model


# ------------------------------
# Prediction
# ------------------------------

def predict(texts, model, vocab, act2id):
    id2act = {v:k for k,v in act2id.items()}
    results = []
    with torch.no_grad():
        for text in texts:
            tokens = torch.tensor([vocab.get(t, 0) for t in tokenize(text)], dtype=torch.long)
            offsets = torch.tensor([0], dtype=torch.long)
            logits = model(tokens, offsets)
            pred = logits.argmax(-1).item()
            results.append(id2act[pred])
    return results


# ------------------------------
# Main
# ------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train","predict"], required=True)
    ap.add_argument("--infile", default=os.getenv("CANONICAL_PAIRS", "data/canonicalized_pairs.jsonl"))
    ap.add_argument("--outdir", default=os.getenv("RL_OUTDIR", "runs/rl_bandit"))
    ap.add_argument("--epochs", type=int, default=int(os.getenv("RL_EPOCHS", "5")))
    ap.add_argument("--lr", type=float, default=float(os.getenv("RL_LR", "1e-3")))
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--val_batch_size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verbose", action="store_true", help="Print batch-level info and dataset stats")
    ap.add_argument("--log_every", type=int, default=10, help="Print every N batches when --verbose")
    ap.add_argument("--no_bar", action="store_true", help="Disable tqdm progress bars")
    ap.add_argument("--text", help="Single text input (for predict)")
    args = ap.parse_args()

    use_bar = (not args.no_bar) and _HAVE_TQDM

    if args.mode == "train":
        data = load_pairs(args.infile, verbose=args.verbose)
        random.shuffle(data)
        n = max(1, int(0.8*len(data)))
        train, val = data[:n], data[n:]

        # build vocab + actions from TRAIN only (safer)
        vocab = {"<unk>":0}
        for text, act, reward in train:
            for t in tokenize(text):
                if t not in vocab:
                    vocab[t] = len(vocab)
        actions_all = sorted(set(a for _, a, _ in data))
        act2id = {a:i for i,a in enumerate(actions_all)}

        _ = train_model(
            train, val, vocab, act2id, args.outdir,
            epochs=args.epochs, lr=args.lr, seed=args.seed,
            verbose=args.verbose, log_every=args.log_every, use_bar=use_bar,
            batch_size=args.batch_size, val_batch_size=args.val_batch_size
        )

    elif args.mode == "predict":
        ckpt_path = Path(args.outdir) / "policy.pt"
        if not ckpt_path.exists():
            raise SystemExit(f"Checkpoint not found at {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model = PolicyNet(len(ckpt["vocab"]), len(ckpt["act2id"]))
        model.load_state_dict(ckpt["model"])
        model.eval()

        vocab, act2id = ckpt["vocab"], ckpt["act2id"]

        if args.text:
            preds = predict([args.text], model, vocab, act2id)
            log(preds[0])
        else:
            with open(args.infile, "r", encoding="utf-8") as f:
                texts = [json.loads(line)["input"] for line in f if line.strip()]
            preds = predict(texts, model, vocab, act2id)
            for t,p in zip(texts, preds):
                log(f"{t[:60]}... → {p}")

if __name__ == "__main__":
    main()
