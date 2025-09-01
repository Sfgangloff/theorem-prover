#!/usr/bin/env python3
import os, json, argparse, random
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# ------------------------------
# Utilities
# ------------------------------

def load_pairs(path):
    """
    Expect JSONL with fields:
      - input: str
      - action: str
      - meta.status: "ok" or "error"
      - done: bool  (True when goal closed / terminal)
    Reward = 1.0 iff (done and status == ok), else 0.0
    """
    data = []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            text = rec["input"]
            act  = rec["action"]
            done = rec.get("done", False)
            ok   = (rec.get("meta", {}).get("status") == "ok")
            reward = 1.0 if (done and ok) else 0.0
            data.append((text, act, reward))
    return data

def tokenize(s):
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
    token_tensors, actions, rewards = zip(*batch)  # lists of tensors / ints / tensors
    lengths = torch.tensor([len(t) for t in token_tensors], dtype=torch.long)
    offsets = torch.zeros(len(token_tensors), dtype=torch.long)
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
# Training loop
# ------------------------------

def train_model(train_data, val_data, vocab, act2id, outdir, epochs=5, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    random.seed(seed)

    train_ds = TextDataset(train_data, vocab, act2id)
    val_ds   = TextDataset(val_data,   vocab, act2id)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_batch)
    val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False, collate_fn=collate_batch)

    model = PolicyNet(len(vocab), len(act2id))
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs+1):
        # ---- Train ----
        model.train()
        running_loss = 0.0
        n_batches = 0

        for tokens, offsets, actions, rewards in train_loader:
            logits = model(tokens, offsets)                   # [B, A]
            ce = nn.functional.cross_entropy(logits, actions, reduction="none")  # [B]
            sum_rewards = rewards.sum()
            if sum_rewards.item() > 0:
                loss = (ce * rewards).sum() / (sum_rewards + 1e-8)
            else:
                loss = ce.mean()  # fallback when no terminal steps in batch

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        # ---- Validate ----
        model.eval()
        val_correct, val_total = 0, 0
        val_rw_ce_sum, val_rw_norm = 0.0, 0.0
        with torch.no_grad():
            for tokens, offsets, actions, rewards in val_loader:
                logits = model(tokens, offsets)
                preds = logits.argmax(dim=-1)
                val_correct += (preds == actions).sum().item()
                val_total   += actions.numel()

                ce = nn.functional.cross_entropy(logits, actions, reduction="none")
                val_rw_ce_sum += (ce * rewards).sum().item()
                val_rw_norm   += rewards.sum().item()

        avg_train_loss = running_loss / max(n_batches, 1)
        val_acc = val_correct / val_total if val_total else 0.0
        val_rw_ce = (val_rw_ce_sum / val_rw_norm) if val_rw_norm > 0 else 0.0
        print(f"Epoch {epoch:02d} | train_loss={avg_train_loss:.4f} | val_acc={val_acc:.3f} | val_rwCE={val_rw_ce:.4f}")

    # save
    Path(outdir).mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "vocab": vocab, "act2id": act2id},
               Path(outdir) / "policy.pt")
    print(f"Saved model to {outdir}/policy.pt")

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
    ap.add_argument("--text", help="Single text input (for predict)")
    args = ap.parse_args()

    if args.mode == "train":
        data = load_pairs(args.infile)
        if not data:
            raise SystemExit(f"No data found at {args.infile}")

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

        train_model(train, val, vocab, act2id, args.outdir, epochs=args.epochs, lr=args.lr)

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
            print(preds[0])
        else:
            with open(args.infile) as f:
                texts = [json.loads(line)["input"] for line in f]
            preds = predict(texts, model, vocab, act2id)
            for t,p in zip(texts, preds):
                print(f"{t[:60]}... → {p}")

if __name__ == "__main__":
    main()