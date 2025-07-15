import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import json

# Load data
df = pd.read_pickle("preprocessed.pkl")
token_to_id = json.load(open("token_to_id.json"))
tactic_to_id = json.load(open("tactic_to_id.json"))
id_to_tactic = json.load(open("id_to_tactic.json"))

class GoalDataset(Dataset):
    def __init__(self, df, token_to_id, tactic_to_id, max_len=20):
        self.data = df
        self.token_to_id = token_to_id
        self.tactic_to_id = tactic_to_id
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens = self.data.iloc[idx]['tokenized_goal']
        token_ids = [self.token_to_id.get(tok, 0) for tok in tokens][:self.max_len]
        token_ids += [0] * (self.max_len - len(token_ids))
        label = self.tactic_to_id[self.data.iloc[idx]['main_tactic']]
        return torch.tensor(token_ids), torch.tensor(label)

# Model
class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_classes):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.cls = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        emb = self.embed(x).transpose(0, 1)  # (seq_len, batch, embed_dim)
        encoded = self.encoder(emb)
        pooled = encoded.mean(dim=0)  # (batch, embed_dim)
        return self.cls(pooled)

# Training
dataset = GoalDataset(df, token_to_id, tactic_to_id)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TransformerClassifier(len(token_to_id), embed_dim=64, num_heads=4, num_classes=len(tactic_to_id)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(5):
    model.train()
    total_loss = 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader):.4f}")

torch.save(model.state_dict(), "tactic_predictor.pt")