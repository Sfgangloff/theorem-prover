import torch
import torch.nn as nn
import json

# Load vocab and model
token_to_id = json.load(open("token_to_id.json"))
id_to_tactic = json.load(open("id_to_tactic.json"))

class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_classes):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.cls = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        emb = self.embed(x).transpose(0, 1)
        encoded = self.encoder(emb)
        pooled = encoded.mean(dim=0)
        return self.cls(pooled)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerClassifier(len(token_to_id), 64, 4, len(id_to_tactic))
model.load_state_dict(torch.load("tactic_predictor.pt", map_location=device))
model.to(device)
model.eval()

def predict(goal_str, max_len=20):
    tokens = goal_str.strip().split()
    ids = [token_to_id.get(tok, 0) for tok in tokens][:max_len]
    ids += [0] * (max_len - len(ids))
    x = torch.tensor([ids]).to(device)
    with torch.no_grad():
        logits = model(x)
        pred = torch.argmax(logits, dim=1).item()
        return id_to_tactic[str(pred)]

# Example
print(predict("m + 0 = n"))