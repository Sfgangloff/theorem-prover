import os
import subprocess
import random
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.distributions import Categorical

# TODO: Repair a filtered as progressively larger supposed to be valid files (break down into environments). Maybe LLM can generate skeletons this way and then fill them.

# === Environment ===

class LeanRepairEnvChar:
    def __init__(self, filepath: str, allowed_chars: List[str]):
        self.original_path = filepath
        with open(filepath) as f:
            self.lines = f.readlines()
        self.allowed_chars = allowed_chars
        self.reset()

    def reset(self):
        self.current_lines = self.lines[:]
        return self.get_state()

    def get_state(self) -> str:
        return "".join(self.current_lines)

    def apply_action(self, action: Tuple[int, int, int]):
        line_idx, char_idx, char_id = action
        line = self.current_lines[line_idx]
        if 0 <= char_idx < len(line):
            new_char = self.allowed_chars[char_id]
            self.current_lines[line_idx] = (
                line[:char_idx] + new_char + line[char_idx + 1:]
            )

    def write_temp_file(self):
        with open("temp.lean", "w") as f:
            f.writelines(self.current_lines)

    def step(self, action: Tuple[int, int, int]):
        self.apply_action(action)
        self.write_temp_file()
        result = subprocess.run(["lean", "--make", "temp.lean"],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
        success = result.returncode == 0
        reward = 1.0 if success else 0.0
        return self.get_state(), reward, success

# === Policy Model ===

class FullContextEditPolicy(nn.Module):
    def __init__(self, vocab_size, max_lines, max_chars, hidden_dim=128):
        super().__init__()
        self.char_embed = nn.Embedding(vocab_size, hidden_dim)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4), num_layers=2
        )
        self.line_head = nn.Linear(hidden_dim, max_lines)
        self.char_head = nn.Linear(hidden_dim, max_chars)
        self.replacement_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, file_tensor):  # shape: [seq_len]
        x = self.char_embed(file_tensor)  # [seq_len, hidden_dim]
        x = self.encoder(x.unsqueeze(1)).squeeze(1)  # [seq_len, hidden_dim]
        summary = x.mean(dim=0)  # [hidden_dim]
        return (
            self.line_head(summary),
            self.char_head(summary),
            self.replacement_head(summary)
        )
# === Training Loop ===

def train(file_list: List[str], allowed_chars: List[str], policy, char_to_id: dict, max_file_len: int, num_episodes: int = 100):
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    for episode in range(num_episodes):
        # Sample a file each episode
        file_path = random.choice(file_list)
        env = LeanRepairEnvChar(file_path, allowed_chars)

        log_probs = []
        rewards = []

        state = env.reset()

        def file_to_tensor(code: str, char_to_id: dict, max_len: int):
            indices = [char_to_id.get(c, char_to_id[' ']) for c in code[:max_len]]
            if len(indices) < max_len:
                indices += [char_to_id[' ']] * (max_len - len(indices))
            return torch.tensor(indices, dtype=torch.long)

        max_steps = 10
        for _ in range(max_steps):
            file_tensor = file_to_tensor(state, char_to_id, max_file_len)

            line_logits, char_logits, repl_logits = policy(file_tensor)

            # Trim logits to current env state
            actual_num_lines = len(env.current_lines)
            actual_char_lens = [len(line) for line in env.current_lines]
            max_char_len = max(actual_char_lens)

            line_logits = line_logits[:actual_num_lines]
            char_logits = char_logits[:max_char_len]

            line_dist = Categorical(logits=line_logits)
            char_dist = Categorical(logits=char_logits)
            repl_dist = Categorical(logits=repl_logits)

            line_idx = line_dist.sample()
            char_idx = char_dist.sample()
            repl_idx = repl_dist.sample()

            log_prob = line_dist.log_prob(line_idx) \
                     + char_dist.log_prob(char_idx) \
                     + repl_dist.log_prob(repl_idx)

            action = (line_idx.item(), char_idx.item(), repl_idx.item())
            state, reward, done = env.step(action)

            log_probs.append(log_prob)
            rewards.append(reward)

            if done:
                break

        G = sum(rewards)
        loss = -sum(log_probs) * G

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"[{episode+1}/{num_episodes}] File: {os.path.basename(file_path)} | Reward={G:.2f} | Steps={len(log_probs)}")

    print("✅ Training complete.")
    return policy

# === Entry Point ===

if __name__ == "__main__":
    allowed_chars = list(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 :;,.=_()+-*/<>|&![](){}'\"#\n"
    )
    char_to_id = {c: i for i, c in enumerate(allowed_chars)}
    vocab_size = len(allowed_chars)

    max_lines = 100
    max_chars = 200
    max_file_len = 2048

    policy = FullContextEditPolicy(
        vocab_size=vocab_size,
        max_lines=max_lines,
        max_chars=max_chars,
        hidden_dim=128
    )

    # 🔁 List of broken .lean files
    file_list = [
        "lean_env/LeanEnv/invalid_files/definition__eigenvector.lean",
        "lean_env/LeanEnv/invalid_files/example__ring.lean"
        # Add more as you create them
    ]

    trained_policy = train(file_list, allowed_chars, policy, char_to_id, max_file_len, num_episodes=100)
