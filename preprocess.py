import pandas as pd
import re
import json
from collections import Counter

def extract_main_tactic(tactic_str):
    match = re.search(r'\( Tactic \. (\w+)', tactic_str)
    return match.group(1) if match else "UNKNOWN"

def tokenize_goal(goal_str):
    return goal_str.strip().split()

# Load CSV
df = pd.read_csv("goal_tactic_dataset.csv")
df['main_tactic'] = df['tactic'].apply(extract_main_tactic)
df['tokenized_goal'] = df['goal'].apply(tokenize_goal)

# Build vocab
token_counter = Counter(tok for tokens in df['tokenized_goal'] for tok in tokens)
token_to_id = {tok: i + 1 for i, (tok, _) in enumerate(token_counter.items())}
token_to_id['<PAD>'] = 0

tactic_to_id = {tac: i for i, tac in enumerate(sorted(df['main_tactic'].unique()))}
id_to_tactic = {i: tac for tac, i in tactic_to_id.items()}

# Save
df.to_pickle("preprocessed.pkl")
json.dump(token_to_id, open("token_to_id.json", "w"))
json.dump(tactic_to_id, open("tactic_to_id.json", "w"))
json.dump(id_to_tactic, open("id_to_tactic.json", "w"))