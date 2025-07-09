import json
import re
import csv

def tokenize(text):
    spaced = re.sub(r'([:=+\-*/()\[\]{}<>.,;`])', r' \1 ', text)
    tokens = spaced.strip().split()
    return tokens

def parse_goal(goal_str):
    if 'Goal:' in goal_str:
        return goal_str.split('Goal:')[-1].strip()
    return goal_str.strip()

def preprocess_logfile(path):
    with open(path, 'r') as f:
        lines = f.readlines()

    dataset = []

    for i, line in enumerate(lines):
        try:
            entry = json.loads(line)
            raw_goal = entry['goal']
            raw_tactic = entry['tactic']

            goal_text = parse_goal(raw_goal)
            goal_tokens = tokenize(goal_text)
            tactic_tokens = tokenize(raw_tactic)

            dataset.append({
                'goal': " ".join(goal_tokens),
                'tactic': " ".join(tactic_tokens)
            })

        except json.JSONDecodeError:
            print(f"[!] Skipping invalid JSON line {i}")

    return dataset

def save_to_csv(dataset, output_path):
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['goal', 'tactic'])
        writer.writeheader()
        for row in dataset:
            writer.writerow(row)

if __name__ == "__main__":
    dataset = preprocess_logfile("goal_tactic_log.jsonl")
    save_to_csv(dataset, "goal_tactic_dataset.csv")
    print(f"Saved {len(dataset)} entries to goal_tactic_dataset.csv")