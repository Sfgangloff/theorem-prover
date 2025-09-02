#!/bin/bash
set -e  # stop if any script fails

python3 make_episodes.py --in lean_env/data/goal_tactic_log.jsonl --permissive
python3 encode_observations.py
python3 canonalize_actions.py