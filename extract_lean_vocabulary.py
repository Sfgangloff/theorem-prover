import re
import json
from collections import Counter, OrderedDict
from pathlib import Path

# TODO: Improve the tokenizer to account for semantic roles of tokens (e.g., keywords, types, names).
# Ensure that comment tokens are also counted, especially if used as metadata or documentation.

def read_lean_file_as_string(filepath: str) -> str:
    """
    Read the contents of a Lean file and return it as a single string.

    Args:
        filepath (str): Path to the .lean file.

    Returns:
        str: Entire contents of the file.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    return content

def tokenize_lean_code(code: str) -> list[str]:
    """
    Tokenize a Lean source code string into meaningful tokens.
    This includes identifiers, keywords, symbols, and punctuation.

    Args:
        code (str): Lean code as a string.

    Returns:
        list[str]: List of extracted tokens.
    """
    # Regex explanation:
    #   - [A-Za-z_][A-Za-z0-9_']* matches identifiers like `def`, `my_theorem'`, `_foo`
    #   - [^\sA-Za-z0-9_] matches any single symbol (punctuation, operators, etc.)
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_']*|[^\sA-Za-z0-9_]", code)
    return tokens

def collect_tokens_from_dir(path: str) -> Counter:
    """
    Traverse a directory recursively and count token frequencies from all `.lean` files.

    Args:
        path (str): Root directory containing Lean files.

    Returns:
        Counter: A Counter object mapping each token to its frequency.
    """
    token_counter = Counter()
    for file_path in Path(path).rglob("*.lean"):
        with open(file_path, "r", encoding="utf-8") as f:
            code = f.read()
        tokens = tokenize_lean_code(code)
        token_counter.update(tokens)
    return token_counter

def save_token_vocab_json(counter: Counter, out_path: str, min_freq: int = 1):
    """
    Save a filtered and sorted token frequency dictionary to a JSON file.

    Args:
        counter (Counter): Token frequency dictionary.
        out_path (str): Path to output JSON file.
        min_freq (int): Minimum frequency threshold for including a token.
    """
    # Filter out tokens below min frequency and sort by decreasing frequency
    sorted_items = sorted(
        ((token, count) for token, count in counter.items() if count >= min_freq),
        key=lambda x: -x[1]
    )
    ordered = OrderedDict(sorted_items)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(ordered, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    # Example usage: Collect tokens from a Lean project (e.g., mathlib)
    counter = collect_tokens_from_dir("extern")
    save_token_vocab_json(counter, out_path="data/mathlib_vocab_freq.json")
