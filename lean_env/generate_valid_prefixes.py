"""
Module: lean_prefix_generator

This script generates increasing valid prefixes from Lean source files,
intended for use in training models that learn from partially written but syntactically
and semantically correct Lean code.

The prefixes always include all import statements at the top of the file
(since they are required for successful compilation), and strip away any comments
(single-line `--` and nested block `/- ... -/`) before prefix generation.

Usage:
    Run this file directly to process a specific Lean file.
    Alternatively, uncomment and configure the `process_all_lean_files` call to
    recursively process a directory of `.lean` files.

Main Components:
- Lean file validation using `lake build`
- Comment removal for accurate prefix generation
- Import detection to ensure all generated prefixes compile correctly
- Batch processing over directories
"""

import subprocess
from pathlib import Path

RED = "\033[91m"
GREEN = "\033[92m"
RESET = "\033[0m"

# TODO: we will want to learn valid imports or simply generate the imports based on existing ones. 

def is_valid_lean_file(filepath: str) -> bool:
    """
    Check whether a Lean file compiles successfully within a Lake project.

    Parameters:
        filepath (str): Path to the Lean file to be validated.

    Returns:
        bool: True if the file compiles without errors, False otherwise.
    """
    # Convert path like LeanEnv/Main.lean → LeanEnv.Main
    module_name = Path(filepath).with_suffix("").as_posix().replace("/", ".")
    print(module_name)
    
    result = subprocess.run(["lake", "build", module_name],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    
    if result.returncode != 0:
        print("Lean error:\n", result.stderr.decode())
    return result.returncode == 0

def find_last_import_line(lines) -> int:
    """
    Return the index (0-based) of the last import statement at the top of the file.
    Assumes all import lines are grouped at the top with no interleaved non-imports.

    Parameters:
        lines (list[str]): Lines of a Lean source file.

    Returns:
        int: Index of the last import line. Returns -1 if no import found.
    """
    last_import = -1
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("import "):
            last_import = i
    return last_import

def strip_comments(lines: list[str]) -> list[str]:
    """
    Remove Lean comments from a list of lines.
    Handles:
        - single-line comments: `--`
        - block comments: `/- ... -/` with nesting

    Parameters:
        lines (list[str]): Raw lines from a Lean source file.

    Returns:
        list[str]: Lines with comments removed.
    """
    stripped_lines = []
    block_comment_level = 0

    for line in lines:
        i = 0
        result = ''
        while i < len(line):
            if block_comment_level == 0 and line[i:i+2] == '--':
                # Start of single-line comment: ignore rest
                break
            elif line[i:i+2] == '/-':
                block_comment_level += 1
                i += 2
            elif line[i:i+2] == '-/':
                if block_comment_level > 0:
                    block_comment_level -= 1
                i += 2
            elif block_comment_level == 0:
                result += line[i]
                i += 1
            else:
                i += 1
        if result.strip() != "":
            stripped_lines.append(result)
    return stripped_lines

def generate_valid_prefixes(lean_file: str, output_dir: str):
    """
    Generate increasing valid prefixes from a Lean file,
    ensuring that all prefixes contain all required import lines.

    For a file with `n` lines, generates one file for each prefix of length `i`
    (with `last_import + 1 <= i <= n`) such that the resulting `.lean` file is
    syntactically and type-correct when compiled by `lake build`.

    Parameters:
        lean_file (str): Path to the input Lean file.
        output_dir (str): Directory where valid prefixes will be stored.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    with open(lean_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    lines = strip_comments(lines)
    last_import = find_last_import_line(lines)

    base_name = Path(lean_file).stem
    valid_count = 0

    for i in range(last_import + 1, len(lines) + 1):
        prefix_lines = lines[:i]
        prefix_file = Path(output_dir) / f"{base_name}_prefix_{i:03d}.lean"

        with open(prefix_file, "w", encoding="utf-8") as f:
            f.writelines(prefix_lines)

        if is_valid_lean_file(str(prefix_file)):
            valid_count += 1
            print(f"{GREEN}Valid prefix: {prefix_file.name}{RESET}")
        else:
            print(f"{RED}Invalid prefix: {prefix_file.name}{RESET}")
            prefix_file.unlink()  # Delete invalid file

    print(f"{GREEN}{valid_count} valid prefix files generated for {lean_file}.{RESET}")

def process_all_lean_files(source_root: str, output_root: str):
    """
    Recursively process all `.lean` files in `source_root` directory,
    applying `generate_valid_prefixes` and preserving relative paths
    in the `output_root` directory.

    Parameters:
        source_root (str): Path to the root directory to scan for Lean files.
        output_root (str): Path to the root output directory for prefix files.
    """
    for lean_path in Path(source_root).rglob("*.lean"):
        rel_path = lean_path.relative_to(source_root).with_suffix("")
        output_dir = Path(output_root) / rel_path
        print(f"\n🔍 Processing {lean_path}...")
        generate_valid_prefixes(lean_path, output_dir)

# === Entry Point ===
if __name__ == "__main__":
    # For batch processing, uncomment below:
    # source_root = "../extern"
    # output_root = "LeanEnv/prefixes"
    # process_all_lean_files(source_root, output_root)

    # For single-file testing
    generate_valid_prefixes("LeanEnv/Basic.lean", "LeanEnv/prefixes")
