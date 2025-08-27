import os
from dotenv import load_dotenv

def ensure_import(lines, import_stmt):
    """
    Ensure `import_stmt` (e.g., 'import ExtractionTactic') is present.
    If missing, insert it after the last existing 'import ...' line,
    or at the top if there are none.
    Returns a possibly-modified list of lines.
    """
    has_import = any(
        l.strip() == import_stmt or l.strip().startswith(import_stmt + " ") for l in lines
    )
    if has_import:
        return lines

    # Find last import line
    last_import_idx = -1
    for idx, l in enumerate(lines):
        st = l.strip()
        if st.startswith("import "):
            last_import_idx = idx

    new_lines = list(lines)
    insertion_line = import_stmt + "\n"
    if last_import_idx >= 0:
        new_lines.insert(last_import_idx + 1, insertion_line)
    else:
        new_lines.insert(0, insertion_line)
    return new_lines


def instrument_file(input_path, output_path, lean_import_module):
    """
    - Copies file
    - Ensures `import <lean_import_module>` is present
    - Instruments tactic blocks (':= by' ... indented) by prefixing 'logStep ' to each tactic line
    """
    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # 1) Ensure the import
    import_stmt = f"import {lean_import_module}"
    lines = ensure_import(lines, import_stmt)

    # 2) Walk and instrument 'by' blocks
    new_lines = []
    inside_by_block = False
    indent_level = None

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Detect start of tactic block: ':= by'
        # (covers '... := by' with or without trailing spaces/comments in the same line)
        if (":= by" in stripped) or stripped.endswith(":= by"):
            inside_by_block = True
            indent_level = None  # to be detected on first indented line
            new_lines.append(line)
            continue

        if inside_by_block:
            # Allow empty lines inside the tactic block
            if stripped == "":
                new_lines.append(line)
                continue

            current_indent = len(line) - len(line.lstrip(" "))

            # First actual line inside the block fixes the indent level
            if indent_level is None:
                indent_level = current_indent

            # Still inside the block if indentation is >= indent_level
            if current_indent >= indent_level:
                if "logStep" in stripped:
                    # Already instrumented
                    new_lines.append(line)
                else:
                    # Insert 'logStep ' preserving indentation
                    log_line = " " * current_indent + "logStep " + line.lstrip(" ")
                    new_lines.append(log_line)
                continue
            else:
                # Dedented: end of tactic block
                inside_by_block = False
                indent_level = None
                new_lines.append(line)
                continue

        # Outside any tactic block
        new_lines.append(line)

    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)


def instrument_all(folder_in, folder_out, lean_import_module):
    os.makedirs(folder_out, exist_ok=True)

    for fname in os.listdir(folder_in):
        if fname.endswith(".lean"):
            input_path = os.path.join(folder_in, fname)
            output_path = os.path.join(folder_out, fname)
            print(f"Instrumenting {fname}")
            instrument_file(input_path, output_path, lean_import_module)


if __name__ == "__main__":
    load_dotenv(".env")

    RAW_FILES_PATH = os.getenv("RAW_FILES_PATH", "raw_proofs")
    INSTRUMENTED_FILES_PATH = os.getenv("INSTRUMENTED_FILES_PATH", "instrumented_proofs")
    LEAN_LOG_TACTIC_IMPORT = os.getenv("LEAN_LOG_TACTIC_IMPORT", "GoalTacticLogger")

    instrument_all(RAW_FILES_PATH, INSTRUMENTED_FILES_PATH, LEAN_LOG_TACTIC_IMPORT)