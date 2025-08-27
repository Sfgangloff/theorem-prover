import os
import re
from dotenv import load_dotenv

# --- Heuristics & skip lists -------------------------------------------------

# Skip entire subtrees if path contains any of these segments
SKIP_DIR_SEGMENTS = {
    os.sep + "Tactic" + os.sep,
    os.sep + "Meta" + os.sep,
    os.sep + "Widget" + os.sep,
    os.sep + "Tests" + os.sep,
    os.sep + "_private" + os.sep,
}

# Lines that indicate this file likely defines/plays with syntax or macros
# (instrumentation commonly breaks these)
SUSPECT_FILE_TOKENS = [
    r'^\s*syntax\b', r'^\s*macro_rules\b', r'^\s*macro\b',
    r'^\s*elab\b', r'^\s*declare_syntax_cat\b',
    r'^\s*tactic_extension\b', r'^\s*simproc\b', r'^\s*dsimproc\b',
    r'\(tactic\|',  # quasiquoters for tactics
]

# Valid tactic starters (keep small & conservative; we can add more later)
TACTIC_STARTERS = [
    # basic
    r'simp\b', r'simp_all\b', r'simp_rw\b', r'rw\b', r'apply\b', r'exact\b',
    r'refine\b', r'refine\b', r'intro\b', r'intros\b', r'revert\b',
    r'cases\b', r'rcases\b', r'constructor\b', r'rfl\b', r'refl\b',
    r'assumption\b', r'change\b', r'linarith\b', r'nlinarith\b', r'ring\b',
    r'decide\b', r'aesop\b', r'omega\b', r'norm_num\b', r'zify\b',
    # control / combinators
    r'first\b', r'try\b', r'all_goals\b', r'any_goals\b', r'repeat\b',
    # misc often-tactics
    r'show\b', r'have\b', r'clear\b', r'rename_i\b', r'simp\?',
]

# Things that **cannot** appear at the start of a tactic line inside `by`
# (if they do, the by-block is suspicious; stop instrumenting that block)
BAD_TACTIC_PREFIXES = [
    'variable', 'namespace', 'section', 'end', 'theorem', 'lemma', 'def',
    'structure', 'inductive', 'mutual', 'where', 'instance', 'class',
    'macro', 'macro_rules', 'syntax', 'elab', 'tactic_extension',
]

MAIN_DEF_RE = re.compile(r'^\s*def\s+main\b', re.MULTILINE)

def strip_comments(src: str) -> str:
    """Remove Lean line (`-- ...`) and block (`/- ... -/`) comments."""
    no_block = re.sub(r'/-(?:.|\n)*?-/', '', src)
    no_line = re.sub(r'--.*', '', no_block)
    return no_line

def contains_main(path: str) -> bool:
    """True if file (after stripping comments) defines `def main`."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            src = f.read()
        src_nc = strip_comments(src)
        return MAIN_DEF_RE.search(src_nc) is not None
    except OSError:
        return False

def file_has_tactic_blocks(path: str) -> bool:
    """Cheap prefilter: skip files without any `:= by`."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return (":= by" in f.read())
    except OSError:
        return False

def path_should_be_skipped(path: str) -> bool:
    """Skip large/problematic dirs by path heuristics."""
    norm = os.path.normpath(path)
    for seg in SKIP_DIR_SEGMENTS:
        if seg in (norm + os.sep):  # ensure segment matches a dir boundary
            return True
    return False

def file_looks_like_macro_or_syntax(path: str) -> bool:
    """Skip files that likely define macros/syntax/elaborators."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            src = strip_comments(f.read())
        for pat in SUSPECT_FILE_TOKENS:
            if re.search(pat, src, re.MULTILINE):
                return True
    except OSError:
        pass
    return False

# Build a single big regex for tactic starters: ^\s*(one_of)
TACTIC_RE = re.compile(r'^\s*(?:' + '|'.join(TACTIC_STARTERS) + r')\b')

def looks_like_tactic_line(s: str) -> bool:
    """Conservative check: does this line look like a tactic command?"""
    # Accept `{ ... }` tactic blocks as a single tactic line too
    if s.startswith('{') or s.startswith('}'):
        return True
    return TACTIC_RE.match(s) is not None

def starts_with_bad_prefix(s: str) -> bool:
    st = s.lstrip()
    for p in BAD_TACTIC_PREFIXES:
        if st.startswith(p + ' ') or st == p:
            return True
    return False

def ensure_import(lines, import_stmt):
    """Ensure `import_stmt` is present; if missing, insert after last import."""
    has_import = any(
        l.strip() == import_stmt or l.strip().startswith(import_stmt + " ")
        for l in lines
    )
    if has_import:
        return lines

    last_import_idx = -1
    for idx, l in enumerate(lines):
        if l.strip().startswith("import "):
            last_import_idx = idx

    new_lines = list(lines)
    insertion_line = import_stmt + "\n"
    if last_import_idx >= 0:
        new_lines.insert(last_import_idx + 1, insertion_line)
    else:
        new_lines.insert(0, insertion_line)
    return new_lines

def file_is_incomplete(path: str) -> bool:
    try:
        with open(path, "r", encoding="utf-8") as f:
            s = f.read()
        # very lightweight: catch sorry/sorryAx/admit outside identifiers
        return re.search(r'(^|[^A-Za-z_])((sorryAx)|sorry|admit)($|[^A-Za-z_])', s) is not None
    except OSError:
        return False

def instrument_file(input_path, output_path, lean_import_module):
    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    import_stmt = f"import {lean_import_module}"
    lines = ensure_import(lines, import_stmt)

    new_lines = []
    inside_by_block = False
    indent_level = None
    block_mode = None  # None | "tactic" | "term"
    block_suspicious = False

    for line in lines:
        stripped = line.strip()

        # Start of a `by` block
        if (":= by" in stripped) or stripped.endswith(":= by"):
            inside_by_block = True
            indent_level = None
            block_mode = None
            block_suspicious = False
            new_lines.append(line)
            continue

        if inside_by_block:
            # Pass through blank lines
            if stripped == "":
                new_lines.append(line)
                continue

            current_indent = len(line) - len(line.lstrip(" "))

            # First non-empty line sets indent and block kind
            if indent_level is None:
                indent_level = current_indent
                # Decide block kind from the very first line
                if starts_with_bad_prefix(stripped):
                    block_suspicious = True
                    block_mode = "term"  # treat as non-tactic to be safe
                else:
                    # classify by first line only
                    block_mode = "tactic" if looks_like_tactic_line(stripped) else "term"

            # Still inside the block?
            if current_indent >= indent_level:
                if block_mode != "tactic" or block_suspicious:
                    # term block or suspicious: never inject
                    new_lines.append(line)
                else:
                    if "logStep" in stripped:
                        new_lines.append(line)  # already instrumented
                    else:
                        if starts_with_bad_prefix(stripped):
                            # flip to suspicious for the remainder of this block
                            block_suspicious = True
                            new_lines.append(line)
                        elif looks_like_tactic_line(stripped):
                            log_line = " " * current_indent + "logStep " + line.lstrip(" ")
                            new_lines.append(log_line)
                        else:
                            new_lines.append(line)
                continue
            else:
                # Dedent => block ends
                inside_by_block = False
                indent_level = None
                block_mode = None
                block_suspicious = False
                new_lines.append(line)
                continue

        # Outside any `by` block
        new_lines.append(line)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

def instrument_all(folder_in, folder_out, lean_import_module):
    # Clear output folder first (so it mirrors the input)
    if os.path.exists(folder_out):
        for root, dirs, files in os.walk(folder_out, topdown=False):
            for fn in files:
                os.remove(os.path.join(root, fn))
            for d in dirs:
                os.rmdir(os.path.join(root, d))
    os.makedirs(folder_out, exist_ok=True)

    for root, _, files in os.walk(folder_in):
        for fname in files:
            if not fname.endswith(".lean"):
                continue
            input_path = os.path.join(root, fname)
            rel_path = os.path.relpath(input_path, folder_in)
            output_path = os.path.join(folder_out, rel_path)

            # Skip by directory heuristics
            if path_should_be_skipped(input_path):
                print(f"Skipping (dir rule): {rel_path}")
                continue

            # Skip executables
            if contains_main(input_path):
                print(f"Skipping (has main): {rel_path}")
                continue

            # Skip macro/syntax-heavy files
            if file_looks_like_macro_or_syntax(input_path):
                print(f"Skipping (syntax/macro file): {rel_path}")
                continue

            # Skip files without any tactic blocks
            if not file_has_tactic_blocks(input_path):
                print(f"Skipping (no tactic blocks): {rel_path}")
                continue

            if file_is_incomplete(input_path):
                rel = os.path.relpath(input_path, folder_in)
                print(f"Skipping (incomplete: contains sorry/admit): {rel}")
                continue

            # Ensure output subdir exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            print(f"Instrumenting {rel_path}")
            instrument_file(input_path, output_path, lean_import_module)

if __name__ == "__main__":
    load_dotenv(".env")
    RAW_FILES_PATH = os.getenv("RAW_FILES_PATH", "raw_proofs")
    INSTRUMENTED_FILES_PATH = os.getenv("INSTRUMENTED_FILES_PATH", "instrumented_proofs")
    LEAN_LOG_TACTIC_IMPORT = os.getenv("LEAN_LOG_TACTIC_IMPORT", "GoalTacticLogger")
    instrument_all(RAW_FILES_PATH, INSTRUMENTED_FILES_PATH, LEAN_LOG_TACTIC_IMPORT)