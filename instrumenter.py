"""
instrumenter.py — Lean 4 tactic logger injector

This script rewrites Lean source files to insert `logStep` calls in tactic mode.
It is intentionally *syntax-light* (regex/heuristics) rather than a full Lean
parser, but includes enough structure to be robust on Mathlib-style proofs.

High-level behavior
-------------------
• Walk an input folder mirroring its structure to an output folder.
• Skip files and directories likely to contain syntax/macro machinery.
• Ensure each processed file imports the logger module (e.g., `GoalTacticLogger`).
• For each `by`-block:
    - Instrument the **head** of lines ending with `:= by` (e.g. `have … := by`)
      so that the decision to introduce a sub-proof is itself logged:
        `have … := by`  →  `logStep have … := by`
    - Then, inside the `by`-block, prefix `logStep` to each tactic segment
      (including those chained with `;` and those following bullets `·`).
• Bullet handling is Unicode-aware: bullets `·` can be nested and may be
  followed by arbitrary whitespace (including NBSP). Each bullet is normalized
  upon reconstruction as `"· "` while preserving the user's original indentation.
• Top-level `;` splitting respects (), [], {} and string literals, so we never
  split inside tactic arguments or strings.

Why buffer a `by`-block?
------------------------
We keep two parallel buffers per block:
  - `block_raw`   : original lines
  - `block_instr` : instrumented counterparts
We also track `block_has_tactic`: True iff any line in the block contains at
least one instrumentable tactic segment. On flush, we write the instrumented
version if and only if the block actually had tactics; otherwise we keep it
verbatim. This avoids invasive edits to purely term-mode `by`-blocks.

What counts as a "tactic segment"?
----------------------------------
Within a logical line, we split at top-level semicolons `;` (not inside pairs
or strings). Each resulting piece is a segment. A segment is recognized as a
tactic if it:
  • starts with a known tactic keyword from `TACTIC_STARTERS`, or
  • is a tactic-mode `let` (starts with `let ` and contains `:=`), or
  • is a lone `{` / `}` (scope braces in tactic mode).
Segments already starting with `logStep` are left unchanged (idempotent).

Safety & limitations
--------------------
• We conservatively skip files that appear to define syntax/macros/elaborators,
  and those in certain directories, to avoid corrupting quoted tactic code.
• Indentation and bullets are handled with Unicode-aware helpers. Dedent ends
  a `by`-block; blank lines are preserved.
• This tool does not rewrite branches following `=>` (e.g., `| zero => rfl`);
  it will still instrument the `rfl` line in the next line if it appears as a
  separate line, but not the inline RHS unless it matches a leading tactic.
• If you add/remove entries in `TACTIC_STARTERS`, re-run to rebuild `TACTIC_RE`.

Usage
-----
Configure via environment:
  RAW_FILES_PATH            : input directory
  INSTRUMENTED_FILES_PATH   : output directory
  LEAN_LOG_TACTIC_IMPORT    : module name to import for `logStep`

Run as a script: it prints which files are instrumented and writes mirrors
under `INSTRUMENTED_FILES_PATH`.

"""

import os
import re
import subprocess
from pathlib import Path
from dotenv import load_dotenv

# --- Heuristics & skip lists -------------------------------------------------

# Folders to be skipped (files in them do not contain proofs).
SKIP_DIR_SEGMENTS = {
    os.sep + "Tactic" + os.sep,
    os.sep + "Meta" + os.sep,
    os.sep + "Widget" + os.sep,
    os.sep + "Tests" + os.sep,
    os.sep + "_private" + os.sep,
}
# Regex expressions detecting modules containing macros/syntax/tactic machinery.
SUSPECT_FILE_TOKENS = [
    r'^\s*syntax\b', r'^\s*macro_rules\b', r'^\s*macro\b',
    r'^\s*elab\b', r'^\s*declare_syntax_cat\b',
    r'^\s*tactic_extension\b', r'^\s*simproc\b', r'^\s*dsimproc\b',
    r'\(tactic\|',
]

# A reasonably broad set of tactic starters, to answer the question: “does this segment start with a tactic?”
TACTIC_STARTERS = [
    # control / combinators
    r'first\b', r'try\b', r'all_goals\b', r'any_goals\b', r'repeat\b', r'focus\b',
    r'fail_if_success\b', r'success_if_fail\b', r'work_on_goal\b',

    # flow
    r'by_cases\b', r'cases\b', r'rcases\b', r'rintro\b', r'intro\b', r'intros\b',
    r'revert\b', r'clear\b', r'rename_i\b', r'ext\b',
    r'by_contra!\b', r'by_contra\b', r'contrapose!\b', r'contrapose\b', r'use\b',r'induction\b',

    # apply/exact/refine
    r'apply\b', r'eapply\b', r'exact\b', r'exacts\b', r'refine\b', r'refine\'\b', r'constructor\b',
    r'assumption\b',

    # rewriting / simplification / unfolding
    r'rw\b', r'rwa\b', r'simp_all\b', r'simp_rw\b', r'simp\?\b', r'simp\b', r'simpa\b',
    r'dsimp\b', r'unfold\b', r'unfold_rw\b',

    # arithmetic / decision
    r'linarith\b', r'nlinarith\b', r'ring_nf\b', r'ring\b', r'omega\b', r'decide\b', r'aesop\b',
    r'norm_num\b', r'zify\b',

    # exact terms / refl
    r'rfl\b', r'refl\b', r'show\b',

    # declarations inside tactic mode
    r'have\b', r'obtain\b', r'specialize\b', r'choose\b',
    # 'let' is only treated as tactic if ':=' is present (handled separately)
    r'set\b',

    # misc
    r'change\b', r'generalize\b', r'subst\b', r'replace\b',
]

# Things that cannot start a tactic line
BAD_TACTIC_PREFIXES = [
    'variable', 'namespace', 'section', 'end', 'theorem', 'lemma', 'def',
    'structure', 'inductive', 'mutual', 'where', 'instance', 'class',
    'macro', 'macro_rules', 'syntax', 'elab', 'tactic_extension',
]

# Used to skip instrumenting files that define an entry point, e.g. executable examples or tools
MAIN_DEF_RE = re.compile(r'^\s*def\s+main\b', re.MULTILINE)

# Compiled once: checks whether a *segment* begins with any of the starters.
TACTIC_RE = re.compile(r'^(?:' + '|'.join(TACTIC_STARTERS) + r')\b')

# --- Utilities ---------------------------------------------------------------

def strip_comments(src: str) -> str:
    """Remove Lean line (`-- ...`) and block (`/- ... -/`) comments."""
    no_block = re.sub(r'/-(?:.|\n)*?-/', '', src)
    no_line = re.sub(r'--.*', '', no_block)
    return no_line

def contains_main(path: str) -> bool:
    """
    True iff the file declares `def main` (after comment stripping).
    Used to avoid instrumenting runnable entrypoints where logging imports might
    be undesirable or change runtime dependencies.
    """
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
    """
    True if the normalized path contains any skip-directory segment (Tactic/Meta/Widget/Tests/_private).
    """
    norm = os.path.normpath(path)
    for seg in SKIP_DIR_SEGMENTS:
        if seg in (norm + os.sep):
            return True
    return False

def file_looks_like_macro_or_syntax(path: str) -> bool:
    """
    Heuristic: after stripping comments, does the file contain top-level macro/syntax constructs
    (e.g., `syntax`, `macro_rules`, `(tactic| …)` quasiquotes, etc.)? If so, we skip the file.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            src = strip_comments(f.read())
        for pat in SUSPECT_FILE_TOKENS:
            if re.search(pat, src, re.MULTILINE):
                return True
    except OSError:
        pass
    return False

def starts_with_bad_prefix(s: str) -> bool:
    """
    True iff the (left-trimmed) line begins with a declaration/control keyword where instrumentation
    would be inappropriate (e.g., `theorem`, `namespace`, `macro`, …).
    """
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
    """
    True if the file contains `sorry`, `admit`, or `sorryAx` outside identifiers.
    We skip such files to avoid logging half-written proofs that may not elaborate.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            s = f.read()
        return re.search(r'(^|[^A-Za-z_])((sorryAx)|sorry|admit)($|[^A-Za-z_])', s) is not None
    except OSError:
        return False


# --- Core instrumentation helpers -------------------------------------------

def split_by_semicolons_top_level(s: str):
    """
    Split a tactic line by semicolons that are at top level (i.e., not inside (), [], {},
    and not inside string literals). Returns list of segments (whitespace trimmed on both ends).

    Rationale: Lean uses `;` to chain tactics on the main goal. We only split at
    depth zero to avoid breaking tactic arguments and to keep strings intact.
    """
    segs = []
    buf = []
    paren = bracket = brace = 0
    in_string = False
    i = 0
    while i < len(s):
        ch = s[i]
        if in_string:
            buf.append(ch)
            if ch == '"' and (i == 0 or s[i-1] != '\\'):
                in_string = False
            i += 1
            continue
        if ch == '"':
            in_string = True
            buf.append(ch)
        elif ch == '(':
            paren += 1; buf.append(ch)
        elif ch == ')':
            paren = max(0, paren-1); buf.append(ch)
        elif ch == '[':
            bracket += 1; buf.append(ch)
        elif ch == ']':
            bracket = max(0, bracket-1); buf.append(ch)
        elif ch == '{':
            brace += 1; buf.append(ch)
        elif ch == '}':
            brace = max(0, brace-1); buf.append(ch)
        elif ch == ';' and paren == 0 and bracket == 0 and brace == 0:
            segs.append(''.join(buf).strip())
            buf = []
        else:
            buf.append(ch)
        i += 1
    tail = ''.join(buf).strip()
    if tail:
        segs.append(tail)
    return segs

def looks_like_tactic_segment(seg: str) -> bool:
    """
    Decide if `seg` looks like a tactic. We accept if:
    - it matches our tactic starters, OR
    - it starts with 'let ' AND contains ':=' (tactic-mode let),
    - '{' or '}' alone (scope braces in tactic mode).
    """
    s = seg.lstrip()
    if not s:
        return False
    if s[0] in '{}':
        return True
    if s.startswith('let ') and ':=' in s:
        return True
    return TACTIC_RE.match(s) is not None

def inject_logstep_into_segment(seg: str, already_has=False) -> str:
    """
    Prepend 'logStep ' to a segment if appropriate.

    Idempotent: if `already_has` is True or the segment already starts with
    'logStep ', the segment is returned unchanged. Leading whitespace is preserved.
    """
    if already_has:
        return seg
    s = seg.lstrip()
    leading = seg[:len(seg) - len(s)]
    if s.startswith("logStep "):
        return seg
    if looks_like_tactic_segment(s):
        return f"{leading}logStep {s}"
    return seg

def instrument_tactic_line_body(body: str) -> str:
    """
    Instrument a single logical tactic line (no leading indentation and without bullet).
    Split on top-level semicolons and inject 'logStep ' before segments that look like tactics.
    Preserve non-tactic segments unchanged.
    Rejoin with ' ; ' to minimize formatting disruption.

    Example:
      'rw [h]; exact foo'  →  'logStep rw [h] ; logStep exact foo'
    """
    segs = split_by_semicolons_top_level(body)
    out = []
    for seg in segs:
        out.append(inject_logstep_into_segment(seg))
    return " ; ".join(out)

def body_has_instrumentable_segment(body: str) -> bool:
    """True iff any top-level ;-segment looks like a tactic and isn't already logged."""
    for seg in split_by_semicolons_top_level(body):
        s = seg.lstrip()
        if not s or s.startswith("logStep "):
            continue
        if looks_like_tactic_segment(s):
            return True
    return False

NBSP = "\u00A0"

def _is_ws(ch: str) -> bool:
    # Treat Unicode NBSP as whitespace as well.
    return ch.isspace() or ch == NBSP

def _strip_leading_bullets(body: str) -> tuple[str, str]:
    """
    Consume ANY number of leading bullets '·' with any whitespace around them.
    Returns (prefix, remainder), where `prefix` reproduces the exact whitespace
    before each bullet and normalizes each consumed bullet to '· ' (bullet+space).
    The `remainder` is the body after the last consumed bullet (without one
    extra whitespace char after that bullet, if present).

    This lets lines like:
        '·     rintro h'   (NBSP after bullet)
    become:
        '· logStep rintro h'
    while preserving indentation and any whitespace before the bullet.
    """
    prefix = ""
    b = body
    while True:
        # keep the exact whitespace before a potential bullet
        i = 0
        while i < len(b) and _is_ws(b[i]):
            i += 1
        if i >= len(b) or b[i] != "·":
            # no bullet found; DO NOT consume the whitespace — return original body
            return prefix, b
        # accumulate the whitespace and a normalized "· "
        prefix += b[:i] + "· "
        # consume that whitespace + the bullet char
        b = b[i + 1:]
        # consume ONE following whitespace char after the bullet (space/NBSP/tab/etc.)
        if b[:1] and _is_ws(b[0]):
            b = b[1:]

# --- Main per-file instrumenter ---------------------------------------------

def instrument_file(input_path, output_path, lean_import_module):
    """
    Rewrite a single Lean file from `input_path` to `output_path`, inserting `logStep`
    calls in tactic mode while preserving formatting.

    Steps:
      1) Ensure `import {lean_import_module}` is present.
      2) Stream through lines, buffering `by`-blocks:
         • On encountering a line with `:= by`, instrument the head BEFORE `by`
           (so `have … := by` is logged), then start buffering the inner block.
         • While inside the block, compute a Unicode-aware indent baseline and
           continue until a dedent occurs.
         • On each line inside, strip any number of bullets `·` (normalizing to
           `'· '`), split the body at top-level `;`, and prefix `logStep` to
           tactic-looking segments.
         • Remember if any line contained a tactic; on dedent, either write the
           instrumented buffer (if True) or the original (if False).
    """
    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    import_stmt = f"import {lean_import_module}"
    lines = ensure_import(lines, import_stmt)

    new_lines = []
    inside_by_block = False
    base_indent = None

    # Buffers for current `by` block
    block_raw = []      # original lines (including the line with `:= by`)
    block_instr = []    # instrumented counterparts
    block_has_tactic = False

    def flush_block():
        """
        Write the current block to `new_lines`: instrumented if it had any tactic
        segments, otherwise the raw text. Then reset the block buffers.
        """
        nonlocal block_raw, block_instr, block_has_tactic
        if not block_raw:
            return
        new_lines.extend(block_instr if block_has_tactic else block_raw)
        block_raw, block_instr, block_has_tactic = [], [], False

    for line in lines:
        stripped = line.strip()

        # Start of a `by` block?
        # Start of a `by` block?
        if (":= by" in stripped) or stripped.endswith(":= by"):
            # If we were already inside a block, flush it (your current design).
            flush_block()

            inside_by_block = True
            base_indent = None

            # --- instrument the head BEFORE ':= by' ---
            current_indent = len(line) - len(line.lstrip(" "))
            indent_spaces = " " * current_indent
            rest = line[current_indent:]                 # keep punctuation/spacing
            rest_lstrip = rest.lstrip()
            bullet_prefix_len = len(rest) - len(rest_lstrip)
            bullet_indent = rest[:bullet_prefix_len]

            # Normalize/collect bullets and get the remainder after bullets
            norm_bullets, after_bullets = _strip_leading_bullets(rest[bullet_prefix_len:])

            # Split once at the first ':= by'
            head, sep, tail = after_bullets.partition(":= by")
            # Instrument the head like a normal tactic body
            inst_head = instrument_tactic_line_body(head)

            # Build raw vs instrumented versions of this line
            raw_line = line
            inst_line = indent_spaces + bullet_indent + norm_bullets + inst_head + sep + tail
            if not inst_line.endswith("\n"):
                inst_line += "\n"

            # Update block buffers
            block_raw.append(raw_line)
            block_instr.append(inst_line)

            # If the head contains a tactic (e.g., 'have', 'refine', 'show', 'set', 'choose', …),
            # flip the block flag so we emit the instrumented block on flush.
            if body_has_instrumentable_segment(head):
                block_has_tactic = True

            continue

        if not inside_by_block:
            new_lines.append(line)
            continue

        # Inside a by-block
        if stripped == "":
            block_raw.append(line)
            block_instr.append(line)
            continue

        current_indent = len(line) - len(line.lstrip(" "))
        if base_indent is None:
            base_indent = current_indent

        # Dedent => end of block: flush, then process this line as outside
        if current_indent < base_indent:
            inside_by_block = False
            base_indent = None
            flush_block()
            new_lines.append(line)
            continue

        # Lines that clearly start a new declaration: keep as-is in both buffers
        if starts_with_bad_prefix(stripped):
            block_raw.append(line)
            block_instr.append(line)
            continue

        # Prepare to inspect the content:
        indent_spaces = " " * current_indent
        rest = line[current_indent:]           # keep punctuation
        rest_lstrip = rest.lstrip()
        bullet_prefix_len = len(rest) - len(rest_lstrip)

        # Handle ANY number of leading bullets `·`, allowing arbitrary Unicode whitespace
        # between indent and bullet, and after each bullet.
        # Preserve the exact whitespace before each bullet; normalize bullet to "· ".
        bullet_indent = rest[:bullet_prefix_len]  # whitespace between base indent and first non-ws
        norm_bullet_prefix, body = _strip_leading_bullets(rest[bullet_prefix_len:])
        prefix = indent_spaces + bullet_indent + norm_bullet_prefix

        # Decide if this body contains any instrumentable tactic segment
        if body_has_instrumentable_segment(body):
            block_has_tactic = True

        # Build instrumented counterpart for this line (non-destructive)
        instrumented_body = instrument_tactic_line_body(body)
        inst_line = prefix + instrumented_body
        if not inst_line.endswith("\n"):
            inst_line += "\n"

        # Append to buffers
        block_raw.append(line)
        block_instr.append(inst_line)

    # End of file: flush any pending block
    flush_block()

    # Write out
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)


# --- Folder driver -----------------------------------------------------------

def instrument_all(folder_in, folder_out, lean_import_module):
    """
    Mirror `folder_in` into `folder_out`, instrumenting each Lean file that:
      • is not in a skipped directory,
      • does not define `def main`,
      • does not look like a syntax/macro module,
      • contains at least one `:= by`,
      • does not contain `sorry`/`admit`.
    """
    # Clear output folder first
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

            if path_should_be_skipped(input_path):
                print(f"Skipping (dir rule): {rel_path}")
                continue
            if contains_main(input_path):
                print(f"Skipping (has main): {rel_path}")
                continue
            if file_looks_like_macro_or_syntax(input_path):
                print(f"Skipping (syntax/macro file): {rel_path}")
                continue
            if not file_has_tactic_blocks(input_path):
                print(f"Skipping (no tactic blocks): {rel_path}")
                continue
            if file_is_incomplete(input_path):
                print(f"Skipping (incomplete: contains sorry/admit): {rel_path}")
                continue
            # if not preflight_compiles(input_path):
            #     print(f"Skipping (preflight compile failed): {rel_path}")
            #     continue

            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            print(f"Instrumenting {rel_path}")
            instrument_file(input_path, output_path, lean_import_module)

# --- Entrypoint --------------------------------------------------------------

if __name__ == "__main__":
    load_dotenv(".env")
    RAW_FILES_PATH = os.getenv("RAW_FILES_PATH", "raw_proofs")
    INSTRUMENTED_FILES_PATH = os.getenv("INSTRUMENTED_FILES_PATH", "instrumented_proofs")
    LEAN_LOG_TACTIC_IMPORT = os.getenv("LEAN_LOG_TACTIC_IMPORT", "GoalTacticLogger")
    instrument_all(RAW_FILES_PATH, INSTRUMENTED_FILES_PATH, LEAN_LOG_TACTIC_IMPORT)

