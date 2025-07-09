import os

def instrument_file(input_path, output_path):
    with open(input_path, 'r') as f:
        lines = f.readlines()

    new_lines = []
    inside_by_block = False
    indent_level = None

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Detect line ending with ':= by'
        if ':= by' in stripped or stripped.endswith(':= by'):
            inside_by_block = True
            indent_level = None  # to be detected
            new_lines.append(line)
            continue

        if inside_by_block:
            if stripped == '':
                new_lines.append(line)
                continue

            current_indent = len(line) - len(line.lstrip())

            # On first indented line, fix indentation level
            if indent_level is None:
                indent_level = current_indent

            # If indentation is same or deeper, we're still inside tactic block
            if current_indent >= indent_level:
                if 'logStep' in stripped:
                    new_lines.append(line)
                else:
                    # Insert logStep
                    log_line = ' ' * current_indent + 'logStep ' + line.lstrip()
                    new_lines.append(log_line)
                continue
            else:
                # We've exited the block
                inside_by_block = False
                indent_level = None
                new_lines.append(line)
                continue

        else:
            new_lines.append(line)

    with open(output_path, 'w') as f:
        f.writelines(new_lines)

def instrument_all(folder_in, folder_out):
    os.makedirs(folder_out, exist_ok=True)

    for fname in os.listdir(folder_in):
        if fname.endswith('.lean'):
            input_path = os.path.join(folder_in, fname)
            output_path = os.path.join(folder_out, fname)
            print(f"Instrumenting {fname}")
            instrument_file(input_path, output_path)

# Example usage:
instrument_all("raw_proofs", "instrumented_proofs")