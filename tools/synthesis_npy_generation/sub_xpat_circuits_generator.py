import os
import shutil
import re
from collections import defaultdict, deque

def convert_verilog_expr_to_python(expr: str) -> str:
    """
    Converts a Verilog logical expression into Python syntax:
    - Handles the ternary operator (including nested ones) by transforming:
    cond ? a : b  -->  ((a) if (cond) else (b))
    - Replaces bitwise/boolean operators and constants.
    """

    expr = re.sub(r"\b1'[bB]0\b", "0", expr)
    expr = re.sub(r"\b1'[bB]1\b", "1", expr)
    expr = re.sub(r"\b1'[hH]0\b", "0", expr)
    expr = re.sub(r"\b1'[hH]1\b", "1", expr)


    def parse_ternary(s: str) -> str:
        s = s.strip()
        while s.startswith("(") and s.endswith(")"):
            depth = 0
            is_wrapped = True
            for i in range(len(s) - 1):
                if s[i] == "(":
                    depth += 1
                elif s[i] == ")":
                    depth -= 1
                if depth == 0:
                    is_wrapped = False
                    break
            if is_wrapped:
                s = s[1:-1].strip()
            else:
                break

        depth = 0
        q_pos = -1
        col_pos = -1

        for i, ch in enumerate(s):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            elif depth == 0 and ch == "?" and q_pos == -1:
                q_pos = i

        if q_pos == -1:
            return s

        depth = 0
        for i in range(q_pos + 1, len(s)):
            if ch := s[i]:
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                elif depth == 0 and ch == ":":
                    col_pos = i
                    break

        if col_pos != -1:
            cond = parse_ternary(s[:q_pos])
            val_true = parse_ternary(s[q_pos + 1:col_pos])
            val_false = parse_ternary(s[col_pos + 1:])
            return f"(({val_true}) if ({cond}) else ({val_false}))"

        return s

    expr = parse_ternary(expr)


    expr = expr.replace("~", " not ")
    expr = expr.replace("&", " and ")
    expr = expr.replace("|", " or ")
    expr = expr.replace("^", " ^ ")

    return expr


def topological_sort_lines(lines):
    var_pattern = re.compile(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b')
    reserved_words = {'assign', 'wire', 'input', 'output', 'module', 'endmodule', 'not', 'and', 'or'}

    definitions = {}
    dependencies = {}
    in_degree = {}
    graph = defaultdict(list)
    declared_assigns = []

    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("assign"):
            continue

        clean_line = stripped[6:].rstrip(";").strip()
        if "=" not in clean_line:
            continue

        lhs, rhs = clean_line.split("=", 1)
        lhs = lhs.strip()

        # Isola i token identificatori escludendo costanti numeriche e parole chiave
        rhs_tokens = set(var_pattern.findall(rhs)) - reserved_words

        definitions[lhs] = clean_line
        dependencies[lhs] = rhs_tokens
        declared_assigns.append(lhs)

    defined_vars = set(definitions.keys())

    for var in declared_assigns:
        internal_deps = dependencies[var] & defined_vars
        in_degree[var] = len(internal_deps)
        for dep in internal_deps:
            graph[dep].append(var)

    queue = deque([v for v in declared_assigns if in_degree[v] == 0])
    sorted_lhs = []

    while queue:
        curr = queue.popleft()
        sorted_lhs.append(curr)
        for nxt in graph[curr]:
            in_degree[nxt] -= 1
            if in_degree[nxt] == 0:
                queue.append(nxt)

    if len(sorted_lhs) < len(declared_assigns):
        unvisited = [v for v in declared_assigns if v not in set(sorted_lhs)]
        sorted_lhs.extend(unvisited)

    return [definitions[lhs] for lhs in sorted_lhs]


def generate_approx_mult_function(input_verilog_path: str, bitwidth: int):
    output_filename = "./tools/synthesis_npy_generation/sub_x_pat_multiplier.py"
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)

    with open(input_verilog_path, "r") as file:
        raw_lines = file.readlines()

    full_text = "".join(raw_lines)

    if re.search(r'assign\s+r\s*=\s*a\s*\*\s*b\s*;', full_text) or "module mul_i" in full_text:
        with open(output_filename, "w") as destination_file:
            destination_file.write("def approx_mult(a: int, b: int) -> int:\n")
            destination_file.write("\treturn int(a * b)\n")
        return

    sorted_assigns = topological_sort_lines(raw_lines)
    output_len = 2 * bitwidth

    prefix = "a_out" if ("a_out" in full_text) else "out"

    with open(output_filename, "w") as destination_file:
        destination_file.write("def approx_mult(a: int, b: int) -> int:\n")

        input_a_vars = ", ".join([f"in{i}" for i in range(bitwidth - 1, -1, -1)])
        input_b_vars = ", ".join([f"in{i}" for i in range(2 * bitwidth - 1, bitwidth - 1, -1)])

        destination_file.write(f"\t{input_a_vars} = [int(bit) for bit in bin(a)[2:].zfill({bitwidth})]\n")
        destination_file.write(f"\t{input_b_vars} = [int(bit) for bit in bin(b)[2:].zfill({bitwidth})]\n")

        init_outputs = [f"{prefix}{i} = 0" for i in range(output_len)]
        destination_file.write("\t" + "; ".join(init_outputs) + "\n")

        for clean_assign in sorted_assigns:
            lhs, rhs = clean_assign.split("=", 1)
            lhs = lhs.strip()
            py_rhs = convert_verilog_expr_to_python(rhs)
            destination_file.write(f"\t{lhs} = int({py_rhs})\n")

        output_bits_vars = ", ".join([f"str(int({prefix}{i}))" for i in range(output_len - 1, -1, -1)])
        destination_file.write(f"\tbits = [{output_bits_vars}]\n")
        destination_file.write("\treturn int(''.join(bits), 2)\n")