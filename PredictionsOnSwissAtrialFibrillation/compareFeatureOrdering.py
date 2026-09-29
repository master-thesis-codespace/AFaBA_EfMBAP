#!/usr/bin/env python3
import argparse
import os
import sys
import pandas as pd

def read_header(path: str):
    if not os.path.exists(path):
        sys.exit(f"ERROR, file not found:\n {path}")
    return pd.read_csv(path, nrows=0).columns.tolist()

def compare_headers(name_a: str, cols_a, name_b: str, cols_b) -> tuple[bool, list[str]]:
    lines = [f"{name_a} ({len(cols_a)} columns) vs {name_b} ({len(cols_b)} columns):"]

    if cols_a == cols_b:
        lines.append("IDENTICAL (same columns, same order)")
        return True, lines

    if len(cols_a) != len(cols_b):
        lines.append(f"Column count differs: {name_a} has {len(cols_a)}, {name_b} has {len(cols_b)}")

    set_a, set_b = set(cols_a), set(cols_b)
    only_in_a = [c for c in cols_a if c not in set_b]
    only_in_b = [c for c in cols_b if c not in set_a]
    if only_in_a:
        lines.append(f"Column(s) in {name_a} but not in {name_b}: {only_in_a}")
    if only_in_b:
        lines.append(f"Column(s) in {name_b} but not in {name_a}: {only_in_b}")

    if not only_in_a and not only_in_b:
        lines.append("Same set of columns, but different ORDER.")                                   # Same set of columns, so the mismatch must be ordering
        for i, (a, b) in enumerate(zip(cols_a, cols_b)):
            if a != b:
                lines.append(f"First differing position: index {i} -> {name_a} has '{a}', {name_b} has '{b}'")
                break
    return False, lines

def main():
    ap = argparse.ArgumentParser(description="Check whether three CSV files have identical columns (header), in the same order")
    ap.add_argument("--file1", required=True, help="Path to the first (reference) CSV file")
    ap.add_argument("--file2", required=True, help="Path to the second CSV file")
    ap.add_argument("--file3", required=True, help="Path to the third CSV file")
    ap.add_argument("--output", default="structural_comparison_result.txt", help="Path to the output .txt file (default: structural_comparison_result.txt)")
    args = ap.parse_args()

    name1, name2, name3 = os.path.basename(args.file1), os.path.basename(args.file2), os.path.basename(args.file3)

    cols1 = read_header(args.file1)
    cols2 = read_header(args.file2)
    cols3 = read_header(args.file3)

    match_12, report_12 = compare_headers(name1, cols1, name2, cols2)
    match_13, report_13 = compare_headers(name1, cols1, name3, cols3)

    all_match = match_12 and match_13

    lines = []
    if all_match:
        lines.append(f"PASS, {name1}, {name2} and {name3} are structurally identical")
    else:
        lines.append("FAIL")
        lines.append("")
        lines.extend(report_12)
        lines.append("")
        lines.extend(report_13)

    report_text = "\n".join(lines) + "\n"

    with open(args.output, "w") as f:
        f.write(report_text)

    print(report_text)
    print(f"Saved as {args.output}")

if __name__ == "__main__":
    main()