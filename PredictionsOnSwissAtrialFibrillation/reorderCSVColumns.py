#!/usr/bin/env python3
import argparse
import os
import sys
import pandas as pd

def main():
    ap = argparse.ArgumentParser(description="Reorder file2's columns to match file1's column order (same column SET required)")
    ap.add_argument("--file1", required=True, help="Reference CSV - its column order is the target order")
    ap.add_argument("--file2", required=True, help="CSV to be reordered to match file1's column order")
    ap.add_argument("--output", default=None, help="Output path for the reordered file2 (default: '<file2 stem>_reordered.csv' next to file2)")
    ap.add_argument("--in_place", action="store_true", help="Overwrite file2 directly instead of writing a new file (no backup is made - use with care)")
    args = ap.parse_args()

    if not os.path.exists(args.file1):
        sys.exit(f"ERROR: file not found:\n{args.file1}")
    if not os.path.exists(args.file2):
        sys.exit(f"ERROR: file not found:\n{args.file2}")

    cols1 = pd.read_csv(args.file1, nrows=0).columns.tolist()
    cols2 = pd.read_csv(args.file2, nrows=0).columns.tolist()

    set1, set2 = set(cols1), set(cols2)
    if set1 != set2:
        only_in_1 = sorted(set1 - set2)
        only_in_2 = sorted(set2 - set1)
        msg = ["Error, file1 and file2 do not have the same set of columns - refusing to reorder."]
        if only_in_1:
            msg.append(f"Column(s) only in file1: {only_in_1}")
        if only_in_2:
            msg.append(f"Column(s) only in file2: {only_in_2}")
        sys.exit("\n".join(msg))

    if cols1 == cols2:
        print("file1 and file2 already have the same column order - nothing to do.")
        return

    n_diff = sum(1 for a, b in zip(cols1, cols2) if a != b)
    print(f"Reordering {args.file2}: {n_diff}/{len(cols1)} column positions differ from {args.file1}")

    df2 = pd.read_csv(args.file2)
    df2_reordered = df2[cols1]                                                                  # reindex to file1's column order; safe since the sets were already confirmed equal

    if args.in_place:
        out_path = args.file2
    elif args.output:
        out_path = args.output
    else:
        stem, ext = os.path.splitext(args.file2)
        out_path = f"{stem}_reordered{ext}"

    df2_reordered.to_csv(out_path, index=False)
    print(f"Saved - {out_path}")

    check_cols = pd.read_csv(out_path, nrows=0).columns.tolist()                                # Post-write verification, so a partial/broken write is never silently reported as success
    assert check_cols == cols1, "Post-write verification failed - output column order does not match file1"
    print("Verified: output column order exactly matches file1.")

if __name__ == "__main__":
    main()