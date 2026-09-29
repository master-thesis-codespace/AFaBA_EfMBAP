#!/usr/bin/env python3
import argparse
import os
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
INPUT_FILES = [os.path.join(BASE, "some path/file ...")]
META_COLS = {"SITE", "ID", "Age", "Sex"}

def output_path(input_path):
    root, ext = os.path.splitext(input_path)
    return root + "_normalized" + ext

def normalize_zscore(df, feature_cols):
    result = df.copy()
    for col in feature_cols:
        std = df[col].std(ddof=0)
        if std == 0:
            result[col] = 0.0
        else:
            result[col] = (df[col] - df[col].mean()) / std
    return result

def normalize_minmax(df, feature_cols):
    result = df.copy()
    for col in feature_cols:
        col_min = df[col].min()
        col_max = df[col].max()
        if col_max == col_min:
            result[col] = 0.0
        else:
            result[col] = (df[col] - col_min) / (col_max - col_min)
    return result

def process_file(path, method):
    if not os.path.exists(path):
        print(f"SKIPPED (file not found): {path}")
        return

    df = pd.read_csv(path)
    feature_cols = [c for c in df.columns if c not in META_COLS and pd.api.types.is_numeric_dtype(df[c])]

    if method == "zscore":
        df_norm = normalize_zscore(df, feature_cols)
    else:
        df_norm = normalize_minmax(df, feature_cols)

    out = output_path(path)
    df_norm.to_csv(out, index=False)

def main():
    parser = argparse.ArgumentParser(description="Normalize measurement features in CSV files.")
    parser.add_argument("--method", choices=["zscore", "minmax"], required=True, help="zscore: zero mean / unit variance.  minmax: scale to [0, 1].")
    args = parser.parse_args()

    label = "Z-score (mean=0, std=1)" if args.method == "zscore" else "Min-Max [0, 1]"
    print(f"Normalization method: {label}")

    for path in INPUT_FILES:
        process_file(path, args.method)

if __name__ == "__main__":
    main()