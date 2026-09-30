#!/usr/bin/env python3
import argparse
import json
import os
import sys

import pandas as pd


def load_json(file_path: str):
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        sys.exit(f"Error: The configuration file {file_path} was not found.")


def main():
    ap = argparse.ArgumentParser(description="Apply 2nd bias correction")
    ap.add_argument("--output_dir", default=".", help="Folder containing predictions")
    args = ap.parse_args()

    pred_path = os.path.join(args.output_dir, "best", "some path/file ...")
    params_path = os.path.join(args.output_dir, "best", "some path/file ...")

    if not os.path.exists(pred_path):
        sys.exit(f"ERROR: file not found:\n  {pred_path}")

    df = pd.read_csv(pred_path)
    required = {"ID", "Age_Actual", "Age_Predicted", "Age_Corrected"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"ERROR: missing columns in {pred_path}: {missing}")

    correction_params = load_json(params_path)
    slope = correction_params["correction_slope"]
    intercept = correction_params["correction_intercept"]

    Y_data = df["Age_Actual"].values
    Y_predict = df["Age_Corrected"].values

    Y_corrected2 = Y_predict + (Y_data - (slope * Y_data + intercept))

    print(f"Applied correction from {params_path} to {len(df)} subjects from {pred_path}")
    print(f"correction_slope = {slope:.6f}")
    print(f"correction_intercept = {intercept:.6f}")

    resid_before = Y_predict - Y_data
    resid_after = Y_corrected2 - Y_data

    df_out = pd.DataFrame({
        "ID": df["ID"],
        "Age_Actual": df["Age_Actual"],
        "Age_Predicted": df["Age_Predicted"],
        "Age_Corrected": df["Age_Corrected"],
        "Age_Corrected2": Y_corrected2,
        "uncorPAD": df["Age_Predicted"] - df["Age_Actual"],
        "corPAD": Y_corrected2 - Y_data,
    })

    csv_out_path = os.path.join(args.output_dir, "best", "results", "cv_final_predictions_SAFHC.csv")
    df_out.to_csv(csv_out_path, index=False)

if __name__ == "__main__":
    main()