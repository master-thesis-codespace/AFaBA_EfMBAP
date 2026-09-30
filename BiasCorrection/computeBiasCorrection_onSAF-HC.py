#!/usr/bin/env python3
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd


def correct_age(Y_data, Y_predict):
    BAG = Y_predict - Y_data
    z = np.polyfit(Y_data, BAG, 1)
    Y_uncorrected_residue = BAG - (z[0] * Y_data + z[1])
    correction_term = np.polyfit(Y_data, Y_predict, 1)
    Y_corrected = Y_predict + (Y_data - (correction_term[0] * Y_data + correction_term[1]))
    return Y_uncorrected_residue, correction_term, Y_corrected

def main():
    ap = argparse.ArgumentParser(description="2nd bias correction")
    ap.add_argument("--output_dir", default=".", help="Folder containing predictions")
    args = ap.parse_args()

    path = os.path.join(args.output_dir, "best", "some path/file ...")
    if not os.path.exists(path):
        sys.exit(f"ERROOR: file not found:\n  {path}")
    df = pd.read_csv(path)
    required = {"ID", "Age_Actual", "Age_Predicted", "Age_Corrected"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"Error: missing columns in {path}: {missing}")

    Y_data = df["Age_Actual"].values
    Y_predict = df["Age_Corrected"].values 

    _, correction_term, Y_corrected = correct_age(Y_data, Y_predict)

    correction_params = {
        "correction_slope": float(correction_term[0]),
        "correction_intercept": float(correction_term[1]),
    }

    params_out_path = os.path.join(args.output_dir, "best", "some path/file ...")
    with open(params_out_path, "w") as f:
        json.dump(correction_params, f, indent=4)

    print(f"Fit on {len(df)} subjects from {path}")
    print(f"Y_data = Age_Actual")
    print(f"Y_predict = Age_Corrected (already training-corrected)")
    print(f"correction_slope = {correction_params['correction_slope']:.6f}")
    print(f"correction_intercept = {correction_params['correction_intercept']:.6f}")

    resid_before = Y_predict - Y_data
    resid_after = Y_corrected - Y_data
    print(f"Mean (Age_Corrected - Age_Actual) BEFORE this new correction: {resid_before.mean():.4f}")
    print(f"Mean (Age_Corrected2 - Age_Actual) AFTER this new correction: {resid_after.mean():.4f} (should be ~0 by construction)")

    print(f"Saved -> {params_out_path}")

    df_out = pd.DataFrame({
        "ID": df["ID"],
        "Age_Actual": df["Age_Actual"],
        "Age_Predicted": df["Age_Predicted"],
        "Age_Corrected": df["Age_Corrected"],
        "Age_Corrected2": Y_corrected,
        "uncorPAD": df["Age_Predicted"] - df["Age_Actual"],
        "corPAD": Y_corrected - Y_data,
    })

    csv_out_path = os.path.join(args.output_dir, "best", "results", "cv_final_predictions_SAFHC.csv")
    df_out.to_csv(csv_out_path, index=False)

if __name__ == "__main__":
    main()