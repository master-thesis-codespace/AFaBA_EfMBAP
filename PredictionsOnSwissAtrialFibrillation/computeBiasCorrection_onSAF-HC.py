#!/usr/bin/env python3
import argparse
import json
import os
import sys
import numpy as np
import pandas as pd


def correct_age(Y_data, Y_predict):
    # Calculate Brain Age Gap (BAG) as the difference between predicted ages and actual ages.
    BAG = Y_predict - Y_data
    # Fit a linear model to the relationship between actual ages and BAG.
    z = np.polyfit(Y_data, BAG, 1)
    # Calculate the uncorrected residue by removing the linear trend from BAG.
    Y_uncorrected_residue = BAG - (z[0] * Y_data + z[1])
    # Fit a linear model to the relationship between actual ages and predicted ages.
    correction_term = np.polyfit(Y_data, Y_predict, 1)
    # Correct the predicted ages using the correction term.
    Y_corrected = Y_predict + (Y_data - (correction_term[0] * Y_data + correction_term[1]))  # shift predictions to align with actual age line
    return Y_uncorrected_residue, correction_term, Y_corrected

def main():
    ap = argparse.ArgumentParser(description="Locally-fit bias correction (Age_Actual vs Age_Corrected) for a prediction folder")
    ap.add_argument("--output_dir", default=".", help="Folder containing best/results/cv_final_predictions.csv")
    args = ap.parse_args()

    path = os.path.join(args.output_dir, "best", "some path/file ...")
    if not os.path.exists(path):
        sys.exit(f"Error, file not found:\n {path}")
    df = pd.read_csv(path)
    required = {"ID", "Age_Actual", "Age_Predicted", "Age_Corrected"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"Error, missing columns in {path}: {missing}")

    Y_data = df["Age_Actual"].values
    Y_predict = df["Age_Corrected"].values                                                  # fit (and, below, apply) against the ALREADY-corrected predictions, per explicit instruction

    _, correction_term, Y_corrected = correct_age(Y_data, Y_predict)                        # Y_corrected here is ALREADY Age_Corrected2 = Age_Corrected + (Age_Actual - (slope*Age_Actual + intercept)), thus correct_age() fits correction_term from (Y_data, Y_predict) and applies it to Y_predict internally, so no separate application step is needed

    correction_params = {
        "correction_slope": float(correction_term[0]),
        "correction_intercept": float(correction_term[1]),
    }

    params_out_path = os.path.join(args.output_dir, "best", "results", "correction_params_SAFHC.json")
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
    print(f"Mean (Age_Corrected2 - Age_Actual) AFTER this new correction: {resid_after.mean():.4f}")

    print(f"Saved as {params_out_path}")

    df_out = pd.DataFrame({                                                                 #apply the new correction to Age_Corrected -> Age_Corrected2, and save the predictions CSV
        "ID": df["ID"],
        "Age_Actual": df["Age_Actual"],
        "Age_Predicted": df["Age_Predicted"],
        "Age_Corrected": df["Age_Corrected"],
        "Age_Corrected2": Y_corrected,
        "uncorPAD": df["Age_Predicted"] - df["Age_Actual"],                                 # anchored to the raw prediction, matching this project's convention elsewhere
        "corPAD": Y_corrected - Y_data,                                                     # residual after the second-stage correction (Age_Corrected2 - Age_Actual)
    })

    csv_out_path = os.path.join(args.output_dir, "best", "results", "cv_final_predictions_SAFHC.csv")
    df_out.to_csv(csv_out_path, index=False)
    print(f"Saved ... {csv_out_path}")

if __name__ == "__main__":
    main()