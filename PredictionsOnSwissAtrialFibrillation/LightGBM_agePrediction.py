# coding: utf-8
import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def load_json(file_path: str) -> Dict[str, Any]:
    try:
        with open(file_path, "r") as f:                                                 #parse a JSON file into a python dictionary
            return json.load(f)                                                         #to load correction_params.json files
    except FileNotFoundError:
        sys.exit(f"Error: The configuration file {file_path} was not found.")

def preproc_data(file_path: str, feature_columns: Optional[List[str]] = None, return_ID: bool = False):
    df_data = pd.read_csv(file_path)                                                    #reads CSV
    if feature_columns is None:
        if df_data.shape[1] > 4:
            feature_columns = df_data.columns.to_list()[4:]                             #default features: index 4 onwards (SITE, ID, Age, Sex are metadata) - matches LightGBM_age_model_modelling.py preproc_data()
        else:
            sys.exit("Error: Input CSV has fewer than 5 columns.")

    df_data.dropna(axis=1, how="all", inplace=True)                                     #handle missing values
    df_data.dropna(axis=0, how="all", inplace=True)
    df_data.fillna(0, inplace=True)

    if "Age" not in df_data.columns:                                                    #checks for Age column
        sys.exit("Error: Column 'Age' missing from input CSV.")
    target = "Age"
    X_data = df_data.loc[:, feature_columns].values
    Y_data = df_data.loc[:, target].values

    if return_ID:
        if "ID" not in df_data.columns:                                                 #checks for ID column
            sys.exit("Error: Column 'ID' missing from input CSV.")
        return X_data, Y_data, df_data["ID"].values                                     #optionally returns ID for annotating output
    return  X_data, Y_data

def prediction_brain_age_MS(file_path, best_model_pkl_file, HC_prediction_json_file, output_csv_filename):
    selected_feature_list = pd.read_csv(file_path).columns.tolist()[4:]                 #get the selected features (index 4 onwards - matches training script)

    X_data, Y_data, ID_data = preproc_data(file_path, feature_columns=selected_feature_list, return_ID=True)

    Y_predict = np.zeros_like(Y_data)
    Y_corrected = np.zeros_like(Y_data)

    #Setup Model
    bst = joblib.load(best_model_pkl_file)                                              #load the pretrained LightGBM model - joblib.load() returns the fully fitted estimator, no reconstruction needed

    #Predict
    Y_predict_cv = bst.predict(X_data)                                                  #generate brain age estimates
    correction_dict = load_json(HC_prediction_json_file)                                #load correction from healthy controls

    #Correct Bias
    Y_corrected_cv = Y_predict_cv + (
        Y_data
        - (correction_dict["correction_slope"] * Y_data + correction_dict["correction_intercept"])
    )                                                                                   #apply healthy controls correction to all predictions with MS

    Y_predict += Y_predict_cv
    Y_corrected += Y_corrected_cv

    #Visualize and SAVE
    axis_lo = min(Y_data.min(), Y_predict.min(), Y_corrected.min()) - 2                 #shared axis range across both subplots, derived from the data itself (not hardcoded) and padded by 2 years
    axis_hi = max(Y_data.max(), Y_predict.max(), Y_corrected.max()) + 2
    identity_line = np.array([axis_lo, axis_hi])

    fig, ax = plt.subplots(1, 2, figsize=(30, 20))                                      #side by side scatter plot (raw compared with corrected predictions)
    ax = ax.ravel()
    ax[0].scatter(Y_data, Y_predict)
    ax[0].plot(identity_line, identity_line, c="black")
    ax[0].set_xlim(axis_lo, axis_hi)
    ax[0].set_ylim(axis_lo, axis_hi)

    ax[1].scatter(Y_data, Y_corrected)
    ax[1].plot(identity_line, identity_line, c="black")
    ax[1].set_xlim(axis_lo, axis_hi)
    ax[1].set_ylim(axis_lo, axis_hi)
    output_plot_filename = Path(output_csv_filename).with_suffix('.png')
    plt.savefig(output_plot_filename)
    print(f"Plot saved to {output_plot_filename}")
    plt.close()

    df_age = pd.DataFrame({"ID": ID_data, "Age_Actual": Y_data, "Age_Predicted": Y_predict, "Age_Corrected": Y_corrected})
    df_age.to_csv(output_csv_filename, index=False)                                     #save final results as a CSV
    print(f"Results saved to {output_csv_filename}")

def build_argparser():                                                                  #run per model fold
    p = argparse.ArgumentParser(description="Brain Age Prediction Script")
    p.add_argument("--file_path", help="CSV file with columns SITE, ID, Age, Sex, then morphometrics (features start at col 5).", required=True)
    p.add_argument("--best_model_pkl_file",help="Pickled (.pkl) file of the best trained LightGBM model in cross-validation", required=True)
    p.add_argument("--HC_prediction_json_file", help="JSON file containing the slope and the intercept of the correction based on healthy controls", required=True)
    p.add_argument("--output_csv_filename", help="Filename for the output CSV.",required=True)
    return p

def main():
    args = build_argparser().parse_args()

    prediction_brain_age_MS(
        file_path=args.file_path,
        best_model_pkl_file=args.best_model_pkl_file,
        HC_prediction_json_file=args.HC_prediction_json_file,
        output_csv_filename=args.output_csv_filename,
    )

if __name__ == "__main__":
    main()