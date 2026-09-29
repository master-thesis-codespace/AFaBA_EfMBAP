#!/usr/bin/env python3
# coding: utf-8

import argparse
import json
import os
import sys
import pandas as pd

def get_selected_feature_list(importance_file_path):
    try:
        print(f"Loading importance data from: {importance_file_path}")                  # Load the specific importance file
        df_XG_results = pd.read_csv(importance_file_path)                               #load permutation_cv{i}.csv

        # Logic from original script: Remove first column (assumed to be index/names)
        df_XG_results = df_XG_results.iloc[:, 1:]                                       #drop first column

        # Calculate threshold (1% of the maximum importance found at index 0,0)
        max_importance = df_XG_results.iloc[0, 0]                                       #highest importance (row is mean, sorted descendent)
        thr = 0.01                                                                      #one percent of maximum
        imp_thr = max_importance * thr

        # Filter features where importance is >= threshold
        selected_feature_list = df_XG_results.loc[                                      #filter out low-contributing features
            :, df_XG_results.iloc[0, :] >= imp_thr
        ].columns.to_list()

        discarded_feature_list = df_XG_results.loc[                                      #filter out low-contributing features
            :, df_XG_results.iloc[0, :] < imp_thr
        ].columns.to_list()
        
        print(f"Selected {len(selected_feature_list)} features based on threshold {imp_thr:.4f}")
        print(f"Kept features: {selected_feature_list}")
        print(f"\nDiscarded features: {discarded_feature_list}")
        return selected_feature_list, discarded_feature_list

    except FileNotFoundError:
        sys.exit(f"Error: Importance file '{importance_file_path}' not found.")
    except Exception as e:
        sys.exit(f"An error occurred during feature selection: {e}")

def save_feature_selection_json(output_csv, kept_features, excluded_features):
    json_path = os.path.splitext(output_csv)[0] + "_feature_selection.json"
    data = {
        "kept_features": kept_features,
        "excluded_features": excluded_features,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Feature selection summary saved to: {json_path}")

def filter_and_save_data(input_csv, output_csv, selected_features):
    try:
        df_data = pd.read_csv(input_csv)                                                #load full feature csv

        # Assuming first 5 columns are metadata (ID, Age, Sex, Site, etc.)
        metadata_columns = df_data.columns.tolist()[:4]                                 #extract metadata

        # Verify selected features exist in the data file
        valid_features = [f for f in selected_features if f in df_data.columns]         #guarantee feature names exist in dataset
        
        missing_count = len(selected_features) - len(valid_features)
        if missing_count > 0:
            print(f"Warning: {missing_count} selected features were not found in the data file.")

        # Combine metadata and valid features
        final_columns = metadata_columns + valid_features                               #concatenates metadata and valid selected features
        df_filtered = df_data[final_columns]

        df_filtered.to_csv(output_csv, index=False)                                     # Save
        print(f"Successfully saved filtered data to: {output_csv}")

    except FileNotFoundError:
        sys.exit(f"Error: Input data file '{input_csv}' not found.")

def build_argparser():
    p = argparse.ArgumentParser(description="Filter brain morphometry data using a specific importance file.")
    p.add_argument("--data_file", required=True, help="The raw csv file containing ID, Age, and all brain morphometrics.")
    p.add_argument("--imp_file", required=True, help="The specific importance/permutation CSV file to use for selection.")
    p.add_argument("--output_file", required=True, help="The filename for the new filtered output csv file.")
    return p

def main():
    args = build_argparser().parse_args()

    features, excluded = get_selected_feature_list(args.imp_file)                   # Get list of features directly from the importance file

    filter_and_save_data(args.data_file, args.output_file, features)                # Filter data file and save

    save_feature_selection_json(args.output_file, features, excluded)               # Save JSON summary of kept vs. excluded features

if __name__ == "__main__":
    main()