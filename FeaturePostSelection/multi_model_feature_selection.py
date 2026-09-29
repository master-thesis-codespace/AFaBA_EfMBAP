#!/usr/bin/env python3
# coding: utf-8

import argparse
import json
import os
import sys
import pandas as pd

def load_importance_series(path):
    df = pd.read_csv(path)
    df = df.iloc[:, 1:]                                                                 # drop the row-label index column
    return df.iloc[0, :]                                                                # row 0 = mean importance; Series: index=feature, value=mean

def get_union_selected_features(importance_file_paths):
    all_feature_names = set()
    per_model_passing = []                                                              # one set of feature names per model
    per_model_means = {}                                                                # feature -> list of mean-importance values across models

    for path in importance_file_paths:
        print(f"Loading importance data from: {path}")
        mean_row = load_importance_series(path)

        all_feature_names.update(mean_row.index)

        max_imp = float(mean_row.iloc[0])                                                # The CSV is sorted descending, so iloc[0] == max
        imp_thr = max_imp * 0.01                                                        # Threshold: 1% of the highest-importance feature in this model.
        passing = set(mean_row[mean_row >= imp_thr].index)
        per_model_passing.append(passing)

        label = os.path.basename(path)
        print(f"{label}: threshold = {imp_thr:.6f} -> {len(passing)} / {len(mean_row)} features pass")

        for feat, val in mean_row.items():
            per_model_means.setdefault(feat, []).append(float(val))

    kept_set = set().union(*per_model_passing)                                          # Union: kept if it passes in ANY model
    excluded_set = all_feature_names - kept_set

    def avg_importance(feat):                                                           #sort kept features by average importance descending, excluded alphabetically
        vals = per_model_means.get(feat, [0.0])
        return sum(vals) / len(vals)

    kept_features = sorted(kept_set, key=avg_importance, reverse=True)
    excluded_features = sorted(excluded_set)

    print(f"\nUnion result: {len(kept_features)} features kept, {len(excluded_features)} features excluded")
    print(f"Kept features: {kept_features}")
    print(f"Excluded features: {excluded_features}")
    return kept_features, excluded_features

def save_feature_selection_json(output_csv, kept_features, excluded_features):
    json_path = os.path.splitext(output_csv)[0] + "_feature_selection.json"
    data = {"kept_features":     kept_features, "excluded_features": excluded_features,}
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Feature selection summary saved to {json_path}")


def filter_and_save_data(input_csv, output_csv, selected_features):
    try:
        df_data = pd.read_csv(input_csv)

        metadata_columns = df_data.columns.tolist()[:4]                                 # First 4 columns are metadata (ID, Age, Sex, Site, etc.)

        valid_features = [f for f in selected_features if f in df_data.columns]         # Guard against feature names present in the importance files but absent from the data CSV

        missing_count = len(selected_features) - len(valid_features)
        if missing_count > 0:
            print(f"Achtung: {missing_count} selected features not found in the data file.")

        final_columns = metadata_columns + valid_features
        df_filtered = df_data[final_columns]

        df_filtered.to_csv(output_csv, index=False)
        print(f"Successfully saved filtered data {output_csv}")

    except FileNotFoundError:
        sys.exit(f"Error! Input data file '{input_csv}' not found.")

def build_argparser():
    p = argparse.ArgumentParser(description=("Filter a feature dataset using a union threshold rule across multiple permutation importance files. A feature is kept if it passes the 1%%-of-max threshold in at least one of the provided models."))
    p.add_argument("--data_file", required=True, help="Raw CSV containing ID, Age, and all feature columns.",)
    p.add_argument("--imp_file", required=True, nargs="+", metavar="IMPORTANCE_CSV", help=("One or more permutation importance CSV files, one per model. A feature is kept if it passes the 1%% threshold in any of them."),)
    p.add_argument("--output_file", required=True, help="Output path for the filtered CSV.",)
    return p

def main():
    args = build_argparser().parse_args()
    features, excluded = get_union_selected_features(args.imp_file)                     #Determine kept/excluded features using the union threshold rule
    filter_and_save_data(args.data_file, args.output_file, features)                    # Filter data file and save
    save_feature_selection_json(args.output_file, features, excluded)                   # Save JSON summary of kept vs. excluded features

if __name__ == "__main__":
    main()