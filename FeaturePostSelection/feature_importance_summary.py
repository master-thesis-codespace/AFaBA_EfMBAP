#!/usr/bin/env python3

import json
import os
import glob
import numpy as np
import pandas as pd

HARMONIZED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))
FOLDERS = ["SC", "BOTH", "BRAIN"]
MODEL_CODES = ["CB", "ET", "LG", "XB", "RF"]

def find_file(folder_path: str, model: str, kind: str):
    candidates = [
        f for f in os.listdir(folder_path)
        if model in f and kind in f and not f.startswith(".")
    ]
    if len(candidates) == 0:
        return None
    if len(candidates) > 1:
        raise ValueError(f"Ambiguous: multiple files in {folder_path} match model='{model}' kind='{kind}': {candidates}")
    return os.path.join(folder_path, candidates[0])

def load_permutation(path: str) -> pd.Series:
    df = pd.read_csv(path, index_col=0)                                                     # first column -> index ("mean", "std")
    mean_row = df.loc["mean"]                                                               # Series: index=feature names, value=mean importance
    return mean_row                                                                         # already sorted descending (most important first)

def load_json_status(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    status = {}
    for feat in data.get("kept_features", []):
        status[feat] = "kept"
    for feat in data.get("excluded_features", []):
        status[feat] = "discarded"
    return status

def build_summary(folder_path: str, folder_name: str) -> pd.DataFrame:                      #Build the summary DataFrame for one dataset folder
    #Collect data for every available model
    model_data = {}                                                                         # model -> {"mean": Series, "status": dict}

    for model in MODEL_CODES:
        perm_path = find_file(folder_path, model, "permutation")
        json_path = find_file(folder_path, model, "feature_selection.json")

        if perm_path is None or json_path is None:                                          # Model not present in this folder
            continue

        print(f"[{folder_name}] {model}: {os.path.basename(perm_path)}")
        model_data[model] = {"mean": load_permutation(perm_path), "status": load_json_status(json_path),}

    if not model_data:
        raise RuntimeError(f"No model data found in {folder_path}")

    all_features = []                                                                       #Collect all feature names (union; same across models here)
    seen = set()
    for md in model_data.values():
        for feat in md["mean"].index:
            if feat not in seen:
                all_features.append(feat)
                seen.add(feat)

    # Compute Avg_Mean across available models
    mean_matrix = pd.DataFrame({model: md["mean"] for model, md in model_data.items()}, index=all_features,)    # rows=features, cols=models; NaN if missing
    avg_mean = mean_matrix.mean(axis=1)                                                     # NaN rows excluded from mean automatically

    rows = pd.DataFrame({"Feature" : all_features, "Avg_Mean": avg_mean.values,})           #build output DataFrame sorted by Avg_Mean descending
    rows = rows.sort_values("Avg_Mean", ascending=False).reset_index(drop=True)

    for model in MODEL_CODES:                                                               #Append three columns per model
        if model not in model_data:
            continue

        mean_series = model_data[model]["mean"]                                             # Series, feature -> mean, sorted desc
        status_map = model_data[model]["status"]
        max_mean = mean_series.iloc[0]                                                      # highest mean in this model
        rank_map = {feat: rank + 1 for rank, feat in enumerate(mean_series.index)}

        pct_col = f"{model}_pct"
        rank_col = f"{model}_rank"
        status_col = f"{model}_status"

        pct_vals = []
        rank_vals = []
        status_vals = []

        for feat in rows["Feature"]:
            if feat in mean_series.index:
                pct_vals.append(round(mean_series[feat] / max_mean * 100, 2))
                rank_vals.append(rank_map[feat])
            else:
                pct_vals.append(np.nan)
                rank_vals.append(np.nan)

            status_vals.append(status_map.get(feat, "-"))

        rows[pct_col] = pct_vals
        rows[rank_col] = rank_vals
        rows[status_col] = status_vals
    return rows

def main():
    for folder_name in FOLDERS:
        folder_path = os.path.join(HARMONIZED_DIR, folder_name)
        if not os.path.isdir(folder_path):
            print(f"Skipping {folder_name}: directory not found at {folder_path}")
            continue

        print(f"\nProcessing {folder_name} …")
        summary = build_summary(folder_path, folder_name)

        out_path = os.path.join(folder_path, f"{folder_name}_feature_importance_summary.csv")
        summary.to_csv(out_path, index=False)
        print(f"Saved {out_path}  ({len(summary)} features)")

if __name__ == "__main__":
    main()