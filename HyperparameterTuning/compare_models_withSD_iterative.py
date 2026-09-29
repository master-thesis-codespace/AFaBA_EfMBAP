#!/usr/bin/env python3
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerTuple
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))

def _path(subdir, ds_dir, run):
    if subdir is None:
        return os.path.join(BASE, ds_dir, run, "some path/file ...", "some path/file ...")
    return os.path.join(BASE, subdir, ds_dir, run, "some path/file ...", "some path/file ...")

def _load(path):
    if path is None or not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    return {"RMSE": float(df["RMSE"].mean()), "MAE": float(df["MAE"].mean())}

MODELS = [
    ("XGBoost",
     "XGBoost",
     "outputWithSD_28052026",
     ["outputWithSD_28052026",
      "outputWithSD_29052026",
      "outputWithSD_02062026",
      "outputWithSD_02062026_run2",
      "outputWithSD_02062026_run3",
      "outputWithSD_02062026_run4"]),

    ("LightGBM",
     "LightGBM",
     "outputWithSD_28052026",
     ["outputWithSD_28052026",
      "outputWithSD_02062026",
      "outputWithSD_02062026_run2",
      "outputWithSD_02062026_run3",
      "outputWithSD_02062026_run4"]),

    ("CatBoost",
     "CatBoost",
     "outputWithSD_28052026",
     ["outputWithSD_28052026",
      "outputWithSD_03062026_run2",
      "outputWithSD_03062026_run3",
      "outputWithSD_03062026_run4",
      "outputWithSD_03062026_run5"]),

    ("Random Forest",
     "RandomForest",
     "outputWithSD_27052026",
     ["outputWithSD_22072026",
      "outputWithSD_22072026_run2",
      "outputWithSD_22072026_run3",
      "outputWithSD_22072026_run4",
      "outputWithSD_22072026_run5"]),

    ("Extra Tree",
     "ExtraTree",
     "outputWithSD_28052026",
     ["outputWithSD_22072026",
      "outputWithSD_22072026_run2",
      "outputWithSD_22072026_run3",
      "outputWithSD_22072026_run4",
      "outputWithSD_22072026_run5",
      "outputWithSD_22072026_run6"]),
]

MAX_RUNS = 7                                                                            # maximum number of iterative runs across all models
DS_COLOR = "#3498db"                                                                  # blue - outlier-free with SD
results = {}
for name, subdir, default_dir, run_dirs in MODELS:
    results[name] = {}
    results[name]["default"] = _load(_path(subdir, default_dir, "default"))
    results[name]["runs"] = [_load(_path(subdir, d, "best")) for d in run_dirs]

BAR_W = 0.11
GROUP_GAP = 0.09

OFFSET_DEFAULT = -0.32                                                                  # Default bar sits at the far left; best bars follow after the gap
OFFSETS_BEST = np.array([-0.12, -0.01, 0.10, 0.21, 0.32, 0.43, 0.54])

METRIC_CONFIGS = [
    ("RMSE", "RMSE (years)", "comparison_RMSE_withSD_22072026_rerunETRP.png", [2.54, 2.72]),
    ("MAE", "MAE (years)", "comparison_MAE_withSD_22072026_rerunETRP.png", [1.93, 2.12]),
    # ("R2", "R^2 score", "comparison_R2_withSD_22072026_rerunETRP.png"),               # not in cv_performance_summary.csv
]

MODEL_NAMES = [m[0] for m in MODELS]
N_MODELS = len(MODEL_NAMES)
MODEL_POS = np.arange(N_MODELS, dtype=float)

def _legend_handles_labels():
    p_default = mpatches.Patch(facecolor=DS_COLOR, alpha=0.55, edgecolor="black", linewidth=0.6)
    p_best = mpatches.Patch(facecolor=DS_COLOR, alpha=1.0, edgecolor="black", linewidth=0.6)
    handles = [(p_default, p_best)]
    labels = ["Outlier-free (with SD)"]
    return handles, labels

for metric_key, metric_label, filename, y_limit in METRIC_CONFIGS:
    fig, ax = plt.subplots(figsize=(15, 6))

    for mi, (name, subdir, default_dir, run_dirs) in enumerate(MODELS):
        cx = MODEL_POS[mi]

        m_def = results[name]["default"]                                                #default bar (left, lighter alpha)
        if m_def is not None:
            bc = ax.bar(cx + OFFSET_DEFAULT, m_def[metric_key], BAR_W, color=DS_COLOR, alpha=0.55, edgecolor="black", linewidth=0.5)
            ax.bar_label(bc, fmt="%.3f", fontsize=5.5, padding=2, rotation=90)

        for ri, m_run in enumerate(results[name]["runs"]):                              #iterative best bars (right, full alpha)
            if m_run is not None:
                bc = ax.bar(cx + OFFSETS_BEST[ri], m_run[metric_key], BAR_W, color=DS_COLOR, alpha=1.0, edgecolor="black", linewidth=0.5)
                ax.bar_label(bc, fmt="%.3f", fontsize=5.5, padding=2, rotation=90)

    y_sub = -0.065                                                                      #sub-labels below the x-axis
    for mi, (name, subdir, default_dir, run_dirs) in enumerate(MODELS):
        cx = MODEL_POS[mi]
        def_center = cx + OFFSET_DEFAULT
        best_center = cx + OFFSETS_BEST[:len(run_dirs)].mean()
        for xc, lbl in [(def_center, "Default-"), (best_center, "VS GS-hyperparameters")]:
            ax.annotate(lbl, xy=(xc, 0), xycoords=("data", "axes fraction"), xytext=(0, -22), textcoords="offset points", ha="center", va="top", fontsize=7, color="dimgray")

    for mi in range(N_MODELS - 1):                                                      # Vertical separators between model groups
        ax.axvline(MODEL_POS[mi] + 0.5, color="lightgray", linestyle="--", linewidth=0.8, zorder=0)

    ax.set_xticks(MODEL_POS)
    ax.set_xticklabels(MODEL_NAMES, fontsize=11)
    ax.tick_params(axis="x", pad=28)
    ax.set_ylabel(metric_label, fontsize=12)
    ax.set_title(f"Brain Age Prediction - {metric_label} Comparison (OOF Cross-Validation)", fontsize=13, pad=10)
    _leg_handles, _leg_labels = _legend_handles_labels()
    ax.legend(handles=_leg_handles, labels=_leg_labels, handler_map={tuple: HandlerTuple(ndivide=None, pad=0.5)}, fontsize=9, ncol=3, loc="lower right", framealpha=0.85,)
    ax.grid(axis="y", alpha=0.25, linewidth=0.7)
    ax.set_xlim(MODEL_POS[0] - 0.6, MODEL_POS[-1] + 0.6)
    #ax.set_ylim(y_limit[0], y_limit[1])

    plt.tight_layout()
    out_path = os.path.join(BASE, filename)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")