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
    ("XGBoost", "XGBoost", "some path/file ..."),
    ("Random Forest", "some path/file ..."),
    ("Extra Tree", "ExtraTree", "some path/file ..."),
    ("LightGBM", "LightGBM", "some path/file ..."),
    ("CatBoost", "CatBoost", "some path/file ..."),
]

DS_KEYS = ["a", "b", "c"]
DS_LABELS = {"a": "Outlier dataset", "b": "Outlier-free (with SD)", "c": "Outlier-free (without SD)"}
DS_COLORS = {"a": "#e74c3c", "b": "#3498db", "c": "#2ecc71"}

results = {}
for name, subdir, dir_a, dir_b, dir_c in MODELS:
    results[name] = {}
    for ds_key, ds_dir in zip(DS_KEYS, [dir_a, dir_b, dir_c]):
        results[name][ds_key] = {}
        for run in ("default", "best"):
            results[name][ds_key][run] = _load(_path(subdir, ds_dir, run))

BAR_W = 0.11                                                                                            # width of each individual bar
GROUP_GAP = 0.09                                                                                        # gap between default and best sub-groups within one model

_off = np.array([-1.0, 0.0, 1.0]) * BAR_W                                                               # Offsets (from model-centre) for 3 default bars then 3 best bars
OFFSETS_DEFAULT = _off - 1.5 * BAR_W - GROUP_GAP / 2
OFFSETS_BEST = _off + 1.5 * BAR_W + GROUP_GAP / 2

METRIC_CONFIGS = [
    ("RMSE", "RMSE (years)", "comparison_RMSE_19072026.png"),
    ("MAE", "MAE (years)", "comparison_MAE_19072026.png"),
    # ("R2", "R^2 score", "comparison_R2_19072026.png"),                                                # not in cv_performance_summary.csv
]

MODEL_NAMES = [m[0] for m in MODELS]
N_MODELS = len(MODEL_NAMES)
MODEL_POS = np.arange(N_MODELS, dtype=float)


def _legend_handles_labels():
    handles = []
    labels = []
    for ds_key in DS_KEYS:
        p_default = mpatches.Patch(facecolor=DS_COLORS[ds_key], alpha=0.55, edgecolor="black", linewidth=0.6)
        p_best = mpatches.Patch(facecolor=DS_COLORS[ds_key], alpha=1.0, edgecolor="black", linewidth=0.6)
        handles.append((p_default, p_best))
        labels.append(DS_LABELS[ds_key])
    return handles, labels

for metric_key, metric_label, filename in METRIC_CONFIGS:
    fig, ax = plt.subplots(figsize=(15, 6))

    for mi, model_name in enumerate(MODEL_NAMES):
        cx = MODEL_POS[mi]

        for di, ds_key in enumerate(DS_KEYS):
            color = DS_COLORS[ds_key]

            m_def = results[model_name][ds_key]["default"]                                              #default bar
            if m_def is not None:
                bc = ax.bar(cx + OFFSETS_DEFAULT[di], m_def[metric_key], BAR_W, color=color, alpha=0.55, edgecolor="black", linewidth=0.5)
                ax.bar_label(bc, fmt="%.3f", fontsize=5.5, padding=2, rotation=90)

            m_best = results[model_name][ds_key]["best"]                                                #best bar
            if m_best is not None:
                bc = ax.bar(cx + OFFSETS_BEST[di], m_best[metric_key], BAR_W, color=color, alpha=1.0, edgecolor="black", linewidth=0.5)
                ax.bar_label(bc, fmt="%.3f", fontsize=5.5, padding=2, rotation=90)

    y_sub = -0.065                                                                                      #sub-labels below the x-axis
    for mi in range(N_MODELS):
        cx = MODEL_POS[mi]
        def_center = cx + OFFSETS_DEFAULT.mean()
        best_center = cx + OFFSETS_BEST.mean()
        for xc, lbl in [(def_center, "Default-"), (best_center, "VS GS-hyperparameters")]:
            ax.annotate(lbl, xy=(xc, 0), xycoords=("data", "axes fraction"), xytext=(0, -22), textcoords="offset points", ha="center", va="top", fontsize=7, color="dimgray")

    for mi in range(N_MODELS - 1):                                                                       # Vertical separators between model groups
        ax.axvline(MODEL_POS[mi] + 0.5, color="lightgray", linestyle="--", linewidth=0.8, zorder=0)

    ax.set_xticks(MODEL_POS)
    ax.set_xticklabels(MODEL_NAMES, fontsize=11)
    ax.tick_params(axis="x", pad=28)                                                                    # push model labels down to clear sub-labels
    ax.set_ylabel(metric_label, fontsize=12)
    ax.set_title(f"Brain Age Prediction — {metric_label} Comparison (OOF Cross-Validation)", fontsize=13, pad=10,)
    _leg_handles, _leg_labels = _legend_handles_labels()
    ax.legend(handles=_leg_handles, labels=_leg_labels, handler_map={tuple: HandlerTuple(ndivide=None, pad=0.5)}, fontsize=9, ncol=3, loc="lower right", framealpha=0.85)
    ax.grid(axis="y", alpha=0.25, linewidth=0.7)
    ax.set_xlim(MODEL_POS[0] - 0.6, MODEL_POS[-1] + 0.6)

    plt.tight_layout()
    out_path = os.path.join(BASE, filename)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")
