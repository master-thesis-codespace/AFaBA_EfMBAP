#!/usr/bin/env python3
import argparse
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

_BLUE_DOT = "#2a78d6"                                                                           # uncorrected - bars
_BLUE_LINE = "#0d366b"                                                                          # uncorrected - mean line
_RED_DOT = "#e34948"                                                                            # corrected - bars
_RED_LINE = "#7c1111"                                                                           # corrected - mean line
_SURFACE = "#fcfcfb"
_PRIMARY = "#0b0b0b"
_MUTED = "#898781"
_GRIDLINE = "#e1e0d9"

AGE_GROUPS = [("10-19", 10, 20), ("70-79", 70, 80)]                                                  #Age groups to plot: (label used in filenames/titles, lo inclusive, hi exclusive)

DATA_SOURCES = [                                                                                    #Data sources: (tag used in filenames, CSV filename, title phrase)
    ("train", "some path/file ...", "Training Set (OOF Cross-Validation)"),
    ("test",  "some path/file ...",     "Test Set"),
]

def load_predictions(folder: str, csv_name: str) -> pd.DataFrame:
    path = os.path.join(folder, "best", "results", csv_name)
    if not os.path.exists(path):
        sys.exit(f"Error: file not found:\n{path}")
    df = pd.read_csv(path)
    missing = {"Age_Actual", "Age_Predicted", "Age_Corrected"} - set(df.columns)
    if missing:
        sys.exit(f"Error: missing columns in {path}: {missing}")
    if "uncorPAD" not in df.columns:
        df["uncorPAD"] = df["Age_Predicted"] - df["Age_Actual"]
    if "corPAD" not in df.columns:
        df["corPAD"] = df["Age_Corrected"] - df["Age_Actual"]
    return df

def _style_ax(ax: plt.Axes) -> None:
    ax.set_facecolor(_SURFACE)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(_MUTED)
        ax.spines[sp].set_linewidth(0.8)
    ax.tick_params(axis="both", labelsize=9, colors=_PRIMARY)
    ax.xaxis.grid(True, color=_GRIDLINE, linewidth=0.35, zorder=0)
    ax.yaxis.grid(True, color=_GRIDLINE, linewidth=0.35, zorder=0)
    ax.set_axisbelow(True)

def plot_histogram(df: pd.DataFrame, out_path: str, title: str) -> None:
    uncor = df["uncorPAD"].values
    cor = df["corPAD"].values

    all_pad = np.concatenate([uncor, cor])                                                          # Build half-integer bin edges covering both distributions
    lo = int(np.floor(all_pad.min()))
    hi = int(np.ceil(all_pad.max()))
    bins = np.arange(lo - 0.5, hi + 1.5, 1.0)

    w_uncor = np.ones(len(uncor)) / len(uncor)
    w_cor = np.ones(len(cor)) / len(cor)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.patch.set_facecolor(_SURFACE)

    ax.hist(uncor, bins=bins, weights=w_uncor, color=_BLUE_DOT, alpha=0.55, edgecolor="white", linewidth=0.5, label="Uncorrected", zorder=2)
    ax.hist(cor, bins=bins, weights=w_cor, color=_RED_DOT, alpha=0.55, edgecolor="white", linewidth=0.5, label="Corrected", zorder=3)

    ax.axvline(np.mean(uncor), color=_BLUE_LINE, linestyle="--", linewidth=1.5, zorder=4)
    ax.axvline(np.mean(cor), color=_RED_LINE, linestyle="--", linewidth=1.5, zorder=4)

    ax.set_xlabel("Predicted age difference (years)", fontsize=11, color=_PRIMARY, labelpad=5)
    ax.set_ylabel("Frequency (normalized)", fontsize=11, color=_PRIMARY, labelpad=5)
    ax.set_title(title, fontsize=12, color=_PRIMARY, pad=10)

    _style_ax(ax)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=_BLUE_DOT, alpha=0.55, label="Uncorrected"),
        Line2D([0], [0], color=_BLUE_LINE, linestyle="--", linewidth=1.5, label=f"Uncorrected mean ({np.mean(uncor):.1f})"),
        plt.Rectangle((0, 0), 1, 1, color=_RED_DOT, alpha=0.55, label="Corrected"),
        Line2D([0], [0], color=_RED_LINE, linestyle="--", linewidth=1.5, label=f"Corrected mean ({np.mean(cor):.1f})"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=9, framealpha=0.90, edgecolor=_MUTED, borderpad=0.6)

    plt.tight_layout(pad=1.0)
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saving to {out_path}")

def main() -> None:
    ap = argparse.ArgumentParser(description="Brain-age-gap histograms: uncorrected vs corrected, split by age group (10-19, 70-79) and by training/test set")
    ap.add_argument("--output_dir", default=".", help="Model output folder containing best/results/cv_final_predictions.csv and test_predictions.csv")
    args = ap.parse_args()

    dir_label = os.path.basename(os.path.abspath(args.output_dir))

    for source_tag, csv_name, source_title in DATA_SOURCES:
        print(f"Loading {source_title} predictions ({csv_name}): {args.output_dir}")
        df = load_predictions(args.output_dir, csv_name)

        for age_label, age_lo, age_hi in AGE_GROUPS:
            df_age = df[(df["Age_Actual"] >= age_lo) & (df["Age_Actual"] < age_hi)]
            if len(df_age) == 0:
                print(f"Achtung: no subjects with Age_Actual in [{age_lo},{age_hi}) in {csv_name} - skipped")
                continue

            print(f"Age {age_label}: n={len(df_age)}")
            title = f"Brain-Age Gap - Differences on the SAF set, Age {age_label} - {dir_label}"
            out_path = os.path.join(args.output_dir, f"{dir_label}_hist_uncorVsCor_SAF_age{age_label}.png")
            plot_histogram(df_age, out_path, title=title)
        print()

if __name__ == "__main__":
    main()