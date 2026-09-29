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

def load_predictions(folder: str) -> pd.DataFrame:
    path = os.path.join(folder, "best", "results", "test_predictions.csv")
    if not os.path.exists(path):
        sys.exit(f"Error, file not found:\n{path}")
    df = pd.read_csv(path)
    missing = {"Age_Actual", "Age_Predicted", "Age_Corrected"} - set(df.columns)
    if missing:
        sys.exit(f"Error, missing columns in {path}: {missing}")
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
    ap = argparse.ArgumentParser(description="Brain-age-gap histogram: uncorrected vs corrected (test set)")
    ap.add_argument("--output_dir", default=".", help="Model output folder containing best/results/test_predictions.csv")
    args = ap.parse_args()

    dir_label = os.path.basename(os.path.abspath(args.output_dir))

    print(f"Loading test-set predictions: {args.output_dir}")
    df = load_predictions(args.output_dir)

    print("Generating histogram ...")
    title = f"Brain-Age Gap — Differences on the Test Set — {dir_label}"
    out_path = os.path.join(args.output_dir, f"{dir_label}_hist_uncorVsCor_test.png")
    plot_histogram(df, out_path, title=title)

if __name__ == "__main__":
    main()