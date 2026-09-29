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

_BLUE = "#2a78d6"                                                                                   # categorical slot 1  - corrected Brain-PAD
_GREY_DOT = "#c3c2b7"                                                                               # secondary/muted ink  - uncorrected Brain-PAD
_RED_DASH = "#e34948"                                                                               # categorical slot 8   - percentile lines
_SURFACE = "#fcfcfb"                                                                                # chart surface
_PRIMARY = "#0b0b0b"                                                                                # primary ink
_MUTED = "#898781"                                                                                  # muted / axis chrome
_GRIDLINE = "#e1e0d9"                                                                               # hairline grid

def _load_pred_csv(path: str, save_back: bool = False) -> pd.DataFrame:
    if not os.path.exists(path):
        sys.exit(f"Error - file not found:\n{path}")
    df = pd.read_csv(path)
    required = {"Age_Actual", "Age_Predicted", "Age_Corrected"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"Error - missing columns in {path}: {missing}")
    df["uncorPAD"] = df["Age_Predicted"] - df["Age_Actual"]
    df["corPAD"] = df["Age_Corrected"] - df["Age_Actual"]
    if save_back:
        df.to_csv(path, index=False)
        print(f"uncorPAD / corPAD computed -> saved to {path}")
    return df


def load_predictions(output_dir: str) -> pd.DataFrame:
    path = os.path.join(output_dir, "best", "some path/file ...")
    return _load_pred_csv(path, save_back=True)


def load_test_predictions(output_dir: str) -> pd.DataFrame:
    path = os.path.join(output_dir, "best", "some path/file ...")
    return _load_pred_csv(path, save_back=False)

def _style_ax(ax: plt.Axes) -> None:
    ax.set_facecolor(_SURFACE)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(_MUTED)
        ax.spines[sp].set_linewidth(0.8)
    ax.tick_params(axis="both", labelsize=9, colors=_PRIMARY)
    ax.yaxis.grid(True, color=_GRIDLINE, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)

def plot_A(df: pd.DataFrame, output_dir: str, title: str, rmse: float, mae: float, inset_label: str, filename: str) -> None:
    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(11, 4.8))
    fig.patch.set_facecolor(_SURFACE)

    ax.scatter(x, df["uncorPAD"], color=_MUTED, alpha=0.55, s=18, linewidths=0, zorder=2)           # Grey (uncorrected) first so blue dots render on top
    ax.scatter(x, df["corPAD"], color=_BLUE, alpha=0.70, s=18, linewidths=0.4, edgecolors="white", zorder=3)    # Blue (corrected) with a hairline white ring so overlapping dots separate

    for p in [10, 25, 50, 75, 90]:                                                                  #Percentile dashed lines
        val = np.percentile(df["corPAD"], p)
        ax.axhline(val, color=_RED_DASH, linestyle="--", linewidth=0.9, alpha=0.80, zorder=1)

    ax.axhline(0, color=_MUTED, linewidth=0.5, zorder=1)                                            #zero reference
    ax.set_xlabel("Participants", fontsize=11, color=_PRIMARY, labelpad=5)
    ax.set_ylabel("Brain-PAD", fontsize=11, color=_PRIMARY, labelpad=5)
    pad = len(df) * 0.012                                                                           # Slight horizontal padding so leftmost/rightmost dots are not clipped
    ax.set_xlim(-pad, len(df) - 1 + pad)
    ax.set_xticks([])                                                                               # no numeric x-scale

    _style_ax(ax)
    ax.set_title(title)

    inset_txt = f"       RMSE       MAE\n{inset_label}   {rmse:.4f}   {mae:.4f}"                    #RMSE / MAE table inset
    ax.text(
        0.985, 0.04, inset_txt,
        transform=ax.transAxes, fontsize=8.5, color=_PRIMARY,
        va="bottom", ha="right", family="monospace",
        bbox=dict(
            boxstyle="round,pad=0.45",
            facecolor=_GRIDLINE, edgecolor=_MUTED,
            linewidth=0.7, alpha=0.95,
        ),
    )

    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=_GREY_DOT, markersize=7, label="Uncorrected Brain-PAD"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=_BLUE, markersize=7, label="Corrected Brain-PAD"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=9, framealpha=0.90, edgecolor=_MUTED, borderpad=0.6)

    plt.tight_layout(pad=1.0)
    out = os.path.join(output_dir, filename)
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Plot A saved as {out}")

def plot_C(df: pd.DataFrame, output_dir: str, title: str, filename: str) -> None:
    age = df["Age_Actual"].values
    pred_uncor = df["Age_Predicted"].values
    pred_cor = df["Age_Corrected"].values

    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_facecolor(_SURFACE)

    ax.scatter(age, pred_uncor, color=_GREY_DOT, alpha=0.55, s=18, linewidths=0, zorder=2)              # Scatter: (Age_Actual, Age_Predicted) - uncorrected, same grey style as uncorPAD in Plot A
    ax.scatter(age, pred_cor, color=_BLUE, alpha=0.60, s=18, linewidths=0.4, edgecolors="white", zorder=3)  # Scatter: (Age_Actual, Age_Corrected) - same blue style as corPAD in Plot A
    x_line = np.linspace(age.min(), age.max(), 300)                                                     # Regression lines drawn over the full observed age range

    m_unc, b_unc = np.polyfit(age, pred_uncor, 1)
    ax.plot(x_line, m_unc * x_line + b_unc, color=_PRIMARY, linestyle="--", linewidth=1.5, zorder=4)
    m_cor, b_cor = np.polyfit(age, pred_cor, 1)
    ax.plot(x_line, m_cor * x_line + b_cor, color=_RED_DASH, linestyle="--", linewidth=1.5, zorder=4)

    ax.set_xlabel("Chronological age (years)", fontsize=11, color=_PRIMARY, labelpad=5)
    ax.set_ylabel("Predicted age (years)", fontsize=11, color=_PRIMARY, labelpad=5)

    _style_ax(ax)
    ax.set_title(title)
    ax.xaxis.grid(True, color=_GRIDLINE, linewidth=0.35, zorder=0)
    ax.yaxis.grid(True, color=_GRIDLINE, linewidth=0.35, zorder=0)

    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=_GREY_DOT, markersize=7, label="Uncorrected predicted age"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=_BLUE, markersize=7, label="Corrected predicted age"),
        Line2D([0], [0], color=_PRIMARY, linestyle="--", linewidth=1.5, label="Uncorrected (linear fit)"),
        Line2D([0], [0], color=_RED_DASH, linestyle="--", linewidth=1.5, label="Corrected (linear fit)"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=9, framealpha=0.90, edgecolor=_MUTED, borderpad=0.6)

    rmse_uncor = float(((pred_uncor - age)**2).mean()**(1/2))                                           #RMSE / MAE table inset
    mae_uncor = float(np.abs( pred_uncor - age).mean())
    rmse_cor = float(((pred_cor - age)**2).mean()**(1/2))
    mae_cor = float(np.abs(pred_cor - age).mean())

    inset_txt = (f"{'':11}{'RMSE':>7}{'MAE':>8}\n{'uncor. Age':<11}{rmse_uncor:>7.4f}{mae_uncor:>8.4f}\n{'cor. Age':<11}{rmse_cor:>7.4f}{mae_cor:>8.4f}"
    )
    ax.text(
        0.985, 0.04, inset_txt,
        transform=ax.transAxes, fontsize=8.5, color=_PRIMARY,
        va="bottom", ha="right", family="monospace",
        bbox=dict(
            boxstyle="round,pad=0.45",
            facecolor=_GRIDLINE, edgecolor=_MUTED,
            linewidth=0.7, alpha=0.95,
        ),
    )

    plt.tight_layout(pad=1.0)
    out = os.path.join(output_dir, filename)
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Plot C saving as {out}")

def main() -> None:
    ap = argparse.ArgumentParser(description="Brain-PAD plots")
    ap.add_argument("--output_dir", required=True, help=("Model output folder that contains best/results/cv_final_predictions.csv"))
    args = ap.parse_args()
    out = args.output_dir
    dir_label = os.path.basename(os.path.abspath(out))

    print("Loading OOF predictions ...")
    df_oof = load_predictions(out)                                                                          #OOF cross-validation predictions

    summary_path = os.path.join(out, "best", "results", "cv_performance_summary.csv")
    cv_df = pd.read_csv(summary_path)
    rmse_oof = float(cv_df["RMSE"].mean())
    mae_oof = float(cv_df["MAE"].mean())

    print("Generating OOF plots ...")
    plot_A(df_oof, out, title=f"Brain-PAD per Participant - OOF Cross-Validation ({dir_label})", rmse=rmse_oof, mae=mae_oof, inset_label="Mean uncor. OOF performance", filename=f"{dir_label}_plot_brainPAD_oof.png")
    plot_C(df_oof, out, title=f"Chronological vs Predicted Age - OOF Cross-Validation ({dir_label})", filename=f"{dir_label}_plot_ageScatter_oof.png")

    print("Loading test-set predictions …")
    df_test = load_test_predictions(out)                                                                    #Held-out test-set predictions
    age_test = df_test["Age_Actual"].values
    rmse_test = float(np.sqrt(((df_test["Age_Predicted"].values - age_test) ** 2).mean()))
    mae_test = float(np.abs( df_test["Age_Predicted"].values - age_test).mean())

    print("Generating test-set plots ...")
    plot_A(df_test, out, title=f"Brain-PAD per Participant - Test Set ({dir_label})", rmse=rmse_test, mae=mae_test, inset_label="Uncor. test performance    ", filename=f"{dir_label}_plot_brainPAD_test.png")
    plot_C(df_test, out, title=f"Chronological vs Predicted Age - Test Set ({dir_label})", filename=f"{dir_label}_plot_ageScatter_test.png")

if __name__ == "__main__":
    main()