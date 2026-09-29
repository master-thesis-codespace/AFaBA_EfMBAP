#!/usr/bin/env python3
import argparse
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from scipy.stats import gaussian_kde
import numpy as np
import pandas as pd

_BLUE_DOT = "#2a78d6"                                                                           # healthy controls - dots
_BLUE_LINE = "#0d366b"                                                                          # healthy controls - regression (dark blue)
_RED_DOT = "#e34948"                                                                            # patients - dots
_RED_LINE = "#7c1111"                                                                           # patients - regression (dark red)
_SURFACE = "#fcfcfb"
_PRIMARY = "#0b0b0b"
_MUTED = "#898781"
_GRIDLINE = "#e1e0d9"

def load_corrected(folder: str) -> pd.DataFrame:
    path = os.path.join(folder, "best", "results", "cv_final_predictions.csv")
    if not os.path.exists(path):
        sys.exit(f"Error, file not found:\n{path}")
    df = pd.read_csv(path)
    missing = {"Age_Actual", "Age_Predicted", "Age_Corrected"} - set(df.columns)
    if missing:
        sys.exit(f"Error, missing columns in {path}: {missing}")
    added = False
    if "uncorPAD" not in df.columns:
        df["uncorPAD"] = df["Age_Predicted"] - df["Age_Actual"]
        added = True
    if "corPAD" not in df.columns:
        df["corPAD"] = df["Age_Corrected"] - df["Age_Actual"]
        added = True
    if added:
        df.to_csv(path, index=False)                                                                #persist newly computed uncorPAD/corPAD back to disk
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

def plot_disease_gap(df_hc: pd.DataFrame, df_pt: pd.DataFrame, healthy_dir: str, out_path: str, title: str) -> None:
    age_hc = df_hc["Age_Actual"].values
    cor_hc = df_hc["Age_Corrected"].values
    age_pt = df_pt["Age_Actual"].values
    cor_pt = df_pt["Age_Corrected"].values

    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_facecolor(_SURFACE)

    ax.scatter(age_hc, cor_hc, color=_BLUE_DOT, alpha=0.60, s=18, linewidths=0.4, edgecolors="white", zorder=3)     # Scatter dots - healthy controls (blue), patients (red)
    ax.scatter(age_pt, cor_pt, color=_RED_DOT, alpha=0.60, s=18, linewidths=0.4, edgecolors="white", zorder=3)

    x_hc = np.linspace(age_hc.min(), age_hc.max(), 300)                                                             # Regression lines - each spans its own group's age range
    m_hc, b_hc = np.polyfit(age_hc, cor_hc, 1)
    ax.plot(x_hc, m_hc * x_hc + b_hc, color=_BLUE_LINE, linestyle="--", linewidth=1.5, zorder=4)

    x_pt = np.linspace(age_pt.min(), age_pt.max(), 300)
    m_pt, b_pt = np.polyfit(age_pt, cor_pt, 1)
    ax.plot(x_pt, m_pt * x_pt + b_pt, color=_RED_LINE, linestyle="--", linewidth=1.5, zorder=4)

    ax.set_xlabel("Chronological age (years)", fontsize=11, color=_PRIMARY, labelpad=5)
    ax.set_ylabel("Predicted age (years)", fontsize=11, color=_PRIMARY, labelpad=5)
    ax.set_title(title, fontsize=12, color=_PRIMARY, pad=10)

    _style_ax(ax)

    # RMSE / MAE inset from healthy-controls cv_performance_summary.csv
    summary_path = os.path.join(healthy_dir, "best", "results", "cv_performance_summary.csv")
    cv_df = pd.read_csv(summary_path)
    rmse = float(cv_df["RMSE"].mean())
    mae = float(cv_df["MAE"].mean())
    inset_txt = f"       RMSE    MAE\nMean   {rmse:.2f}   {mae:.2f}"
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
        Line2D([0], [0], marker="o", color="w", markerfacecolor=_BLUE_DOT, markersize=7, label="Healthy Controls"),
        Line2D([0], [0], color=_BLUE_LINE, linestyle="--", linewidth=1.5, label="HC linear fit"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=_RED_DOT, markersize=7, label="SAF HC"),
        Line2D([0], [0], color=_RED_LINE, linestyle="--", linewidth=1.5, label="SAF linear fit"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=9, framealpha=0.90, edgecolor=_MUTED, borderpad=0.6)

    plt.tight_layout(pad=1.0)
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved ... {out_path}")

def plot_histogram(df_hc: pd.DataFrame, df_pt: pd.DataFrame, out_path: str, title: str) -> None:
    cor_hc = df_hc["corPAD"].values
    cor_pt = df_pt["corPAD"].values

    all_pad = np.concatenate([cor_hc, cor_pt])                                                      # Build odd-integer bin edges covering both distributions
    lo = int(np.floor(all_pad.min()))
    #if lo % 2 == 0:
    #    lo -= 1
    hi = int(np.ceil(all_pad.max()))
    #if hi % 2 == 0:
    #    hi += 1
    bins = np.arange(lo - 0.5, hi + 1.5, 1.0)

    w_hc = np.ones(len(cor_hc)) / len(cor_hc)
    w_pt = np.ones(len(cor_pt)) / len(cor_pt)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.patch.set_facecolor(_SURFACE)

    ax.hist(cor_hc, bins=bins, weights=w_hc, color=_BLUE_DOT, alpha=0.55, edgecolor="white", linewidth=0.5, label="Healthy Controls", zorder=2)
    ax.hist(cor_pt, bins=bins, weights=w_pt, color=_RED_DOT, alpha=0.55, edgecolor="white", linewidth=0.5, label="SAF HC", zorder=3)

    ax.axvline(np.mean(cor_hc), color=_BLUE_LINE, linestyle="--", linewidth=1.5, zorder=4)
    ax.axvline(np.mean(cor_pt), color=_RED_LINE, linestyle="--", linewidth=1.5, zorder=4)

    ax.set_xlabel("Predicted age difference (years)", fontsize=11, color=_PRIMARY, labelpad=5)
    ax.set_ylabel("Frequency (normalized)", fontsize=11, color=_PRIMARY, labelpad=5)
    ax.set_title(title, fontsize=12, color=_PRIMARY, pad=10)

    _style_ax(ax)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=_BLUE_DOT, alpha=0.55, label="Healthy Controls"),
        Line2D([0], [0], color=_BLUE_LINE, linestyle="--", linewidth=1.5, label=f"Healthy Control mean ({np.mean(cor_hc):.1f})"),
        plt.Rectangle((0, 0), 1, 1, color=_RED_DOT, alpha=0.55, label="SAF HC"),
        Line2D([0], [0], color=_RED_LINE, linestyle="--", linewidth=1.5, label=f"SAF mean ({np.mean(cor_pt):.1f})"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=9, framealpha=0.90, edgecolor=_MUTED, borderpad=0.6)

    plt.tight_layout(pad=1.0)
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved ... {out_path}")

def plot_violin(df_hc: pd.DataFrame, df_pt: pd.DataFrame, healthy_name: str, patient_name: str, out_path: str, title: str) -> None:
    groups = [
        (0, df_hc["Age_Actual"].values, df_hc["Age_Corrected"].values, healthy_name),
        (1, df_pt["Age_Actual"].values, df_pt["Age_Corrected"].values, patient_name),
    ]

    all_vals = np.concatenate([v for _, a, c, _ in groups for v in (a, c)])
    buf = (all_vals.max() - all_vals.min()) * 0.05
    y_grid = np.linspace(all_vals.min() - buf, all_vals.max() + buf, 400)

    vwidth = 0.38                                                                                   # half-width of each violin in x-axis units

    fig, ax = plt.subplots(figsize=(6, 7))
    fig.patch.set_facecolor(_SURFACE)

    for x_pos, actual, corrected, _ in groups:
        for data, side, fill_col, outline_col in [
            (actual, "left", _BLUE_DOT, _BLUE_LINE),
            (corrected, "right", _RED_DOT, _RED_LINE),
        ]:
            kde = gaussian_kde(data, bw_method="scott")
            raw = kde(y_grid)
            density = raw / raw.max() * vwidth

            med = np.median(data)
            d_med = float(kde([med])[0]) / raw.max() * vwidth

            if side == "left":
                ax.fill_betweenx(y_grid, x_pos, x_pos - density, color=fill_col, alpha=0.6, zorder=2)
                ax.plot(x_pos - density, y_grid, color=outline_col, linewidth=0.8, zorder=3)
                ax.plot([x_pos - d_med * 0.5, x_pos], [med, med], color=outline_col, linewidth=2.5, solid_capstyle="round", zorder=5)
            else:
                ax.fill_betweenx(y_grid, x_pos, x_pos + density, color=fill_col, alpha=0.6, zorder=2)
                ax.plot(x_pos + density, y_grid, color=outline_col, linewidth=0.8, zorder=3)
                ax.plot([x_pos, x_pos + d_med * 0.5], [med, med], color=outline_col, linewidth=2.5, solid_capstyle="round", zorder=5)
        ax.axvline(x_pos, color=_MUTED, linewidth=0.6, zorder=1)                                     # Centre dividing line between the two halves

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Healthy Cohort", "SAF HC"], fontsize=9)
    ax.set_xlabel("Phenotype group", fontsize=11, color=_PRIMARY, labelpad=5)
    ax.set_ylabel("Age (years)", fontsize=11, color=_PRIMARY, labelpad=5)
    ax.set_xlim(-0.6, 1.6)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title(title, fontsize=12, color=_PRIMARY, pad=10)

    _style_ax(ax)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=_BLUE_DOT, alpha=0.6, label="Chronological age (Age_Actual)"),
        plt.Rectangle((0, 0), 1, 1, color=_RED_DOT, alpha=0.6, label="Corrected predicted age (Age_Corrected)"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=9, framealpha=0.90, edgecolor=_MUTED, borderpad=0.6)

    plt.tight_layout(pad=1.0)
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved as {out_path}")

def main() -> None:
    ap = argparse.ArgumentParser(description="Plot corrected predicted age vs chronological age for two groups")
    ap.add_argument("--healthy", required=True, help="Output folder for healthy controls")
    ap.add_argument("--patient", required=True, help="Output folder for patients")
    args = ap.parse_args()

    healthy_name = os.path.basename(os.path.normpath(args.healthy))
    patient_name = os.path.basename(os.path.normpath(args.patient))

    abs_hc = os.path.abspath(args.healthy)                                                          # Common parent directory (handles sibling folders like XGBoost/outputA + XGBoost/outputB)
    abs_pt = os.path.abspath(args.patient)
    parent = os.path.commonpath([abs_hc, abs_pt])
    if parent in (abs_hc, abs_pt):                                                                  # Step up if commonpath resolves to one of the folders itself
        parent = os.path.dirname(parent)

    stem = f"{healthy_name}-{patient_name}"

    print(f"Loading healthy controls: {args.healthy}")
    df_hc = load_corrected(args.healthy)
    print(f"Loading patients: {args.patient}")
    df_pt = load_corrected(args.patient)

    print("Generating plots ...")
    title = f"Brain Age Gap - {healthy_name}"
    plot_disease_gap(df_hc, df_pt, args.healthy, os.path.join(parent, f"diseasePAD_{stem}.png"), title=title)
    plot_histogram(df_hc, df_pt, os.path.join(parent, f"diseasePAD_hist_{stem}.png"), title=title)
    plot_violin(df_hc, df_pt, healthy_name, patient_name, os.path.join(parent, f"diseasePAD_violin_{stem}.png"), title=title)

if __name__ == "__main__":
    main()