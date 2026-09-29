# coding: utf-8
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

FEATURES = [
    "aseg_VentricleChoroidVol_Vol",
    "aseg_Right-Pallidum_normStdDev",
    "lh_a2009s_G_front_sup_ThickAvg",
]

AGE_LO, AGE_HI = 60, 80                                                                                 # decades 60s + 70s, [60, 80)
COHORT_FILES = [
    ("HC", "some path/file ..."),
    ("SAF-HC", "some path/file ...v"),
    ("SAF-Patient", "some path/file ..."),
]
COLORS = {"HC": "#2a78d6", "SAF-HC": "#e34948", "SAF-Patient": "#7c1111"}

def load(path: str, label: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[(df["Age"] >= AGE_LO) & (df["Age"] < AGE_HI)].copy()
    df["cohort"] = label
    return df

def main():
    cohorts = [(label, load(path, label)) for label, path in COHORT_FILES]

    print(f"Subjects aged [{AGE_LO},{AGE_HI}) per cohort:")
    for label, df in cohorts:
        print(f"{label:12s} n={len(df):4d}")
    print()

    for feat in FEATURES:
        print(f"=== {feat} ===")
        positions, box_data, box_colors, xticklabels = [], [], [], []
        block_bounds = []                                                                               # (start, end, label, grand_mean, n)
        pos = 0

        for label, df in cohorts:
            sites = sorted(df["SITE"].unique())
            start = pos
            for s in sites:
                vals = df.loc[df["SITE"] == s, feat].dropna().values
                if len(vals) == 0:
                    continue
                positions.append(pos)
                box_data.append(vals)
                box_colors.append(COLORS[label])
                xticklabels.append(f"S{s}\n(n={len(vals)})")
                pos += 1
            end = pos - 1
            grand_mean = df[feat].mean()
            grand_std = df[feat].std()
            block_bounds.append((start, end, label, grand_mean, len(df)))
            print(f"{label:12s} mean={grand_mean:10.4f}  std={grand_std:8.4f}  n={len(df)}")
            pos += 1                                                                                    # gap between cohort blocks

        hc_mean = block_bounds[0][3]
        for start, end, label, gm, n in block_bounds[1:]:
            print(f"{label} - HC delta: {gm - hc_mean:+.4f}  ({(gm - hc_mean) / hc_mean * 100:+.1f}% of HC mean)" if hc_mean != 0 else "")
        print()

        fig, ax = plt.subplots(figsize=(14, 5.5))
        bp = ax.boxplot(box_data, positions=positions, widths=0.6, patch_artist=True, showfliers=False)
        for patch, color in zip(bp["boxes"], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.55)
        for median in bp["medians"]:
            median.set_color("black")

        for start, end, label, gm, n in block_bounds:
            ax.hlines(gm, start - 0.5, end + 0.5, colors=COLORS[label], linestyles="--", linewidth=2, zorder=5)
            ax.annotate(f"{label} mean = {gm:.2f}  (n={n})", xy=((start + end) / 2, 1.10), xycoords=("data", "axes fraction"), ha="center", va="bottom", fontsize=9.5, color=COLORS[label], fontweight="bold", annotation_clip=False)

        for start, end, label, gm, n in block_bounds[:-1]:
            ax.axvline(end + 0.5, color="grey", linestyle=":", linewidth=1)

        ax.set_xticks(positions)
        ax.set_xticklabels(xticklabels, fontsize=7)
        ax.set_xlabel("Site (internal ID within each study - NOT comparable across studies)", fontsize=9)
        ax.set_ylabel(feat, fontsize=10)
        ax.set_title(f"{feat} - harmonized values, ages {AGE_LO}-{AGE_HI - 1}, by site within cohort", fontsize=11, pad=45)
        ax.grid(axis="y", linewidth=0.3, alpha=0.6)

        plt.tight_layout()
        safe_name = feat.replace("(", "").replace(")", "").replace("/", "_")
        out_path = f"harmonized/harmonization_check_{safe_name}.png"
        fig.savefig(out_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"Saving to {out_path}\n")


def main_collapsed():
    cohorts = [(label, load(path, label)) for label, path in COHORT_FILES]

    fig, axes = plt.subplots(1, len(FEATURES), figsize=(5 * len(FEATURES), 5.5))
    if len(FEATURES) == 1:
        axes = [axes]

    for ax, feat in zip(axes, FEATURES):
        box_data, box_colors, labels, means, ns = [], [], [], [], []
        for label, df in cohorts:
            vals = df[feat].dropna().values
            box_data.append(vals)
            box_colors.append(COLORS[label])
            labels.append(label)
            means.append(vals.mean())
            ns.append(len(vals))

        bp = ax.boxplot(box_data, positions=range(len(cohorts)), widths=0.55, patch_artist=True, showfliers=False)
        for patch, color in zip(bp["boxes"], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.55)
        for median in bp["medians"]:
            median.set_color("black")

        for i, (label, m, n) in enumerate(zip(labels, means, ns)):
            ax.hlines(m, i - 0.3, i + 0.3, colors=COLORS[label], linestyles="--", linewidth=2, zorder=5)
            ax.annotate(f"mean={m:.2f}\nn={n}", xy=(i, 1.02), xycoords=("data", "axes fraction"), ha="center", va="bottom", fontsize=8.5, color=COLORS[label], fontweight="bold", annotation_clip=False)
        ax.set_xticks(range(len(cohorts)))
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_title(feat, fontsize=10, pad=38)
        ax.grid(axis="y", linewidth=0.3, alpha=0.6)

    fig.suptitle(f"Harmonized values, ages {AGE_LO}-{AGE_HI - 1}", fontsize=12, y=1.12)
    plt.tight_layout()
    out_path = "harmonized/harmonization_check_collapsed.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saving as {out_path}")

if __name__ == "__main__":
    main()
    print()
    main_collapsed()