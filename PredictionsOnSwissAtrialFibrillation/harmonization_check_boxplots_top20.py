# coding: utf-8
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

FEATURES = [
    "aseg_VentricleChoroidVol_Vol",
    "aseg_Right-Pallidum_normStdDev",
    "lh_a2009s_G_front_sup_ThickAvg",
    "aseg_SubCortGrayVol_Vol",
    "aseg_3rd-Ventricle_normStdDev",
    "STD(solidity)_1",
    "aseg_Left-Pallidum_normStdDev",
    "aseg_Left-Lateral-Ventricle_normStdDev",
    "lh_a2009s_G_front_sup_GrayVol",
    "rh_a2009s_S_central_SurfArea",
    "aseg_Brain-Stem_normStdDev",
    "aseg_rhSurfaceHoles_Vol",
    "rh_a2009s_Lat_Fis-post_ThickAvg",
    "lh_a2009s_G_cingul-Post-dorsal_ThickAvg",
    "lh_a2009s_Pole_occipital_ThickAvg",
    "lh_a2009s_S_circular_insula_inf_ThickStd",
    "rh_a2009s_S_circular_insula_sup_ThickAvg",
    "aseg_Right-Pallidum_normMean",
    "lh_a2009s_G_front_inf-Triangul_ThickStd",
    "aseg_Right-Thalamus-Proper_normMean",
]

OUT_PATH = "harmonized/harmonization_check_top20_collapsed.png"
AGE_LO, AGE_HI = 60, 80                                                                         # decades 60s + 70s, [60, 80)

COHORT_FILES = [
    ("HC", "some path/file ..."),
    ("SAF-HC", "some path/file ..."),
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

    ncols = 5
    nrows = -(-len(FEATURES) // ncols)                                                          # ceil division
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 5.2 * nrows))
    axes = axes.flatten()
    for ax in axes[len(FEATURES):]:                                                             # hide any unused grid cells
        ax.axis("off")

    for ax, feat in zip(axes, FEATURES):
        print(f"=== {feat} ===")
        box_data, box_colors, labels, means, ns = [], [], [], [], []
        for label, df in cohorts:
            vals = df[feat].dropna().values
            box_data.append(vals)
            box_colors.append(COLORS[label])
            labels.append(label)
            means.append(vals.mean())
            ns.append(len(vals))
            print(f"{label:12s} mean={vals.mean():10.4f}  std={vals.std():8.4f}  n={len(vals)}")

        hc_mean = means[0]
        for label, m in zip(labels[1:], means[1:]):
            if hc_mean != 0:
                print(f"{label} - HC delta: {m - hc_mean:+.4f}  ({(m - hc_mean) / hc_mean * 100:+.1f}% of HC mean)")
        print()

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

    fig.suptitle(f"Harmonized values, ages {AGE_LO}-{AGE_HI - 1} - top 20 features by permutation importance, all sites pooled per cohort", fontsize=13)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT_PATH, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saving as {OUT_PATH}")

if __name__ == "__main__":
    main()