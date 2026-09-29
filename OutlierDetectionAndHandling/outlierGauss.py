import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('some path/file ...')

attributes = ['aseg_lhCortexVol_Vol', 'aseg_rhCortexVol_Vol', 'aseg_lhCerebralWhiteMatterVol_Vol', 'aseg_rhCerebralWhiteMatterVol_Vol', 'aseg_eTIV_Vol', 'aseg_BrainSegVol-to-eTIV_Vol']

band_styles = [                                                                         # Std-band colors (increasingly intense)
    (3, '#d4edda', '3SD'),
    (2, '#fff3cd', '2SD'),
    (1, '#f8d7da', '1SD'),
]

for attr in attributes:
    data = pd.to_numeric(df[attr], errors='coerce').dropna()
    subjects = df.loc[data.index, 'Subject']
    ages = df.loc[data.index, 'Age']

    mu = data.mean()
    sigma = data.std()

    is_outlier = (data < mu - 3*sigma) | (data > mu + 3*sigma)
    outlier_values = data[is_outlier].values
    outlier_subjects = subjects[is_outlier].values
    outlier_ages = ages[is_outlier].values
    
    upper_outliers = [(subj, age, val) for subj, age, val in                            #Console output: grouped by attribute, then upper/lower
                      zip(outlier_subjects, outlier_ages, outlier_values)
                      if val > mu + 3*sigma]
    lower_outliers = [(subj, age, val) for subj, age, val in 
                      zip(outlier_subjects, outlier_ages, outlier_values)
                      if val < mu - 3*sigma]

    print(f"Attribute: {attr}")
    print(f"mu = {mu:.3f}  |  sigma = {sigma:.3f}  |  3sigma-bounds: [{mu-3*sigma:.3f}, {mu+3*sigma:.3f}]")

    print(f"UPPER outliers (> mu+3sigma): {len(upper_outliers)}")
    for subj, age, val in sorted(upper_outliers, key=lambda x: x[2], reverse=True):
        print(f"{subj}  (age: {age:.0f}y)  value: {val:.3f}")

    print(f"LOWER outliers (< mu-3sigma): {len(lower_outliers)}")
    for subj, age, val in sorted(lower_outliers, key=lambda x: x[2]):
        print(f"{subj}  (age: {age:.0f}y)  value: {val:.3f}")

    fig, ax = plt.subplots(figsize=(6, 18))

    ax.boxplot(data, patch_artist=True,                                                  # Draw boxplot (no built-in fliers, we handle them manually)
               boxprops=dict(facecolor='lightsteelblue', color='steelblue', alpha=0.6),
               medianprops=dict(color='red', linewidth=2),
               whiskerprops=dict(color='steelblue'),
               capprops=dict(color='steelblue'),
               flierprops=dict(marker='', markersize=0),                                # hide default fliers
               showfliers=False)

    for n_std, face_color, lbl in band_styles:                                          # shaded std bands (3SD outermost -> 1SD innermost)
        ax.axhspan(mu - n_std*sigma, mu + n_std*sigma, color=face_color, alpha=0.45, zorder=0, label=f'±{lbl}')

    ax.axhline(mu, color='navy', linewidth=1.2, linestyle='-',  label=f'Mean ({mu:.2f})')   #Mean line

    line_styles = {1: ('--', 0.9), 2: ('-.', 0.9), 3: (':', 1.1)}                       #Std boundary lines
    for n_std, (ls, lw) in line_styles.items():
        ax.axhline(mu + n_std*sigma, color='dimgray', linestyle=ls, linewidth=lw, label=f'+{n_std}SD ({mu + n_std*sigma:.2f})')
        ax.axhline(mu - n_std*sigma, color='dimgray', linestyle=ls, linewidth=lw, label=f'-{n_std}SD ({mu - n_std*sigma:.2f})')

    normal_mask = ~is_outlier                                                           #Scatter all non-outlier points (jittered x for visibility)
    normal_values = data[normal_mask].values
    np.random.seed(42)
    jitter_normal = np.random.uniform(-0.08, 0.08, size=len(normal_values))
    jitter_outlier = np.random.uniform(-0.08, 0.08, size=len(outlier_values))

    ax.scatter(1 + jitter_normal, normal_values, color='steelblue', alpha=0.4, s=18, zorder=3)
    ax.scatter(1 + jitter_outlier, outlier_values, color='black', marker='x', s=60, linewidths=1.8, zorder=4, label='Outlier (>3SD)')

    y_pad = (data.max() - data.min()) * 0.05
    ax.set_ylim(data.min() - y_pad, data.max() + y_pad)
    ax.autoscale(enable=False)

    sort_idx = np.argsort(outlier_values)                                               # Annotate outliers with vertical jitter to avoid overlap
    outlier_values = outlier_values[sort_idx]
    outlier_subjects = outlier_subjects[sort_idx]
    outlier_ages = outlier_ages[sort_idx]

    y_range = data.max() - data.min()
    min_gap = y_range * 0.04                                                            # 4% of y-range between labels

    y_placed = []
    for val, subj, age in zip(outlier_values, outlier_subjects, outlier_ages):
        y_text = val
        for prev_y in sorted(y_placed):                                                 ## Nudge upward until no overlap with already-placed labels
            if abs(y_text - prev_y) < min_gap:
                y_text = prev_y + min_gap
        y_placed.append(y_text)

        label = f"{subj} ({age:.0f}y)"

        if abs(y_text - val) > min_gap * 0.3:                                          # Thin connector if nudged
            ax.annotate("", xy=(1.06, val), xytext=(1.09, y_text), xycoords='data', textcoords='data', arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))

        ax.text(1.10, y_text, label, fontsize=7.5, va='center', color='black', fontweight='bold', clip_on=False)

    ax.set_xlim(0.5, 2.6)
    ax.set_title(f"Gaussian Outlier Detection: {attr}", fontsize=13)
    ax.set_ylabel(attr)
    ax.set_xticks([])
    ax.grid(axis='y', alpha=0.2)

    handles, labels = ax.get_legend_handles_labels()                                    # Legend, deduplicate and place outside plot
    ax.legend(handles, labels, fontsize=7.5, loc='upper left', bbox_to_anchor=(0.01, 0.99), framealpha=0.8)

    plt.savefig(f"./BoxPlot/{attr}_GAUSS.png")
    plt.show()