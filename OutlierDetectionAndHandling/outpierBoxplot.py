import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('some path/file ...')

attributes = ['aseg_lhCortexVol_Vol', 'aseg_rhCortexVol_Vol', 'aseg_lhCerebralWhiteMatterVol_Vol', 'aseg_rhCerebralWhiteMatterVol_Vol', 'aseg_eTIV_Vol', 'aseg_BrainSegVol-to-eTIV_Vol']

for attr in attributes:
    fig, ax = plt.subplots(figsize=(6, 18))

    data = pd.to_numeric(df[attr], errors='coerce').dropna()
    subjects = df.loc[data.index, 'Subject']
    ages = df.loc[data.index, 'Age']

    bp = ax.boxplot(data, patch_artist=True,                                                # Draw boxplot
                    boxprops=dict(facecolor='lightsteelblue', color='steelblue'),
                    medianprops=dict(color='red', linewidth=2),
                    whiskerprops=dict(color='steelblue'),
                    capprops=dict(color='steelblue'),
                    flierprops=dict(marker='x', color='black', markersize=6))

    Q1, Q3 = np.percentile(data, [25, 75])                                                  # IQR-based outlier detection
    IQR = Q3 - Q1
    lower_fence = Q1 - 1.5 * IQR
    upper_fence = Q3 + 1.5 * IQR

    is_outlier = (data < lower_fence) | (data > upper_fence)
    outlier_values = data[is_outlier].values
    outlier_subjects = subjects[is_outlier].values
    outlier_ages = ages[is_outlier].values

    upper_outliers = [(subj, age, val) for subj, age, val in
                      zip(outlier_subjects, outlier_ages, outlier_values)
                      if val > upper_fence]
    lower_outliers = [(subj, age, val) for subj, age, val in
                      zip(outlier_subjects, outlier_ages, outlier_values)
                      if val < lower_fence]

    print(f"Attribute: {attr}")
    print(f"Q1 = {Q1:.3f}  |  Q3 = {Q3:.3f}  |  IQR = {IQR:.3f}")
    print(f"Fences: [{lower_fence:.3f}, {upper_fence:.3f}]")

    print(f"UPPER outliers (> Q3+1.5*IQR): {len(upper_outliers)}")
    for subj, age, val in sorted(upper_outliers, key=lambda x: x[2], reverse=True):
        print(f"{subj}  (age: {age:.0f}y)  value: {val:.3f}")

    print(f"LOWER outliers (< Q1-1.5*IQR): {len(lower_outliers)}")
    for subj, age, val in sorted(lower_outliers, key=lambda x: x[2]):
        print(f"{subj}  (age: {age:.0f}y)  value: {val:.3f}")

    sort_idx = np.argsort(outlier_values)                                                       # Sort outliers by value so jitter is applied consistently
    outlier_values = outlier_values[sort_idx]
    outlier_subjects = outlier_subjects[sort_idx]
    outlier_ages = outlier_ages[sort_idx]

    min_gap = (data.max() - data.min()) * 0.025                                             # Vertical jitter: if two labels are too close, nudge the second one up
    y_placed = []

    for val, subj, age in zip(outlier_values, outlier_subjects, outlier_ages):
        y_text = val    
        for prev_y in y_placed:                                                             #push label up until it no longer overlaps a previously placed one
            if abs(y_text - prev_y) < min_gap:
                y_text = prev_y + min_gap
        y_placed.append(y_text)

        label = f"{subj} ({age:.0f}y)"
        ax.annotate(label, xy=(1, val),
                    xytext=(1, y_text),
                    xycoords=('data', 'data'),
                    textcoords=('data', 'data'),
                    fontsize=7,
                    color='black',
                    va='center',
                    arrowprops=dict(arrowstyle='-', color='gray', lw=0.5) if abs(y_text - val) > min_gap * 0.5 else None)
        # Shift text to the right of the box
        ax.annotate(label, xy=(1, val), xytext=(8, (y_text - val)), textcoords=('offset points' if False else 'offset points',), fontsize=7, color='black', va='center')

    plt.close()

    fig, ax = plt.subplots(figsize=(6, 18))
    ax.boxplot(data, patch_artist=True,
               boxprops=dict(facecolor='lightsteelblue', color='steelblue'),
               medianprops=dict(color='red', linewidth=2),
               whiskerprops=dict(color='steelblue'),
               capprops=dict(color='steelblue'),
               flierprops=dict(marker='x', color='black', markersize=6))

    y_pad = (data.max() - data.min()) * 0.05
    ax.set_ylim(data.min() - y_pad, data.max() + y_pad)
    ax.autoscale(enable=False)

    y_placed = []
    for val, subj, age in zip(outlier_values, outlier_subjects, outlier_ages):
        y_text = val
        for prev_y in y_placed:
            if abs(y_text - prev_y) < min_gap:
                y_text = prev_y + min_gap
        y_placed.append(y_text)

        label = f"{subj} ({age:.0f}y)"
        if abs(y_text - val) > min_gap * 0.3:                                               # Draw a thin connector if the label was nudged
            ax.annotate("", xy=(1.05, val), xytext=(1.08, y_text), xycoords='data', textcoords='data',  arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
        ax.text(1.09, y_text, label, fontsize=7, va='center', color='black', clip_on=False)

    ax.set_xlim(0.5, 2.5)                                                                   # extra room on the right for labels
    ax.set_title(f"{attr} - ALL", fontsize=13)
    ax.set_ylabel(attr)
    ax.set_xticks([])
    ax.grid(axis='y', alpha=0.3)
    plt.savefig(f"./BoxPlot/{attr}.png")
    plt.show()