import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

outlier_counter = Counter()

SCT = pd.read_excel('csa_perlevel_final_age_gender_Outliererased.ods', engine='odf')
#SCT = pd.read_csv('some path/file ...')

datasets = {
    #'AHDC': 'AHDC',
    #'Calgary': 'Calgary_Campinas',
    #'DVPFC': 'DVPFC',
    #'IXI': 'IXI',
    #'MRART': 'MR-ART',
    #'NIMH': 'NIMH',
    #'OASIS_4': 'OASIS_4',
    #'SALD': 'SALD',
    #'Paingen': 'paingen',
    #'UCSF_NODDI': '01_ucsf-noddi',
    #'SWU_SLIM': '03_swu-slim',
    #'MMRR': '08_mmrr',
    #'ABRIM': '16_ABRIM',
    #'QTIM': '99_QTIM',
    #'HBN': '12_HBN',
    #'SwedishStudy': 'SwedishStudy',
    #'OASIS_3': 'OASIS_3' ,
    'ALL': None  # entire corpus
}

import numpy as np

def plot_gender(data, dataset_label, gender_val, gender_label, color):
    global outlier_counter
    gender_data = data.loc[data['Gender'] == gender_val]
    measurements = ["MEAN(area)", "MEAN(angle_AP)", "MEAN(angle_RL)", "MEAN(diameter_AP)", "MEAN(diameter_RL)", "MEAN(eccentricity)", "MEAN(orientation)", "MEAN(solidity)", "SUM(length)"]

    for attribute in measurements:
        fig, axes = plt.subplots(1, 4, figsize=(26, 6), sharey=True)
        fig.suptitle(f"{attribute} - {dataset_label} | {gender_label} [Cleaned Outliers]")
        #print(f"-> Attribute selected: {attribute}")

        for i, ax in enumerate(axes, start=1):
            level_data = gender_data.loc[gender_data['VertLevel'] == i].dropna(subset=['Age', attribute])
            ages = level_data['Age'].values
            y_value = level_data[attribute].values
            subjects = level_data['Subject'].values

            if len(ages) < 4:                                                               # too few points for regression
                ax.scatter(ages, y_value, color=color)
                ax.set_title(f"VertLevel {i}")
                ax.set_xlabel("Age [Years]")
                ax.set_ylabel(attribute)
                continue

            coeffs = np.polyfit(ages, y_value, deg=2)                                       #Linear or quadratic regression
            poly_fn = np.poly1d(coeffs)
            age_line = np.linspace(ages.min(), ages.max(), 200)
            fitted = poly_fn(ages)

            residuals = y_value - fitted                                                    #Residual-based IQR outlier detection
            Q1, Q3 = np.percentile(residuals, [25, 75])
            IQR = Q3 - Q1
            lower_fence = Q1 - 1.5 * IQR
            upper_fence = Q3 + 1.5 * IQR

            is_outlier = (residuals < lower_fence) | (residuals > upper_fence)
            normal_mask = ~is_outlier

            outlier_counter.update(subjects[is_outlier])                                    #Track outlier subjects

            ax.plot(age_line, poly_fn(age_line), color='black', linewidth=1.2, label='Regression')      #regression line

            ax.fill_between(age_line, poly_fn(age_line) + lower_fence, poly_fn(age_line) + upper_fence, color=color, alpha=0.15, label='Normal band')   #shaded prediction band (regression +/- fence width)

            ax.scatter(ages[normal_mask], y_value[normal_mask], color=color, label='Normal')    #Normal & outlier points
            ax.scatter(ages[is_outlier], y_value[is_outlier], color='black', marker='x', s=60, linewidths=1.5, label='Outlier')

            subjects = level_data['Subject'].values                                         #Annotation
            for j, (x, y, subj) in enumerate(zip(ages, y_value, subjects)):
                if is_outlier[j]:
                    ax.annotate(subj, (x, y), fontsize=8,
                                xytext=(3, 3),                                              # offset in points
                                textcoords='offset points',
                                #color='black' if is_outlier[j] else 'dimgray')
                                color='black')

            ax.set_title(f"VertLevel {i}")
            ax.set_xlabel("Age [Years]")
            ax.set_ylabel(attribute)
            ax.grid(alpha=0.2)

        axes[-1].legend(fontsize=8, loc='upper right')
        plt.tight_layout()
        #plt.savefig(f"./2D-Regression_plotting/{attribute}_{dataset_label}_{gender_label}_2D-Regression_annotated.png")
        plt.savefig(f"./2D-Regression_plotting/Annotated/{dataset_label}_{gender_label}_{attribute}_2D-Regression_annotated_outlierCleaned.png")
        #plt.show()

for label, dataset_key in datasets.items():
    if dataset_key is None:
        data = SCT[['Subject', 'Age', 'Gender', 'VertLevel', 'MEAN(area)', "MEAN(angle_AP)", "MEAN(angle_RL)", "MEAN(diameter_AP)", "MEAN(diameter_RL)", "MEAN(eccentricity)", "MEAN(orientation)", "MEAN(solidity)", "SUM(length)"]]
    else:
        data = SCT.loc[SCT['Dataset'] == dataset_key, ['Subject', 'Age', 'Gender', 'VertLevel', 'MEAN(area)', "MEAN(angle_AP)", "MEAN(angle_RL)", "MEAN(diameter_AP)", "MEAN(diameter_RL)", "MEAN(eccentricity)", "MEAN(orientation)", "MEAN(solidity)", "SUM(length)"]]
    #print(f"Label selectedü: {label}")
    #print(f"Dataset selectedü: {dataset_key}")
    plot_gender(data, label, gender_val=0, gender_label="Female", color="tomato")
    
    print(f"\nOutliers for {label} | Female")
    for subject, count in outlier_counter.most_common():
        if count>3:
            print(f"{subject}: {count} time(s)")
    outlier_counter.clear()
    plot_gender(data, label, gender_val=1, gender_label="Male", color="steelblue")
    print(f"\nOutliers for {label} | Male")
    for subject, count in outlier_counter.most_common():
        if count>3:
            print(f"{subject}: {count} time(s)")
    outlier_counter.clear()