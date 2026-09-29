#!/usr/bin/env python3
import os

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from scipy import stats

BASE = os.path.dirname(os.path.abspath(__file__))
HC_PATH = os.path.join(BASE, "some files/paths ...")
PT_PATH = os.path.join(BASE, "some files/paths ...")
OUT_PATH = os.path.join(BASE, "ancova_results_top10.csv")
DIAG_OUT_PATH = os.path.join(BASE, "ancova_results_top10_diagnostics.csv")
SENS_OUT_PATH = os.path.join(BASE, "ancova_results_top10_sensitivity.csv")

# Top-10 permutation-importance features
TOP10_FEATURES = ["aseg_VentricleChoroidVol_Vol", "aseg_Right-Pallidum_normStdDev", "lh_a2009s_G_front_sup_ThickAvg", "aseg_SubCortGrayVol_Vol", "aseg_3rd-Ventricle_normStdDev", "STD(solidity)_1", "aseg_Left-Pallidum_normStdDev", "aseg_Left-Lateral-Ventricle_normStdDev", "lh_a2009s_G_front_sup_GrayVol", "rh_a2009s_S_central_SurfArea"]

ALPHA = 0.05
N_TESTS = len(TOP10_FEATURES)

def load_combined():                                                            #Load SAF-HC / SAF-Patient, tag with group (0=HC/no AF, 1=Patient/AF), concat
    hc = pd.read_csv(HC_PATH)
    pt = pd.read_csv(PT_PATH)
    hc = hc.copy()
    pt = pt.copy()
    hc["group"] = 0
    pt["group"] = 1
    df = pd.concat([hc, pt], ignore_index=True)

    required = TOP10_FEATURES + ["Age", "Sex", "SITE"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected column(s): {missing}")

    n_missing = df[required].isna().sum().sum()
    if n_missing:
        raise ValueError(f"Found {n_missing} missing value(s) across Age/Sex/SITE/top-10 features - resolve before running ANCOVA (this script does not impute).")
    return df, hc, pt

def describe_groups(hc, pt):                                                    #Print descriptive Age/Sex balance between the two cohorts, for context
    print("Cohort sizes and Age/Sex balance")
    print(f"SAF-HC (group=0, no AF): n={len(hc)} Age mean={hc['Age'].mean():.2f} (SD={hc['Age'].std():.2f}) Sex=1 fraction={hc['Sex'].mean():.3f}")
    print(f"SAF-Patient (group=1, AF): n={len(pt)} Age mean={pt['Age'].mean():.2f} (SD={pt['Age'].std():.2f}) Sex=1 fraction={pt['Sex'].mean():.3f}")

def run_vif(df):                                                                #VIF/Tolerance for Age/Sex/group - feature-independent, computed once
    X = df[["Age", "Sex", "group"]].copy()
    X = pd.concat([pd.Series(1.0, index=X.index, name="const"), X], axis=1)
    vif = {col: variance_inflation_factor(X.values, i) for i, col in enumerate(X.columns) if col != "const"}
    tolerance = {col: 1.0 / v for col, v in vif.items()}
    return vif, tolerance

def run_standardized_beta_group(df, feature):                                   #Standardized group coefficient: z-score feature/Age/Sex/group, refit, read off group's beta
    z = df[[feature, "Age", "Sex", "group"]].copy()
    z = (z - z.mean()) / z.std()                                                 # z-score every column (mean 0, SD 1) - includes the two 0/1 dummies
    model = smf.ols(f'Q("{feature}") ~ Age + Sex + group', data=z).fit()         # no C(): already numeric (0/1), now z-scored like the others
    return model.params["group"]                                                # standardized regression coefficient for group

def run_hc3(model, term):                                                       #HC3-robust se/t/p/CI for one term of a fitted OLS model
    robust = model.get_robustcov_results(cov_type="HC3")
    pos = list(model.params.index).index(term)                                  # get_robustcov_results returns plain arrays, position- not name-indexed
    ci_lo, ci_hi = robust.conf_int()[pos]
    return robust.bse[pos], robust.tvalues[pos], robust.pvalues[pos], ci_lo, ci_hi

def run_sensitivity(df, feature, beta_main, p_main, lin_model):                 #Group effect under 3 specifications: main, +SITE covariate, +Age^2 term (reuses lin_model)
    rows = [{"feature": feature, "model": "main", "beta_group": beta_main, "p_group": p_main}]

    site_model = smf.ols(f'Q("{feature}") ~ Age + C(Sex) + C(group) + C(SITE)', data=df).fit()  # adds study center as a covariate
    rows.append({"feature": feature, "model": "with_SITE", "beta_group": site_model.params["C(group)[T.1]"], "p_group": site_model.pvalues["C(group)[T.1]"]})

    rows.append({"feature": feature, "model": "with_Age2", "beta_group": lin_model.params["C(group)[T.1]"], "p_group": lin_model.pvalues["C(group)[T.1]"]})
    return rows

def run_ancova(df, feature):                                                    #Fit `feature ~ Age + C(Sex) + C(group)` and return the group-term test stats plus the 5 standard MLR/ANCOVA diagnostics
    formula = f'Q("{feature}") ~ Age + C(Sex) + C(group)'
    model = smf.ols(formula, data=df).fit()

    group_term = "C(group)[T.1]"
    beta = model.params[group_term]
    se = model.bse[group_term]
    tval = model.tvalues[group_term]
    pval = model.pvalues[group_term]
    ci_lo, ci_hi = model.conf_int().loc[group_term]                              # 95% CI for the group coefficient (physician-review addition)
    beta_std = run_standardized_beta_group(df, feature)                          # standardized effect size, comparable across features
    se_hc3, t_hc3, p_hc3, ci_lo_hc3, ci_hi_hc3 = run_hc3(model, group_term)       # HC3-robust inference, alongside the classical values above

    resid = model.resid

    #Normality - informational; with n=957/1590
    shapiro_p = stats.shapiro(resid.sample(min(len(resid), 5000), random_state=42))[1] \
        if len(resid) > 3 else np.nan

    #Homoscedasticity - one dimension only (residual variance, by group)
    levene_resid_p = stats.levene(resid[df["group"] == 0], resid[df["group"] == 1])[1]
    #Homoscedasticity - comprehensive (residual variance vs. all predictors at once).
    _, bp_p, _, _ = het_breuschpagan(resid, model.model.exog)

    #Linearity - is a quadratic Age term significant in a richer, separate model?
    lin_model = smf.ols(f'Q("{feature}") ~ Age + I(Age**2) + C(Sex) + C(group)', data=df).fit()
    age2_p = lin_model.pvalues["I(Age ** 2)"]

    #Independence - do residuals cluster by SITE (the real risk in multi-site harmonized data)
    site_tmp = df[["SITE"]].copy()
    site_tmp["resid"] = resid.values
    site_model = smf.ols("resid ~ C(SITE)", data=site_tmp).fit()
    site_p = site_model.f_pvalue

    # Homogeneity-of-regression-slopes diagnostic (ANCOVA-specific, not one of the 5)
    inter_formula = f'Q("{feature}") ~ Age * C(group) + C(Sex)'
    inter_model = smf.ols(inter_formula, data=df).fit()
    inter_term = [t for t in inter_model.params.index if "Age:C(group)" in t][0]
    age_group_interaction_p = inter_model.pvalues[inter_term]

    sensitivity_rows = run_sensitivity(df, feature, beta, pval, lin_model)       # group effect under +SITE / +Age^2 specifications

    result = {
        "feature": feature,
        "n_HC": int((df["group"] == 0).sum()),
        "n_Patient": int((df["group"] == 1).sum()),
        "mean_HC_raw": df.loc[df["group"] == 0, feature].mean(),
        "mean_Patient_raw": df.loc[df["group"] == 1, feature].mean(),
        "beta_group_adj": beta,                                                 # adjusted (Age, Sex-controlled) Patient - HC difference
        "se_group": se,
        "t_group": tval,
        "p_raw": pval,
        "ci95_lo": ci_lo,
        "ci95_hi": ci_hi,
        "beta_std_group": beta_std,                                             # standardized effect size (z-scored fit)
        "se_hc3_group": se_hc3, "t_hc3_group": t_hc3, "p_hc3_group": p_hc3,       # HC3-robust inference, same beta as above
        "ci95_lo_hc3_group": ci_lo_hc3, "ci95_hi_hc3_group": ci_hi_hc3,
        "significant_hc3_group_0.05": p_hc3 < ALPHA,
        "1_linearity_age2_p": age2_p,
        "2_independence_site_anova_p": site_p,
        "3_homoscedasticity_levene_resid_p": levene_resid_p,
        "3_homoscedasticity_breuschpagan_p": bp_p,
        "4_normality_shapiro_p": shapiro_p,
        "extra_age_group_interaction_p": age_group_interaction_p,
    }
    return result, sensitivity_rows

def main():
    df, hc, pt = load_combined()
    describe_groups(hc, pt)

    pairs = [run_ancova(df, f) for f in TOP10_FEATURES]                          # each call returns (result_dict, sensitivity_rows_list)
    rows = [p[0] for p in pairs]
    sens_rows = [r for p in pairs for r in p[1]]                                 # flatten: 10 features x 3 specs = 30 sensitivity rows
    res = pd.DataFrame(rows)
    sens = pd.DataFrame(sens_rows)

    #Multicollinearity - feature-independent (same Age/Sex/group design matrix for every feature's model)
    vif, tolerance = run_vif(df)
    for name in ("Age", "Sex", "group"):
        res[f"5_multicollinearity_vif_{name.lower()}"] = vif[name]
        res[f"5_multicollinearity_tolerance_{name.lower()}"] = tolerance[name]

    # Bonferroni correction across the N_TESTS pre-specified features
    res["p_bonferroni"] = (res["p_raw"] * N_TESTS).clip(upper=1.0)
    res["significant_raw_0.05"] = res["p_raw"] < ALPHA
    res["significant_bonferroni_0.05"] = res["p_bonferroni"] < ALPHA

    # Benjamini-Hochberg FDR correction, added alongside Bonferroni for comparison
    _, p_fdr, _, _ = multipletests(res["p_raw"].values, alpha=ALPHA, method="fdr_bh")
    res["p_fdr_bh"] = p_fdr
    res["significant_fdr_0.05"] = res["p_fdr_bh"] < ALPHA

    res = res.sort_values("p_raw").reset_index(drop=True)

    diag_cols = ["feature", "1_linearity_age2_p", "2_independence_site_anova_p", "3_homoscedasticity_levene_resid_p", "3_homoscedasticity_breuschpagan_p", "4_normality_shapiro_p", "5_multicollinearity_vif_age", "5_multicollinearity_tolerance_age", "5_multicollinearity_vif_sex", "5_multicollinearity_tolerance_sex", "5_multicollinearity_vif_group", "5_multicollinearity_tolerance_group", "extra_age_group_interaction_p"]
    diag = res[diag_cols]
    results_only = res.drop(columns=[c for c in diag_cols if c != "feature"])

    results_only.to_csv(OUT_PATH, index=False)
    diag.to_csv(DIAG_OUT_PATH, index=False)
    sens.to_csv(SENS_OUT_PATH, index=False)

    print(f"ANCOVA: feature ~ Age + Sex + group   (group: 0=SAF-HC, 1=SAF-Patient)")
    print(f"Bonferroni threshold: alpha/{N_TESTS} = {ALPHA / N_TESTS:.4f} | FDR (BH): alpha = {ALPHA}")
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(res[["feature", "n_HC", "n_Patient", "beta_group_adj", "p_raw", "p_bonferroni", "significant_bonferroni_0.05", "p_fdr_bh", "significant_fdr_0.05"]].to_string(index=False))

    print("Standardized effect size + HC3 robust inference (physician-review addition):")
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(res[["feature", "beta_std_group", "p_hc3_group", "significant_hc3_group_0.05"]].to_string(index=False))

    print("Diagnostics - the 5 standard MLR/ANCOVA prerequisites, per feature:")
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(res[["feature", "1_linearity_age2_p", "2_independence_site_anova_p", "3_homoscedasticity_levene_resid_p", "3_homoscedasticity_breuschpagan_p", "4_normality_shapiro_p"]].to_string(index=False))

    print(f"Multicollinearity (feature-independent, same for every row):")
    for name in ("Age", "Sex", "group"):
        v, t = vif[name], tolerance[name]
        print(f"{name:6s}: VIF={v:.2f}  Tolerance={t:.2f}  {'VIF HURT' if v > 5 else '(fine)'}")

    slope_violations = res.loc[res["extra_age_group_interaction_p"] < ALPHA, "feature"].tolist()
    if slope_violations:
        print(f"homogeneity-of-slopes assumption VIOLATED (Age x group interaction p<{ALPHA}) for {len(slope_violations)} feature(s): {slope_violations}")
        print("For these, beta_group_adj is an average over an age-varying group effect, not a constant one.")

    print("Sensitivity analysis - group effect stability across specifications (main / +SITE / +Age^2):")
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(sens.to_string(index=False))

if __name__ == "__main__":
    main()