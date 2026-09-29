#!/usr/bin/env python3
import os

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

BASE = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(BASE, "some path/file ...")
OUT_PATH = os.path.join(BASE, "mlr_corPAD_vs_AgeSexGroup_results.csv")
DIAG_OUT_PATH = os.path.join(BASE, "mlr_corPAD_vs_AgeSexGroup_diagnostics.csv")     # where the 5-prerequisites + interaction diagnostics get saved
SENS_OUT_PATH = os.path.join(BASE, "mlr_corPAD_vs_AgeSexGroup_sensitivity.csv")     # where the +SITE / +Age^2 sensitivity-analysis coefficients get saved
PLOT_AGE_PATH = os.path.join(BASE, "corPAD_vs_Age.png")
PLOT_SEX_PATH = os.path.join(BASE, "corPAD_vs_Sex.png")
PLOT_GROUP_PATH = os.path.join(BASE, "corPAD_vs_Group.png")

REQUIRED_COLS = ["Age", "corPAD", "Sex", "Group", "SITE"]                         # SITE added for the independence/clustering diagnostic
ALPHA = 0.05
COLOR_BLUE = "#3498db"

def load_data():
    df = pd.read_csv(INPUT_PATH)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]                   # any of the needed columns absent from this file?
    if missing:
        raise ValueError(f"Missing expected column(s): {missing}")
    n_missing = df[REQUIRED_COLS].isna().sum().sum()                              # total NaN count across just the columns this script uses
    if n_missing:
        raise ValueError(f"Found {n_missing} missing value(s) across {REQUIRED_COLS} - resolve before running the regression (this script does not impute).")
    return df

def describe_data(df):
    print("SAF-study cohort - descriptive stats")
    print(f"n = {len(df)}  (Group=0/HC: {(df['Group']==0).sum()}, Group=1/Patient: {(df['Group']==1).sum()})")
    print(f"Age: mean={df['Age'].mean():.2f} (SD={df['Age'].std():.2f})")
    print(f"corPAD: mean={df['corPAD'].mean():.2f} (SD={df['corPAD'].std():.2f})")
    print(f"Sex=1 fraction overall: {df['Sex'].mean():.3f}   (HC: {df.loc[df['Group']==0,'Sex'].mean():.3f}, Patient: {df.loc[df['Group']==1,'Sex'].mean():.3f})")
    print(f"SITEs: {df['SITE'].nunique()} distinct, sizes {df['SITE'].value_counts().min()}-{df['SITE'].value_counts().max()}")

def run_vif(df):
    X = df[["Age", "Sex", "Group"]].copy()                                       # the 3 predictors whose mutual collinearity we're checking
    X = pd.concat([pd.Series(1.0, index=X.index, name="const"), X], axis=1)       # VIF needs an explicit intercept column to be meaningful
    vif = {col: variance_inflation_factor(X.values, i)                            # one VIF value per real predictor (skip the intercept itself)
           for i, col in enumerate(X.columns) if col != "const"}                  # enumerate gives the column INDEX variance_inflation_factor needs
    tolerance = {col: 1.0 / v for col, v in vif.items()}
    return vif, tolerance

def run_interaction_check(df, term_formula, interaction_name):                      #Fit a model with one extra interaction term; return that term's p-value
    model = smf.ols(term_formula, data=df).fit()
    matches = [t for t in model.params.index if interaction_name in t]            # patsy names it e.g. "Age:C(Group)[T.1]" - find it by substring
    if not matches:
        return np.nan                                                             # defensive fallback; should not trigger for a 2-level factor
    # if the interacting factor has 2 levels there is exactly one interaction term
    return model.pvalues[matches[0]]                                              # p-value for "does Age's slope differ across this factor"

def run_linearity_check(df):                                                        #Fit corPAD ~ Age + Age^2 + Sex + Group; return the Age^2 term's p-value
    model = smf.ols('Q("corPAD") ~ Age + I(Age**2) + C(Sex) + C(Group)', data=df).fit()  # I() lets patsy evaluate a literal power term
    return model.pvalues["I(Age ** 2)"]                                          # significant => curvature => plain-linear Age term is too simple

def run_standardized_betas(df):                                                     #Fit corPAD ~ Age + Sex + Group on z-scored columns
    z = df[["Age", "Sex", "Group", "corPAD"]].copy()
    z = (z - z.mean()) / z.std()                                                 # z-score every column (mean 0, SD 1) - includes the two 0/1 dummies
    model = smf.ols('Q("corPAD") ~ Age + Sex + Group', data=z).fit()             # no C() here: Sex/Group are already numeric (0/1, now z-scored)
    return {"Age": model.params["Age"], "Sex": model.params["Sex"], "Group": model.params["Group"]}  # one std-beta per term

def run_hc3_inference(model):                                                       #Refit the main model's inference with HC3 (heteroscedasticity-robust) covariance
    robust = model.get_robustcov_results(cov_type="HC3")                         # same point estimates, robust SE/t/p/CI instead of classical
    return robust

def run_sensitivity_models(df):                                                     #Refit the Group/Age/Sex effect with (a) SITE added, (b) a quadratic Age term; return per-model coefficients
    rows = []
    main = smf.ols('Q("corPAD") ~ Age + C(Sex) + C(Group)', data=df).fit()
    rows.append({"model": "main", "beta_age": main.params["Age"], "p_age": main.pvalues["Age"], "beta_sex": main.params["C(Sex)[T.1]"], "p_sex": main.pvalues["C(Sex)[T.1]"], "beta_group": main.params["C(Group)[T.1]"], "p_group": main.pvalues["C(Group)[T.1]"]})

    with_site = smf.ols('Q("corPAD") ~ Age + C(Sex) + C(Group) + C(SITE)', data=df).fit()  # adds study center as a covariate
    rows.append({"model": "with_SITE", "beta_age": with_site.params["Age"], "p_age": with_site.pvalues["Age"], "beta_sex": with_site.params["C(Sex)[T.1]"], "p_sex": with_site.pvalues["C(Sex)[T.1]"], "beta_group": with_site.params["C(Group)[T.1]"], "p_group": with_site.pvalues["C(Group)[T.1]"]})

    with_age2 = smf.ols('Q("corPAD") ~ Age + I(Age**2) + C(Sex) + C(Group)', data=df).fit()  # adds a quadratic Age term
    rows.append({"model": "with_Age2", "beta_age": with_age2.params["Age"], "p_age": with_age2.pvalues["Age"], "beta_sex": with_age2.params["C(Sex)[T.1]"], "p_sex": with_age2.pvalues["C(Sex)[T.1]"], "beta_group": with_age2.params["C(Group)[T.1]"], "p_group": with_age2.pvalues["C(Group)[T.1]"]})

    return pd.DataFrame(rows)                                                    # one row per specification, same 3 terms' beta/p in every row

def run_site_independence_check(df, resid):                                         #One-way ANOVA of residuals on SITE; overall F-test p-value
    tmp = df[["SITE"]].copy()                                                    # only SITE is needed for this diagnostic model
    tmp["resid"] = resid.values                                                  # attach the main model's residuals as the outcome to explain
    site_model = smf.ols("resid ~ C(SITE)", data=tmp).fit()                      # does mean residual differ across the 12 sites?
    return site_model.f_pvalue                                                    # low p => residuals cluster by site => independence assumption shaky

def plot_age_vs_corpad(df, age2_p, out_path):                                       #Scatter + linear fit + LOWESS smoothed curve - visualizes the linearity check
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.regplot(x="Age", y="corPAD", data=df, ax=ax, scatter_kws=dict(color=COLOR_BLUE, alpha=0.35, s=18, linewidths=0), line_kws=dict(color=COLOR_BLUE, linewidth=2), ci=95)
    sns.regplot(x="Age", y="corPAD", data=df, ax=ax, scatter=False, lowess=True, line_kws=dict(color="#e67e22", linewidth=2, linestyle="--"), ci=None) 
    ax.set_xlabel("Age", fontsize=11)
    ax.set_ylabel("corPAD (years)", fontsize=11)
    ax.set_title(f"corPAD vs. Age - linear fit (blue) vs. LOWESS smoothed curve (orange, dashed)\nAge^2 term p={age2_p:.3g} - divergence between the two curves is exactly what that p-value is testing", fontsize=10)
    ax.grid(alpha=0.25, linewidth=0.6)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def plot_binary_scatter_vs_corpad(df, group_col, group_labels, title, out_path):    #Scatter (with horizontal jitter) + linear fit of corPAD vs. a 2-level predictor (Sex/Group)
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.regplot(x=group_col, y="corPAD", data=df, ax=ax, x_jitter=0.08, scatter_kws=dict(color=COLOR_BLUE, alpha=0.35, s=18, linewidths=0), line_kws=dict(color=COLOR_BLUE, linewidth=2), ci=95)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(group_labels)
    ax.set_xlabel(group_col, fontsize=11)
    ax.set_ylabel("corPAD (years)", fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.25, linewidth=0.6)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def main():
    df = load_data()
    describe_data(df)

    formula = 'Q("corPAD") ~ Age + C(Sex) + C(Group)'                            # Age continuous (un-wrapped); Sex/Group categorical (C()-wrapped)
    model = smf.ols(formula, data=df).fit()                                       # the one model all 3 reported coefficients come from
    beta_std = run_standardized_betas(df)                                         # standardized effect size per term (physician-review addition)
    robust = run_hc3_inference(model)                                             # HC3-robust SE/t/p/CI, same point estimates as `model`

    rows = []
    for term, label in [("Age", "Age"), ("C(Sex)[T.1]", "Sex"), ("C(Group)[T.1]", "Group")]:  # patsy's actual term name -> human-readable label
        beta = model.params[term]                                                 # coefficient (effect size) for this term
        se = model.bse[term]                                                      # standard error of that coefficient
        tval = model.tvalues[term]                                                # t-statistic (beta / se)
        pval = model.pvalues[term]                                                # two-sided p-value for this term alone
        ci_lo, ci_hi = model.conf_int().loc[term]                                 # 95% confidence interval bounds for beta
        pos = list(model.params.index).index(term)                               # get_robustcov_results returns plain arrays, indexed by POSITION not name
        se_hc3 = robust.bse[pos]                                                  # HC3-robust standard error (same beta, different SE)
        t_hc3 = robust.tvalues[pos]                                               # HC3-robust t-statistic
        p_hc3 = robust.pvalues[pos]                                               # HC3-robust p-value
        ci_lo_hc3, ci_hi_hc3 = robust.conf_int()[pos]                             # robust results' conf_int is also position-indexed
        rows.append({"term": label, "beta": beta, "se": se, "t": tval, "p_value": pval, "ci95_lo": ci_lo, "ci95_hi": ci_hi, "significant_0.05": pval < ALPHA, "beta_std": beta_std[label], "se_hc3": se_hc3, "t_hc3": t_hc3, "p_hc3": p_hc3, "ci95_lo_hc3": ci_lo_hc3, "ci95_hi_hc3": ci_hi_hc3, "significant_hc3_0.05": p_hc3 < ALPHA})
    res = pd.DataFrame(rows)                                                      # one row per term: Age, Sex, Group
    res.to_csv(OUT_PATH, index=False)

    sens = run_sensitivity_models(df)                                            # Group/Age/Sex coefficients under +SITE and +Age^2 specifications
    sens.to_csv(SENS_OUT_PATH, index=False)

    resid = model.resid                                                          # main model's residuals - most diagnostics below use these
    age2_p = run_linearity_check(df)                                              #linearity: is Age^2 significant in a richer model?
    site_p = run_site_independence_check(df, resid)                               #independence: do residuals cluster by SITE?
    levene_p = stats.levene(resid[df["Group"] == 0], resid[df["Group"] == 1])[1]  #homoscedasticity, Group dimension only
    _, bp_p, _, _ = het_breuschpagan(resid, model.model.exog)                    #homoscedasticity, ALL predictors at once (comprehensive)
    shapiro_p = stats.shapiro(resid.sample(min(len(resid), 5000), random_state=42))[1]  #normality (cap inert here, n=2523<5000)
    vif, tolerance = run_vif(df)                                                  #multicollinearity: VIF and its reciprocal, Tolerance

    age_group_p = run_interaction_check(df, 'Q("corPAD") ~ Age * C(Group) + C(Sex)', "Age:C(Group)")  #does Age's slope differ HC vs. Patient?
    age_sex_p = run_interaction_check(df, 'Q("corPAD") ~ Age * C(Sex) + C(Group)', "Age:C(Sex)")      #does Age's slope differ by Sex?

    diag = pd.DataFrame([{
        "1_linearity_age2_p": age2_p,
        "1_linearity_flag": age2_p < ALPHA,
        "2_independence_site_anova_p": site_p,
        "2_independence_flag": site_p < ALPHA,
        "3_homoscedasticity_levene_group_p": levene_p,
        "3_homoscedasticity_breuschpagan_p": bp_p,
        "3_homoscedasticity_flag": bp_p < ALPHA,                                  # Breusch-Pagan is the comprehensive one; Levene is secondary
        "4_normality_shapiro_p": shapiro_p,
        "5_multicollinearity_vif_age": vif["Age"],
        "5_multicollinearity_vif_sex": vif["Sex"],
        "5_multicollinearity_vif_group": vif["Group"],
        "5_multicollinearity_tolerance_age": tolerance["Age"],
        "5_multicollinearity_tolerance_sex": tolerance["Sex"],
        "5_multicollinearity_tolerance_group": tolerance["Group"],
        "5_multicollinearity_flag": max(vif.values()) > 5,                        #more conservative VIF>5 threshold
        "extra_age_x_group_interaction_p": age_group_p,
        "extra_age_x_sex_interaction_p": age_sex_p,
    }])
    diag.to_csv(DIAG_OUT_PATH, index=False)

    plot_age_vs_corpad(df, age2_p, PLOT_AGE_PATH)
    plot_binary_scatter_vs_corpad(df, "Sex", ["Sex=0", "Sex=1"], "corPAD vs. Sex", PLOT_SEX_PATH)
    plot_binary_scatter_vs_corpad(df, "Group", ["Group=0 (HC)", "Group=1 (Patient)"], "corPAD vs. Group", PLOT_GROUP_PATH)

    print('MLR: corPAD ~ Age + Sex + Group   (Group: 0=SAF-HC, 1=SAF-Patient) - no multiple-comparison')
    print('correction applied across the 3 terms (3 different questions, not repeated tests)')
    with pd.option_context("display.width", 160, "display.max_columns", None):
        print(res.to_string(index=False))

    print(f"R^2 = {model.rsquared:.4f}   n = {int(model.nobs)}")                   # overall model fit + sample size actually used

    print("Robust inference (HC3) - does the conclusion hold under heteroscedasticity-robust SEs?")
    for term, label in [("Age", "Age"), ("C(Sex)[T.1]", "Sex"), ("C(Group)[T.1]", "Group")]:
        p_hc3 = robust.pvalues[list(model.params.index).index(term)]             # HC3-robust p-value for this term (position-indexed, see above)
        print(f"  {label:6s}: p_classical={model.pvalues[term]:.3g}  p_HC3={p_hc3:.3g}  {'(significant under both - conclusion stable)' if p_hc3 < ALPHA else 'NOT significant under HC3'}")

    print("Diagnostics - the 5 standard MLR prerequisites:")
    print(f"Linearity (Age^2 term) p = {age2_p:.3g}  {'relationship may be NON-linear' if age2_p < ALPHA else '(linear Age term looks adequate)'}")
    print(f"Independence (SITE ANOVA) p = {site_p:.3g}  {'residuals DO cluster by SITE' if site_p < ALPHA else '(no evidence of site clustering)'}")
    print(f"Homoscedasticity - Levene (Group only) p = {levene_p:.3g}")
    print(f"Homoscedasticity - Breusch-Pagan (all 3) p = {bp_p:.3g}  {'residual variance DEPENDS on predictors' if bp_p < ALPHA else '(no evidence of heteroscedasticity)'}")
    print(f"Normality (Shapiro-Wilk) p = {shapiro_p:.3g}  (expected low at n={int(model.nobs)}, not disqualifying)")
    print(f"Multicollinearity - VIF / Tolerance (=1/VIF):")
    for name in ("Age", "Sex", "Group"):
        print(f"{name:6s}: VIF={vif[name]:.2f}  Tolerance={tolerance[name]:.2f}  {'VIF hurt' if vif[name] > 5 else '(fine)'}")

    print("Extra (not one of the 5, but relevant to interpreting Age's coefficient):")
    print(f"Age x Group interaction p = {age_group_p:.3g}  {'slope DIFFERS by Group' if age_group_p < ALPHA else ''}")
    print(f"Age x Sex interaction p = {age_sex_p:.3g}  {'slope DIFFERS by Sex' if age_sex_p < ALPHA else ''}")

    print("Sensitivity analyses - Group effect across model specifications (should stay stable):")
    with pd.option_context("display.width", 160, "display.max_columns", None):
        print(sens.to_string(index=False))                                       # main / +SITE / +Age^2, side by side

if __name__ == "__main__":
    main()