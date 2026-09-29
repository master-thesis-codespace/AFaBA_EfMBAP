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
OUT_PATH = os.path.join(BASE, "mlr_corPAD_vs_AgeSexWMH_results.csv")
DIAG_OUT_PATH = os.path.join(BASE, "mlr_corPAD_vs_AgeSexWMH_diagnostics.csv")       # where the 5-prerequisites + interaction diagnostics get saved
SENS_OUT_PATH = os.path.join(BASE, "mlr_corPAD_vs_AgeSexWMH_sensitivity.csv")       # where the +SITE / +Age^2 / +wmh^2 sensitivity coefficients get saved
PLOT_AGE_PATH = os.path.join(BASE, "corPAD_vs_Age_AgeSexWMH.png")
PLOT_SEX_PATH = os.path.join(BASE, "corPAD_vs_Sex_AgeSexWMH.png")
PLOT_WMH_PATH = os.path.join(BASE, "corPAD_vs_WMH_AgeSexWMH.png")

REQUIRED_COLS = ["Age", "corPAD", "Sex", "SITE", "wmh_volume_ml"]                 # SITE for independence check; no Group needed (not in this model)
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
    print("SAF-study cohort - descriptive stats (Group NOT used as a model term here)")
    print(f"n = {len(df)}")
    print(f"Age: mean={df['Age'].mean():.2f} (SD={df['Age'].std():.2f})")
    print(f"corPAD: mean={df['corPAD'].mean():.2f} (SD={df['corPAD'].std():.2f})")
    print(f"wmh_volume_ml: mean={df['wmh_volume_ml'].mean():.3f} (SD={df['wmh_volume_ml'].std():.3f})  skewness={df['wmh_volume_ml'].skew():.3f}")
    print(f"Sex=1 fraction overall: {df['Sex'].mean():.3f}")
    print(f"SITEs: {df['SITE'].nunique()} distinct, sizes {df['SITE'].value_counts().min()}-{df['SITE'].value_counts().max()}")

def run_vif(df):
    X = df[["Age", "Sex", "wmh_volume_ml"]].copy()                               # the 3 predictors whose mutual collinearity we're checking
    X = pd.concat([pd.Series(1.0, index=X.index, name="const"), X], axis=1)       # VIF needs an explicit intercept column to be meaningful
    vif = {col: variance_inflation_factor(X.values, i)                            # one VIF value per real predictor (skip the intercept itself)
           for i, col in enumerate(X.columns) if col != "const"}                  # enumerate gives the column INDEX variance_inflation_factor needs
    tolerance = {col: 1.0 / v for col, v in vif.items()}
    return vif, tolerance

def run_interaction_check(df, term_formula, interaction_name):                      #Fit a model with one extra interaction term; return that term's p-value
    model = smf.ols(term_formula, data=df).fit()
    matches = [t for t in model.params.index if interaction_name in t]            # patsy names it e.g. "Age:C(Sex)[T.1]" - find it by substring
    if not matches:
        return np.nan                                                             # defensive fallback; should not trigger for a 2-level factor
    # if the interacting factor has 2 levels there is exactly one interaction term
    return model.pvalues[matches[0]]                                              # p-value for "does Age's slope differ across this factor"

def run_linearity_check_age(df):                                                    #Fit corPAD ~ Age + Age^2 + Sex + wmh; return the Age^2 term's p-value
    model = smf.ols('Q("corPAD") ~ Age + I(Age**2) + C(Sex) + wmh_volume_ml', data=df).fit()
    return model.pvalues["I(Age ** 2)"]                                          # significant => curvature => plain-linear Age term is too simple

def run_linearity_check_wmh(df):                                                    #Fit corPAD ~ Age + Sex + wmh + wmh^2; return the wmh^2 term's p-value
    model = smf.ols('Q("corPAD") ~ Age + C(Sex) + wmh_volume_ml + I(wmh_volume_ml**2)', data=df).fit()
    return model, model.pvalues["I(wmh_volume_ml ** 2)"]                         # model returned too - reused by the with_wmh2 sensitivity row

def run_standardized_betas(df):                                                     #Fit corPAD ~ Age + Sex + wmh on z-scored columns
    z = df[["Age", "Sex", "wmh_volume_ml", "corPAD"]].copy()
    z = (z - z.mean()) / z.std()                                                 # z-score every column (mean 0, SD 1) - includes the 0/1 Sex dummy
    model = smf.ols('Q("corPAD") ~ Age + Sex + wmh_volume_ml', data=z).fit()     # no C() here: Sex already numeric (0/1), now z-scored
    return {"Age": model.params["Age"], "Sex": model.params["Sex"], "wmh_volume_ml": model.params["wmh_volume_ml"]} # one std-beta per term

def run_hc3_inference(model):                                                       #Refit the main model's inference with HC3 (heteroscedasticity-robust) covariance
    robust = model.get_robustcov_results(cov_type="HC3")                         # same point estimates, robust SE/t/p/CI instead of classical
    return robust


def run_sensitivity_models(df, wmh2_model):                                         #Refit all 3 terms' effects with (a) SITE added, (b) a quadratic Age term,  (c) a quadratic wmh_volume_ml term; return per-model coefficients
    rows = []

    def _row(label, model):
        return {"model": label, "beta_age": model.params["Age"], "p_age": model.pvalues["Age"], "beta_sex": model.params["C(Sex)[T.1]"], "p_sex": model.pvalues["C(Sex)[T.1]"], "beta_wmh": model.params["wmh_volume_ml"], "p_wmh": model.pvalues["wmh_volume_ml"]}

    main = smf.ols('Q("corPAD") ~ Age + C(Sex) + wmh_volume_ml', data=df).fit()  # re-fit here, self-contained sensitivity table
    rows.append(_row("main", main))

    with_site = smf.ols('Q("corPAD") ~ Age + C(Sex) + wmh_volume_ml + C(SITE)', data=df).fit()  # adds study center as a covariate
    rows.append(_row("with_SITE", with_site))

    with_age2 = smf.ols('Q("corPAD") ~ Age + I(Age**2) + C(Sex) + wmh_volume_ml', data=df).fit()  # adds a quadratic Age term
    rows.append(_row("with_Age2", with_age2))

    rows.append(_row("with_wmh2", wmh2_model))                                   # reuses the already-fit wmh linearity-check model, no extra fit needed

    return pd.DataFrame(rows)                                                    # one row per specification, all 3 terms' beta/p in every row

def run_site_independence_check(df, resid):                                         #One-way ANOVA of residuals on SITE; overall F-test p-value
    tmp = df[["SITE"]].copy()
    tmp["resid"] = resid.values                                                  # attach the main model's residuals as the outcome to explain
    site_model = smf.ols("resid ~ C(SITE)", data=tmp).fit()                      # does mean residual differ across the 12 sites?
    return site_model.f_pvalue                                                    # low p => residuals cluster by site => independence assumption shaky

def plot_continuous_vs_corpad(df, xcol, xlabel, term2_p, term2_label, out_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.regplot(x=xcol, y="corPAD", data=df, ax=ax, scatter_kws=dict(color=COLOR_BLUE, alpha=0.35, s=18, linewidths=0),
                line_kws=dict(color=COLOR_BLUE, linewidth=2), ci=95)
    sns.regplot(x=xcol, y="corPAD", data=df, ax=ax, scatter=False, lowess=True, line_kws=dict(color="#e67e22", linewidth=2, linestyle="--"), ci=None)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("corPAD (years)", fontsize=11)
    ax.set_title(f"corPAD vs. {xlabel} - linear fit (blue) vs. LOWESS smoothed curve (orange, dashed)\n{term2_label}^2 term p={term2_p:.3g} - divergence between the two curves is exactly what that p-value is testing", fontsize=10)
    ax.grid(alpha=0.25, linewidth=0.6)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def plot_binary_scatter_vs_corpad(df, group_col, group_labels, title, out_path):
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

    formula = 'Q("corPAD") ~ Age + C(Sex) + wmh_volume_ml'                       # Age/wmh continuous (un-wrapped); Sex categorical (C()-wrapped)
    model = smf.ols(formula, data=df).fit()                                       # the one model all 3 reported coefficients come from
    beta_std = run_standardized_betas(df)                                         # standardized effect size per term (physician-review addition)
    robust = run_hc3_inference(model)                                             # HC3-robust SE/t/p/CI, same point estimates as `model`

    term_labels = [("Age", "Age"), ("C(Sex)[T.1]", "Sex"), ("wmh_volume_ml", "wmh_volume_ml")]  # patsy term -> label
    rows = []
    for term, label in term_labels:
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
    res = pd.DataFrame(rows)                                                      # one row per term: Age, Sex, wmh_volume_ml
    res.to_csv(OUT_PATH, index=False)

    resid = model.resid                                                          # main model's residuals - most diagnostics below use these
    age2_p = run_linearity_check_age(df)                                          #linearity: is Age^2 significant in a richer model?
    wmh2_model, wmh2_p = run_linearity_check_wmh(df)                              #linearity: is wmh_volume_ml^2 significant in a richer model?
    site_p = run_site_independence_check(df, resid)                               #independence: do residuals cluster by SITE?
    levene_p = stats.levene(resid[df["Sex"] == 0], resid[df["Sex"] == 1])[1]      #homoscedasticity, Sex dimension only (no Group factor here)
    _, bp_p, _, _ = het_breuschpagan(resid, model.model.exog)                     #homoscedasticity, ALL 3 predictors at once (comprehensive)
    shapiro_p = stats.shapiro(resid.sample(min(len(resid), 5000), random_state=42))[1]  #normality (cap inert here, n=2523<5000)
    vif, tolerance = run_vif(df)                                                  #multicollinearity: VIF and its reciprocal, Tolerance

    age_sex_p = run_interaction_check(df, 'Q("corPAD") ~ Age * C(Sex) + wmh_volume_ml', "Age:C(Sex)")  #does Age's slope differ by Sex?

    diag = pd.DataFrame([{
        "1_linearity_age2_p": age2_p,
        "1_linearity_age_flag": age2_p < ALPHA,
        "1_linearity_wmh2_p": wmh2_p,
        "1_linearity_wmh_flag": wmh2_p < ALPHA,
        "2_independence_site_anova_p": site_p,
        "2_independence_flag": site_p < ALPHA,
        "3_homoscedasticity_levene_sex_p": levene_p,                              # Sex-split Levene (Group not in this model)
        "3_homoscedasticity_breuschpagan_p": bp_p,
        "3_homoscedasticity_flag": bp_p < ALPHA,                                  # Breusch-Pagan is the comprehensive one; Levene is secondary
        "4_normality_shapiro_p": shapiro_p,
        "5_multicollinearity_vif_age": vif["Age"],
        "5_multicollinearity_vif_sex": vif["Sex"],
        "5_multicollinearity_vif_wmh": vif["wmh_volume_ml"],
        "5_multicollinearity_tolerance_age": tolerance["Age"],
        "5_multicollinearity_tolerance_sex": tolerance["Sex"],
        "5_multicollinearity_tolerance_wmh": tolerance["wmh_volume_ml"],
        "5_multicollinearity_flag": max(vif.values()) > 5,                        #more conservative VIF>5 threshold
        "extra_age_x_sex_interaction_p": age_sex_p,
    }])
    diag.to_csv(DIAG_OUT_PATH, index=False)

    sens = run_sensitivity_models(df, wmh2_model)                                # all 3 terms' coefficients under +SITE / +Age^2 / +wmh^2 specifications
    sens.to_csv(SENS_OUT_PATH, index=False)

    plot_continuous_vs_corpad(df, "Age", "Age", age2_p, "Age", PLOT_AGE_PATH)
    plot_binary_scatter_vs_corpad(df, "Sex", ["Sex=0", "Sex=1"], "corPAD vs. Sex", PLOT_SEX_PATH)
    plot_continuous_vs_corpad(df, "wmh_volume_ml", "WMH volume (mL)", wmh2_p, "wmh_volume_ml", PLOT_WMH_PATH)

    print('MLR: corPAD ~ Age + Sex + wmh_volume_ml   (Group NOT included as a model term)')
    print('- no multiple-comparison correction applied across the 3 terms (3 different questions, not repeated tests)')
    with pd.option_context("display.width", 160, "display.max_columns", None):
        print(res.to_string(index=False))

    print(f"R^2 = {model.rsquared:.4f}   n = {int(model.nobs)}")                   # overall model fit + sample size actually used
          
    print("Robust inference (HC3) - does the conclusion hold under heteroscedasticity-robust SEs?")
    for term, label in term_labels:
        p_hc3 = robust.pvalues[list(model.params.index).index(term)]             # HC3-robust p-value for this term (position-indexed, see above)
        print(f"{label:14s}: p_classical={model.pvalues[term]:.3g}  p_HC3={p_hc3:.3g}  {'(significant under both - conclusion stable)' if p_hc3 < ALPHA else 'NOT significant under HC3'}")

    print("Diagnostics - the 5 standard MLR prerequisites:")
    print(f"Linearity (Age^2 term) p = {age2_p:.3g}  {'relationship may be NON-linear' if age2_p < ALPHA else '(linear Age term looks adequate)'}")
    print(f"Linearity (wmh_volume_ml^2 term) p = {wmh2_p:.3g}  {'relationship may be NON-linear' if wmh2_p < ALPHA else '(linear wmh term looks adequate)'}")
    print(f"Independence (SITE ANOVA)  p = {site_p:.3g}  {'residuals DO cluster by SITE' if site_p < ALPHA else '(no evidence of site clustering)'}")
    print(f"Homoscedasticity - Levene (Sex only, no Group in this model) p = {levene_p:.3g}")
    print(f"Homoscedasticity - Breusch-Pagan (all 3) p = {bp_p:.3g}  {'residual variance DEPENDS on predictors' if bp_p < ALPHA else '(no evidence of heteroscedasticity)'}")
    print(f"Normality (Shapiro-Wilk) p = {shapiro_p:.3g}  (expected low at n={int(model.nobs)}, not disqualifying)")
    print(f"Multicollinearity - VIF / Tolerance (=1/VIF):")
    for name, key in (("Age", "Age"), ("Sex", "Sex"), ("wmh_volume_ml", "wmh_volume_ml")):
        print(f"{name:14s}: VIF={vif[key]:.2f}  Tolerance={tolerance[key]:.2f}  {'VIF hurt' if vif[key] > 5 else '(fine)'}")

    print("Relevant to interpreting Age's coefficient:")
    print(f"Age x Sex interaction p = {age_sex_p:.3g}  {'slope DIFFERS by Sex' if age_sex_p < ALPHA else ''}")
    print("Sensitivity analyses - all 3 terms across model specifications (should stay stable):")
    with pd.option_context("display.width", 160, "display.max_columns", None):
        print(sens.to_string(index=False))                                       # main / +SITE / +Age^2 / +wmh^2, side by side

if __name__ == "__main__":
    main()