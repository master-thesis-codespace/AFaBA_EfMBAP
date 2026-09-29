import logging                                                                 
import os                                                                      
import numpy as np                                                             
import pandas as pd                                                              
import matplotlib                                                                
matplotlib.use('Agg')                                                          
import matplotlib.pyplot as plt                                                
import shap                                                                  
from sklearn.inspection import permutation_importance                        

logger = logging.getLogger(__name__)                                             

def plot_xgb_importance(model, plots_dir, tag=""):
    import xgboost as xgb                                                          # lazy import: only needed when running XGBoost
    fig, ax = plt.subplots(figsize=(10, 8))                                        # create figure with enough height for up to 20 feature bars
    xgb.plot_importance(model, ax=ax, max_num_features=20, importance_type='gain', title=f"Feature Importance (gain) - {tag}")      # draw horizontal bar chart of gain-based importance scores
    plt.tight_layout()                                                           
    fname = os.path.join(plots_dir, f"feature_importance_{tag}.png")             
    fig.savefig(fname, dpi=150)   
    plt.close(fig)                                                           
    logger.info(f"  Feature importance plot saved: {fname}")

def plot_loss_curves(eval_histories, plots_dir, tag=""):
    all_train_rmse = [h['validation_0']['rmse'] for h in eval_histories]            # per-fold training RMSE sequences (validation_0 = first eval_set entry = train)
    all_val_rmse = [h['validation_1']['rmse'] for h in eval_histories]              # per-fold validation RMSE sequences (validation_1 = second eval_set entry = val)
    max_len = max(len(x) for x in all_train_rmse)                                  # length of the longest sequence, used for padding shorter ones

    def pad_and_mean(arr_list, length):                                            # inner helper: pad sequences to equal length, then average column-wise
        padded = [np.pad(np.array(a, dtype=float), (0, length - len(a)), constant_values=np.nan) for a in arr_list]               # extend each sequence to max_len with NaN so all have equal length
        return np.nanmean(padded, axis=0)                                          # column-wise mean ignoring NaN padding

    mean_train = pad_and_mean(all_train_rmse, max_len)                             # averaged training RMSE curve across all 5 folds
    mean_val = pad_and_mean(all_val_rmse, max_len)                                # averaged validation RMSE curve across all 5 folds

    fig, ax = plt.subplots(figsize=(10, 5))                                        
    ax.plot(mean_train, label='Train RMSE (mean over 5 folds)')                    # draw mean training curve
    ax.plot(mean_val, label='Validation RMSE (mean over 5 folds)')                  # draw mean validation curve
    ax.set_xlabel("Boosting round")                                               
    ax.set_ylabel("RMSE (years)")                                                 
    ax.set_yscale('log')                                                          
    ax.set_title(f"Training vs. Validation Loss - {tag}")                        
    ax.legend()                                                                    
    plt.tight_layout()                                                            
    fname = os.path.join(plots_dir, f"loss_curve_{tag}.png")                      
    fig.savefig(fname, dpi=150)                                                    
    plt.close(fig)                                                              
    logger.info(f"Loss curve plot saved: {fname}")

def plot_tree_importance(model, feature_names, plots_dir, tag="", model_name=""):
    importances = model.feature_importances_                                       # mean decrease in impurity for each feature, averaged over all trees
    sorted_idx = importances.argsort()                                              # sort ascending so most important feature ends up at top of barh chart
    top20_idx = sorted_idx[-20:]                                                    # take top-20 most important features

    fig, ax = plt.subplots(figsize=(10, 8))                                      
    ax.barh(np.array(feature_names)[top20_idx], importances[top20_idx], align='center')  # horizontal bar chart of impurity-based importance for top-20 features
    ax.set_xlabel("Mean decrease in impurity")                                    
    ax.set_title(f"{model_name} Feature Importance (impurity) - {tag}")           
    plt.tight_layout()                                                             
    fname = os.path.join(plots_dir, f"feature_importance_{tag}.png")              
    fig.savefig(fname, dpi=150)                                                   
    plt.close(fig)                                                                
    logger.info(f"Feature importance plot saved: {fname}")

def plot_oob_curve(oob_errors, plots_dir, tag=""):
    n_trees = [x[0] for x in oob_errors]
    rmse = [x[1] for x in oob_errors] 

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(n_trees, rmse, marker='o', markersize=4)
    ax.set_xlabel("Number of trees")
    ax.set_ylabel("OOB RMSE (years)")
    ax.set_title(f"OOB Error vs. Number of Trees - {tag}")
    plt.tight_layout()
    fname = os.path.join(plots_dir, f"oob_curve_{tag}.png")
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    logger.info(f"OOB curve plot saved: {fname}")

def compute_shap_values(model, X_sample, feature_names):
    logger.info(f"Computing SHAP values ({len(X_sample)} samples) ...")
    explainer = shap.TreeExplainer(model)                                          # build a fast tree-based SHAP explainer from the trained model
    X_df = pd.DataFrame(X_sample, columns=feature_names)                            # named DataFrame -> SHAP uses column names in all labels
    return explainer(X_df)                                                         # Explanation object with .values, .base_values, .data, .feature_names

def plot_shap_waterfall(shap_vals, plots_dir, tag="", sample_idx=0):
    shap.plots.waterfall(shap_vals[sample_idx], max_display=20, show=False)       # draw waterfall for one sample
    plt.title(f"SHAP Waterfall - test sample {sample_idx} ({tag})")
    fname = os.path.join(plots_dir, f"shap_waterfall_{tag}.png")
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"SHAP waterfall plot saved: {fname}")

def plot_shap_beeswarm(shap_vals, plots_dir, tag="", max_display=20, data_source="test", fold=None):
    n_features = shap_vals.shape[1]
    n_shown = min(max_display, n_features)                                          # actual number of rows that will appear in the plot
    show_all = (n_shown >= n_features)
    fig_height = max(6, n_shown * 0.35)
    shap.plots.beeswarm(shap_vals, max_display=max_display, show=False)             # draw beeswarm
    plt.gcf().set_size_inches(10, fig_height)                                      # resize after SHAP draws: applies dynamic height regardless of which figure SHAP used
    title_suffix = "all features" if show_all else f"top {n_shown} features"
    fold_label = f" - fold {fold}" if fold is not None else ""
    plt.title(f"SHAP Beeswarm - {title_suffix}, all {data_source} samples{fold_label} ({tag})")
    fname_suffix = "all" if (max_display >= n_features) else f"top{max_display}"
    fold_part = f"_cv{fold}" if fold is not None else ""
    fname = os.path.join(plots_dir, f"shap_beeswarm_{fname_suffix}_{data_source}{fold_part}_{tag}.png")
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  SHAP beeswarm plot saved: {fname}")

def plot_lgbm_importance(model, plots_dir, tag="", feature_names=None):
    import lightgbm as lgb                                                           # lazy import: only needed when running LightGBM
    importances = model.booster_.feature_importance(importance_type='gain')         # gain-based feature importance from the trained booster
    feature_names = feature_names if feature_names is not None else model.booster_.feature_name()  
    sorted_idx = importances.argsort()                                              # sort ascending so most important feature ends up at top of barh chart
    top20_idx = sorted_idx[-20:]                                                    # take the top-20 most important features

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(np.array(feature_names)[top20_idx], importances[top20_idx], align='center') # horizontal bar chart of gain-based importance for top-20 features
    ax.set_xlabel("Gain")
    ax.set_title(f"LightGBM Feature Importance (gain) - {tag}")
    plt.tight_layout()
    fname = os.path.join(plots_dir, f"feature_importance_{tag}.png")
    fig.savefig(fname, dpi=150)
    plt.close(fig)                                                                   # release figure memory
    logger.info(f"Feature importance plot saved: {fname}")

def plot_lgbm_loss_curves(eval_histories, plots_dir, tag=""):
    all_train_rmse = [h['train']['rmse'] for h in eval_histories]                   # per-fold training RMSE sequences
    all_val_rmse = [h['val']['rmse'] for h in eval_histories]                       # per-fold validation RMSE sequences
    max_len = max(len(x) for x in all_train_rmse)                                   # length of the longest sequence, used for padding shorter ones

    def pad_and_mean(arr_list, length):                                              # pad sequences to equal length, then average column-wise
        padded = [np.pad(np.array(a, dtype=float), (0, length - len(a)), constant_values=np.nan) for a in arr_list]                 # extend each sequence to max_len with NaN so all have equal length
        return np.nanmean(padded, axis=0)                                            # column-wise mean ignoring NaN padding

    mean_train = pad_and_mean(all_train_rmse, max_len)                               # averaged training RMSE curve across all 5 folds
    mean_val = pad_and_mean(all_val_rmse, max_len)                                   # averaged validation RMSE curve across all 5 folds

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(mean_train, label='Train RMSE (mean over 5 folds)')
    ax.plot(mean_val, label='Validation RMSE (mean over 5 folds)')
    ax.set_xlabel("Boosting round")
    ax.set_ylabel("RMSE (years)")
    ax.set_yscale('log')
    ax.set_title(f"Training vs. Validation Loss - {tag}")
    ax.legend()
    plt.tight_layout()
    fname = os.path.join(plots_dir, f"loss_curve_{tag}.png")
    fig.savefig(fname, dpi=150) 
    plt.close(fig) 
    logger.info(f"Loss curve plot saved: {fname}")

def plot_catboost_importance(model, plots_dir, tag="", feature_names=None):
    import catboost                                                                  # lazy import: only needed when running CatBoost
    importances = model.get_feature_importance()                                    # PredictionValuesChange importance, one value per feature
    feature_names = feature_names if feature_names is not None else model.feature_names_ 
    sorted_idx = importances.argsort()                                              # sort ascending so most important feature ends up at top of barh chart
    top20_idx = sorted_idx[-20:]                                                    # take the top-20 most important features

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(np.array(feature_names)[top20_idx], importances[top20_idx], align='center') # horizontal bar chart of importance for top-20 features
    ax.set_xlabel("PredictionValuesChange (gain)")
    ax.set_title(f"CatBoost Feature Importance (gain) - {tag}")
    plt.tight_layout()
    fname = os.path.join(plots_dir, f"feature_importance_{tag}.png")
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    logger.info(f"Feature importance plot saved: {fname}")

def plot_catboost_loss_curves(eval_histories, plots_dir, tag=""):
    all_train_rmse = [h['learn']['RMSE'] for h in eval_histories]                   # per-fold training RMSE sequences
    all_val_rmse = [h['validation']['RMSE'] for h in eval_histories]                # per-fold validation RMSE sequences
    max_len = max(len(x) for x in all_train_rmse)                                    # length of the longest sequence

    def pad_and_mean(arr_list, length):
        padded = [np.pad(np.array(a, dtype=float), (0, length - len(a)), constant_values=np.nan) for a in arr_list]                 # extend each sequence to max_len with NaN so all have equal length
        return np.nanmean(padded, axis=0)                                            # column-wise mean ignoring NaN padding

    mean_train = pad_and_mean(all_train_rmse, max_len)                               # averaged training RMSE curve across all 5 folds
    mean_val = pad_and_mean(all_val_rmse, max_len)                                   # averaged validation RMSE curve across all 5 folds

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(mean_train, label='Train RMSE (mean over 5 folds)')
    ax.plot(mean_val, label='Validation RMSE (mean over 5 folds)')
    ax.set_xlabel("Boosting round")
    ax.set_ylabel("RMSE (years)")
    ax.set_yscale('log')
    ax.set_title(f"Training vs. Validation Loss - {tag}")
    ax.legend()
    plt.tight_layout()
    fname = os.path.join(plots_dir, f"loss_curve_{tag}.png")
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    logger.info(f"Loss curve plot saved: {fname}")

def plot_permutation_importance(model, X_test, Y_test, feature_names, plots_dir, tag=""):
    logger.info(f"  Computing permutation importance on test data ({len(X_test)} subjects) ...")
    result = permutation_importance(
        model, X_test, Y_test,                                                     # evaluate on the held-out test subjects
        n_repeats=30,                                                              # shuffle each feature 30 times for stable importance estimates
        random_state=42,                                                           # fixed seed for reproducibility
        scoring='neg_root_mean_squared_error'                                      # positive importance = RMSE rises when feature is shuffled = feature matters
    )

    sorted_idx = result.importances_mean.argsort()                                 # sort ascending so most important feature ends up at top of barh chart
    feat_names_sorted = np.array(feature_names)[sorted_idx]                        # reorder feature name labels to match sorted indices
    means_sorted = result.importances_mean[sorted_idx]                             # reorder mean importance values to match sorted indices
    stds_sorted = result.importances_std[sorted_idx]                               # reorder std values to match sorted indices

    fig_height = max(6, len(feature_names) * 0.35)                                 # scale figure height so each feature row has at least 0.35 inches
    fig, ax = plt.subplots(figsize=(10, fig_height))
    ax.barh(feat_names_sorted, means_sorted, xerr=stds_sorted, align='center')    # horizontal bar chart; error bars show variability across 30 shuffles
    ax.axvline(x=0, color='black', linewidth=0.8, linestyle='--')
    ax.set_xlabel("Mean decrease in RMSE (years) when feature is permuted")
    ax.set_title(f"Permutation Importance on Test Data - {tag}")
    plt.tight_layout()
    fname = os.path.join(plots_dir, f"permutation_importance_test_{tag}.png")
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    logger.info(f"Permutation importance (test) plot saved: {fname}")

def plot_permutation_importance_oof(pi_mean, pi_std, plots_dir, tag="", fold=0):
    logger.info(f"Plotting permutation importance (OOF fold {fold}) ...")
    sorted_idx = pi_mean.values.argsort()                                           # sort ascending: least important at bottom, most important at top of barh
    feat_names_sorted = pi_mean.index[sorted_idx]
    means_sorted = pi_mean.values[sorted_idx]
    stds_sorted = pi_std.values[sorted_idx]

    fig_height = max(6, len(pi_mean) * 0.35)                                        # scale figure height so each feature row has at least 0.35 inches
    fig, ax = plt.subplots(figsize=(10, fig_height))
    ax.barh(feat_names_sorted, means_sorted, xerr=stds_sorted, align='center')     # horizontal bar chart
    ax.axvline(x=0, color='black', linewidth=0.8, linestyle='--')
    ax.set_xlabel("Mean decrease in RMSE (years) when feature is permuted")
    ax.set_title(f"Permutation Importance on OOF Data - fold {fold} ({tag})")
    plt.tight_layout()
    fname = os.path.join(plots_dir, f"permutation_importance_oof_cv{fold}_{tag}.png")
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    logger.info(f"Permutation importance (OOF fold {fold}) plot saved: {fname}")