import argparse
import json
import logging
import os
import sys
import time
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plotting import (
    plot_catboost_importance, plot_catboost_loss_curves,
    compute_shap_values, plot_shap_waterfall,
    plot_shap_beeswarm, plot_permutation_importance, plot_permutation_importance_oof,
)

logging.basicConfig(                                                                #Setup Logging, track training process
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def preproc_data(file_path, feature_columns=None):
    if not os.path.exists(file_path):                                               # verify the file exists before attempting to read it
        raise FileNotFoundError(f"Input file not found: {file_path}")               # abort early with a clear error message

    df_data = pd.read_csv(file_path)                                                # load the entire CSV into a DataFrame

    # Features: Sex + following columns (index 3 onwards)
    if not feature_columns:                                                         # use default feature set if caller did not specify one
        feature_columns = df_data.columns.to_list()[3:]                             # Sex at index 3, MEAN(area)_1 at index 4; both included as features

    # Clean missing values
    df_data.dropna(axis=1, how="all", inplace=True)                                 # drop columns where every value is NaN
    df_data.dropna(axis=0, how="all", inplace=True)                                 # drop rows where every value is NaN
    df_data.fillna(0, inplace=True)                                                 # replace remaining NaN cells with 0
    df_data = df_data.copy()                                                        # consolidate fragmented memory layout before adding stratification columns

    target = "Age"                                                                  # name of the regression target column

    # Stratification on Age (10-year bins) + Site (1-17).
    if "batch_vector" in df_data.columns:
        df_data.rename(columns={"batch_vector": "SITE"}, inplace=True)
    try:
        df_data["Age_category"] = (df_data["Age"] // 10).astype(int)                # 10-year bin: 0-9->0, 10-19->1, 20-29->2
        strat_label = (df_data["Age_category"].values.astype(int)
                       + df_data["SITE"].values.astype(int) * 10)                   # site 1-17 -> tens/hundreds place; step 10 > max(age_bin)=9
        _counts = pd.Series(strat_label).value_counts()
        strat_label = np.where(pd.Series(strat_label).map(_counts) < 2, -1, strat_label)  # pool singleton (age, site, sex) combos into -1; StratifiedShuffleSplit requires >= 2 members per class
    except KeyError as e:
        logger.error(f"Missing column required for stratification: {e}")            # report which column is missing
        raise                                                                      

    X_data = df_data.loc[:, feature_columns].values                                 # feature matrix
    Y_data = df_data.loc[:, target].values                                          # regression target (Age in years)
    IDs = df_data["ID"].values                                                      # subject identifiers

    return feature_columns, strat_label, X_data, Y_data, IDs

def correct_age(Y_data, Y_predict):                                                 #Correct predicted ages based on the relationship between actual and predicted ages
    # Calculate Brain Age Gap (BAG) as the difference between predicted ages and actual ages.
    BAG = Y_predict - Y_data
    # Fit a linear model to the relationship between actual ages and BAG.
    z = np.polyfit(Y_data, BAG, 1)
    # Calculate the uncorrected residue by removing the linear trend from BAG.
    Y_uncorrected_residue = BAG - (z[0] * Y_data + z[1])
    # Fit a linear model to the relationship between actual ages and predicted ages.
    correction_term = np.polyfit(Y_data, Y_predict, 1)
    # Correct the predicted ages using the correction term.
    Y_corrected = Y_predict + (Y_data - (correction_term[0] * Y_data + correction_term[1]))  # shift predictions to align with actual age line
    return Y_uncorrected_residue, correction_term, Y_corrected

def run_grid_search(X_train, Y_train, strat_train, grid_params):                    #Performs Grid Search to find best hyperparameters
    logger.info("--- Starting Grid Search ---")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)                # stratified 5-fold on training data only
    cv_ind_list = [a for a in skf.split(strat_train, strat_train)]                #     materialise fold index pairs as a list for GridSearchCV

    cb = CatBoostRegressor(                                                         # base estimator; hyperparams overridden by grid search
        loss_function='RMSE',                                                       # minimise RMSE during boosting
        random_seed=42,
        thread_count=1,                                                             # single-threaded per fit so GridSearchCV (n_jobs=-1) owns all parallelism
        verbose=0,                                                                  # suppress per-round console output inside GridSearchCV
    )

    search = GridSearchCV(
        estimator=cb,
        param_grid=grid_params,
        scoring={                                                                   # compute all three metrics for every combination
            'neg_rmse': 'neg_root_mean_squared_error',
            'neg_mae': 'neg_mean_absolute_error',
            'r2': 'r2',
        },
        n_jobs=32,
        cv=cv_ind_list,
        refit=False,                                                                # only finds best params; retraining done in train_and_evaluate
        verbose=1
    )
    search.fit(X_train, Y_train)                                                    # run all combinations x 5 folds on training data only

    best_idx = np.argmax(search.cv_results_['mean_test_neg_rmse'])                  # highest neg_rmse = lowest RMSE = best combination
    best_params = search.cv_results_['params'][best_idx]
    best_rmse = -search.cv_results_['mean_test_neg_rmse'][best_idx]

    logger.info(f"Grid Search Best Params: {best_params}")                          # print the winning hyperparameter combination
    logger.info(f"Grid Search Best RMSE: {best_rmse:.4f}")                          #print the best cross-validated RMSE

    cv_df = pd.DataFrame(search.cv_results_['params'])
    cv_df['mean_cv_rmse'] = -search.cv_results_['mean_test_neg_rmse']               # convert neg scores back to positive RMSE
    cv_df['std_cv_rmse'] = search.cv_results_['std_test_neg_rmse']
    cv_df['mean_cv_mae'] = -search.cv_results_['mean_test_neg_mae']
    cv_df['std_cv_mae'] = search.cv_results_['std_test_neg_mae']
    cv_df['mean_cv_r2'] = search.cv_results_['mean_test_r2']
    cv_df['std_cv_r2'] = search.cv_results_['std_test_r2']
    cv_df.sort_values('mean_cv_rmse', inplace=True, ignore_index=True)              # sort best (lowest RMSE) first
    return best_params, cv_df

def train_and_evaluate(X_train_all, Y_train_all, strat_train, ID_train, X_test, Y_test, ID_test, feature_columns, output_dir, param_dict, tag): #Run 5-fold CV on training data, then train a final CatBoost model on all training subjects and evaluate it on the held-out test set
    logger.info(f"--- Starting Training Loop [{tag}] ---")

    clean_params = {k: v for k, v in param_dict.items() if k != 'early_stopping_rounds'}  # strip early_stopping_rounds for final model only

    # Directories
    model_dir = os.path.join(output_dir, tag, "models")                             # sub-folder for per-fold and final model files
    results_dir = os.path.join(output_dir, tag, "results")                          # sub-folder for metrics CSVs, predictions, and correction params
    plots_dir = os.path.join(output_dir, tag, "plots")                              # sub-folder for all PNG plots
    splits_dir = os.path.join(output_dir, tag, "splits")                            # sub-folder for per-fold train/val subject ID JSON files
    for d in [model_dir, results_dir, plots_dir, splits_dir]:
        os.makedirs(d, exist_ok=True)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)                # stratified 5-fold on training subjects
    cv_ind_list = [a for a in skf.split(strat_train, strat_train)]                  # materialise fold index pairs as a list for the loop

    rmse_arr = np.zeros(5)                                                          # pre-allocate array to store per-fold RMSE
    mae_arr = np.zeros(5)                                                           # pre-allocate array to store per-fold MAE
    eval_histories = []                                                            # store per-fold eval history dicts for loss-curve plot
    best_iters = []                                                                 # best iteration (tree count) per fold; mean used for final model iterations

    all_val_ids = []                                                                # accumulate subject IDs of validation samples across all folds
    all_val_actual = []                                                             # accumulate true ages of validation samples across all folds
    all_val_preds = []                                                              # accumulate predicted ages of validation samples across all folds
    # Cross Validation Loop
    for cv_ind, (tr_ind, val_ind) in enumerate(cv_ind_list):                        # iterate over 5 stratified folds
        X_tr, Y_tr = X_train_all[tr_ind, :], Y_train_all[tr_ind]                    # training subjects for this fold
        X_val, Y_val = X_train_all[val_ind, :], Y_train_all[val_ind]                # validation subjects for this fold

        # Save fold split IDs
        split_info = {"train_ids": ID_train[tr_ind].tolist(), "val_ids": ID_train[val_ind].tolist()}
        with open(os.path.join(splits_dir, f"cv{cv_ind}_ids.json"), "w") as f:
            json.dump(split_info, f, indent=4)                                      # persist fold split to JSON so the split is auditable
        logger.info(f"Fold {cv_ind}: train={len(tr_ind)}, val={len(val_ind)}")

        # Train
        cb = CatBoostRegressor(
            loss_function='RMSE',
            random_seed=42,
            thread_count=-1,
            verbose=0,                                                              # suppress per-round console output
            **param_dict                                                            # unpack iterations, learning_rate, depth, l2_leaf_reg, min_data_in_leaf, early_stopping_rounds
        )
        # eval_set passes both train and val so we can track both curves.
        # validation_0 -> train, validation_1 -> val in evals_result()
        cb.fit(
            X_tr, Y_tr,
            eval_set=(X_val, Y_val),                                                # track RMSE on validation set at every boosting round
        )

        eval_histories.append(cb.get_evals_result())
        bi = cb.get_best_iteration()                                                 # 0-indexed best iteration; None if all iterations completed without early stopping firing
        best_iters.append((bi + 1) if bi is not None else param_dict['iterations'])  # +1 converts 0-indexed to tree count; fall back to full iterations when None

        # Evaluate
        preds = cb.predict(X_val)                                                   # predict ages on the held-out validation fold
        all_val_ids.extend(ID_train[val_ind])                                       # collect subject IDs for this fold's validation set
        all_val_actual.extend(Y_val)                                                # collect true ages for this fold's validation set
        all_val_preds.extend(preds)                                                 # collect predicted ages for this fold's validation set

        rmse_arr[cv_ind] = ((preds - Y_val)**2).mean()**(1/2)                       # root mean square error for this fold's validation predictions
        mae_arr[cv_ind] = np.abs(preds - Y_val).mean()                              # mean absolute error for this fold's validation predictions
        logger.info(f"Fold {cv_ind}: RMSE={rmse_arr[cv_ind]:.4f}, MAE={mae_arr[cv_ind]:.4f}")

        # Save
        cb.save_model(os.path.join(model_dir, f"cb_model_cv{cv_ind}.cbm"))          # CatBoost native binary format; load with CatBoostRegressor().load_model()

        # Permutation importance per fold (on training data, consistent with other pipelines)
        pi = permutation_importance(cb, X_tr, Y_tr, n_repeats=5, random_state=123, scoring="neg_mean_squared_error")  # shuffle each feature 5 times and measure the rise in MSE
        perm_df = pd.DataFrame(data=np.stack([pi.importances_mean, pi.importances_std], axis=0), columns=feature_columns, index=["mean", "std"]).sort_values(axis=1, by="mean", ascending=False)                          # sort features descending by mean importance score
        perm_df.to_csv(os.path.join(results_dir, f"permutation_cv{cv_ind}.csv"))    # save permutation importance table for this fold

    # Loss curve visualisation (averaged over folds)
    plot_catboost_loss_curves(eval_histories, plots_dir, tag=tag)

    # Best-fold OOF diagnostics: permutation importance and SHAP beeswarm for the fold with lowest validation RMSE
    best_fold = int(np.argmin(rmse_arr))                                            # index of fold with lowest OOF validation RMSE
    df_pi_best = pd.read_csv(os.path.join(results_dir, f"permutation_cv{best_fold}.csv"), index_col=0)  # load pre-saved PI for best fold
    plot_permutation_importance_oof(df_pi_best.loc["mean"], df_pi_best.loc["std"], plots_dir, tag=tag, fold=best_fold)  # single PI plot for best fold

    tr_ind_best = cv_ind_list[best_fold][0]                                         # training indices for the best fold
    X_tr_best = X_train_all[tr_ind_best, :]                                         # full training subset for the best fold
    cb_best = CatBoostRegressor()
    cb_best.load_model(os.path.join(model_dir, f"cb_model_cv{best_fold}.cbm"))      # restore best fold's trained CatBoost from disk
    logger.info(f"  Computing SHAP beeswarm on best CV fold (fold {best_fold}, RMSE={rmse_arr[best_fold]:.4f}) training data ...")
    shap_vals_best = compute_shap_values(cb_best, X_tr_best, feature_columns)       # exact Shapley values via TreeExplainer on training subjects
    plot_shap_beeswarm(shap_vals_best, plots_dir, tag=tag, max_display=20, data_source="train", fold=best_fold)  # top-20 beeswarm for best fold
    if len(feature_columns) > 20:                                                   # only add all-features plot when dataset has more than 20 features
        plot_shap_beeswarm(shap_vals_best, plots_dir, tag=tag, max_display=len(feature_columns), data_source="train", fold=best_fold)  # all-features beeswarm

    logger.info("--- Performing Global Age Correction ---")

    # Age correction on combined CV validation predictions
    Y_cv_actual = np.array(all_val_actual)                                          # convert accumulated true ages to a single array
    Y_cv_predict = np.array(all_val_preds)                                          # convert accumulated predicted ages to a single array
    
    # Calculate correction based on all folds combined
    _, correction_term, Y_cv_corrected = correct_age(Y_cv_actual, Y_cv_predict)     # fit and apply linear age-bias correction to CV predictions

    # Save Correction Parameters to JSO
    correction_params = {
        "correction_slope": float(correction_term[0]),                              # slope of the predicted-vs-actual linear fit
        "correction_intercept": float(correction_term[1])                           # intercept of the predicted-vs-actual linear fit
    }
    with open(os.path.join(results_dir, "correction_params.json"), "w") as f:
        json.dump(correction_params, f, indent=4)                                   # save correction coefficients for use at inference time

    # Save the actual, predicted, and corrected ages to a CSV
    pd.DataFrame({
        "ID": all_val_ids,
        "Age_Actual": Y_cv_actual,
        "Age_Predicted": Y_cv_predict,
        "Age_Corrected": Y_cv_corrected
    }).to_csv(os.path.join(results_dir, "cv_final_predictions.csv"), index=False)   # save per-subject actual, predicted, and corrected ages from CV

    # Save Summary
    pd.DataFrame({"CV": range(5), "RMSE": rmse_arr, "MAE": mae_arr}).to_csv(
        os.path.join(results_dir, "cv_performance_summary.csv"), index=False        # save per-fold RMSE and MAE summary table
    )

    # Train final model on ALL training subjects
    mean_best_iter = max(1, int(round(np.mean(best_iters))))                        # average tree count at best val RMSE across the 5 CV folds
    logger.info(f"Training final model on all {len(Y_train_all)} training subjects ...")
    logger.info(f"iterations={mean_best_iter} (mean early-stopping iteration across CV folds)")
    final_params = {k: v for k, v in clean_params.items() if k != 'iterations'}     # exclude iterations; replaced by mean_best_iter
    final_model = CatBoostRegressor(
        loss_function='RMSE',
        random_seed=42,
        thread_count=-1,
        verbose=0,                                                                  # suppress per-round console output
        iterations=mean_best_iter,                                                  # fixed at mean CV best iteration; no early stopping without eval_set
        **final_params                                                              # remaining hyperparameters from CV
    )
    final_model.fit(X_train_all, Y_train_all)                                       # train on all training subjects without a validation split
    final_model_path = os.path.join(model_dir, "final_model.cbm")
    final_model.save_model(final_model_path)
    logger.info(f"Final model saved to {final_model_path}")

    # Evaluate final model on held-out test set
    test_preds = final_model.predict(X_test)                                         # predict ages for the held-out test subjects
    slope = correction_params["correction_slope"]
    intercept = correction_params["correction_intercept"]
    test_preds_corrected = test_preds + (Y_test - (slope * Y_test + intercept))     # apply same linear bias correction used on CV predictions
    test_rmse = ((test_preds - Y_test)**2).mean()**(1/2)                            # root mean square prediction error in years
    test_mae = np.abs(test_preds - Y_test).mean()                                   # mean absolute prediction error in years
    test_r2 = r2_score(Y_test, test_preds)                                          # proportion of age variance explained by the model (1.0 = perfect)

    logger.info("")
    logger.info(f"TEST SET RESULTS [{tag.upper()}]")
    logger.info(f"RMSE: {test_rmse:.4f} years")
    logger.info(f"MAE: {test_mae:.4f} years")
    logger.info(f"R^2: {test_r2:.4f}")
    logger.info("")

    test_metrics = {"RMSE": test_rmse, "MAE": test_mae, "R2": test_r2}            # pack all three test metrics into a dict for return and saving
    with open(os.path.join(results_dir, "test_metrics.json"), "w") as f:
        json.dump(test_metrics, f, indent=4)                                       # persist test metrics as JSON for later comparison

    pd.DataFrame({
        "ID": ID_test,
        "Age_Actual": Y_test,
        "Age_Predicted": test_preds,
        "Age_Corrected": test_preds_corrected,
    }).to_csv(os.path.join(results_dir, "test_predictions.csv"), index=False)      # save per-subject test predictions alongside ground-truth ages

    plot_catboost_importance(final_model, plots_dir, tag=tag, feature_names=feature_columns)  # feature_columns supplies meaningful names; model stores auto-generated '0','1',... when trained on numpy
    plot_permutation_importance(final_model, X_test, Y_test, feature_columns, plots_dir, tag=tag)  # importance on unseen test data
    shap_vals = compute_shap_values(final_model, X_test, feature_columns)         # compute once, reuse for all SHAP plots
    plot_shap_waterfall(shap_vals, plots_dir, tag=tag, sample_idx=0)              # single-subject decomposition: how each feature pushes one prediction
    plot_shap_beeswarm(shap_vals, plots_dir, tag=tag, max_display=20)             # top-20 overview across all test subjects
    if len(feature_columns) > 20:                                                  # only add a second plot when more than 20 features exist
        plot_shap_beeswarm(shap_vals, plots_dir, tag=tag, max_display=len(feature_columns))  # all features

    return final_model, test_metrics

def main():
    parser = argparse.ArgumentParser(description="CatBoost Brain Age: Grid Search & Train")
    parser.add_argument('--input', type=str, required=True, help="Path to input CSV file")
    parser.add_argument('--output_dir', type=str, default="./output", help="Directory to save results")

    # Hyperparameter search ranges
    # CatBoost uses 'iterations' (= n_estimators), 'depth' (= max_depth), and 'l2_leaf_reg'
    # (regularisation). No unlimited-depth sentinel - depth is always a positive integer.
    parser.add_argument('--iterations', type=int, nargs='+', default=[100], help="Values for iterations (number of boosting rounds)")
    parser.add_argument('--learning_rate', type=float, nargs='+', default=[0.03], help="Values for learning_rate (step size shrinkage)")
    parser.add_argument('--depth', type=int, nargs='+', default=[6], help="Values for depth (symmetric tree depth)")
    parser.add_argument('--l2_leaf_reg', type=float, nargs='+', default=[3.0], help="Values for l2_leaf_reg (L2 regularisation on leaf weights)")
    parser.add_argument('--min_data_in_leaf', type=int, nargs='+', default=[1], help="Values for min_data_in_leaf (minimum samples per leaf)")
    parser.add_argument('--early_stopping_rounds', type=int, default=100, help="Rounds without val-RMSE improvement before stopping (applied during training, not grid search)")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)                                    # create the root output directory if it does not yet exist
    t_start = time.time()                                                           # record program start for total duration reporting

    logger.info("Loading data ...")                                                #Load data and perform 90/10 stratified train/test split
    feature_columns, strat_label, X_data, Y_data, ID_data = preproc_data(args.input)  # load CSV, extract features/targets/IDs and stratification labels
    logger.info(f"Total subjects: {len(Y_data)}")
    logger.info(f"Number of features: {len(feature_columns)}")
    logger.info(f"Age range: {Y_data.min():.1f} - {Y_data.max():.1f} years")

    # Stratified 90/10 split on compound Age-bin + Site + Sex label
    (X_train, X_test,
     Y_train, Y_test,
     strat_train, _,
     ID_train, ID_test) = train_test_split(
        X_data, Y_data, strat_label, ID_data,
        test_size=0.10,                                                             # 10% test subjects
        random_state=42,                                                            # fixed seed ensures the same split every run
        stratify=strat_label                                                        # preserve age-bin x site x sex proportions in both splits
    )
    logger.info(f"Train set: {len(Y_train)} subjects (90 %)")
    logger.info(f"Test set: {len(Y_test)} subjects (10 %) - held out until after training")

    # Save test-set IDs so the split is reproducible
    pd.DataFrame({"ID": ID_test, "Age_Actual": Y_test}).to_csv(os.path.join(args.output_dir, "test_set_ids.csv"), index=False) # record which subjects are in the test set for auditability

    DEFAULT_PARAMS = {                                                              #Run 1 - Default CatBoost hyperparameters
        'iterations': 1000,
        'learning_rate': 0.03,
        'depth': 6,
        'l2_leaf_reg': 3.0,
        'min_data_in_leaf': 1,
        'early_stopping_rounds': args.early_stopping_rounds,
    }
    logger.info("RUN 1 - DEFAULT HYPERPARAMETERS")
    logger.info(f"  {DEFAULT_PARAMS}")
    _, default_metrics = train_and_evaluate(
        X_train, Y_train, strat_train, ID_train,
        X_test, Y_test, ID_test,
        feature_columns,
        args.output_dir, DEFAULT_PARAMS, tag="default"
    )

    grid_params = {                                                                 #Grid Search over candidate hyperparameter ranges
        'iterations': args.iterations, # default [100]
        'learning_rate': args.learning_rate, # default [0.03]
        'depth': args.depth, # default [6]
        'l2_leaf_reg': args.l2_leaf_reg, # default [3.0]
        'min_data_in_leaf': args.min_data_in_leaf, # default [1]
    }
    n_combinations = (len(args.iterations) * len(args.learning_rate) * len(args.depth) * len(args.l2_leaf_reg) * len(args.min_data_in_leaf)) # total number of hyperparameter combinations to evaluate
    logger.info("")
    logger.info("3 - GRID SEARCH")
    logger.info(f"Grid: {grid_params}")
    logger.info(f"Total combinations: {n_combinations} x 5 folds = {n_combinations * 5} fits")

    t_gs_start = time.time()                                                        # record grid search start for duration reporting
    best_params, cv_results_df = run_grid_search(X_train, Y_train, strat_train, grid_params)  # search for best params; also returns per-combination CV metrics
    gs_duration = round(time.time() - t_gs_start, 1)                                # seconds elapsed during grid search (stored separately; not passed to estimator)
    cv_results_df.to_csv(os.path.join(args.output_dir, "grid_search_cv_results.csv"), index=False)  # save RMSE, MAE, R^2 for every hyperparameter combination
    best_params['early_stopping_rounds'] = args.early_stopping_rounds

    logger.info("")                                                                 #Run 2 - Best hyperparameters Train with best params and evaluate on test set
    logger.info("RUN 2 - BEST HYPERPARAMETERS")
    logger.info(f"{best_params}")

    _, best_metrics = train_and_evaluate(                                           # train with best params; discard model object, keep test metrics
        X_train, Y_train, strat_train, ID_train,
        X_test, Y_test, ID_test,
        feature_columns,
        args.output_dir, best_params, tag="best"
    )

    logger.info("")                                                                 #Final comparison summary
    logger.info("FINAL COMPARISON - TEST SET")
    logger.info(f"{'Metric':<8}  {'Default':>10}  {'Best':>10}")
    logger.info("")
    logger.info(f"{'RMSE':<8}  {default_metrics['RMSE']:>10.4f}  {best_metrics['RMSE']:>10.4f}  years")
    logger.info(f"{'MAE':<8}  {default_metrics['MAE']:>10.4f}  {best_metrics['MAE']:>10.4f}  years")
    logger.info(f"{'R2':<8}  {default_metrics['R2']:>10.4f}  {best_metrics['R2']:>10.4f}")
    logger.info("")
    logger.info(f"  Final model (best params) stored at: {args.output_dir}/best/models/final_model.cbm")
    pd.DataFrame([{"run": "default", **default_metrics}, {"run": "best", **best_metrics}]).to_csv(os.path.join(args.output_dir, "comparison_test_metrics.csv"), index=False)

    logger.info("")
    logger.info("FINAL RESULTS - TEST SET")
    logger.info(f"{'Metric':<8}  {'Best':>10}")
    logger.info("")
    logger.info(f"{'RMSE':<8}  {best_metrics['RMSE']:>10.4f}  years")
    logger.info(f"{'MAE':<8}  {best_metrics['MAE']:>10.4f}  years")
    logger.info(f"{'R2':<8}  {best_metrics['R2']:>10.4f}")
    logger.info("")
    logger.info(f"Final model (best params) stored at: {args.output_dir}/best/models/final_model.cbm")

    best_params['grid_search_duration_s'] = gs_duration                             # add timing to params dict only now, after training is done
    best_params['total_duration_s'] = round(time.time() - t_start, 1)               # total program runtime in seconds
    logger.info("")
    logger.info(f"Grid search duration: {gs_duration:.1f} s")
    logger.info(f"Total duration: {best_params['total_duration_s']:.1f} s")
    with open(os.path.join(args.output_dir, "best_search_params.json"), "w") as f:
        json.dump(best_params, f, indent=4)                                         # persist best params and timing to JSON

if __name__ == "__main__":
    main()