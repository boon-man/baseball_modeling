import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from hyperopt import fmin, tpe, Trials, STATUS_OK
from tqdm.auto import tqdm
from IPython.display import display


def split_data_random(df: pd.DataFrame, test_size=0.10, val_size=0.10, random_state=62820):
    """
    Splits a DataFrame into training, validation, and test sets.

    The function separates predictors and target, then performs a two-step split:
    first into train+val and test, then splits train+val into train and validation sets.
    The split is stratified by the specified random seed for reproducibility.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing features and the target variable.
    test_size : float, default 0.15
        Proportion of the data to allocate to the test set.
    val_size : float, default 0.05
        Proportion of the data to allocate to the validation set (relative to the full dataset).
    random_state : int, default 62820
        Random seed for reproducibility.

    Returns
    -------
    x_train : pd.DataFrame
        Training predictors.
    x_val : pd.DataFrame
        Validation predictors.
    x_test : pd.DataFrame
        Test predictors.
    y_train : pd.Series
        Training target.
    y_val : pd.Series
        Validation target.
    y_test : pd.Series
        Test target.
    """
    x = df.drop(columns=["Name", "fantasy_points_future"])
    y = df["fantasy_points_future"]

    # First split: train+val and test
    x_temp, x_test, y_temp, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state
    )
    # Second split: train and val
    val_relative_size = val_size / (1 - test_size)
    x_train, x_val, y_train, y_val = train_test_split(
        x_temp, y_temp, test_size=val_relative_size, random_state=random_state
    )

    return x_train, x_val, x_test, y_train, y_val, y_test


def create_baseline(
    x_train: pd.DataFrame,
    x_val: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_val: pd.DataFrame,
    y_test: pd.DataFrame,
) -> tuple[XGBRegressor, np.ndarray]:
    """
    Trains a baseline XGBoost regression model and evaluates its performance.

    This function fits a baseline XGBoost regressor on the training data, then evaluates
    the model on the test set using RMSE, MAE, and R^2 metrics. The function returns the
    trained model and the test set predictions.

    Parameters
    ----------
    x_train : pd.DataFrame
        Training predictors.
    x_val : pd.DataFrame
        Validation predictors (not used in this baseline).
    x_test : pd.DataFrame
        Test predictors.
    y_train : pd.Series
        Training target.
    y_val : pd.Series
        Validation target (not used in this baseline).
    y_test : pd.Series
        Test target.

    Returns
    -------
    model : XGBRegressor
        Trained XGBoost regression model.
    y_test_pred : np.ndarray
        Predictions for the test set.
    """

    # Initialize the baseline XGBoost Regressor
    model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=1234,
        enable_categorical=True,
    )

    # Combining train and val for baseline fitting
    x_train = pd.concat([x_train, x_val], axis=0)
    y_train = pd.concat([y_train, y_val], axis=0)

    # Fit the model, drop ID column from datasets prior to fitting
    model.fit(x_train, y_train)

    # --- Test set evaluation ---
    y_test_pred = model.predict(x_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_spearman = spearmanr(y_test, y_test_pred, nan_policy="omit").correlation

    print(f"[Test] RMSE: {test_rmse:.3f} | MAE: {test_mae:.3f} | R^2: {test_r2:.3f} | Spearman: {test_spearman:.3f}")

    return model, y_test_pred


def tune_xgb(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    space: dict,
    model_objective: str = "reg:absoluteerror",
    metric: str = "rmse",
    alpha: float = 1.5,
    evals: int = 75,
    random_state: int = 62820,
    id_cols: list[str] | None = None,
) -> dict:
    """
    Performs hyperparameter optimization for an XGBoost regressor using Hyperopt.

    This function tunes XGBoost model hyperparameters by minimizing a specified loss metric
    (RMSE, MAE, or an asymmetric loss) on the validation set. The search is performed using
    the Hyperopt library and the Tree of Parzen Estimators (TPE) algorithm. The function
    returns the best set of hyperparameters found.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training predictors.
    X_val : pd.DataFrame
        Validation predictors.
    y_train : pd.Series
        Training target.
    y_val : pd.Series
        Validation target.
    space : dict
        Hyperparameter search space for Hyperopt.
    metric : str, default "asymmetric"
        Metric to optimize ("rmse", "mae", or "asymmetric").
    alpha : float, default 1.5
        Penalty multiplier for under-predictions in the asymmetric loss.
    evals : int, default 75
        Number of Hyperopt evaluations.
    random_state : int, default 62820
        Random seed for reproducibility.
    id_cols : list of str, optional
        Columns to exclude from predictors (e.g., player IDs).

    Returns
    -------
    best_params : dict
        Dictionary of the best hyperparameters found.
    """

    id_cols = id_cols or ["IDfg"]

    def _drop_ids(df: pd.DataFrame) -> pd.DataFrame:
        return df.drop(columns=id_cols, errors="ignore")

    X_tr = _drop_ids(X_train)
    X_v = _drop_ids(X_val)

    def asymmetric_loss(y_true, y_pred, alpha=1.5):
        residuals = y_true - y_pred
        loss = np.where(residuals > 0, alpha * (residuals**2), residuals**2)
        return float(np.mean(loss))

    def objective(params):
        model = XGBRegressor(
            objective=model_objective,
            learning_rate=float(params["learning_rate"]),

            # leaf-based tree growth
            grow_policy="lossguide",
            max_leaves=int(params["max_leaves"]),

            subsample=float(params["subsample"]),
            colsample_bytree=float(params["colsample_bytree"]),
            min_child_weight=float(params["min_child_weight"]),
            reg_lambda=float(params["reg_lambda"]),
            reg_alpha=float(params["reg_alpha"]),
            gamma=float(params["gamma"]),

            enable_categorical=True,
            n_estimators=5000,
            random_state=random_state,
            n_jobs=-1,
            tree_method="hist",
            eval_metric=metric,
            early_stopping_rounds=100,
        )

        model.fit(X_tr, y_train, eval_set=[(X_v, y_val)], verbose=False)
        y_pred = model.predict(X_v)

        # --- compute all metrics for visibility ---
        rmse = float(np.sqrt(mean_squared_error(y_val, y_pred)))
        mae = float(mean_absolute_error(y_val, y_pred))
        r2 = float(r2_score(y_val, y_pred))
        spearman = float(spearmanr(y_val, y_pred, nan_policy="omit").correlation)
        asym = float(asymmetric_loss(y_val, y_pred, alpha=alpha))

        # --- choose the one to optimize ---
        if metric == "rmse":
            loss = rmse
        elif metric == "mae":
            loss = mae
        elif metric == "asymmetric":
            loss = asym
        else:
            raise ValueError(f"Unknown metric: {metric}")

        # Hyperopt will store this in trials; handy for later analysis
        return {
            "loss": loss,
            "status": STATUS_OK,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "spearman": spearman,
            "asym": asym,
            "best_iteration": getattr(model, "best_iteration", None),
        }

    trials = Trials()
    best = fmin(
        fn=objective, space=space, algo=tpe.suggest, max_evals=evals, trials=trials
    )

    best_params = {
        "learning_rate": float(best["learning_rate"]),
        "max_leaves": int(best["max_leaves"]),
        "grow_policy": "lossguide",

        "subsample": float(best["subsample"]),
        "colsample_bytree": float(best["colsample_bytree"]),
        "min_child_weight": float(best["min_child_weight"]),
        "reg_lambda": float(best["reg_lambda"]),
        "reg_alpha": float(best["reg_alpha"]),
        "gamma": float(best["gamma"]),
    }

    # Print out iteration results
    best_iters = [
        t["result"].get("best_iteration")
        for t in trials.trials
        if t["result"].get("best_iteration") is not None
    ]

    if best_iters:
        print(
            f"[Early stopping summary] "
            f"mean={int(np.mean(best_iters))}, "
            f"min={int(np.min(best_iters))}, "
            f"max={int(np.max(best_iters))}"
        )

    # print the best trial's metrics
    best_trial = trials.best_trial["result"]
    best_iteration = best_trial.get("best_iteration")
    print("Best Parameters:", best_params)
    print(
        f"[Best trial @ val] optimized={metric} "
        f"| RMSE={best_trial.get('rmse', float('nan')):.3f} "
        f"| MAE={best_trial.get('mae', float('nan')):.3f} "
        f"| R^2={best_trial.get('r2', float('nan')):.3f} "
        f"| SPEAR={best_trial.get('spearman', float('nan')):.3f} "
    )

    return best_params, best_iteration


def create_model(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
    final_params: dict,
    n_estimators: int,
    model_objective: str = "reg:absoluteerror",
    metric: str = "rmse",
    random_state: int = 62820,
    id_cols: list[str] | None = None,
    alpha: float = 1.5,
) -> tuple[XGBRegressor, np.ndarray]:
    """
    Trains an XGBoost regression model with hyperparameter tuning and evaluates its performance.

    This function fits an XGBoost regressor using the provided training data and hyperparameters,
    evaluates the model on both validation and test sets using RMSE, MAE, R^2, and asymmetric loss metrics,
    and prints the results. The function returns the trained model and test set predictions.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training predictors.
    X_val : pd.DataFrame
        Validation predictors.
    X_test : pd.DataFrame
        Test predictors.
    y_train : pd.Series
        Training target.
    y_val : pd.Series
        Validation target.
    y_test : pd.Series
        Test target.
    final_params : dict
        Dictionary of hyperparameters for the XGBoost regressor.
    random_state : int, default 62820
        Random seed for reproducibility.
    id_cols : list of str, optional
        Columns to exclude from predictors (e.g., player IDs).
    alpha : float, default 1.5
        Penalty multiplier for under-predictions in the asymmetric loss.

    Returns
    -------
    model : XGBRegressor
        Trained XGBoost regression model.
    test_pred : np.ndarray
        Predictions for the test set.
    """

    id_cols = id_cols or ["IDfg"]

    def _drop_ids(df: pd.DataFrame) -> pd.DataFrame:
        return df.drop(columns=id_cols, errors="ignore")

    def asymmetric_loss(y_true, y_pred, alpha=1.5):
        residuals = y_true - y_pred
        loss = np.where(residuals > 0, alpha * (residuals**2), residuals**2)
        return float(np.mean(loss))

    X_tr = _drop_ids(X_train)
    X_v = _drop_ids(X_val)
    X_te = _drop_ids(X_test)

    # Combining train + val for final training
    X_tr = pd.concat([X_tr, X_v], axis=0)
    y_train = pd.concat([y_train, y_val], axis=0)

    # Setting a final_n_estimators slightly higher to account for lack of early stopping in final training
    final_n_estimators = int(n_estimators * 1.15)

    model = XGBRegressor(
        objective=model_objective,
        **final_params,
        enable_categorical=True,
        n_estimators=final_n_estimators,
        random_state=random_state,
        n_jobs=-1,
        tree_method="hist",
        eval_metric=metric,
    )

    model.fit(X_tr, y_train, verbose=False)

    # --- Test metrics ---
    test_pred = model.predict(X_te)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    test_mae = mean_absolute_error(y_test, test_pred)
    test_r2 = r2_score(y_test, test_pred)
    test_spearman = float(spearmanr(y_test, test_pred, nan_policy="omit").correlation)

    print(
        f"[Test] RMSE: {test_rmse:.3f} | MAE: {test_mae:.3f} | R^2: {test_r2:.3f} | SPEAR: {test_spearman:.3f}"
    )

    return model, test_pred

# Function for extracting simulated bootstrap prediction intervals
def generate_prediction_intervals(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_pred: pd.DataFrame,
    *,
    base_params: dict,
    model_objective: str = "reg:absoluteerror",
    metric: str = "mae",
    n_bootstrap: int = 30,
    random_state: int = 62820,
    id_cols: list[str] | None = None,
    n_estimators: int = 5000,
    early_stopping_rounds: int = 50,
) -> pd.DataFrame:
    """
    Estimate prediction intervals via bootstrap-resampled XGBoost models.

    Default strategy:
      - Bootstrap sample at the *player* level (IDfg by default).
      - Use out-of-bag (OOB) *players* each iteration for early stopping.
      - Aggregate predictions across bootstraps to produce percentile intervals.

    Notes:
      - If OOB set is too small in an iteration, falls back to training without early stopping.
    """
    id_cols = id_cols or ["IDfg"]

    def _drop_ids(df: pd.DataFrame) -> pd.DataFrame:
        return df.drop(columns=[c for c in id_cols if c in df.columns], errors="ignore")

    # --- Prepare matrices ---
    X_tr = _drop_ids(X_train).reset_index(drop=True)
    X_p = _drop_ids(X_pred).reset_index(drop=True)

    y_tr_full = y_train.reset_index(drop=True)

    # --- Params ---
    params = {
        "objective": model_objective,
        "eval_metric": metric,
        "tree_method": "hist",
        "enable_categorical": True,
    }
    params.update(base_params or {})

    # --- Out of Bag (OOB) grouping (player-level) ---
    group_col = "IDfg"
    if group_col not in X_train.columns:
        raise ValueError(
            "OOB-by-player default requires 'IDfg' to be present in X_train. "
            "Either include it in X_train or change group_col in the function body."
        )

    group_ids = X_train[group_col].reset_index(drop=True).values
    unique_players = pd.unique(group_ids)

    # Guardrails
    min_oob_rows = 200  # keep early stopping stable; adjust inside function if desired

    preds_list: list[np.ndarray] = []

    for b in tqdm(range(n_bootstrap), desc="Bootstrapping prediction intervals"):
        rng_b = np.random.default_rng(random_state + b)

        # Sample players with replacement, then include all rows for sampled players
        boot_players = rng_b.choice(unique_players, size=len(unique_players), replace=True)
        boot_set = set(boot_players)

        in_bag_mask = np.isin(group_ids, list(boot_set))
        idx_boot = np.where(in_bag_mask)[0]
        idx_oob = np.where(~in_bag_mask)[0]

        X_fit = X_tr.iloc[idx_boot]
        y_fit = y_tr_full.iloc[idx_boot]

        model = XGBRegressor(
            n_estimators=n_estimators,
            random_state=random_state + b,
            early_stopping_rounds=early_stopping_rounds,
            n_jobs=-1,
            **params,
        )

        # Use OOB players for early stopping when we have enough rows
        use_oob = idx_oob.size >= min_oob_rows
        if use_oob:
            X_oob = X_tr.iloc[idx_oob]
            y_oob = y_tr_full.iloc[idx_oob]
            model.fit(X_fit, y_fit, eval_set=[(X_oob, y_oob)], verbose=False)
        else:
            # Fallback: fit without eval_set (early stopping won't activate)
            model.fit(X_fit, y_fit, verbose=False)

        base_preds = model.predict(X_p)

        # --- Add residual noise from OOB predictions to generate player prediction interavals ---
        if use_oob:
            oob_preds = model.predict(X_oob)
            residuals = (y_oob.values - oob_preds)
            residuals = residuals - residuals.mean()  # de-bias
            noise = rng_b.choice(residuals, size=len(base_preds), replace=True)
            preds = base_preds + noise
        else:
            preds = base_preds

        preds_list.append(preds)

    pred_mat = np.vstack(preds_list)  # (n_bootstrap, n_rows_pred)

    out = pd.DataFrame(
        {
            "pred_mean": pred_mat.mean(axis=0),
            "pred_p10": np.percentile(pred_mat, 10, axis=0),
            "pred_p50": np.percentile(pred_mat, 50, axis=0),
            "pred_p90": np.percentile(pred_mat, 90, axis=0),
        },
        index=X_pred.index,
    )

    downside_floor = 0.02 * out["pred_mean"].abs().clip(lower=1.0)  # 2% of mean

    out["pred_width_80"] = out["pred_p90"] - out["pred_p10"]
    out["pred_upside"] = out["pred_p90"] - out["pred_mean"]
    out["pred_downside"] = out["pred_mean"] - out["pred_p10"]
    out["implied_upside"] = out["pred_upside"] / (out["pred_downside"] + downside_floor)

    # Prepend id columns from original X_pred when available
    id_present = [c for c in id_cols if c in X_pred.columns]
    if id_present:
        ids = X_pred[id_present].reset_index(drop=True)
        out = pd.concat([ids, out.reset_index(drop=True)], axis=1)
        out.index = X_pred.index

    return out

def compile_predictions(
    complete_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y_test: pd.Series,
    y_pred,
    *,
    id_col: str = "IDfg",
    name_col: str = "Name",
    season_col: str = "Season",
    actual_col: str = "fantasy_points_future",
) -> pd.DataFrame:
    """
    Compile a results dataframe that aligns with the NBA plotting conventions.

    Output columns (core):
      - Name
      - Season
      - predicted_fantasy_points
      - fantasy_points_future
      - prediction_diff (predicted - actual)
      - absolute_diff (abs(predicted - actual))

    This output plugs directly into:
      - plot_actual_vs_pred_mlb(..., pred_col="predicted_fantasy_points")
      - plot_resid_vs_pred_mlb(..., pred_col="predicted_fantasy_points")
      - plot_resid_hist_mlb(...)
    """

    # Copy so we don't mutate caller dataframes
    test_df_out = test_df.copy()

    # Attach predictions + actuals (aligned naming)
    test_df_out["predicted_fantasy_points"] = y_pred
    test_df_out[actual_col] = y_test

    # Pull minimal metadata from the full dataset for Name/Season join
    meta_df = complete_df[[id_col, name_col, season_col]].drop_duplicates(
        subset=[id_col, season_col]
    )

    # Merge metadata onto test rows
    results = test_df_out.merge(meta_df, on=[id_col, season_col], how="left")

    # Compute diffs
    results["prediction_diff"] = (
        results["predicted_fantasy_points"] - results[actual_col]
    )
    results["absolute_diff"] = results["prediction_diff"].abs()

    # Select output columns (only keep what exists)
    desired_cols = [
        name_col,
        actual_col,
        "predicted_fantasy_points",
        "prediction_diff",
        "absolute_diff",
        season_col,
        "Age",
        "fantasy_points",
    ]
    desired_cols = [c for c in desired_cols if c in results.columns]

    results = results[desired_cols].sort_values(
        by="predicted_fantasy_points", ascending=False
    )

    return results

def combine_projections(
    prediction_df: pd.DataFrame, projection_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Combines the model predictions with the FantasyPros projections.

    Parameters:
    prediction_df (pd.DataFrame): The DataFrame containing the model predictions.
    projection_df (pd.DataFrame): The DataFrame containing the FantasyPros projections.

    Returns:
    pd.DataFrame: The combined DataFrame.
    """
    # Merge the model predictions with the FantasyPros projections
    combined = prediction_df.merge(
        projection_df, on=["first_name", "last_name"], how="outer"
    )

    # Filter to rows where IDfg contains duplicate values (excluding NaN)
    duplicates = combined[
        combined["IDfg"].notna() & combined.duplicated("IDfg", keep=False)
    ]

    if not duplicates.empty:
        print("Duplicate rows found:")
        display(duplicates)
    else:
        print("No duplicate rows found.")

    return combined
