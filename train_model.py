import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from hyperopt import fmin, tpe, Trials, STATUS_OK


def scale_numeric_columns(dfs: list, target_variable: str) -> list:
    scaler = StandardScaler()
    scaled_dfs = []
    for df in dfs:
        numeric_cols = df.select_dtypes(include=["number"]).columns
        cols_to_scale = [
            col for col in numeric_cols if col not in ["IDfg", target_variable]
        ]  # Scaling all numeric columns except the player ID & target variable
        df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
        scaled_dfs.append(df)
    return scaled_dfs


def split_data(df: pd.DataFrame, test_size=0.2, val_size=0.1, random_state=62820):
    """
    Splits the data into train, validation, and test sets.
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
):
    """
    This function creates a baseline model for the data
    """

    # Initialize the baseline XGBoost Regressor
    model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=1234,
    )

    # Fit the model, drop ID column from datasets prior to fitting
    model.fit(x_train, y_train)

    # # --- Validation set evaluation ---
    # y_val_pred = model.predict(x_val)
    # val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    # val_mae = mean_absolute_error(y_val, y_val_pred)
    # val_r2 = r2_score(y_val, y_val_pred)
    # print(
    #     f"[Test] RMSE: {test_rmse:.3f} | MAE: {test_mae:.3f} | R^2: {test_r2:.3f}"
    # )

    # --- Test set evaluation ---
    y_test_pred = model.predict(x_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    print(f"[Test] RMSE: {test_rmse:.3f} | MAE: {test_mae:.3f} | R^2: {test_r2:.3f}")

    return model, y_test_pred


def tune_xgb(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    space: dict,
    metric: str = "asymmetric",
    alpha: float = 1.5,
    evals: int = 75,
    random_state: int = 62820,
    id_cols: list[str] | None = None,
    max_depth_choices: list[int] | None = None,
):
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
            objective="reg:squarederror",
            learning_rate=float(params["learning_rate"]),
            max_depth=int(params["max_depth"]),
            subsample=float(params["subsample"]),
            colsample_bytree=float(params["colsample_bytree"]),
            min_child_weight=float(params["min_child_weight"]),
            reg_lambda=float(params["reg_lambda"]),
            reg_alpha=float(params["reg_alpha"]),
            gamma=float(params["gamma"]),
            n_estimators=2000,
            random_state=random_state,
            n_jobs=-1,
            tree_method="hist",
            eval_metric="rmse",
            early_stopping_rounds=100,
        )

        model.fit(X_tr, y_train, eval_set=[(X_v, y_val)], verbose=False)
        y_pred = model.predict(X_v)

        # --- compute all metrics for visibility ---
        rmse = float(np.sqrt(mean_squared_error(y_val, y_pred)))
        mae = float(mean_absolute_error(y_val, y_pred))
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
            "asym": asym,
            "best_iteration": getattr(model, "best_iteration", None),
        }

    trials = Trials()
    best = fmin(
        fn=objective, space=space, algo=tpe.suggest, max_evals=evals, trials=trials
    )

    # Map hp.choice back to actual max_depth
    if max_depth_choices is not None:
        best["max_depth"] = int(max_depth_choices[best["max_depth"]])
    else:
        best["max_depth"] = int(best["max_depth"])

    best_params = {
        "learning_rate": float(best["learning_rate"]),
        "max_depth": int(best["max_depth"]),
        "subsample": float(best["subsample"]),
        "colsample_bytree": float(best["colsample_bytree"]),
        "min_child_weight": float(best["min_child_weight"]),
        "reg_lambda": float(best["reg_lambda"]),
        "reg_alpha": float(best["reg_alpha"]),
        "gamma": float(best["gamma"]),
    }

    # print the best trial's metrics
    best_trial = trials.best_trial["result"]
    print("Best Parameters:", best_params)
    print(
        f"[Best trial @ val] optimized={metric} "
        f"| RMSE={best_trial.get('rmse', float('nan')):.3f} "
        f"| MAE={best_trial.get('mae', float('nan')):.3f} "
        f"| ASYM={best_trial.get('asym', float('nan')):.3f}"
    )

    return best_params


# def tune_model(
#     x_train: pd.DataFrame,
#     x_test: pd.DataFrame,
#     y_train: pd.DataFrame,
#     y_test: pd.DataFrame,
#     space: dict,
#     n_estimators_list: list,
#     max_depth_list: list,
#     metric: str,
#     alpha: float = 1.5,  # Under-predictions penalty multiplier,
#     evals: int = 30,
# ):

#     # Define an asymmetric loss function
#     def asymmetric_loss(y_true, y_pred, alpha=1.5):
#         residuals = y_true - y_pred
#         loss = np.where(residuals > 0, alpha * (residuals**2), residuals**2)
#         return np.mean(loss)

#     # Define the objective function for Hyperopt
#     def objective(params):
#         # Initialize the model with current parameters
#         model = XGBRegressor(
#             n_estimators=params["n_estimators"],
#             learning_rate=params["learning_rate"],
#             max_depth=params["max_depth"],
#             subsample=params["subsample"],
#             colsample_bytree=params["colsample_bytree"],
#             random_state=1234,
#         )

#         # Fit the model
#         model.fit(x_train, y_train)

#         # Predict on validation set
#         y_pred = model.predict(x_test)

#         if metric == "rmse":
#             # Calculate RMSE
#             loss = np.sqrt(mean_squared_error(y_test, y_pred))
#             # Return negative RMSE as Hyperopt minimizes the objective
#             return {"loss": loss, "status": STATUS_OK}
#         if metric == "mae":
#             # Calculate MAE
#             loss = mean_absolute_error(y_test, y_pred)
#             # Return negative MAE as Hyperopt minimizes the objective
#             return {"loss": loss, "status": STATUS_OK}
#         if metric == "asymmetric":
#             # Use asymmetric loss function
#             loss = asymmetric_loss(y_test, y_pred, alpha=alpha)
#             return {"loss": loss, "status": STATUS_OK}

#     # Create a Trials object to store results
#     trials = Trials()

#     # Run optimization
#     best_params = fmin(
#         fn=objective,  # Objective function
#         space=space,  # Search space
#         algo=tpe.suggest,  # Tree of Parzen Estimators (TPE) algorithm
#         max_evals=evals,  # Number of iterations
#         trials=trials,  # Store trials for analysis, no fixed rstate to make testing dynamic
#         #        rstate=np.random.default_rng(1234),  # For reproducibility
#     )

#     # Map the final parameters
#     final_params = {
#         "n_estimators": n_estimators_list[best_params["n_estimators"]],
#         "learning_rate": best_params["learning_rate"],
#         "max_depth": max_depth_list[best_params["max_depth"]],
#         "subsample": best_params["subsample"],
#         "colsample_bytree": best_params["colsample_bytree"],
#     }

#     print("Best Parameters:", final_params)

#     return final_params


def create_model(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
    final_params: dict,
    random_state: int = 62820,
    id_cols: list[str] | None = None,
    alpha: float = 1.5,  # <-- same asymmetry parameter
):
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

    model = XGBRegressor(
        objective="reg:squarederror",
        **final_params,
        n_estimators=2000,
        random_state=random_state,
        n_jobs=-1,
        tree_method="hist",
        eval_metric="rmse",
        early_stopping_rounds=50,
    )

    model.fit(X_tr, y_train, eval_set=[(X_v, y_val)], verbose=False)

    if hasattr(model, "best_iteration") and model.best_iteration is not None:
        print(f"Best iteration: {model.best_iteration}")

    # --- Validation metrics (optional but useful) ---
    val_pred = model.predict(X_v)
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    val_mae = mean_absolute_error(y_val, val_pred)
    val_r2 = r2_score(y_val, val_pred)
    val_asym = asymmetric_loss(y_val, val_pred, alpha=alpha)

    print(
        f"[Val] RMSE: {val_rmse:.3f} | MAE: {val_mae:.3f} | R^2: {val_r2:.3f} | ASYM: {val_asym:.3f}"
    )

    # --- Test metrics ---
    test_pred = model.predict(X_te)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    test_mae = mean_absolute_error(y_test, test_pred)
    test_r2 = r2_score(y_test, test_pred)
    test_asym = asymmetric_loss(y_test, test_pred, alpha=alpha)

    print(
        f"[Test] RMSE: {test_rmse:.3f} | MAE: {test_mae:.3f} | R^2: {test_r2:.3f} | ASYM: {test_asym:.3f}"
    )

    return model, test_pred


# def create_model(
#     x_train: pd.DataFrame,
#     x_test: pd.DataFrame,
#     y_train: pd.DataFrame,
#     y_test: pd.DataFrame,
#     final_params: dict,
# ):
#     """
#     This function creates a model using the best training parameters
#     """
#     # Initialize the XGBoost Regressor with best parameters
#     model = XGBRegressor(**final_params, random_state=1234)

#     # Fit the model
#     model.fit(x_train, y_train)

#     # Make predictions
#     y_pred = model.predict(x_test)

#     # Calculate the root mean squared error
#     rmse = np.sqrt(mean_squared_error(y_test, y_pred))
#     print(f"Root Mean Squared Error: {rmse}")

#     # Calculate the mean absolute error
#     mae = mean_absolute_error(y_test, y_pred)
#     print(f"Mean Absolute Error: {mae}")

#     # Calculate the R^2 score
#     r2 = r2_score(y_test, y_pred)
#     print(f"R^2 Score: {r2}")

#     return model, y_pred
