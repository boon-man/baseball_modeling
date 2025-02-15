import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
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


def split_data(df: pd.DataFrame):
    """
    This function splits the data into the target variable and the predictors
    """
    # Defining the target variable and the predictors, removing the Name and IDFG from the predictors
    x = df.drop(columns=["Name", "fantasy_points_future"])
    y = df["fantasy_points_future"]

    # Split data into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=1234
    )

    return x_train, x_test, y_train, y_test


def create_baseline(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.DataFrame,
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

    # Make predictions
    y_pred = model.predict(x_test)

    # Calculate the root mean squared error
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Root Mean Squared Error: {rmse}")

    # Calculate the R^2 score
    r2 = r2_score(y_test, y_pred)
    print(f"R^2 Score: {r2}")

    return model, y_pred


def tune_model(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    space: dict,
    n_estimators_list: list,
    max_depth_list: list,
):

    # Define the objective function for Hyperopt
    def objective(params):
        # Initialize the model with current parameters
        model = XGBRegressor(
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            max_depth=params["max_depth"],
            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
            random_state=1234,
        )

        # Fit the model
        model.fit(x_train, y_train)

        # Predict on validation set
        y_pred = model.predict(x_test)

        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        # Return negative RMSE as Hyperopt minimizes the objective
        return {"loss": rmse, "status": STATUS_OK}

    # Create a Trials object to store results
    trials = Trials()

    # Run optimization
    best_params = fmin(
        fn=objective,  # Objective function
        space=space,  # Search space
        algo=tpe.suggest,  # Tree of Parzen Estimators (TPE) algorithm
        max_evals=20,  # Number of iterations
        trials=trials,  # Store trials for analysis
        rstate=np.random.default_rng(1234),  # For reproducibility
    )

    # Map the final parameters
    final_params = {
        "n_estimators": n_estimators_list[best_params["n_estimators"]],
        "learning_rate": best_params["learning_rate"],
        "max_depth": max_depth_list[best_params["max_depth"]],
        "subsample": best_params["subsample"],
        "colsample_bytree": best_params["colsample_bytree"],
    }

    print("Best Parameters:", final_params)

    return final_params


def create_model(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    final_params: dict,
):
    """
    This function creates a model using the best training parameters
    """
    # Initialize the XGBoost Regressor with best parameters
    model = XGBRegressor(**final_params, random_state=1234)

    # Fit the model
    model.fit(x_train, y_train)

    # Make predictions
    y_pred = model.predict(x_test)

    # Calculate the root mean squared error
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Root Mean Squared Error: {rmse}")

    # Calculate the R^2 score
    r2 = r2_score(y_test, y_pred)
    print(f"R^2 Score: {r2}")

    return model, y_pred
