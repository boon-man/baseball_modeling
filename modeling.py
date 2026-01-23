import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from hyperopt import fmin, tpe, Trials, STATUS_OK
from IPython.display import display
import warnings

def calculate_delta(
    df: pd.DataFrame,
    fantasy_points_col: str = 'fantasy_points',
    agg_fantasy_points_col: str = 'fantasy_points_agg',
    agg_years: int = 3,
    core_cols: list | None = None,
    output_col: str = 'fantasy_points_delta'
) -> pd.DataFrame:
    """
    Calculate deltas between current season values and multi-year averages for multiple columns.
    
    This metric reveals whether a player is performing above or below their historical average
    across multiple statistics, useful for identifying breakout/breakdown seasons.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing player season data.
    fantasy_points_col : str, default 'fantasy_points'
        Column name containing current season fantasy points.
    agg_fantasy_points_col : str, default 'fantasy_points_agg'
        Column name containing aggregated fantasy points over multiple years.
    agg_years : int, default 3
        Number of years used in the aggregation (for calculating average).
    core_cols : list, optional
        List of column names to calculate deltas for. If provided, deltas will be calculated
        for these columns in addition to fantasy_points. Column names should not include the
        year suffix (e.g., 'HR' not 'HR3'). The function expects aggregated columns named
        as 'column_name' + str(agg_years).
    output_col : str, default 'fantasy_points_delta'
        Name of the output column for fantasy points delta. For career_cols, output columns
        are named as 'column_name_delta'.
    
    Returns
    -------
    pd.DataFrame
        Input dataframe with new columns containing deltas for fantasy_points and career_cols.
        Delta = Current Season Value - (Aggregated Value / Years)
    
    Notes
    -----
    Positive deltas indicate above-average performance; negative deltas indicate below-average.
    """
    df = df.copy()
    
    # Calculate fantasy points delta
    avg_fantasy_points = df[agg_fantasy_points_col] / agg_years
    df[output_col] = df[fantasy_points_col] - avg_fantasy_points
    
    # Calculate deltas for aggregate columns if provided
    if core_cols:
        for col in core_cols:
            if col == 'IDfg':
                continue
            
            agg_col = f'{col}{agg_years}'
            delta_col = f'{col}_delta'
            
            # Only calculate if aggregated column exists
            if agg_col in df.columns:
                avg_col = df[agg_col] / agg_years
                df[delta_col] = df[col] - avg_col
    
    return df

def calculate_productivity_score(
    df: pd.DataFrame,
    fantasy_points_col: str = "fantasy_points",
    age_col: str = "Age",
    output_col: str = "productivity_score",
    agg_years: int = 3,
) -> pd.DataFrame:
    df = df.copy()

    # Ensure sorting
    df = df.sort_values(["IDfg", "Season"]).reset_index(drop=True)

    # Base productivity
    df["age_squared"] = df[age_col] ** 2
    df[output_col] = df[fantasy_points_col] / df["age_squared"]

    short_w = agg_years
    long_w = agg_years * 2
    # Season-aware rolling function, catching missing seasons (due to injury/relegation/suspension/inactivity)
    def _season_aware(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("Season").copy()

        full_seasons = pd.Index(
            range(int(g["Season"].min()), int(g["Season"].max()) + 1),
            name="Season",
        )

        s = (
            g.set_index("Season")[output_col]
            .reindex(full_seasons)
            .astype(float)
        )

        played = s.notna()

        # missed_prev_season: whether previous season was missed
        prev_played = played.shift(1).fillna(True).astype(bool)   
        missed_prev = (~prev_played).astype("Int64")

        # years_since_last_season: gap since previous played season (consecutive years -> 1)
        played_years = pd.Series(full_seasons, index=full_seasons).where(played)
        prev_played_year = played_years.ffill().shift(1)
        years_since_last = (pd.Series(full_seasons, index=full_seasons) - prev_played_year).astype("Float64")

        # map back to observed seasons
        g["missed_prev_season"] = g["Season"].map(missed_prev).fillna(0).astype("int8")
        g["years_since_last_season"] = g["Season"].map(years_since_last).fillna(0).astype("int16")

        # Rolling means (missing seasons ignored in mean)
        roll_short = s.rolling(window=short_w, min_periods=1).mean()
        roll_long = s.rolling(window=long_w, min_periods=1).mean()

        cov_short = played.rolling(window=short_w, min_periods=1).sum()
        cov_long = played.rolling(window=long_w, min_periods=1).sum()

        trend = s.diff()

        g[f"productivity_{short_w}yr"] = g["Season"].map(roll_short)
        g[f"productivity_{long_w}yr"] = g["Season"].map(roll_long)

        g[f"productivity_covered_{short_w}yr"] = g["Season"].map(cov_short).astype("Int64")
        g[f"productivity_covered_{long_w}yr"] = g["Season"].map(cov_long).astype("Int64")

        g["productivity_trend"] = g["Season"].map(trend)

        g["recent_weight"] = (
            g[f"productivity_covered_{short_w}yr"]
            / g[f"productivity_covered_{long_w}yr"].replace({0: pd.NA})
        )

        return g

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        warnings.simplefilter("ignore", category=UserWarning)

        df = (
            df.groupby("IDfg", group_keys=False)
            .apply(_season_aware)
            .reset_index(drop=True)
        )

    return df.drop(columns=["age_squared"])

# Calculate efficiency statistics regarding fantasy points per game
def add_efficiency_stats(
    df: pd.DataFrame,
    fantasy_points_col: str = 'fantasy_points',
    agg_years: int = 3,
) -> pd.DataFrame:
    """
    Calculate efficiency statistics: fantasy points per game for current year and prior windows.
    
    Creates fantasy_points_pg, fantasy_points_pg_prior{agg_years}, and fantasy_points_pg_prior{agg_years*2} for batters,
    and fantasy_points_per_start, fantasy_points_per_start_prior{agg_years}, and fantasy_points_per_start_prior{agg_years*2} for pitchers.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing player season data with aggregated columns.
    fantasy_points_col : str, default 'fantasy_points'
        Column name containing fantasy points.
    agg_years : int, default 3
        Number of years in the base aggregation window.
    
    Returns
    -------
    pd.DataFrame
        Input dataframe with new efficiency columns added.
    """
    df = df.copy()
    
    w1 = agg_years
    w2 = agg_years * 2
    
    # Current year efficiency stats
    df['fantasy_points_pg'] = df[fantasy_points_col] / df['G'].replace(0, np.nan)
    df['fantasy_points_pg'] = df['fantasy_points_pg'].fillna(0)
    
    # Prior window 1
    fantasy_col_w1 = f'{fantasy_points_col}_prior{w1}'
    games_col_w1 = f'G_prior{w1}'
    
    if fantasy_col_w1 in df.columns and games_col_w1 in df.columns:
        df[f'fantasy_points_pg_prior{w1}'] = df[fantasy_col_w1] / df[games_col_w1].replace(0, np.nan)
        df[f'fantasy_points_pg_prior{w1}'] = df[f'fantasy_points_pg_prior{w1}'].fillna(0)
    
    # Prior window 2
    fantasy_col_w2 = f'{fantasy_points_col}_prior{w2}'
    games_col_w2 = f'G_prior{w2}'
    
    if fantasy_col_w2 in df.columns and games_col_w2 in df.columns:
        df[f'fantasy_points_pg_prior{w2}'] = df[fantasy_col_w2] / df[games_col_w2].replace(0, np.nan)
        df[f'fantasy_points_pg_prior{w2}'] = df[f'fantasy_points_pg_prior{w2}'].fillna(0)
    
    return df

# Adding per-year features to normalize aggregated counting stats to a per-year basis
def add_per_year_features(
    df: pd.DataFrame,
    *,
    agg_years: int,
    sum_cols: list[str],
    coverage_prefix: str = "years_covered_prior",
) -> pd.DataFrame:
    out = df.copy()
    w = agg_years

    coverage_col = f"{coverage_prefix}{w}"

    for c in sum_cols:
        agg_col = f"{c}_prior{w}"
        if agg_col in out.columns and coverage_col in out.columns:
            out[f"{c}_per_year_prior{w}"] = (
                out[agg_col] / out[coverage_col].replace(0, pd.NA)
            )

    return out

def calculate_growths(
    df: pd.DataFrame,
    agg_years: int,
    *,
    add_recent: bool = True,
) -> pd.DataFrame:
    """
    Add growth / trend features for workload-style statistics.

    Growth types added:
    1) Prior-window growth:
       per_year_prior{agg_years} - per_year_prior{2 * agg_years}

    2) Recent-vs-baseline growth (optional):
       current_season_value - per_year_prior{agg_years}

    The function is schema-aware and only creates columns when
    the required inputs exist in the dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Batting or pitching dataframe.
    agg_years : int
        Base aggregation window (e.g. 3).
    add_recent : bool, default True
        Whether to add recent-vs-baseline growth features.

    Returns
    -------
    pd.DataFrame
        DataFrame with growth features added.
    """
    out = df.copy()
    w = agg_years
    w2 = agg_years * 2

    # --- PRIOR vs PRIOR growths (per-year pace changes) ---
    prior_growth_specs = {
        # Batters
        "AB": "AB",
        "G": "G",

        # Pitchers
        "IP": "IP",
        "GS": "GS",
    }

    for stat, base in prior_growth_specs.items():
        c1 = f"{base}_per_year_prior{w}"
        c2 = f"{base}_per_year_prior{w2}"

        if c1 in out.columns and c2 in out.columns:
            out[f"{base}_growth"] = out[c1] - out[c2]

    # --- RECENT vs BASELINE growths (role / workload jumps) ---
    if add_recent:
        recent_growth_specs = {
            # Batters
            "AB": "AB",
            "G": "G",

            # Pitchers
            "IP": "IP",
            "GS": "GS",
        }

        for stat, base in recent_growth_specs.items():
            cur = base
            prior = f"{base}_per_year_prior{w}"

            if cur in out.columns and prior in out.columns:
                out[f"{base}_growth_recent"] = out[cur] - out[prior]

    return out

# Function to add "years since peak" feature to dataset
def calculate_years_since_peak(
    df: pd.DataFrame,
    player_col="IDfg",
    year_col="Season",
    value_col="fantasy_points",
    output_before_col="years_before_peak",
    output_after_col="years_after_peak",
) -> pd.DataFrame:
    """
    Adds columns indicating years before and after each player's peak season.
    
    The peak is defined as the season with the maximum value_col for each player.
    
    - years_before_peak: years until peak (positive for pre-peak seasons, 0 for peak and after)
    - years_after_peak: years since peak (0 for peak and before, positive for post-peak seasons)
    - pct_of_peak_year: current season value as percentage of peak season value (0-100+)
    
    This separation avoids penalizing young players with negative values and allows
    the model to learn different patterns for rising vs. declining phases.
    pct_of_peak_year captures performance relative to career best.
    """
    df = df.copy()
    
    # Get the index of the row with max value per player
    peak_idx = df.groupby(player_col)[value_col].idxmax()
    
    # Pull peak year and peak value from those rows
    peak_years = df.loc[peak_idx, year_col]
    peak_values = df.loc[peak_idx, value_col]
    
    # Map both to every row
    df["peak_year"] = df[player_col].map(peak_years)
    df["peak_value"] = df[player_col].map(peak_values)
    
    # Calculate years relative to peak
    years_diff = df[year_col] - df["peak_year"]
    
    # years_before_peak: positive when before peak, 0 otherwise
    df[output_before_col] = np.where(years_diff < 0, -years_diff, 0)
    
    # years_after_peak: positive when after peak, 0 otherwise
    df[output_after_col] = np.where(years_diff > 0, years_diff, 0)

    # pct_of_peak_year: current season as % of peak (avoid division by zero)
    df["pct_of_peak_year"] = np.where(
        (df["peak_value"].notna()) & (df["peak_value"] > 0),
        (df[value_col] / df["peak_value"]) * 100,
        0.0
    )
    
    return df.drop(columns=["peak_year", "peak_value"])

def calculate_fantasy_points_percentile(
    df: pd.DataFrame,
    season_col: str = "Season",
    fantasy_points_col: str = "fantasy_points",
    output_col: str = "fantasy_points_percentile",
) -> pd.DataFrame:
    """
    Calculate each player's fantasy points percentile within their season cohort.
    
    Percentile is computed relative to all other players in the same season,
    allowing the model to learn how position relative to peers affects future performance.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with player season data.
    player_col : str, default "IDfg"
        Column name for player identifier.
    season_col : str, default "Season"
        Column name for season year.
    fantasy_points_col : str, default "fantasy_points"
        Column name containing fantasy points.
    output_col : str, default "fantasy_points_percentile"
        Name of the output percentile column.
    
    Returns
    -------
    pd.DataFrame
        Input dataframe with new percentile column (0-100 scale).
    """
    df = df.copy()
    
    df[output_col] = (
        df.groupby(season_col)[fantasy_points_col]
        .rank(pct=True, method="average")
        .mul(100)
    )
    
    return df

# Function to add player tier based on recent WAR per season performance
def add_player_tier(
    df: pd.DataFrame,
    *,
    agg_years: int,
    war_col: str = "WAR",
    coverage_prefix: str = "years_covered_prior",
    tier_col: str = "player_tier_recent",
) -> pd.DataFrame:
    """
    Adds a categorical player tier based on WAR per season
    over the most recent prior window (agg_years).
    """
    out = df.copy()

    w = agg_years
    war_sum_col = f"{war_col}_prior{w}"
    coverage_col = f"{coverage_prefix}{w}"

    if war_sum_col not in out.columns or coverage_col not in out.columns:
        raise ValueError(f"Missing required columns: {war_sum_col}, {coverage_col}")

    out["war_per_year_recent"] = np.where(
        out[coverage_col] > 0,
        out[war_sum_col] / out[coverage_col],
        0.0,
    )

    out[tier_col] = pd.cut(
        out["war_per_year_recent"],
        bins=[-np.inf, 1.0, 2.0, 4.0, np.inf],
        labels=["replacement", "avg", "above_avg", "star"],
    ).astype("category")

    return out.drop(columns=["war_per_year_recent"])

# Determines the pitcher role based on games started and innings pitched
def add_pitcher_role_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["starter_rate"] = np.where(out["G"] > 0, out["GS"] / out["G"], 0.0)
    out["ip_per_game"] = np.where(out["G"] > 0, out["IP"] / out["G"], 0.0)

    # Defining starter vs reliever based on starter rate
    out["is_starter"] = (out["starter_rate"] >= 0.5).astype(int)
    out["is_reliever"] = (out["starter_rate"] < 0.5).astype(int)

    return out

def scale_numeric_columns(dfs: list, target_variable: str) -> list:
    """
    Scales numeric columns in a list of DataFrames using standardization.

    This function applies sklearn's StandardScaler to all numeric columns in each DataFrame,
    except for the player ID column ("IDfg") and the specified target variable. The scaling
    is performed independently for each DataFrame.

    Parameters
    ----------
    dfs : list of pd.DataFrame
        List of DataFrames to scale.
    target_variable : str
        Name of the target variable column to exclude from scaling.

    Returns
    -------
    list of pd.DataFrame
        List of DataFrames with scaled numeric columns.
    """
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
    Splits a DataFrame into training, validation, and test sets.

    The function separates predictors and target, then performs a two-step split:
    first into train+val and test, then splits train+val into train and validation sets.
    The split is stratified by the specified random seed for reproducibility.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing features and the target variable.
    test_size : float, default 0.2
        Proportion of the data to allocate to the test set.
    val_size : float, default 0.1
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

    # Fit the model, drop ID column from datasets prior to fitting
    model.fit(x_train, y_train)

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
    max_depth_choices : list of int, optional
        List of possible max_depth values (for mapping Hyperopt choices).

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
            objective="reg:squarederror",
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

    model = XGBRegressor(
        objective="reg:squarederror",
        **final_params,
        enable_categorical=True,
        n_estimators=5000,
        random_state=random_state,
        n_jobs=-1,
        tree_method="hist",
        eval_metric="rmse",
        early_stopping_rounds=100,
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