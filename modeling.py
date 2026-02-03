import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from hyperopt import fmin, tpe, Trials, STATUS_OK
from tqdm.auto import tqdm
from IPython.display import display

def _add_deltas(df: pd.DataFrame, *, agg_years: int, core_cols: list[str]) -> pd.DataFrame:
    return calculate_delta(
        df,
        fantasy_points_col="fantasy_points",
        agg_fantasy_points_col=f"fantasy_points_prior{agg_years}",
        agg_years=agg_years,
        core_cols=core_cols,
    )
    
def calculate_delta(
    df: pd.DataFrame,
    fantasy_points_col: str = 'fantasy_points',
    agg_fantasy_points_col: str = 'fantasy_points_agg',
    agg_years: int = 3,
    core_cols: list | None = None,
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

    out = df.copy()

    out["fantasy_points_delta"] = out[fantasy_points_col] - (out[agg_fantasy_points_col] / agg_years)

    if core_cols:
        cols = [c for c in core_cols if c != "IDfg"]
        agg_cols = [f"{c}{agg_years}" for c in cols]
        # only keep ones that exist
        pairs = [(c, a) for c, a in zip(cols, agg_cols) if a in out.columns and c in out.columns]

        for c, a in pairs:
            out[f"{c}_delta"] = out[c] - (out[a] / agg_years)

    return out

def calculate_productivity_score(
    df: pd.DataFrame,
    fantasy_points_col: str = "fantasy_points",
    age_col: str = "Age",
    agg_years: int = 3,
) -> pd.DataFrame:
    out = df.copy()

    out["productivity_score"] = out[fantasy_points_col] / (out[age_col] ** 2)

    # assumes already sorted by IDfg, Season
    g = out.groupby("IDfg", sort=False)

    out["missed_prev_season"] = (g["Season"].diff().fillna(1) > 1).astype("int8")
    out["years_since_last_season"] = g["Season"].diff().fillna(1).astype("int16")

    w = agg_years
    w2 = agg_years * 2

    out[f"productivity_{w}yr"] = g["productivity_score"].rolling(w, min_periods=1).mean().reset_index(level=0, drop=True)
    out[f"productivity_{w2}yr"] = g["productivity_score"].rolling(w2, min_periods=1).mean().reset_index(level=0, drop=True)

    # coverage counts (how many seasons observed in window)
    out[f"productivity_covered_{w}yr"] = g["productivity_score"].rolling(w, min_periods=1).count().reset_index(level=0, drop=True).astype("int16")
    out[f"productivity_covered_{w2}yr"] = g["productivity_score"].rolling(w2, min_periods=1).count().reset_index(level=0, drop=True).astype("int16")

    out["productivity_trend"] = g["productivity_score"].diff()

    out["recent_weight"] = (
        out[f"productivity_covered_{w}yr"]
        / out[f"productivity_covered_{w2}yr"].replace({0: pd.NA})
    )

    return out

# Calculate efficiency statistics regarding fantasy points per game
def add_efficiency_stats(
    df: pd.DataFrame,
    fantasy_points_col: str = 'fantasy_points',
    agg_years: int = 3,
    *,
    eps: float = 1e-9,
) -> pd.DataFrame:
    """
    Calculate efficiency statistics: Add efficiency-style features for both batters and pitchers
    
      - fantasy_points_pg (and prior windows if present)
      - pitcher-like: fp_per_ip, fp_per_tbf, pitches_per_ip, pitches_per_tbf (if cols exist)
      - batter-like: PA_approx, fp_per_pa_approx (if cols exist)
      - SB_per_game (+ prior + jump) (if cols exist)
      - role-jump: starter_rate_prior/jump, ip_per_game_prior/jump (if cols exist)
      - damage_index (if cols exist)
      - barrels_per_bip (if cols exist)
      - discipline_index (if cols exist)

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
    out = df.copy()

    w1 = agg_years
    w2 = agg_years * 2

    # -------------------------
    # Base: fantasy points per game (current + priors)
    # -------------------------
    if fantasy_points_col in out.columns and "G" in out.columns:
        out["fantasy_points_pg"] = out[fantasy_points_col] / out["G"].replace(0, np.nan)
        out["fantasy_points_pg"] = out["fantasy_points_pg"].fillna(0)

    # Prior window 1
    fantasy_col_w1 = f"{fantasy_points_col}_prior{w1}"
    games_col_w1 = f"G_prior{w1}"
    if fantasy_col_w1 in out.columns and games_col_w1 in out.columns:
        out[f"fantasy_points_pg_prior{w1}"] = out[fantasy_col_w1] / out[games_col_w1].replace(0, np.nan)
        out[f"fantasy_points_pg_prior{w1}"] = out[f"fantasy_points_pg_prior{w1}"].fillna(0)

    # Prior window 2
    fantasy_col_w2 = f"{fantasy_points_col}_prior{w2}"
    games_col_w2 = f"G_prior{w2}"
    if fantasy_col_w2 in out.columns and games_col_w2 in out.columns:
        out[f"fantasy_points_pg_prior{w2}"] = out[fantasy_col_w2] / out[games_col_w2].replace(0, np.nan)
        out[f"fantasy_points_pg_prior{w2}"] = out[f"fantasy_points_pg_prior{w2}"].fillna(0)

    # -------------------------
    # Pitcher efficiency 
    # -------------------------

    if all(c in out.columns for c in ["G", "GS"]):
        out["starter_rate"] = np.where(out["G"] > 0, out["GS"] / out["G"], 0.0)
        out["ip_per_game"] = np.where(out["G"] > 0, out["IP"] / out["G"], 0.0)

    if fantasy_points_col in out.columns and "IP" in out.columns:
        out["fp_per_ip"] = out[fantasy_points_col] / out["IP"].replace(0, np.nan)
        out["fp_per_ip"] = out["fp_per_ip"].fillna(0)

    if "Pitches" in out.columns and "IP" in out.columns:
        out["pitches_per_ip"] = out["Pitches"] / out["IP"].replace(0, np.nan)
        out["pitches_per_ip"] = out["pitches_per_ip"].fillna(0)

        # Role jump features
    if "starter_rate" in out.columns:
        gs_prior = f"GS_prior{w1}"
        g_prior = f"G_prior{w1}"
        if gs_prior in out.columns and g_prior in out.columns:
            out["starter_rate_prior"] = out[gs_prior] / out[g_prior].replace(0, np.nan)
            out["starter_rate_prior"] = out["starter_rate_prior"].fillna(0)
            out["starter_rate_jump"] = out["starter_rate"] - out["starter_rate_prior"]

    if "ip_per_game" in out.columns:
        ip_prior = f"IP_prior{w1}"
        g_prior = f"G_prior{w1}"
        if ip_prior in out.columns and g_prior in out.columns:
            out["ip_per_game_prior"] = out[ip_prior] / out[g_prior].replace(0, np.nan)
            out["ip_per_game_prior"] = out["ip_per_game_prior"].fillna(0)
            out["ip_per_game_jump"] = out["ip_per_game"] - out["ip_per_game_prior"]

    # Simple pitcher "damage index" proxy
    # (Barrel% + Hard% + HR/FB) — only if all present
    if all(c in out.columns for c in ["Barrel%", "Hard%", "HR/FB"]):
        out["damage_index"] = out["Barrel%"] + out["Hard%"] + out["HR/FB"]

    # -------------------------
    # Batter efficiency (if cols exist)
    # -------------------------
    # PA proxy + FP/PA proxy
    if all(c in out.columns for c in ["AB", "BB", "HBP"]):
        out["PA_approx"] = out["AB"] + out["BB"] + out["HBP"]
        if fantasy_points_col in out.columns:
            out["fp_per_pa_approx"] = out[fantasy_points_col] / out["PA_approx"].replace(0, np.nan)
            out["fp_per_pa_approx"] = out["fp_per_pa_approx"].fillna(0)

    # SB intent: SB/G current + prior + jump
    if "SB" in out.columns and "G" in out.columns:
        out["SB_per_game"] = out["SB"] / out["G"].replace(0, np.nan)
        out["SB_per_game"] = out["SB_per_game"].fillna(0)

        sb_prior = f"SB_prior{w1}"
        g_prior = f"G_prior{w1}"
        if sb_prior in out.columns and g_prior in out.columns:
            out["SB_per_game_prior"] = out[sb_prior] / out[g_prior].replace(0, np.nan)
            out["SB_per_game_prior"] = out["SB_per_game_prior"].fillna(0)
            out["SB_jump_vs_prior"] = out["SB_per_game"] - out["SB_per_game_prior"]

    # Barrels per BIP proxy: Barrels / (GB + FB)
    if all(c in out.columns for c in ["Barrels", "GB", "FB"]):
        denom = (out["GB"] + out["FB"]).replace(0, np.nan)
        out["barrels_per_bip"] = out["Barrels"] / denom
        out["barrels_per_bip"] = out["barrels_per_bip"].fillna(0)

    # Discipline index: (BB/K * Contact%) / (SwStr% + eps)
    if all(c in out.columns for c in ["BB/K", "Contact%", "SwStr%"]):
        out["discipline_index"] = (out["BB/K"] * out["Contact%"]) / (out["SwStr%"] + eps)
        out["discipline_index"] = out["discipline_index"].replace([np.inf, -np.inf], 0).fillna(0)
    
    return out

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

def split_data(
    df: pd.DataFrame,
    *,
    season_col: str = "Season",
    target_col: str = "fantasy_points_future",
    drop_cols: list[str] | None = None,
    test_season_frac: float = 0.10,   
    val_frac: float = 0.10,  
    random_state: int = 62820,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Mixed split:
      - Test = most recent `test_season_frac` of unique seasons (time-based)
      - Train/Val = random split within remaining seasons

    Default val_frac_within_trainval is set so total proportions are ~80/5/15.
    """
    drop_cols = drop_cols or ["Name"]

    seasons = np.array(sorted(df[season_col].dropna().astype(int).unique()))
    n_seasons = len(seasons)

    n_test = max(1, int(round(n_seasons * test_season_frac)))
    test_seasons = seasons[-n_test:]

    df_test = df[df[season_col].isin(test_seasons)].copy()
    df_trainval = df[~df[season_col].isin(test_seasons)].copy()

    X = df_trainval.drop(columns=[*drop_cols, target_col], errors="ignore")
    y = df_trainval[target_col]

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=val_frac,
        random_state=random_state,
    )

    X_test = df_test.drop(columns=[*drop_cols, target_col], errors="ignore")
    y_test = df_test[target_col]

    return X_train, X_val, X_test, y_train, y_val, y_test

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
            early_stopping_rounds=150,
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