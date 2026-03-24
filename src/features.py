import pandas as pd
import numpy as np


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
    Adds columns indicating years after each player's peak season.

    The peak is defined as the season with the maximum value_col for each player.

    - years_before_peak: years until peak (positive for pre-peak seasons, 0 for peak and after)
    - pct_of_peak_year: current season value as percentage of peak season value (0-100+)

    This separation avoids penalizing young players with negative values and allows
    the model to learn different patterns for rising vs. declining phases.
    pct_of_peak_year captures performance relative to career best.
    """
    df_out = (
        df
        .copy()
        .sort_values([player_col, year_col])
    )

    # Running best-to-date value for each player
    df_out["_peak_value_to_date"] = (
        df_out
        .groupby(player_col, sort=False)[value_col]
        .cummax()
    )

    # Year in which the running max last occurred (ffill within player)
    df_out["_peak_year_to_date"] = (
        df_out[year_col]
        .where(df_out[value_col] == df_out["_peak_value_to_date"])
    )
    df_out["_peak_year_to_date"] = (
        df_out
        .groupby(player_col, sort=False)["_peak_year_to_date"]
        .ffill()
    )

    # years_after_peak: positive when after best-to-date, 0 otherwise
    # (with best-to-date peak, this is always >= 0)
    df_out[output_after_col] = (df_out[year_col] - df_out["_peak_year_to_date"]).astype("int64")

    # pct_of_peak_year: current season as % of best-to-date peak (avoid division by zero)
    df_out["pct_of_peak_year"] = np.where(
        df_out["_peak_value_to_date"].notna() & (df_out["_peak_value_to_date"] > 0),
        (df_out[value_col] / df_out["_peak_value_to_date"]) * 100.0,
        0.0,
    )

    return df_out.drop(columns=["_peak_year_to_date", "_peak_value_to_date"])

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

# Function to add era buckets based on season
def add_era_bucket(df):
    s = df["Season"]
    return df.assign(
        era=np.select(
            [
                s < 2005,
                s < 2010,
                s < 2015,
                s < 2020,
                s < 2030,
            ],
            [
                "early_2000s",
                "mid_2000s",
                "early_2010s",
                "mid_2010s",
                "2020s",
            ],
            default="other",
        )
    )

# Function to create a feature that will denote player years available in aggregate season pulls (e.g., player only has 3 years of data but agg_years=5)
def add_history_coverage(
    df: pd.DataFrame,
    *,
    agg_years: int,
    years_in_league_col: str = "years_in_league",
    prefix: str = "years_covered_prior",
) -> pd.DataFrame:
    """
    Adds:
      - years_covered_prior{agg_years}
      - years_covered_prior{agg_years*2}

    Coverage = min(years_in_league + 1, window)
    """
    w1 = agg_years
    w2 = agg_years * 2

    out = (
        df
        .assign(
            seasons_available=lambda d: pd.to_numeric(d[years_in_league_col], errors="coerce").fillna(0) + 1,
            **{
                f"{prefix}{w1}": lambda d, w=w1: d["seasons_available"].clip(upper=w).astype("Int64"),
                f"{prefix}{w2}": lambda d, w=w2: d["seasons_available"].clip(upper=w).astype("Int64"),
            },
        )
        .drop(columns=["seasons_available"])
    )

    return out
