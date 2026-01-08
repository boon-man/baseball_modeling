import pandas as pd
from pybaseball import batting_stats, pitching_stats, playerid_reverse_lookup
from helper import (
    calc_fantasy_points_batting,
    calc_fantasy_points_pitching,
    add_suffix_to_columns,
    save_data,
    split_name,
)
import os
from pathlib import Path
from typing import Callable, Literal, Optional, Iterable
import time
import requests
from bs4 import BeautifulSoup

StatsFn = Callable[..., pd.DataFrame]
FantasyFn = Callable[[pd.DataFrame, str], pd.DataFrame]


def pull_projections(url: str):
    """
    Fetches and parses player projection data from the FantasyPros website.

    This function sends an HTTP request to the specified FantasyPros projections URL,
    parses the HTML table containing player projections, and returns the data as a pandas DataFrame.
    The DataFrame includes extracted and cleaned columns for player name, team, and positions,
    with all numeric columns converted to appropriate types.

    Parameters
    ----------
    url : str
        The URL of the FantasyPros projections page to scrape.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing player projections with columns for player name, team, positions,
        and all available projection statistics.
    """
    # Send a request to fetch the webpage
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")

    # Find the table containing player projections
    table = soup.find("table", {"id": "data"})

    # Extract table headers
    headers = [th.text.strip() for th in table.find("thead").find_all("th")]

    # Extract table rows
    rows = []
    for tr in table.find("tbody").find_all("tr"):
        cells = [td.text.strip() for td in tr.find_all("td")]
        rows.append(cells)

    # Convert data into a DataFrame
    df = pd.DataFrame(rows, columns=headers)

    # Convert the rest of the columns to numeric type
    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Extract Player, Team, and Positions
    df["Team"] = (
        df["Player"].str.extract(r"\(([^-]+) -")[0].str.strip()
    )  # Extract team, strings following the "(" and before the "-"
    df["Positions"] = (
        df["Player"].str.extract(r"- (.+)\)")[0].str.strip()
    )  # Extract positions, strings following the "-" and before the ")"
    df["Player"] = df["Player"].str.extract(r"^(.+?) \(")[
        0
    ]  # Extract player name, strings at the beginning of the string and before the "("

    # Splitting the Player name into first name and last name columns joining onto core dataframe
    df = split_name(df, "Player")

    return df

# Casting categorical feature dtypes for modeling, when reading in files from CSV
def cast_feature_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # This one is fine as category (comes from pd.cut labels)
    if "overall_pick_bucket" in out.columns:
        out["overall_pick_bucket"] = out["overall_pick_bucket"].astype("category")

    # Convert to python object first, THEN to category (avoids string[python] categories)
    if "birth_country" in out.columns:
        out["birth_country"] = (
            out["birth_country"]
            .astype("object")          # <--- important
            .fillna("Unknown")
            .astype("category")
        )

    return out

# Ensure only numeric columns are filled when cleaning NaNs
def fillna_numeric_only(df: pd.DataFrame, value: float = 0) -> pd.DataFrame:
    num_cols = df.select_dtypes(include=["number"]).columns
    return df.assign(**{c: df[c].fillna(value) for c in num_cols})

def _fetch_player_origin(mlbam_ids: pd.Series) -> pd.DataFrame:
    ids = (
        mlbam_ids
        .dropna()
        .astype(int)
        .drop_duplicates()
        .tolist()
    )

    rows = []
    url = "https://statsapi.mlb.com/api/v1/people"

    for i in range(0, len(ids), 50):
        chunk = ids[i:i+50]
        r = requests.get(url, params={"personIds": ",".join(map(str, chunk))}, timeout=30)
        r.raise_for_status()

        for p in r.json().get("people", []):
            rows.append({
                "mlbam_id": p.get("id"),
                "birth_country": p.get("birthCountry"),
            })

    return (
        pd.DataFrame(rows)
        .assign(
            mlbam_id=lambda d: pd.to_numeric(d["mlbam_id"], errors="coerce").astype("Int64"),
            birth_country=lambda d: d["birth_country"].astype("string"),
        )
    )

def _fetch_draft_year(year: int) -> pd.DataFrame:
    """
    Pull /api/v1/draft/{year} and return one row per pick.
    """
    url = f"https://statsapi.mlb.com/api/v1/draft/{year}"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    payload = r.json()

    rows = []
    drafts = (payload or {}).get("drafts", {})
    for rnd in drafts.get("rounds", []):
        for pick in rnd.get("picks", []):
            person = pick.get("person") or {}
            team = pick.get("team") or {}
            school = pick.get("school") or {}

            rows.append(
                {
                    "draft_year": year,
                    "mlbam_id": person.get("id"),
                    "pick_round": pick.get("pickRound"),
                    "round_pick_number": pick.get("roundPickNumber"),
                    "overall_pick_number": pick.get("pickNumber"),  # overall pick :contentReference[oaicite:2]{index=2}
                    "team_id": team.get("id"),
                    "team_name": team.get("name"),
                    "draft_type": (pick.get("draftType") or {}).get("code"),
                    "is_drafted": pick.get("isDrafted"),
                    "is_pass": pick.get("isPass"),
                    "signing_bonus": pick.get("signingBonus"),
                    "pick_value": pick.get("pickValue"),
                    "school_name": school.get("name"),
                    "school_class": school.get("schoolClass"),
                }
            )

    return (
        pd.DataFrame(rows)
        .assign(
            mlbam_id=lambda d: pd.to_numeric(d["mlbam_id"], errors="coerce").astype("Int64"),
            overall_pick_number=lambda d: pd.to_numeric(d["overall_pick_number"], errors="coerce").astype("Int64"),
            round_pick_number=lambda d: pd.to_numeric(d["round_pick_number"], errors="coerce").astype("Int64"),
            draft_year=lambda d: pd.to_numeric(d["draft_year"], errors="coerce").astype("Int64"),
        )
    )


def build_or_update_draft_cache(
    year_start: int,
    year_end: int,
    cache_path: str | Path = "data/draft/draft_picks_cache.csv",
) -> pd.DataFrame:
    """
    Builds/updates a cache of draft picks across years [year_start, year_end].
    Only fetches missing years if cache exists.
    """
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        cached = pd.read_csv(cache_path)
        cached["draft_year"] = pd.to_numeric(cached["draft_year"], errors="coerce").astype("Int64")
        existing_years = set(cached["draft_year"].dropna().astype(int).unique().tolist())
    else:
        cached = pd.DataFrame()
        existing_years = set()

    target_years = set(range(year_start, year_end + 1))
    missing_years = sorted(list(target_years - existing_years))

    new_parts = []
    for y in missing_years:
        try:
            new_parts.append(_fetch_draft_year(y))
        except Exception:
            # if a year fails, skip it (you can log/raise if you prefer)
            continue

    if new_parts:
        updated = pd.concat([cached, *new_parts], ignore_index=True) if not cached.empty else pd.concat(new_parts, ignore_index=True)
        updated.to_csv(cache_path, index=False)
        return updated

    return cached


def add_overall_pick_features(
    df: pd.DataFrame,
    year_start: int = 1975,
    cache_path: str | Path = "data/draft/draft_picks_cache.csv",
) -> pd.DataFrame:
    """
    Adds overall pick info to a season-level dataframe via mlbam_id.
    Uses the player's earliest draft year (min draft_year) as canonical.
    """
    draft_cache = build_or_update_draft_cache(
        year_start=year_start,
        year_end=int(df["Season"].max()),
        cache_path=cache_path,
    )

    # Reduce to 1 row per player (earliest draft year, keep that pick)
    draft_one = (
        draft_cache
        .dropna(subset=["mlbam_id", "draft_year", "overall_pick_number"])
        .sort_values(["mlbam_id", "draft_year", "overall_pick_number"])
        .groupby("mlbam_id", as_index=False)
        .head(1)
        .rename(
            columns={
                "draft_year": "draft_year",
                "overall_pick_number": "draft_overall_pick",
                "pick_round": "draft_round",
                "round_pick_number": "draft_round_pick",
                "team_id": "draft_team_id",
                "team_name": "draft_team_name",
            }
        )
        .filter(items=[
            "mlbam_id",
            "draft_year",
            "draft_overall_pick",
        ])
    )

    out = (
        df.merge(draft_one, on="mlbam_id", how="left")
        .assign(
            is_drafted=lambda d: d["draft_overall_pick"].notna().astype(int),
            years_since_draft=lambda d: (d["Season"] - d["draft_year"]).astype("Float64"),
            # modeling-friendly buckets
            overall_pick_bucket=lambda d: pd.cut(
                d["draft_overall_pick"].astype("Float64"),
                bins=[-float("inf"), 10, 50, 100, 200, float("inf")],
                labels=["top10", "11-50", "51-100", "101-200", "200+"],
            ),
        )
    )

    # Step to incorporate player country of origin into the dataset
    origin_df = _fetch_player_origin(out["mlbam_id"])

    out = (
        out
        .merge(origin_df, on="mlbam_id", how="left")
        .assign(
            is_international=lambda d: (
                d["birth_country"].notna()
                & ~d["birth_country"].isin(["USA", "United States", "US"])
            ).astype(int),

            # Undrafted international player flag
            is_intl_undrafted=lambda d: (
                (d["draft_overall_pick"].isna()) & (d["is_international"] == 1)
            ).astype(int),
        )
    )
    return out

def _player_history_lookup(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pulls player data from the pybaseball API and merges it with the provided DataFrame to obtain player rookie seasons.

    Parameters:
    df (pd.DataFrame): The DataFrame to which the player data will be added.

    Returns:
    pd.DataFrame: The DataFrame with the player data added.
    """

    player_ids = (
        df["IDfg"]
        .dropna()
        .unique()
        .tolist()
    )

    id_map = (
        playerid_reverse_lookup(player_ids, key_type="fangraphs")
        .filter(items=["key_fangraphs", "key_mlbam", "mlb_played_first"])
        .rename(
            columns={
                "key_fangraphs": "IDfg",
                "key_mlbam": "mlbam_id",
                "mlb_played_first": "rookie_year",
            }
        )
        .assign(
            rookie_year=lambda d: pd.to_numeric(d["rookie_year"], errors="coerce"),
            mlbam_id=lambda d: pd.to_numeric(d["mlbam_id"], errors="coerce").astype("Int64"),
        )
    )

    df = (
        df.merge(id_map, on="IDfg", how="left")
        .assign(
            years_in_league=lambda d: d["Season"] - d["rookie_year"],
        )
    )

    return df


def _validate_covid_impact(season, years):
    """
    This function checks if the year 2020 occurred during the specified number of years leading up to the given season.
    """
    start_year = season - years
    end_year = season
    return 2020 in range(start_year, end_year + 1)


def _prior_year_bounds(year: int, window: int) -> tuple[int, int]:
    # year=2022, window=3 => 2020-2022
    end_y = year
    start_y = end_y - window + 1
    return start_y, end_y

def _career_year_bounds(end_year: int, career_years_back: int) -> tuple[int, int]:
    # end_year=2023, career_years_back=10 => 2014-2023
    start_y = end_year - career_years_back + 1
    return start_y, end_year

# Helper data pull function for identifying year suffixes for saved filename recognition
def _year_suffix(start_year: int, end_year: int) -> str:
    return f"{start_year}" if start_year == end_year else f"{start_year}_{end_year}"

def pull_agg_stats(
    *,
    stats_fn: StatsFn,
    stat_cols: list[str],
    mode: Literal["career", "prior"],
    player_id_col: str = "IDfg",
    year: int | None = None,  # required for prior
    end_year: int | None = None,  # required for career
    window: int | None = None,  # required for prior
    career_years_back: int = 10,  # used for career
    qual: int = 1,
    suffix: str,
    fantasy_fn: Optional[FantasyFn] = None,
    fantasy_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Pulls and aggregates player statistics over a specified window or career span.

    Depending on the mode, this function retrieves player statistics for either a prior window of seasons
    or a career window, applies optional fantasy point calculations, and appends a suffix to column names
    (excluding player ID and optionally the fantasy column).

    Parameters
    ----------
    stats_fn : Callable[..., pd.DataFrame]
        Function to pull player statistics (e.g., batting_stats or pitching_stats).
    stat_cols : list of str
        List of columns to retain from the stats pull.
    mode : {'career', 'prior'}
        Aggregation mode. 'prior' for a window of previous seasons, 'career' for a career window.
    player_id_col : str, default 'IDfg'
        Name of the player ID column to exclude from suffixing.
    year : int, optional
        The reference year for 'prior' mode (required if mode='prior').
    end_year : int, optional
        The end year for 'career' mode (required if mode='career').
    window : int, optional
        Number of seasons to aggregate for 'prior' mode (required if mode='prior').
    career_years_back : int, default 10
        Number of years to look back for 'career' mode.
    qual : int, default 1
        Minimum qualification threshold for stats pull.
    suffix : str
        Suffix to append to column names (except player ID and optionally the fantasy column).
    fantasy_fn : Callable[[pd.DataFrame, str], pd.DataFrame], optional
        Function to calculate fantasy points and add as a column.
    fantasy_col : str, optional
        Name of the fantasy points column to exclude from suffixing.

    Returns
    -------
    pd.DataFrame
        Aggregated player statistics DataFrame with suffixed columns.
    """
    if mode == "prior":
        if year is None or window is None:
            raise ValueError("mode='prior' requires year and window.")
        start_y, end_y = _prior_year_bounds(year, window)

    if mode == "career":
        if end_year is None:
            raise ValueError("mode='career' requires end_year.")
        start_y, end_y = _career_year_bounds(end_year, career_years_back)

    df = (
        stats_fn(
            start_season=start_y,
            end_season=end_y,
            qual=qual,
            split_seasons=False,
        )
        .filter(items=stat_cols)
        .drop(columns=["Name", "Age"], errors="ignore")
    )

    if fantasy_fn is not None:
        if fantasy_col is None:
            raise ValueError(
                "If fantasy_fn is provided, fantasy_col must be provided too."
            )
        fantasy_fn(df, fantasy_col)

    exclude_cols = [player_id_col]
    if fantasy_col is not None:
        exclude_cols.append(fantasy_col)

    df = add_suffix_to_columns(
        df=df,
        suffix=suffix,
        exclude_columns=exclude_cols,
    )

    return df


def pull_data(
    start_year: int,
    end_year: int,
    agg_years: int,
    batting_stat_cols: list,
    pitching_stat_cols: list,
    batting_career_cols: list,
    pitching_career_cols: list,
    include_future_target: bool,
    career_window_years: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Pulls, processes, and aggregates multi-year batting and pitching data for MLB players.

    This function loads existing processed data from disk if available. Otherwise, it pulls season-by-season
    batting and pitching statistics (current, future, prior windows, and career aggregates) for the specified
    year range, computes fantasy points, merges all relevant features, and adds COVID-19 and player metadata.
    Optionally, the processed data is saved to disk for future use.

    Parameters
    ----------
    start_year : int
        The first season to include in the data pull.
    end_year : int
        The last season to include in the data pull.
    agg_years : int
        The number of prior seasons to aggregate for rolling features.
    batting_stat_cols : list of str
        List of columns to include for batting features.
    pitching_stat_cols : list of str
        List of columns to include for pitching features.
    batting_career_cols : list of str
        List of columns to include for batting career aggregates.
    pitching_career_cols : list of str
        List of columns to include for pitching career aggregates.
    include_future_target : bool
        Whether to include future season fantasy points as target variable.
    career_window_years : int, default 10
        Number of years to use for career aggregates.

    Returns
    -------
    tuple of pd.DataFrame
        Tuple containing (batting_df, pitching_df) with all features and targets for modeling.
    """

    year_suffix = _year_suffix(start_year, end_year)

    batting_path = f"data/batting_{year_suffix}.csv"
    pitching_path = f"data/pitching_{year_suffix}.csv"

    if os.path.exists(batting_path) and os.path.exists(pitching_path):
        batting_df = pd.read_csv(batting_path)
        pitching_df = pd.read_csv(pitching_path)
        print("Loaded existing data files.")
        return batting_df, pitching_df

    years = list(range(start_year, end_year + 1))

    # Recommended: remove Season from aggregate pulls (career/prior)
    batting_agg_cols = [c for c in batting_stat_cols if c != "Season"]
    pitching_agg_cols = [c for c in pitching_stat_cols if c != "Season"]

    batting_all = []
    pitching_all = []

    # Loop through each input year and pull data
    for year in years:
        print(f"Pulling data for year: {year}")

        # =========================
        # Batting pulls
        # =========================

        # Current season (features)
        batting_current = batting_stats(
            start_season=year,
            qual=50,
            split_seasons=True,
        ).filter(items=batting_stat_cols)
        calc_fantasy_points_batting(batting_current, "fantasy_points")

        if include_future_target:
            # Future season (target)
            batting_future = batting_stats(
                start_season=year + 1,
                qual=50,
                split_seasons=True,
            ).filter(items=batting_stat_cols)
            calc_fantasy_points_batting(batting_future, "fantasy_points_future")
            batting_future = batting_future[["IDfg", "fantasy_points_future"]]

        # Prior k
        bat_prior_k = pull_agg_stats(
            stats_fn=batting_stats,
            stat_cols=batting_agg_cols,
            mode="prior",
            year=year,
            window=agg_years,
            qual=50,
            suffix=f"_prior{agg_years}",
            fantasy_fn=calc_fantasy_points_batting,
            fantasy_col=f"fantasy_points_prior{agg_years}",
        )

        # Prior 2k
        bat_prior_2k = pull_agg_stats(
            stats_fn=batting_stats,
            stat_cols=batting_agg_cols,
            mode="prior",
            year=year,
            window=agg_years * 2,
            qual=50,
            suffix=f"_prior{agg_years * 2}",
            fantasy_fn=calc_fantasy_points_batting,
            fantasy_col=f"fantasy_points_prior{agg_years * 2}",
        )

        batting_priors = bat_prior_k.merge(bat_prior_2k, on="IDfg", how="outer")

        # "Career" stats (last 10 years, Fangraphs does not allow more than 10 years in a pull)
        career_batting = pull_agg_stats(
            stats_fn=batting_stats,
            stat_cols=batting_career_cols,
            mode="prior",
            year=year,
            window=career_window_years,
            qual=1,
            suffix="_career",
            fantasy_fn=calc_fantasy_points_batting,
            fantasy_col="fantasy_points_career",
        )

        # Combine batting
        batting_year = (
            batting_current
                .merge(batting_priors, on="IDfg", how="left")
                .merge(career_batting, on="IDfg", how="left")
        )

        # Append future target if pulling training data
        if include_future_target:
            batting_year = batting_year.merge(
                batting_future, on="IDfg", how="left"
        )

        # =========================
        # Pitching pulls
        # =========================

        pitching_current = pitching_stats(
            start_season=year,
            qual=15,
            split_seasons=True,
        ).filter(items=pitching_stat_cols)
        calc_fantasy_points_pitching(pitching_current, "fantasy_points")

        if include_future_target:
            pitching_future = pitching_stats(
                start_season=year + 1,
                qual=15,
                split_seasons=True,
            ).filter(items=pitching_stat_cols)

            calc_fantasy_points_pitching(pitching_future, "fantasy_points_future")
            pitching_future = pitching_future[["IDfg", "fantasy_points_future"]]

        pit_prior_k = pull_agg_stats(
            stats_fn=pitching_stats,
            stat_cols=pitching_agg_cols,
            mode="prior",
            year=year,
            window=agg_years,
            qual=15,
            suffix=f"_prior{agg_years}",
            fantasy_fn=calc_fantasy_points_pitching,
            fantasy_col=f"fantasy_points_prior{agg_years}",
        )

        pit_prior_2k = pull_agg_stats(
            stats_fn=pitching_stats,
            stat_cols=pitching_agg_cols,
            mode="prior",
            year=year,
            window=agg_years * 2,
            qual=15,
            suffix=f"_prior{agg_years * 2}",
            fantasy_fn=calc_fantasy_points_pitching,
            fantasy_col=f"fantasy_points_prior{agg_years * 2}",
        )

        pitching_priors = pit_prior_k.merge(pit_prior_2k, on="IDfg", how="outer")

        # "Career" stats (last 10 years, Fangraphs does not allow more than 10 years in a pull)
        career_pitching = pull_agg_stats(
            stats_fn=pitching_stats,
            stat_cols=pitching_career_cols,
            mode="prior",
            year=year,
            window=career_window_years,
            qual=1,
            suffix="_career",
            fantasy_fn=calc_fantasy_points_pitching,
            fantasy_col="fantasy_points_career",
        )

        # Combine pitching
        pitching_year = (
            pitching_current
                .merge(pitching_priors, on="IDfg", how="left")
                .merge(career_pitching, on="IDfg", how="left")
        )

        # Append future target if pulling training data
        if include_future_target:
            pitching_year = pitching_year.merge(
                pitching_future, on="IDfg", how="left"
        )

        batting_all.append(batting_year)
        pitching_all.append(pitching_year)

    batting_df = pd.concat(batting_all, ignore_index=True)
    pitching_df = pd.concat(pitching_all, ignore_index=True)

    # -------------------------
    # COVID flags (keep both windows explicit)
    # -------------------------
    batting_df["covid_season"] = batting_df["Season"] == 2020
    pitching_df["covid_season"] = pitching_df["Season"] == 2020

    batting_df[f"covid_impact_prior{agg_years}"] = batting_df["Season"].apply(
        lambda x: _validate_covid_impact(x, agg_years)
    )
    pitching_df[f"covid_impact_prior{agg_years}"] = pitching_df["Season"].apply(
        lambda x: _validate_covid_impact(x, agg_years)
    )

    batting_df[f"covid_impact_prior{agg_years * 2}"] = batting_df["Season"].apply(
        lambda x: _validate_covid_impact(x, agg_years * 2)
    )
    pitching_df[f"covid_impact_prior{agg_years * 2}"] = pitching_df["Season"].apply(
        lambda x: _validate_covid_impact(x, agg_years * 2)
    )

    # -------------------------
    # Player metadata + cleanup
    # -------------------------
    batting_df = (
        batting_df
        .pipe(_player_history_lookup)
        .pipe(add_overall_pick_features)
        .drop(columns=["mlbam_id"], errors="ignore")
    )

    pitching_df = (
        pitching_df
        .pipe(_player_history_lookup)
        .pipe(add_overall_pick_features)
        .drop(columns=["mlbam_id"], errors="ignore")
    )

    batting_df = fillna_numeric_only(batting_df, value=0)
    pitching_df = fillna_numeric_only(pitching_df, value=0)

    save_data(
            dataframes=[batting_df, pitching_df],
            file_names=["batting", "pitching"],
            start_year=start_year,
            end_year=end_year,
    )

    print("Data pull complete.")
    return batting_df, pitching_df


# -------------------------
# Wrappers for pulling training & testing data
# -------------------------
def pull_training_data(
    start_year: int,
    end_year: int,  # last feature year (e.g., 2024)
    agg_years: int,
    batting_stat_cols: list,
    pitching_stat_cols: list,
    batting_career_cols: list,
    pitching_career_cols: list,
    career_window_years: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Pull training data with future-season targets included.
    """
    return pull_data(
        start_year=start_year,
        end_year=end_year,
        agg_years=agg_years,
        batting_stat_cols=batting_stat_cols,
        pitching_stat_cols=pitching_stat_cols,
        batting_career_cols=batting_career_cols,
        pitching_career_cols=pitching_career_cols,
        career_window_years=career_window_years,
        include_future_target=True, # Include future target for training data
    )


def pull_prediction_data(
    year: int,
    agg_years: int,
    batting_stat_cols: list,
    pitching_stat_cols: list,
    batting_career_cols: list,
    pitching_career_cols: list,
    career_window_years: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Pull a scoring/prediction dataset (features only; no future target pull).
    """
    return pull_data(
        start_year=year,
        end_year=year,
        agg_years=agg_years,
        batting_stat_cols=batting_stat_cols,
        pitching_stat_cols=pitching_stat_cols,
        batting_career_cols=batting_career_cols,
        pitching_career_cols=pitching_career_cols,
        career_window_years=career_window_years,
        include_future_target=False, # No future target for prediction data
    )

# def pull_prediction_data(
#     prediction_year: int,
#     agg_years: int,
#     batting_stat_cols: list,
#     pitching_stat_cols: list,
# ) -> tuple:
#     """
#     Pulls and processes batting and pitching data for the specified years.

#     Parameters:
#     end_year (int): The end year for the data pull.
#     agg_years (int): The number of years to aggregate for prior data.
#     batting_stat_cols (list): List of columns to include in the batting data.
#     pitching_stat_cols (list): List of columns to include in the pitching data.

#     Returns:
#     tuple: A tuple containing two DataFrames, one for batting data and one for pitching data.
#     """
#     # Initialize empty DataFrames
#     batting_df = pd.DataFrame()
#     pitching_df = pd.DataFrame()

#     # Creating start and end years for the aggregated data pull of prior player seasons
#     end_year_prior = prediction_year - 1
#     start_year_prior = end_year_prior - agg_years

#     # Pulling batting stats
#     batting_df_current = batting_stats(
#         start_season=prediction_year,  # Selecting a single season for most recent stats
#         qual=50,
#         split_seasons=True,
#     ).filter(items=batting_stat_cols)
#     calc_fantasy_points_batting(batting_df_current, "fantasy_points")

#     batting_df_prior = batting_stats(
#         start_season=start_year_prior,
#         end_season=end_year_prior,
#         qual=50,
#         split_seasons=False,
#     ).filter(items=batting_stat_cols)
#     batting_df_prior = batting_df_prior.drop(
#         columns=["Name", "Age"]
#     )  # Dropping redundant columns for joining
#     calc_fantasy_points_batting(batting_df_prior, "fantasy_points_prior")
#     batting_df_prior = add_suffix_to_columns(
#         batting_df_prior, "_prior", exclude_columns=["IDfg", "fantasy_points_prior"]
#     )

#     # Combining batting features into single dataframe and replace NaN values with 0
#     batting_df_current = batting_df_current.merge(
#         batting_df_prior, on="IDfg", how="left"
#     )

#     # Pulling pitching stats
#     pitching_df_current = pitching_stats(
#         start_season=prediction_year,  # Selecting a single season for most recent stats
#         qual=15,
#         split_seasons=True,
#     ).filter(items=pitching_stat_cols)
#     calc_fantasy_points_pitching(pitching_df_current, "fantasy_points")

#     pitching_df_prior = pitching_stats(
#         start_season=start_year_prior,
#         end_season=end_year_prior,
#         qual=15,
#         split_seasons=False,
#     ).filter(items=pitching_stat_cols)
#     pitching_df_prior = pitching_df_prior.drop(
#         columns=["Name", "Age"]
#     )  # Dropping redundant columns for joining
#     calc_fantasy_points_pitching(pitching_df_prior, "fantasy_points_prior")
#     pitching_df_prior = add_suffix_to_columns(
#         pitching_df_prior, "_prior", exclude_columns=["IDfg", "fantasy_points_prior"]
#     )

#     # Combining pitching features into single dataframe & replace NaN values with 0
#     pitching_df_current = pitching_df_current.merge(
#         pitching_df_prior, on="IDfg", how="left"
#     )

#     # Append the results to the main DataFrames
#     batting_df = pd.concat([batting_df, batting_df_current], ignore_index=True)
#     pitching_df = pd.concat([pitching_df, pitching_df_current], ignore_index=True)

#     # Add a column to indicate if the season is during the COVID-19 pandemic
#     batting_df["covid_season"] = batting_df["Season"] == 2020
#     pitching_df["covid_season"] = pitching_df["Season"] == 2020

#     # Add a column to indicate if the prior seasons were during the COVID-19 pandemic
#     batting_df["covid_impact"] = batting_df["Season"].apply(
#         lambda x: _validate_covid_impact(x, agg_years)
#     )
#     pitching_df["covid_impact"] = pitching_df["Season"].apply(
#         lambda x: _validate_covid_impact(x, agg_years)
#     )

#     # Add player rookie seasons onto the data, helps with modeling new players vs veterans
#     batting_df = player_data(batting_df)
#     pitching_df = player_data(pitching_df)

#     # Replacing NaN values with 0
#     batting_df.fillna(0, inplace=True)
#     pitching_df.fillna(0, inplace=True)

#     return batting_df, pitching_df
