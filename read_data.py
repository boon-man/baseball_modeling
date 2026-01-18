import pandas as pd
import numpy as np
from pybaseball import batting_stats, pitching_stats, playerid_reverse_lookup
import os
from pathlib import Path
from typing import Callable, Literal, Optional
import requests
from bs4 import BeautifulSoup

from modeling import (calculate_productivity_score, add_per_year_features, calculate_growths, calculate_years_since_peak, 
    add_player_tier, add_pitcher_role_flags)
from helper import (
    calc_fantasy_points_batting,
    calc_fantasy_points_pitching,
    add_suffix_to_columns,
    save_data,
    split_name,
    _add_deltas,
)
from config import AGG_YEARS

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

    # Certifying categorical dtypes
    if "overall_pick_bucket" in out.columns:
        out["overall_pick_bucket"] = out["overall_pick_bucket"].astype("category")
    if "era" in out.columns:
        out["era"] = out["era"].astype("category")
    if "primary_pos" in out.columns:
        out["primary_pos"] = (
            out["primary_pos"]
            .astype("object")
            .fillna("Unknown")
            .astype(str) # primary_pos is an integer format, needs to be string first before category conversion for XGB
            .astype("category")
        )
    if "pos_type" in out.columns:
        out["pos_type"] = (
            out["pos_type"]
            .astype("object")
            .fillna("Unknown")
            .astype("category")
        )
    if "birth_country" in out.columns:
        out["birth_country"] = (
            out["birth_country"]
            .astype("object")
            .fillna("Unknown")
            .astype("category")
        )
    if "player_tier_recent" in out.columns:
        out["player_tier_recent"] = (
        out["player_tier_recent"]
        .astype("object")
        .fillna("Unknown")
        .astype("category")
    )

    return out

# Ensure only numeric columns are filled when cleaning NaNs
def fillna_numeric_only(df: pd.DataFrame, value: float = 0) -> pd.DataFrame:
    num_cols = df.select_dtypes(include=["number"]).columns
    return df.assign(**{c: df[c].fillna(value) for c in num_cols})

def _fetch_player_origin_and_position(mlbam_ids: pd.Series) -> pd.DataFrame:
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
        r = requests.get(
            url,
            params={"personIds": ",".join(map(str, chunk))},
            timeout=30,
        )
        r.raise_for_status()

        for p in r.json().get("people", []):
            pos = p.get("primaryPosition") or {}
            rows.append({
                "mlbam_id": p.get("id"),
                "primary_pos": pos.get("code"),        # e.g. "P", "OF", "1B"
                "pos_type": pos.get("type"),           # "Pitcher" or "Player"
                "birth_country": p.get("birthCountry"),
            })

    return (
        pd.DataFrame(rows)
        .assign(
            mlbam_id=lambda d: pd.to_numeric(d["mlbam_id"], errors="coerce").astype("Int64"),
            primary_pos=lambda d: d["primary_pos"].astype("string"),
            pos_type=lambda d: d["pos_type"].astype("string"),
        )
    )

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

def patch_missing_mlbam_ids(
    df: pd.DataFrame,
    name_col: str = "Name",
    idfg_col: str = "IDfg",
    mlbam_col: str = "mlbam_id",
    timeout: int = 30,
) -> pd.DataFrame:
    out = df.copy()

    missing = out[mlbam_col].isna()
    if not missing.any():
        return out

    # One lookup per unique name among missing IDs
    names = (
        out.loc[missing, name_col]
        .dropna()
        .drop_duplicates()
        .tolist()
    )

    rows = []
    for nm in names:
        r = requests.get(
            "https://statsapi.mlb.com/api/v1/people/search",
            params={"search": nm},
            timeout=timeout,
        )
        r.raise_for_status()
        people = r.json().get("people", [])

        # pick best match; simplest: exact match on fullName
        match = next((p for p in people if (p.get("fullName") or "").lower() == nm.lower()), None)
        if match is None and people:
            match = people[0]  # fallback

        if match:
            rows.append({"Name": nm, "mlbam_id_patch": match.get("id")})

    patch = (
        pd.DataFrame(rows)
        .assign(mlbam_id_patch=lambda d: pd.to_numeric(d["mlbam_id_patch"], errors="coerce").astype("Int64"))
    )

    out = out.merge(patch, on="Name", how="left")
    out[mlbam_col] = out[mlbam_col].fillna(out["mlbam_id_patch"])
    out = out.drop(columns=["mlbam_id_patch"])

    return out


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

    # Ensuring data types are assigned correctly before merging
    df = df.assign(
        mlbam_id=pd.to_numeric(df["mlbam_id"], errors="coerce").astype("Int64")
    )
    draft_cache = draft_cache.assign(
        mlbam_id=pd.to_numeric(draft_cache["mlbam_id"], errors="coerce").astype("Int64"),
        overall_pick_number=pd.to_numeric(draft_cache["overall_pick_number"], errors="coerce").astype("Int64"),
        draft_year=pd.to_numeric(draft_cache["draft_year"], errors="coerce").astype("Int64"),
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
    origin_df = _fetch_player_origin_and_position(out["mlbam_id"])

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
    batting_agg_cols: list,
    pitching_agg_cols: list,
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
    batting_agg_cols : list of str
        List of columns to include for batting aggregates.
    pitching_agg_cols : list of str
        List of columns to include for pitching aggregates.
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

        if year < end_year:
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
            stat_cols=["IDfg", "REW", "wOBA", "wRC+", "OPS", "ISO"], # limiting to key rate stats for batting career
            mode="prior",
            year=year,
            window=career_window_years,
            qual=1,
            suffix="_career",
            fantasy_fn=None,
            fantasy_col=None,
        )
        # Calculate REW per year and dropping raw REW career total
        career_batting["REW_career_per_year"] = (
            career_batting["REW_career"] / career_window_years
        )
        career_batting = career_batting.drop(columns=["REW_career"])

        # Combine batting
        batting_year = (
            batting_current
                .merge(batting_priors, on="IDfg", how="left")
                .merge(career_batting, on="IDfg", how="left")
        )

        # Append future target if pulling training data
        if year < end_year:
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

        if year < end_year:
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
            stat_cols=["IDfg", "REW", "WPA", "FIP", "K-BB%", "SIERA"], # limiting to key rate stats for pitching career
            mode="prior",
            year=year,
            window=career_window_years,
            qual=1,
            suffix="_career",
            fantasy_fn=None,
            fantasy_col=None,
        )
        # Calculate REW per year and dropping raw REW career total
        career_pitching["REW_career_per_year"] = (
            career_pitching["REW_career"] / career_window_years
        )
        career_pitching = career_pitching.drop(columns=["REW_career"])

        # Combine pitching
        pitching_year = (
            pitching_current
                .merge(pitching_priors, on="IDfg", how="left")
                .merge(career_pitching, on="IDfg", how="left")
        )

        # Append future target if pulling training data
        if year < end_year:
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
    # Player metadata and draft info additions & final feature engineering
    # -------------------------
    batting_df = (
        batting_df
        .pipe(_player_history_lookup)
        .pipe(patch_missing_mlbam_ids)
        .pipe(add_overall_pick_features)
        .drop(columns=["mlbam_id"], errors="ignore")
        .pipe(_add_deltas, agg_years=AGG_YEARS, core_cols=batting_agg_cols)
        .pipe(calculate_productivity_score, agg_years=AGG_YEARS)
        .pipe(calculate_years_since_peak)
        .pipe(add_era_bucket)
        .pipe(add_history_coverage, agg_years=AGG_YEARS)
        .pipe(add_per_year_features, agg_years=AGG_YEARS, sum_cols=["G", "AB", "R", "H", "HR", "SB", "BB", "SO", "WAR"],)
        .pipe(calculate_growths, agg_years=AGG_YEARS)
        .pipe(add_player_tier, agg_years=AGG_YEARS)
    )

    pitching_df = (
        pitching_df
        .pipe(_player_history_lookup)
        .pipe(patch_missing_mlbam_ids)
        .pipe(add_overall_pick_features)
        .drop(columns=["mlbam_id"], errors="ignore")
        .pipe(_add_deltas, agg_years=AGG_YEARS, core_cols=pitching_agg_cols)
        .pipe(calculate_productivity_score, agg_years=AGG_YEARS)
        .pipe(calculate_years_since_peak)
        .pipe(add_pitcher_role_flags)
        .pipe(add_era_bucket)
        .pipe(add_history_coverage, agg_years=AGG_YEARS)
        .pipe(add_per_year_features, agg_years=AGG_YEARS, sum_cols=["G", "GS", "IP", "W", "SO", "BB", "HR", "ER", "WAR"],)
        .pipe(calculate_growths, agg_years=AGG_YEARS)
        .pipe(add_player_tier, agg_years=AGG_YEARS)
    )

    save_data(
            dataframes=[batting_df, pitching_df],
            file_names=["batting", "pitching"],
            start_year=start_year,
            end_year=end_year,
    )

    print("Data pull & feature engineering complete.")
    return batting_df, pitching_df
