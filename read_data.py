import pandas as pd
from pybaseball import batting_stats, pitching_stats, playerid_reverse_lookup
from helper import (
    calc_fantasy_points_batting,
    calc_fantasy_points_pitching,
    add_suffix_to_columns,
    save_data,
    split_name,
)
import requests
from bs4 import BeautifulSoup


def pull_projections(url: str):
    """
    This function pulls projections from FantasyPros
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


def player_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pulls player data from the pybaseball API and merges it with the provided DataFrame to obtain player rookie seasons.

    Parameters:
    df (pd.DataFrame): The DataFrame to which the player data will be added.

    Returns:
    pd.DataFrame: The DataFrame with the player data added.
    """

    player_ids = df["IDfg"].unique().tolist()
    player_ids = playerid_reverse_lookup(player_ids, key_type="fangraphs").filter(
        items=["key_fangraphs", "mlb_played_first"]
    )
    player_ids = player_ids.rename(
        columns={"key_fangraphs": "IDfg", "mlb_played_first": "rookie_year"}
    )
    df = df.merge(player_ids, on="IDfg", how="left")

    # Add total years in league to the data for each player for context on eligible playing years
    df["years_in_league"] = df["Season"] - df["rookie_year"]
    return df


def validate_covid_impact(season, years):
    """
    This function checks if the year 2020 occurred during the specified number of years leading up to the given season.
    """
    start_year = season - years
    end_year = season
    return 2020 in range(start_year, end_year + 1)


def pull_data(
    start_year: int,
    end_year: int,
    agg_years: int,
    batting_stat_cols: list,
    pitching_stat_cols: list,
    save_results: bool,
) -> tuple:
    """
    Pulls and processes batting and pitching data for the specified years.

    Parameters:
    end_year (int): The end year for the data pull.
    agg_years (int): The number of years to aggregate for prior data.
    batting_stat_cols (list): List of columns to include in the batting data.
    pitching_stat_cols (list): List of columns to include in the pitching data.

    Returns:
    tuple: A tuple containing two DataFrames, one for batting data and one for pitching data.
    """

    # Initialize empty DataFrames
    batting_df = pd.DataFrame()
    pitching_df = pd.DataFrame()

    years = list(range(start_year, end_year + 1))

    for year in years:
        # Creating start and end years for the aggregated data pull of prior player seasons
        end_year_prior = year - 1
        start_year_prior = end_year_prior - agg_years
        end_year_future = year + 1

        # Pulling batting stats
        batting_df_future = batting_stats(
            start_season=end_year_future,  # Selecting a single season for most recent stats
            qual=50,
            split_seasons=True,
        ).filter(items=batting_stat_cols)
        calc_fantasy_points_batting(batting_df_future, "fantasy_points_future")
        # Selecting player ID and fantasy points for future season
        batting_df_future = batting_df_future[["IDfg", "fantasy_points_future"]]

        # Pulling batting stats
        batting_df_current = batting_stats(
            start_season=year,  # Selecting a single season for most recent stats
            qual=50,
            split_seasons=True,
        ).filter(items=batting_stat_cols)
        calc_fantasy_points_batting(batting_df_current, "fantasy_points")

        batting_df_prior = batting_stats(
            start_season=start_year_prior,
            end_season=end_year_prior,
            qual=50,
            split_seasons=False,
        ).filter(items=batting_stat_cols)
        batting_df_prior = batting_df_prior.drop(
            columns=["Name", "Age"]
        )  # Dropping redundant columns for joining
        calc_fantasy_points_batting(batting_df_prior, "fantasy_points_prior")
        batting_df_prior = add_suffix_to_columns(
            batting_df_prior, "_prior", exclude_columns=["IDfg", "fantasy_points_prior"]
        )

        # Combining batting features into single dataframe and replace NaN values with 0
        batting_df_current = batting_df_current.merge(
            batting_df_prior, on="IDfg", how="left"
        ).merge(batting_df_future, on="IDfg", how="left")

        # Pulling pitching stats
        pitching_df_future = pitching_stats(
            start_season=end_year_future,  # Selecting a single season for most recent stats
            qual=20,
            split_seasons=True,
        ).filter(items=pitching_stat_cols)
        calc_fantasy_points_pitching(pitching_df_future, "fantasy_points_future")
        # Selecting player ID and fantasy points for future season
        pitching_df_future = pitching_df_future[["IDfg", "fantasy_points_future"]]

        # Pulling pitching stats
        pitching_df_current = pitching_stats(
            start_season=year,  # Selecting a single season for most recent stats
            qual=20,
            split_seasons=True,
        ).filter(items=pitching_stat_cols)
        calc_fantasy_points_pitching(pitching_df_current, "fantasy_points")

        pitching_df_prior = pitching_stats(
            start_season=start_year_prior,
            end_season=end_year_prior,
            qual=20,
            split_seasons=False,
        ).filter(items=pitching_stat_cols)
        pitching_df_prior = pitching_df_prior.drop(
            columns=["Name", "Age"]
        )  # Dropping redundant columns for joining
        calc_fantasy_points_pitching(pitching_df_prior, "fantasy_points_prior")
        pitching_df_prior = add_suffix_to_columns(
            pitching_df_prior,
            "_prior",
            exclude_columns=["IDfg", "fantasy_points_prior"],
        )

        # Combining pitching features into single dataframe & replace NaN values with 0
        pitching_df_current = pitching_df_current.merge(
            pitching_df_prior, on="IDfg", how="left"
        ).merge(pitching_df_future, on="IDfg", how="left")

        # Append the results to the main DataFrames
        batting_df = pd.concat([batting_df, batting_df_current], ignore_index=True)
        pitching_df = pd.concat([pitching_df, pitching_df_current], ignore_index=True)

    # Add a column to indicate if the season is during the COVID-19 pandemic
    batting_df["covid_season"] = batting_df["Season"] == 2020
    pitching_df["covid_season"] = pitching_df["Season"] == 2020

    # Add a column to indicate if the prior seasons were during the COVID-19 pandemic
    batting_df["covid_impact"] = batting_df["Season"].apply(
        lambda x: validate_covid_impact(x, agg_years)
    )
    pitching_df["covid_impact"] = pitching_df["Season"].apply(
        lambda x: validate_covid_impact(x, agg_years)
    )

    # Add player rookie seasons onto the data, helps with modeling new players vs veterans
    batting_df = player_data(batting_df)
    pitching_df = player_data(pitching_df)

    # Replacing NaN values with 0
    batting_df.fillna(0, inplace=True)
    pitching_df.fillna(0, inplace=True)

    if save_results == True:
        # Save the DataFrames to CSV files
        save_data([batting_df, pitching_df], ["batting_data", "pitching_data"])

    return batting_df, pitching_df


def pull_prediction_data(
    prediction_year: int,
    agg_years: int,
    batting_stat_cols: list,
    pitching_stat_cols: list,
) -> tuple:
    """
    Pulls and processes batting and pitching data for the specified years.

    Parameters:
    end_year (int): The end year for the data pull.
    agg_years (int): The number of years to aggregate for prior data.
    batting_stat_cols (list): List of columns to include in the batting data.
    pitching_stat_cols (list): List of columns to include in the pitching data.

    Returns:
    tuple: A tuple containing two DataFrames, one for batting data and one for pitching data.
    """
    # Initialize empty DataFrames
    batting_df = pd.DataFrame()
    pitching_df = pd.DataFrame()

    # Creating start and end years for the aggregated data pull of prior player seasons
    end_year_prior = prediction_year - 1
    start_year_prior = end_year_prior - agg_years

    # Pulling batting stats
    batting_df_current = batting_stats(
        start_season=prediction_year,  # Selecting a single season for most recent stats
        qual=50,
        split_seasons=True,
    ).filter(items=batting_stat_cols)
    calc_fantasy_points_batting(batting_df_current, "fantasy_points")

    batting_df_prior = batting_stats(
        start_season=start_year_prior,
        end_season=end_year_prior,
        qual=50,
        split_seasons=False,
    ).filter(items=batting_stat_cols)
    batting_df_prior = batting_df_prior.drop(
        columns=["Name", "Age"]
    )  # Dropping redundant columns for joining
    calc_fantasy_points_batting(batting_df_prior, "fantasy_points_prior")
    batting_df_prior = add_suffix_to_columns(
        batting_df_prior, "_prior", exclude_columns=["IDfg", "fantasy_points_prior"]
    )

    # Combining batting features into single dataframe and replace NaN values with 0
    batting_df_current = batting_df_current.merge(
        batting_df_prior, on="IDfg", how="left"
    )

    # Pulling pitching stats
    pitching_df_current = pitching_stats(
        start_season=prediction_year,  # Selecting a single season for most recent stats
        qual=20,
        split_seasons=True,
    ).filter(items=pitching_stat_cols)
    calc_fantasy_points_pitching(pitching_df_current, "fantasy_points")

    pitching_df_prior = pitching_stats(
        start_season=start_year_prior,
        end_season=end_year_prior,
        qual=20,
        split_seasons=False,
    ).filter(items=pitching_stat_cols)
    pitching_df_prior = pitching_df_prior.drop(
        columns=["Name", "Age"]
    )  # Dropping redundant columns for joining
    calc_fantasy_points_pitching(pitching_df_prior, "fantasy_points_prior")
    pitching_df_prior = add_suffix_to_columns(
        pitching_df_prior, "_prior", exclude_columns=["IDfg", "fantasy_points_prior"]
    )

    # Combining pitching features into single dataframe & replace NaN values with 0
    pitching_df_current = pitching_df_current.merge(
        pitching_df_prior, on="IDfg", how="left"
    )

    # Append the results to the main DataFrames
    batting_df = pd.concat([batting_df, batting_df_current], ignore_index=True)
    pitching_df = pd.concat([pitching_df, pitching_df_current], ignore_index=True)

    # Add a column to indicate if the season is during the COVID-19 pandemic
    batting_df["covid_season"] = batting_df["Season"] == 2020
    pitching_df["covid_season"] = pitching_df["Season"] == 2020

    # Add a column to indicate if the prior seasons were during the COVID-19 pandemic
    batting_df["covid_impact"] = batting_df["Season"].apply(
        lambda x: validate_covid_impact(x, agg_years)
    )
    pitching_df["covid_impact"] = pitching_df["Season"].apply(
        lambda x: validate_covid_impact(x, agg_years)
    )

    # Add player rookie seasons onto the data, helps with modeling new players vs veterans
    batting_df = player_data(batting_df)
    pitching_df = player_data(pitching_df)

    # Replacing NaN values with 0
    batting_df.fillna(0, inplace=True)
    pitching_df.fillna(0, inplace=True)

    return batting_df, pitching_df
