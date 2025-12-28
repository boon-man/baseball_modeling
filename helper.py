import pandas as pd
from datetime import datetime
import os


def calc_fantasy_points_batting(df, column_name):
    """
    Adds a fantasy points column to the DataFrame based on the provided calculation.

    Parameters:
    df (pd.DataFrame): The DataFrame to which the column will be added.
    column_name (str): The name of the new column.

    Returns:
    pd.DataFrame: The DataFrame with the new fantasy points column added.
    """
    df[column_name] = (
        (df["1B"] * 3)
        + (df["2B"] * 6)
        + (df["3B"] * 8)
        + (df["HR"] * 10)
        + (df["BB"] * 3)
        + (df["HBP"] * 3)
        + (df["RBI"] * 2)
        + (df["R"] * 2)
        + (df["SB"] * 4)
    )
    return df


def calc_fantasy_points_pitching(df, column_name):
    """
    Adds a fantasy points column to the DataFrame based on the provided calculation.

    Parameters:
    df (pd.DataFrame): The DataFrame to which the column will be added.
    column_name (str): The name of the new column.

    Returns:
    pd.DataFrame: The DataFrame with the new fantasy points column added.
    """
    df[column_name] = (df["W"] * 5) + (df["SO"] * 3) + (df["IP"] * 3) + (df["ER"] * -3)
    return df


def add_suffix_to_columns(df, suffix, exclude_columns):
    """
    Adds a suffix to all columns in the DataFrame except the specified columns.

    Parameters:
    df (pd.DataFrame): The DataFrame whose columns will be renamed.
    suffix (str): The suffix to add to the column names.
    exclude_columns (list): List of column names to exclude from renaming.

    Returns:
    pd.DataFrame: The DataFrame with renamed columns.
    """
    df = df.rename(columns=lambda x: x + suffix if x not in exclude_columns else x)
    return df


def save_data(dataframes, file_names, start_year, end_year):
    """
    Saves multiple DataFrames to the 'data' folder with start and end year appended to each file name.

    Parameters:
    dataframes (list of pd.DataFrame): List of DataFrames to be saved.
    file_names (list of str): List of file names corresponding to each DataFrame.
    start_year (int): Starting year for the data.
    end_year (int): Ending year for the data.

    Returns:
    None
    """
    data_folder = "data"
    os.makedirs(data_folder, exist_ok=True)

    for df, file_name in zip(dataframes, file_names):
        df.to_csv(
            os.path.join(data_folder, f"{file_name}_{start_year}_{end_year}.csv"),
            index=False,
        )
    print(f"Data saved successfully.")


def load_training_data():
    """
    Loads the 'batting_data' and 'pitching_data' files for the current year from the 'data' directory.

    Returns:
    tuple: A tuple containing two DataFrames, one for batting data and one for pitching data.
    """
    current_year = datetime.now().strftime("%Y")
    data_folder = "data"
    batting_data_path = os.path.join(data_folder, f"batting_data_{current_year}.csv")
    pitching_data_path = os.path.join(data_folder, f"pitching_data_{current_year}.csv")

    batting_df = pd.read_csv(batting_data_path)
    pitching_df = pd.read_csv(pitching_data_path)

    return batting_df, pitching_df


def split_name(df, name_column):
    """
    Splits the player name into first and last name columns.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the player names.
    name_column (str): The name of the column containing the player names.

    Returns:
    pd.DataFrame: The DataFrame with the player names split into first and last name columns.
    """
    df[["first_name", "last_name"]] = df[name_column].str.split(" ", n=1, expand=True)

    # Remove anything after an additional space in the last name
    df["last_name"] = df["last_name"].str.split(" ").str[0]

    df["first_name"] = df["first_name"].str.lower()
    df["last_name"] = df["last_name"].str.lower()
    return df


def get_value_before_comma(value):
    if "," in value:
        return value.split(",")[0]
    return value
