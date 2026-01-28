import pandas as pd
from datetime import datetime
from modeling import calculate_delta
from typing import Literal
import os

def calc_fantasy_points(df: pd.DataFrame, *, rules: dict[str, float], out_col: str) -> pd.DataFrame:
    df[out_col] = 0.0
    for stat, w in rules.items():
        if stat in df.columns:
            df[out_col] += df[stat].fillna(0) * float(w)
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

def _year_suffix(start_year: int, end_year: int) -> str:
    return f"{start_year}" if start_year == end_year else f"{start_year}_{end_year}"


def save_data(
    *,
    dataframes: list[pd.DataFrame],
    file_names: list[str],
    start_year: int,
    end_year: int,
    data_folder: str = "data",
    fmt: str = "UD",               
    file_type: Literal["parquet", "csv"] = "parquet",
    parquet_engine: str = "pyarrow",
    parquet_compression: str = "zstd",
) -> None:
    """
    Save one or more DataFrames to disk with consistent, intention-revealing filenames.

    Filenames:
      - <file_name>_<start>_<end>[_<fmt>].parquet
      - <file_name>_<year>[_<fmt>].parquet

    Examples:
      batting_2017_2024_UD.parquet
      pitching_2025_DK.parquet
    """
    os.makedirs(data_folder, exist_ok=True)

    year_suffix = _year_suffix(start_year, end_year)
    fmt_suffix = f"_{fmt}" if fmt is not None else ""

    for df, file_name in zip(dataframes, file_names):
        if file_type == "parquet":
            out_path = os.path.join(
                data_folder, f"{file_name}_{year_suffix}{fmt_suffix}.parquet"
            )
            df.to_parquet(
                out_path,
                index=False,
                engine=parquet_engine,
                compression=parquet_compression,
            )
        elif file_type == "csv":
            out_path = os.path.join(
                data_folder, f"{file_name}_{year_suffix}{fmt_suffix}.csv"
            )
            df.to_csv(out_path, index=False)
        else:
            raise ValueError(f"Unsupported file_type: {file_type}")

    print(f"Data saved successfully ({file_type}).")


def load_training_data(
    *,
    year: int | None = None,
    data_folder: str = "data",
    fmt: str = "UD",  
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads the batting_data and pitching_data Parquet files for a given year
    and scoring format.

    Expected filenames:
      - batting_data_<year>[_<fmt>].parquet
      - pitching_data_<year>[_<fmt>].parquet

    Parameters
    ----------
    year : int, optional
        Year to load (defaults to current year).
    data_folder : str, default "data"
        Directory containing the data files.
    fmt : {"UD", "DK"}, optional
        Scoring format suffix.

    Returns
    -------
    batting_df : pd.DataFrame
    pitching_df : pd.DataFrame

    Raises
    ------
    FileNotFoundError
        If one or both parquet files are missing.
    """
    year = year or int(datetime.now().strftime("%Y"))
    fmt_suffix = f"_{fmt}" if fmt is not None else ""

    batting_path = os.path.join(
        data_folder, f"batting_data_{year}{fmt_suffix}.parquet"
    )
    pitching_path = os.path.join(
        data_folder, f"pitching_data_{year}{fmt_suffix}.parquet"
    )

    if not os.path.exists(batting_path):
        raise FileNotFoundError(f"Missing batting file: {batting_path}")

    if not os.path.exists(pitching_path):
        raise FileNotFoundError(f"Missing pitching file: {pitching_path}")

    batting_df = pd.read_parquet(batting_path)
    pitching_df = pd.read_parquet(pitching_path)

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
