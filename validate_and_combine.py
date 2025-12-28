import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display


def plot_predictions(test_df: pd.DataFrame, y_test: pd.DataFrame, y_pred: pd):
    """
    This function plots the predictions against the actual values
    """
    test_df["fantasy_points_future"] = y_test
    test_df["fantasy_points_pred"] = y_pred

    sns.regplot(x="fantasy_points_pred", y="fantasy_points_future", data=test_df)
    plt.xlabel("Fantasy Points Prediction")
    plt.ylabel("Fantasy Points Actual")
    plt.title("Predicted vs Actual Player Performance")
    plt.show()


def compile_predictions(
    complete_df: pd.DataFrame, test_df: pd.DataFrame, y_test: pd.DataFrame, y_pred: pd
):
    """
    This function compiles the predictions into a dataframe
    """
    test_df["fantasy_points_pred"] = y_pred
    test_df["fantasy_points_future"] = y_test

    comp_df = complete_df[["IDfg", "Name", "Season"]]

    comp_df2 = test_df.merge(comp_df, on=["IDfg", "Season"], how="left")

    comp_df2["diff"] = abs(
        comp_df2["fantasy_points_future"] - comp_df2["fantasy_points_pred"]
    )

    results = comp_df2[
        [
            "Name",
            "fantasy_points_future",
            "fantasy_points_pred",
            "diff",
            "Season",
            "Age",
            "fantasy_points",
        ]
    ].sort_values(by="fantasy_points_pred", ascending=False)

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
