import pandas as pd
import numpy as np
from plotnine import (
    ggplot,
    aes,
    geom_point,
    geom_smooth,
    geom_abline,
    geom_segment,
    geom_text,
    geom_histogram,
    geom_hline,
    geom_line,
    annotate,
    labs,
    scale_x_continuous,
    scale_y_continuous,
    scale_y_reverse,
    theme_classic,
    theme,
    element_text,
    element_blank,
    element_rect,
    element_line,
)
from config import MLB_COLOR_PALETTE

def theme_mlb():
    """
    Custom plotnine theme for MLB:
    - clean/classic feel
    - slightly bolder titles
    - faint gridlines for readability
    """
    return theme_classic() + theme(
        text=element_text(family="serif"),
        plot_title=element_text(size=20, weight="bold"),
        axis_title=element_text(size=16),
        axis_text=element_text(size=14),
        legend_title=element_text(size=12),
        legend_text=element_text(size=11),
        panel_border=element_blank(),
        panel_background=element_rect(fill="white", color=None),
        panel_grid_major=element_line(color="#e6e6e6", size=0.35),
        panel_grid_minor=element_line(color="#f2f2f2", size=0.25),
        figure_size=(10, 6),
    )

def _ensure_prediction_diff(
    df: pd.DataFrame,
    *,
    actual_col: str,
    pred_col: str,
    diff_col: str,
) -> pd.DataFrame:
    """
    Ensure a signed residual column exists in the results dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Results dataframe.
    actual_col : str
        Column containing actual values.
    pred_col : str
        Column containing predicted values.
    diff_col : str
        Desired column name for signed residuals, defined as (pred - actual).

    Returns
    -------
    pd.DataFrame
        Copy of df with diff_col present.
    """
    out = df.copy()
    if diff_col not in out.columns:
        out[diff_col] = out[pred_col] - out[actual_col]
    return out


def _build_outlier_labels(
    df: pd.DataFrame,
    *,
    name_col: str,
    season_col: str,
    add_one_to_season: bool = True,
) -> pd.Series:
    """
    Construct formatted outlier labels of the form: '<Name> (<Season>)'.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing name and season columns.
    name_col : str
        Player name column.
    season_col : str
        S    Returns
    -------
    pd.Series
        String labels aligned with df rows.
    """
    season = df[season_col]
    if add_one_to_season:
        season = season + 1
    return df[name_col].astype(str) + " (" + season.astype(int).astype(str) + ")"

def plot_actual_vs_pred_mlb(
    results: pd.DataFrame,
    color_palette: list[str] = MLB_COLOR_PALETTE,
    *,
    actual_col: str = "fantasy_points_future",
    pred_col: str = "predicted_fantasy_points",
    diff_col: str = "prediction_diff",
    name_col: str = "Name",
    season_col: str = "Season",
    top_n: int = 15,
    x_offset: float = 15,
    y_offset: float = 40,
    add_one_to_season_in_label: bool = True,
):
    """
    Generate a calibration plot comparing actual vs. predicted fantasy points.

    This visualization includes a scatter plot of predictions against actuals with
    diagnostic overlays to assess model performance and identify systematic biases.
    Labeled outliers highlight the largest prediction errors by absolute residual.

    Parameters
    ----------
    results : pd.DataFrame
        Results dataframe containing prediction and actual outcome columns.
    color_palette : list[str], optional
        List of hex color codes for plot styling. Default uses MLB_COLOR_PALETTE.
    actual_col : str, default "fantasy_points_future"
        Column name containing actual fantasy point outcomes.
    pred_col : str, default "predicted_fantasy_points"
        Column name containing model predictions.
    diff_col : str, default "prediction_diff"
        Column name for prediction residuals (predicted - actual). Computed if absent.
    name_col : str, default "Name"
        Column name containing player display names.
    season_col : str, default "Season"
        Column name containing season identifiers.
    top_n : int, default 15
        Number of largest absolute residuals to label on the plot.
    x_offset : float, default 15
        Horizontal offset for outlier label positioning.
    y_offset : float, default 40
        Vertical offset for outlier label positioning.
    add_one_to_season_in_label : bool, default True
        If True, displays (Season + 1) in outlier labels. Useful when the target
        represents the following season.

    Returns
    -------
    plotnine.ggplot
        A plotnine ggplot object with the following visual elements:
        - Scatter points representing each prediction-actual pair
        - Linear regression fit line with 99% confidence band
        - 45° reference line indicating perfect predictions
        - Callouts for top-N outliers by residual magnitude
        - Region annotations labeling "Overperformers" and "Underperformers"

    Notes
    -----
    Residuals are defined as (predicted - actual). Points above the 45° line
    indicate overpredictions; points below indicate underpredictions.
    """
    df = _ensure_prediction_diff(
        results,
        actual_col=actual_col,
        pred_col=pred_col,
        diff_col=diff_col,
    )

    # ------------------------------
    # Dynamic annotation positions
    # ------------------------------
    max_x = df[pred_col].max()
    max_y = df[actual_col].max()

    # Overperformers: left / upper region
    over_x = 0.3 * max_x
    over_y = 0.8 * max_y

    # Underperformers: right / lower region
    under_x = 0.8 * max_x
    under_y = 0.1 * max_y

    # ------------------------------
    # Top outliers for labeling
    # ------------------------------
    top_outliers = (
        df.reindex(df[diff_col].abs().sort_values(ascending=False).index)
        .head(top_n)
        .copy()
    )

    top_outliers["label"] = _build_outlier_labels(
        top_outliers,
        name_col=name_col,
        season_col=season_col,
        add_one_to_season=add_one_to_season_in_label,
    )
    top_outliers["label_x"] = top_outliers[pred_col] + x_offset
    top_outliers["label_y"] = top_outliers[actual_col] + y_offset

    p = (
        ggplot(df, aes(x=pred_col, y=actual_col))

        # Scatter points (primary visual)
        + geom_point(alpha=0.8, size=2, color=color_palette[0])

        # Fitted line + confidence interval (lighter blue, de-emphasized)
        + geom_smooth(
            method="lm",
            se=True,
            level=0.99,
            color="#4A79B8",
            fill="#AFC7E8",
            alpha=0.18,
            size=0.4,
        )

        # Perfect-prediction reference line
        + geom_abline(
            slope=1,
            intercept=0,
            linetype="dashed",
            color="grey",
            alpha=0.30,
        )

        # Outlier callouts
        + geom_segment(
            top_outliers,
            aes(x=pred_col, y=actual_col, xend="label_x", yend="label_y"),
            color="darkgrey",
            size=0.3,
            alpha=0.85,
        )
        + geom_text(
            top_outliers,
            aes(x="label_x", y="label_y", label="label"),
            size=7,
            ha="left",
            va="bottom",
            fontstyle="italic",
        )

        # Region annotations (now dynamic)
        + annotate(
            "text",
            x=under_x,
            y=under_y,
            label="Underperformers",
            size=14,
            ha="center",
            va="bottom",
            color="black",
            alpha=0.7,
            fontweight="bold",
        )
        + annotate(
            "text",
            x=over_x,
            y=over_y,
            label="Overperformers",
            size=14,
            ha="center",
            va="top",
            color="black",
            alpha=0.7,
            fontweight="bold",
        )

        # Labels + scales + theme
        + labs(
            title="Actual vs Predicted Fantasy Points",
            x="Predicted Fantasy Points",
            y="Actual Fantasy Points (Future Season)",
        )
        + scale_x_continuous(expand=(0.12, 0))
        + scale_y_continuous(limits=(0, None))
        + theme_mlb()
        + theme(figure_size=(12, 10))
    )

    return p

def plot_resid_vs_pred_mlb(
    results: pd.DataFrame,
    color_palette: list[str] = MLB_COLOR_PALETTE,
    *,
    actual_col: str = "fantasy_points_future",
    pred_col: str = "predicted_fantasy_points",
    diff_col: str = "prediction_diff",
    name_col: str = "Name",
    season_col: str = "Season",
    top_n: int = 25,
    x_offset: float = 25,
    y_offset: float = 25,
    band: float = 300,
    add_one_to_season_in_label: bool = True,
    reverse_y: bool = True,
):
    """
    Plot residuals vs predicted values, labeling the largest outliers.

    Residual definition:
        prediction_diff = predicted - actual

    This diagnostic is useful for:
    - checking heteroskedasticity (error magnitude changes by predicted score)
    - identifying systematic bias (residuals drifting above/below zero)
    - quickly spotting extreme misses and their location in prediction space

    Parameters
    ----------
    results : pd.DataFrame
        DataFrame containing prediction results. Must include:
        - `actual_col` (actual values)
        - `pred_col` (predicted values)
        - `name_col` (player display name)
        - `season_col` (season identifier)
        Optionally includes `diff_col`. If absent, it will be computed.
    color_palette : list[str], default MLB_COLOR_PALETTE
        List of hex color codes used for plot styling.
    actual_col : str, default "fantasy_points_future"
        Column name containing actual target values.
    pred_col : str, default "predicted_fantasy_points"
        Column name containing model predictions.
    diff_col : str, default "prediction_diff"
        Column name for signed residuals (predicted - actual).
    name_col : str, default "Name"
        Column name containing player names.
    season_col : str, default "Season"
        Column name containing the season/year.
    top_n : int, default 25
        Number of highest absolute residuals to label.
    x_offset : float, default 25
        Horizontal label offset applied to outlier labels.
    y_offset : float, default 25
        Vertical label offset applied to outlier labels. Sign is automatically
        adjusted to nudge labels away from the point (based on residual sign).
    band : float, default 300
        Symmetric residual band (+/- band) shaded in the background to provide a
        quick view of "typical" prediction errors.
    add_one_to_season_in_label : bool, default False
        If True, labels display (Season + 1). Helpful when the target is the
        following season and you want labels to reflect that year.
    reverse_y : bool, default True
        If True, reverses the y-axis (mirrors the NBA plot convention you used).
        If you prefer standard residual orientation, set to False.

    Returns
    -------
    plotnine.ggplot
        A plotnine ggplot object.
    """
    df = _ensure_prediction_diff(
        results,
        actual_col=actual_col,
        pred_col=pred_col,
        diff_col=diff_col,
    )

    top_outliers = (
        df.reindex(df[diff_col].abs().sort_values(ascending=False).index)
        .head(top_n)
        .copy()
    )

    top_outliers["label"] = _build_outlier_labels(
        top_outliers,
        name_col=name_col,
        season_col=season_col,
        add_one_to_season=add_one_to_season_in_label,
    )
    top_outliers["label_x"] = top_outliers[pred_col] + x_offset
    top_outliers["label_y"] = top_outliers[diff_col] + (
        y_offset * np.sign(top_outliers[diff_col])
    )

    p = (
        ggplot(df, aes(x=pred_col, y=diff_col))
        + labs(
            title="Residuals vs Predicted Fantasy Points",
            x="Predicted Fantasy Points",
            y="Prediction Diff (Predicted - Actual)",
        )
        + geom_point(alpha=0.8, size=1.5, color=color_palette[1])
        + geom_hline(yintercept=0, linetype="dashed", color="grey", alpha=0.7)
        + annotate(
            "rect",
            xmin=-np.inf,
            xmax=np.inf,
            ymin=-band,
            ymax=band,
            alpha=0.18,
            fill="lightgrey",
        )
        + geom_segment(
            top_outliers,
            aes(x=pred_col, y=diff_col, xend="label_x", yend="label_y"),
            color="darkgrey",
            size=0.3,
            alpha=0.85,
        )
        + geom_text(
            top_outliers,
            aes(x="label_x", y="label_y", label="label"),
            size=7,
            ha="left",
            va="bottom",
            fontstyle="italic",
        )
        + theme_mlb()
        + theme(figure_size=(12, 10))
    )

    if reverse_y:
        p = p + scale_y_reverse()

    return p

def plot_resid_hist_mlb(
    results: pd.DataFrame,
    color_palette: list[str] = MLB_COLOR_PALETTE,
    *,
    actual_col: str = "fantasy_points_future",
    pred_col: str = "predicted_fantasy_points",
    diff_col: str = "prediction_diff",
    band: float = 300,
    binwidth: float = 50,
    x_annotate: float | None = None,
    y_annotate: float | None = None,
):
    """
    Plot a histogram of residuals with an annotated +/- error band.

    Residual definition:
        prediction_diff = predicted - actual

    This diagnostic is useful for:
    - understanding the shape and spread of model errors
    - estimating the share of predictions within a practical error tolerance band
    - spotting skew (consistent over/under-prediction)

    Parameters
    ----------
    results : pd.DataFrame
        DataFrame containing prediction results. Must include `diff_col`, or must
        include both `pred_col` and `actual_col` so residuals can be computed.
    color_palette : list[str], default MLB_COLOR_PALETTE
        List of hex color codes used for plot styling.
    actual_col : str, default "fantasy_points_future"
        Column name containing actual target values.
    pred_col : str, default "predicted_fantasy_points"
        Column name containing model predictions.
    diff_col : str, default "prediction_diff"
        Column name for signed residuals (predicted - actual).
    band : float, default 300
        Symmetric residual band (+/- band) shaded in the background.
    binwidth : float, default 50
        Histogram bin width for residual distribution.
    x_annotate : float | None, default None
        X-position for the annotation text. If None, uses an 85th-percentile
        heuristic based on the residual distribution.
    y_annotate : float | None, default None
        Y-position for the annotation text. If None, defaults to 0. You may want
        to set this explicitly if the label overlaps with bins in your dataset.

    Returns
    -------
    plotnine.ggplot
        A plotnine ggplot object.
    """
    df = _ensure_prediction_diff(
        results,
        actual_col=actual_col,
        pred_col=pred_col,
        diff_col=diff_col,
    )

    within_pct = int((df[diff_col].abs() <= band).mean() * 100)

    # Heuristics for annotation placement if not provided
    if x_annotate is None:
        x_annotate = float(df[diff_col].quantile(0.85))
    if y_annotate is None:
        y_annotate = 0

    p = (
        ggplot(df, aes(x=diff_col))
        + geom_histogram(
            binwidth=binwidth,
            fill=color_palette[2],
            alpha=0.95,
            color="white",
        )
        + annotate(
            "rect",
            xmin=-band,
            xmax=band,
            ymin=-float("inf"),
            ymax=float("inf"),
            alpha=0.18,
            fill="lightgrey",
        )
        + annotate(
            "text",
            x=x_annotate,
            y=y_annotate,
            label=f"{within_pct}% of predictions within +/- {int(band)}",
            size=12,
            ha="left",
            va="top",
            color="black",
            alpha=0.75,
        )
        + labs(
            title="Distribution of Prediction Errors",
            x="Prediction Diff (Predicted - Actual)"
        )
        + theme_mlb()
        + theme(axis_title_y=element_blank())
    )

    return p

    
def plot_decile_calib(
    results: pd.DataFrame,
    color_palette: list[str] = MLB_COLOR_PALETTE,
    *,
    pred_col: str = "predicted_fantasy_points",
    actual_col: str = "fantasy_points_future",
    n_deciles: int = 10,
    nudge_y: float = 20,
):
    """
    Plot a decile-based calibration curve for MLB fantasy point predictions.

    The data is binned into deciles based on predicted fantasy points, and for each
    decile we plot:
        - mean predicted fantasy points (x-axis)
        - mean actual fantasy points (y-axis)
        - a percentage label showing (pred - actual) / actual

    Parameters
    ----------
    results : pd.DataFrame
        DataFrame containing prediction results. Must include:
        - `pred_col` (model predictions, default 'predicted_fantasy_points')
        - `actual_col` (actual values, default 'fantasy_points_future')
    color_palette : list[str], default MLB_COLOR_PALETTE
        List of hex color codes used for styling.
    pred_col : str, default "predicted_fantasy_points"
        Column name for model predictions.
    actual_col : str, default "fantasy_points_future"
        Column name for actual target values.
    n_deciles : int, default 10
        Number of quantile bins to use for calibration (e.g., 10 for deciles).
    nudge_y : float, default 20
        Vertical offset for the % labels above each point.

    Returns
    -------
    plotnine.ggplot
        A plotnine ggplot object representing the decile calibration curve.
    """
    df = results[[pred_col, actual_col]].dropna().copy()

    # Assign deciles based on predicted points
    df["pred_decile"] = pd.qcut(
        df[pred_col],
        q=n_deciles,
        labels=False,
        duplicates="drop",
    )

    # Aggregate mean predicted and actual per decile
    decile_calib = (
        df.groupby("pred_decile", as_index=False)
        .agg(
            mean_pred=(pred_col, "mean"),
            mean_actual=(actual_col, "mean"),
        )
        .sort_values("pred_decile")
    )

    # Compute percentage difference (pred - actual) / actual
    decile_diff = decile_calib.assign(
        pct_diff=lambda d: (d["mean_pred"] - d["mean_actual"])
        / d["mean_actual"]
        * 100
    )

    # For labels, flip sign as in your NBA code so that positive % implies
    # “model underpredicted” (actual > predicted) if you prefer that convention.
    decile_diff = decile_diff.assign(
        diff_label=lambda d: (-d["pct_diff"].round(1)).astype(str) + "%"
    )

    p_decile_calib = (
        ggplot(decile_calib, aes(x="mean_pred", y="mean_actual"))
        + geom_point(
            fill=color_palette[3],
            color=color_palette[3],
            size=2.4,
        )
        + geom_line(
            color=color_palette[3],
            alpha=1.0,
            size=0.9,
        )
        + geom_abline(
            slope=1,
            intercept=0,
            linetype="dashed",
            color="lightgrey",
            alpha=0.6,
        )
        + geom_text(
            decile_diff,
            aes(x="mean_pred", y="mean_actual", label="diff_label"),
            va="bottom",
            ha="center",
            color="grey",
            fontweight="bold",
            size=9,
            nudge_y=nudge_y,
        )
        + labs(
            title="Calibration by Predicted Decile",
            x="Mean Predicted Fantasy Points",
            y="Mean Actual Fantasy Points",
        )
        + theme_mlb()
        + theme(
            figure_size=(10, 5),
            panel_grid_minor=element_blank(),
        )
    )

    return p_decile_calib