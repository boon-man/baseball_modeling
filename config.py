from hyperopt import hp
import numpy as np

# Player postition overrides (there may be conflicts between sources)
BAT_POS_OVERRIDES = {
    "Shohei Ohtani": "OF",
    "Rafael Devers": "OF",
    "Tyler Soderstrom": "OF",
}

# Defining scoring formats and their corresponding rules
SCORING_RULES: dict[str, dict[str, float]] = {
    "UD": {
        "bat": {"1B": 3, "2B": 6, "3B": 8, "HR": 10, "BB": 3, "HBP": 3, "RBI": 2, "R": 2, "SB": 4},
        "pit": {"W": 5, "SO": 3, "IP": 3, "ER": -3},
    },
    "DK": {
        "bat": {"1B": 3, "2B": 5, "3B": 8, "HR": 10, "BB": 2, "HBP": 2, "RBI": 2, "R": 2, "SB": 5},
        "pit": {"W": 4, "SO": 2, "IP": 2.25, "ER": -2, "H": -0.6, "BB": -0.6, "HBP": -0.6, "CG": 2.5, "ShO": 2.5},
    },
}

# Defining hyperparameter search space
param_space = {
    "learning_rate": hp.loguniform("learning_rate", np.log(0.01), np.log(0.07)),

    # leaf-based complexity control
    "max_leaves": hp.quniform("max_leaves", 8, 48, 1),

    "subsample": hp.uniform("subsample", 0.75, 0.95),
    "colsample_bytree": hp.uniform("colsample_bytree", 0.6, 0.95),

    "min_child_weight": hp.loguniform("min_child_weight", np.log(0.1), np.log(10.0)),
    "reg_lambda": hp.loguniform("reg_lambda", np.log(1e-3), np.log(10.0)),
    "reg_alpha": hp.loguniform("reg_alpha", np.log(1e-3), np.log(10.0)),
    "gamma": hp.loguniform("gamma", np.log(1e-5), np.log(2.0)),
}

# Defining number of seasons to aggregate on historical pulls
AGG_YEARS = 3

# Defining color palette for visualizations
MLB_COLOR_PALETTE = [
    "#1E5BA8",  # deep blue
    "#C8102E",  # strong red
    "#FFC000",  # golden orange
    "#1F8A70",  # teal/green
    "#6F2DBD",  # bold purple
    "#2F2F2F",  # near-black for accents
]

# Dampening parameter for MLB positional groups, helpful for re-balancing rankings based on positional ADP demands
POS_DAMPENING_MAP = {
    "P": 0.95,    
    "IF": 0.9,   
    "OF": 1.1,   
}

# Columns for modeling player performance
batting_stat_cols = [
    'IDfg', 'Season', 'Name', 'Age', 'G', 'AB', 'H', '1B', '2B', '3B', 'HR', 'R', 'RBI', 'HBP', 'BB', 'IBB', 'SO',
    'GDP', 'GB', 'FB', 'K%', 'SB', 'AVG', 'OBP', 'SLG', 'OPS', 'BABIP', 'REW', 'RAR', 'WAR', 'wRC+', 'BB/K', 'ISO', 'Spd',
    'wFB', 'wSL', 'wCB', 'wCH', 'WPA', 'Contact%', 'Barrels', 'Barrel%', 'HardHit', 'wOBA', 'SwStr%', 'BsR', 'Soft%',
    'Hard%', 'FB%', 'EV', 'HardHit%', 'XBR', 'xwOBA', 'GB%', 'HR/FB', 'Offense', 'Defense'
]

pitching_stat_cols = [
    'IDfg', 'Season', 'Name', 'Age', 'G', 'GS', 'Pitches', 'Strikes', 'W', 'WAR', 'xERA', 'ERA', 'IP', 'TBF', 'H',
    'ER', 'HR', 'BB', 'HBP', 'SO', 'CG', 'ShO', 'GB', 'FB', 'AVG', 'WHIP', 'BABIP', 'K/BB', 'K-BB%', 'FIP', 'SwStr%', 'CSW%', 'HR/FB',
    'FBv', 'FB%', 'wFB', 'wSL', 'wCB', 'wCH', 'WPA', 'REW', 'RAR', 'Swing%', 'HR/9', 'K%', 'BB%', 'SIERA', 'Soft%', 'Barrel%',
    'HardHit', 'Hard%', 'Pitching+', 'Location+', 'Stuff+', 'LOB%', 'GB%'
]

# Features that will only be captured via the aggregated history pulls
batting_agg_cols = [
    "IDfg",
    "G", "AB", "H", "1B", "2B", "3B", "HR", "R", "RBI", "BB", "SO", "SB", 'HBP',
    "AVG", "OBP", "SLG", "OPS", "BABIP", "ISO", "wRC+", 'REW', "RAR", "WAR", "wOBA",
    "Barrels", "Barrel%", "HardHit%", "EV"
]

pitching_agg_cols = [
    "IDfg",
    "G", "GS", "IP", "TBF", "W", "SO", "BB", "HR", "ER", 'CG', 'ShO', 'HBP',
    "ERA", "FIP", "WHIP", "K/BB", "K-BB%", 'REW', "RAR", "WAR", "wOBA", "SIERA",
    'Pitching+', 'Location+', 'Stuff+'
]