# Defining color palette for visualizations
MLB_COLOR_PALETTE = [
    "#0B3D91",  # deep blue
    "#C8102E",  # strong red
    "#FFB81C",  # gold
    "#1F8A70",  # teal/green
    "#6F2DBD",  # bold purple
    "#2F2F2F",  # near-black for accents
]

# Dampening parameter for MLB positional groups
POS_DAMPENING_MAP = {
    "P": 1.00,    
    "IF": 1.00,   
    "OF": 1.00,   
}

# Columns for modeling player performance
batting_stat_cols = [
    'IDfg', 'Season', 'Name', 'Age', 'G', 'AB', 'H', '1B', '2B', '3B', 'HR', 'R', 'RBI', 'HBP', 'BB', 'IBB', 'SO',
    'GDP', 'GB', 'FB', 'K%', 'SB', 'AVG', 'OBP', 'SLG', 'OPS', 'BABIP', 'RAR', 'WAR', 'wRC+', 'BB/K', 'ISO', 'Spd',
    'wFB', 'wSL', 'wCB', 'wCH', 'WPA', 'Contact%', 'Barrels', 'Barrel%', 'HardHit', 'wOBA', 'SwStr%', 'BsR', 'Soft%',
    'Hard%', 'FB%', 'EV', 'HardHit%', 'XBR', 'xwOBA', 'GB%', 'HR/FB', 'Offense'
]

pitching_stat_cols = [
    'IDfg', 'Season', 'Name', 'Age', 'G', 'GS', 'Pitches', 'Strikes', 'W', 'WAR', 'xERA', 'ERA', 'IP', 'TBF', 'H',
    'ER', 'HR', 'BB', 'SO', 'GB', 'FB', 'AVG', 'WHIP', 'BABIP', 'K/BB', 'K-BB%', 'FIP', 'SwStr%', 'CSW%', 'HR/FB',
    'FBv', 'FB%', 'wFB', 'wSL', 'wCB', 'wCH', 'WPA', 'RAR', 'Swing%', 'K%', 'BB%', 'SIERA', 'Soft%', 'Barrel%',
    'HardHit', 'Hard%', 'Pitching+', 'Location+', 'Stuff+', 'LOB%', 'GB%'
]

# Features that will only be captured via the career data pull
batting_career_cols = [
    "IDfg",
    "G", "AB", "H", "1B", "2B", "3B", "HR", "R", "RBI", "BB", "SO", "SB", 'HBP',
    "AVG", "OBP", "SLG", "OPS", "BABIP", "ISO", "wRC+", "WAR",
]

pitching_career_cols = [
    "IDfg",
    "G", "GS", "IP", "TBF", "W", "SO", "BB", "HR", "ER",
    "ERA", "FIP", "WHIP", "K/BB", "K-BB%", "WAR",
]