# Defining color palette for visualizations
MLB_COLOR_PALETTE = [
    "#0B3D91",  # deep blue
    "#C8102E",  # strong red
    "#FFB81C",  # gold
    "#1F8A70",  # teal/green
    "#6F2DBD",  # bold purple
    "#2F2F2F",  # near-black for accents
]

# Dampening parameter for MLB positional groups, helpful for re-balancing rankings based on positional ADP demands
POS_DAMPENING_MAP = {
    "P": 0.85,    
    "IF": 0.85,   
    "OF": 1.25,   
}

# Columns for modeling player performance
batting_stat_cols = [
    'IDfg', 'Season', 'Name', 'Age', 'G', 'AB', 'H', '1B', '2B', '3B', 'HR', 'R', 'RBI', 'HBP', 'BB', 'IBB', 'SO',
    'GDP', 'GB', 'FB', 'K%', 'SB', 'AVG', 'OBP', 'SLG', 'OPS', 'BABIP', 'REW', 'RAR', 'WAR', 'wRC+', 'BB/K', 'ISO', 'Spd',
    'wFB', 'wSL', 'wCB', 'wCH', 'WPA', 'Contact%', 'Barrels', 'Barrel%', 'HardHit', 'wOBA', 'SwStr%', 'BsR', 'Soft%',
    'Hard%', 'FB%', 'EV', 'HardHit%', 'XBR', 'xwOBA', 'GB%', 'HR/FB', 'Offense'
]

pitching_stat_cols = [
    'IDfg', 'Season', 'Name', 'Age', 'G', 'GS', 'Pitches', 'Strikes', 'W', 'WAR', 'xERA', 'ERA', 'IP', 'TBF', 'H',
    'ER', 'HR', 'BB', 'SO', 'GB', 'FB', 'AVG', 'WHIP', 'BABIP', 'K/BB', 'K-BB%', 'FIP', 'SwStr%', 'CSW%', 'HR/FB',
    'FBv', 'FB%', 'wFB', 'wSL', 'wCB', 'wCH', 'WPA', 'REW', 'RAR', 'Swing%', 'K%', 'BB%', 'SIERA', 'Soft%', 'Barrel%',
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
    "G", "GS", "IP", "TBF", "W", "SO", "BB", "HR", "ER",
    "ERA", "FIP", "WHIP", "K/BB", "K-BB%", 'REW', "RAR", "WAR", "wOBA", "SIERA",
    'Pitching+', 'Location+', 'Stuff+'
]