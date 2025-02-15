# flake8: noqa

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def assign_position_group(df: pd.DataFrame, position_col: str):
    """
    Assigns a positional group based on the player's position.

    Parameters:
    df (pd.DataFrame): The DataFrame containing player data.
    position_col (str): The column name indicating the player's position.

    Returns:
    pd.DataFrame: The DataFrame with an additional 'position_group' column.
    """
    # Define position group mappings, Designated Hitters will default as infielders
    position_mapping = {
        'IF': {'1B', '2B', 'SS', '3B', 'C', 'DH'},
        'OF': {'OF'},
        'P': {'SP', 'RP'}
    }


def map_position(df, pos):
    for group, positions in position_mapping.items():
        if pos in positions:
            return group
    return 'Other'  # Fallback for unexpected values

    # Apply mapping function
    df['position_group'] = df[position_col].apply(map_position)
    return df

def calculate_relative_value(df: pd.DataFrame, position_col: str, projection_col: str):
    """
    Calculates the relative value score for each player based on their positional group.

    Parameters:
    df (pd.DataFrame): The DataFrame containing player data.
    position_col (str): The column name indicating the player's positional group.
    projection_col (str): The column name indicating the player's projected value.

    Returns:
    pd.DataFrame: The DataFrame with an additional 'relative_value' column.
    """
    # Group by positional group and calculate mean and standard deviation
    grouped_stats = df.groupby(position_col)[projection_col].agg(['mean', 'std']).rename(columns={'mean': 'group_mean', 'std': 'group_std'})    
    # Merge the calculated stats back into the original dataframe
    df = df.merge(grouped_stats, on=position_col, how='left')   
    # Compute the relative value score
    df['relative_value'] = ((df[projection_col] - df['group_mean']) / df['group_std']) * df[projection_col] 
    # Handle potential division by zero (if std is 0)
    df['relative_value'].fillna(0, inplace=True)

    # Define final player rankings based on relative positional value scores
    df['final_ranking'] = df['relative_value'].rank(ascending=False) 
    # Drop intermediate columns if needed
    df.drop(columns=['group_mean', 'group_std'], inplace=True)

    return df

def determine_optimal_k(data: pd.DataFrame, max_k=10):
    """
    Determines the optimal number of clusters using the elbow method.

    Parameters:
    data (pd.DataFrame): DataFrame containing the relative_value column.
    max_k (int): Maximum number of clusters to consider.
    
    Returns:
    None (Displays the elbow plot)
    """


    kmeans_df = data[['relative_value']]
    wss = []
    # Dropping K values of 1 & 2 (their inherently large WSS values distorts plotting)
    for k in range(3, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(kmeans_df)
        wss.append(kmeans.inertia_)
    # Plot the elbow method results
    plt.figure(figsize=(8, 5))
    # Dropping K values of 1 & 2 (their inherently large WSS values distorts plotting)
    plt.plot(range(3, max_k + 1), wss, marker='o', linestyle='-')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Within-Cluster Sum of Squares (WSS)')
    plt.title('Elbow Plot for Optimal K Selection')
    plt.show()



def segment_players(data: pd.DataFrame, k: int):
    """
    Segments players into K groups using KMeans clustering.
    ***Assumes that a column named "relative_value" will be segmented***

    Parameters:
    data (pd.DataFrame): DataFrame containing the relative_value column.
    k (int): The optimal number of clusters.

    Returns:
    pd.DataFrame: The DataFrame with an additional 'value_segment' column.
    """
    # Extract relevant feature
    features = data[['relative_value']].copy()

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    data['raw_segment'] = kmeans.fit_predict(features)

    # Compute mean relative_value for each cluster
    cluster_means = data.groupby('raw_segment')['relative_value'].mean()

    # Create a mapping from old cluster labels to new ordered labels (1 = highest value, etc.)
    sorted_clusters = cluster_means.sort_values(ascending=False).index
    cluster_mapping = {old_label: new_label+1 for new_label, old_label in enumerate(sorted_clusters)}

    # Apply the mapping
    data['player_value_tier'] = data['raw_segment'].map(cluster_mapping)

    # Drop the temporary raw segment column
    data.drop(columns=['raw_segment'], inplace=True)
    
    return data