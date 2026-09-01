# Part 1: Conditional Probability Modeling

"""
Part 1: Conditional Probability Modeling

Estimate how likely a song is to receive a 5★ rating using conditional probabilities.
"""

import pandas as pd
import numpy as np
import os
import itertools
import sys

def load_data(data_dir='data'):
    """
    Load tracks and ratings data.
    """
    tracks_path = os.path.join(data_dir, 'tracks.csv')
    ratings_path = os.path.join(data_dir, 'ratings.csv')
    
    if not os.path.exists(tracks_path) or not os.path.exists(ratings_path):
        raise FileNotFoundError(f"Could not find data files in {data_dir}")

    tracks = pd.read_csv(tracks_path)
    ratings = pd.read_csv(ratings_path)
    
    # Merge ratings with tracks to get feature info for each rating
    # ratings.csv has 'song_id', tracks.csv has 'track_id'
    merged_df = pd.merge(ratings, tracks, left_on='song_id', right_on='track_id', how='inner', suffixes=('_rating', ''))
    
    return merged_df, tracks, ratings

def create_binned_features(df):
    """
    Create binned versions of continuous features as requested.
    - Popularity: 10-point bins
    - Available Markets Count: 10-point bins
    - Duration: 0-2, 2-3, 3-4, 4-5, 5+ minutes
    """
    df = df.copy()
    
    # 1. Track Popularity (bins of 10)
    # e.g., 89 -> 80-89
    if 'track_popularity' in df.columns:
        # Handle NaN if any
        df['track_popularity'] = df['track_popularity'].fillna(0).astype(int)
        lower = (df['track_popularity'] // 10) * 10
        upper = lower + 9
        df['popularity_bin'] = lower.astype(str) + '-' + upper.astype(str)
        
    # 2. Available Markets Count (bins of 20)
    if 'available_markets_count' in df.columns:
        df['available_markets_count'] = df['available_markets_count'].fillna(0).astype(int)
        lower = (df['available_markets_count'] // 20) * 20
        upper = lower + 19
        df['available_markets_bin'] = lower.astype(str) + '-' + upper.astype(str)
        
    # 3. Duration Bin
    if 'duration_ms' in df.columns:
        def bin_duration(ms):
            minutes = ms / 60000
            if minutes < 2:
                return '0-2 min'
            elif minutes < 3:
                return '2-3 min'
            elif minutes < 4:
                return '3-4 min'
            elif minutes < 5:
                return '4-5 min'
            else:
                return '5+ min'
                
        df['duration_bin'] = df['duration_ms'].apply(bin_duration)
        
    return df

def compute_conditional_prob(df, feature_col, target_rating=5, alpha=1, rating_levels=5):
    """
    Compute P(Rating=target | Feature) with Laplace smoothing.
    
    P(Y=k | X=x) = (Count(Y=k, X=x) + alpha) / (Count(X=x) + alpha * |Y|)
    """
    # Count occurrences of each feature value (Count(X=x))
    feature_counts = df[feature_col].value_counts().reset_index()
    feature_counts.columns = [feature_col, 'total_count']
    
    # Count occurrences of target rating for each feature value (Count(Y=k, X=x))
    target_counts = df[df['rating'] == target_rating][feature_col].value_counts().reset_index()
    target_counts.columns = [feature_col, 'target_count']
    
    # Merge counts
    stats = pd.merge(feature_counts, target_counts, on=feature_col, how='left')
    stats['target_count'] = stats['target_count'].fillna(0)
    
    # Apply Laplace smoothing
    # Numerator: Count(Y=k, X=x) + alpha
    # Denominator: Count(X=x) + alpha * |Y|
    stats['probability'] = (stats['target_count'] + alpha) / (stats['total_count'] + alpha * rating_levels)
    
    return stats.sort_values('probability', ascending=False)

def compute_interaction_prob(df, feature_cols, target_rating=5, alpha=1, rating_levels=5):
    """
    Compute P(Rating=target | Feature1, Feature2, ...)
    """
    # Create a combined feature column
    combined_col = '_'.join(feature_cols)
    df[combined_col] = df[feature_cols].astype(str).agg(' & '.join, axis=1)
    
    return compute_conditional_prob(df, combined_col, target_rating, alpha, rating_levels)

def bayesian_interpretation(df, feature_col, target_rating=5):
    """
    Compute P(Feature | Rating=target) using Bayes' rule.
    """
    target_df = df[df['rating'] == target_rating]
    total_target = len(target_df)
    
    feature_counts = target_df[feature_col].value_counts().reset_index()
    feature_counts.columns = [feature_col, 'count']
    
    feature_counts['probability'] = feature_counts['count'] / total_target
    
    return feature_counts.sort_values('probability', ascending=False)

def compute_rating_distribution(df, feature_col, alpha=1, rating_levels=5):
    """
    Compute P(Rating=k | Feature) for all k in [1, rating_levels].
    """
    # 1. Calculate counts for each feature value
    feature_counts = df[feature_col].value_counts().reset_index()
    feature_counts.columns = [feature_col, 'total_count']
    
    # 2. Calculate counts for each (feature, rating) pair
    # Pivot table: index=feature, columns=rating, values=count
    # We add a dummy column to count
    df_copy = df.copy()
    df_copy['dummy'] = 1
    pivot = df_copy.pivot_table(index=feature_col, columns='rating', values='dummy', aggfunc='count', fill_value=0)
    
    # Ensure all rating columns exist
    for r in range(1, rating_levels + 1):
        if r not in pivot.columns:
            pivot[r] = 0
            
    # Merge total counts
    stats = pd.merge(feature_counts, pivot, on=feature_col, how='left')
    
    # 3. Calculate probabilities with Laplace smoothing
    # P(k|x) = (Count(k,x) + alpha) / (Count(x) + alpha * |Y|)
    denominator = stats['total_count'] + alpha * rating_levels
    
    for r in range(1, rating_levels + 1):
        stats[f'P({r}*)'] = (stats[r] + alpha) / denominator
        
    # Select and reorder columns
    cols = [feature_col, 'total_count'] + [f'P({r}*)' for r in range(1, rating_levels + 1)]
    return stats[cols].sort_values('total_count', ascending=False)

def analyze_group_preferences(tracks_df, data_dir, feature_columns):
    """
    Analyze group preferences by averaging individual user probabilities.
    P_group(5* | F) = (1/M) * sum(P_m(5* | F))
    """
    # Create binned features on the tracks dataframe once, so we have all bins available
    tracks_df = create_binned_features(tracks_df)
    
    group_file = os.path.join(data_dir, 'group.csv')
    
    users_data = {}
    
    # 1. Load Group Data
    if os.path.exists(group_file):
        group_df = pd.read_csv(group_file)
        unique_users = group_df['user_id'].unique()
        for uid in unique_users:
            user_subset = group_df[group_df['user_id'] == uid]
            # Merge with tracks (which is already binned now)
            merged = pd.merge(user_subset, tracks_df, left_on='song_id', right_on='track_id', how='inner')
            users_data[uid] = merged
    else:
        print(f"[!] Group file not found at {group_file}")
        return

    print(f"\n--- Group Analysis (Users: {', '.join(users_data.keys())}) ---")
    print("Computing P_group(5* | F) = Average of P_m(5* | F)...")
    print("Showing Key: Individual User Probabilities and Group Average")
    
    for col in feature_columns:
        # Collect per-user probabilities
        user_probs = []
        
        # Get all unique values for this feature from tracks_df to ensure alignment
        # This works because tracks_df is now binned
        if col not in tracks_df.columns:
             continue
             
        all_values = tracks_df[col].dropna().unique()
        master_df = pd.DataFrame(all_values, columns=[col])
        
        for uid, df in users_data.items():
            if col not in df.columns:
                continue
                
            # Compute P(5* | F) for this user
            stats = compute_conditional_prob(df, col, target_rating=5, alpha=1)
            
            # We only need the probability column. 
            # Calculate default prior for unseen: (0 + 1) / (0 + 5) = 0.2
            smoothed_prior = 1.0 / 5.0
            
            # Prepare user stats for merge
            user_stats = stats[[col, 'probability']].rename(columns={'probability': f'prob_{uid}'})
            
            # Merge
            master_df = pd.merge(master_df, user_stats, on=col, how='left')
            master_df[f'prob_{uid}'] = master_df[f'prob_{uid}'].fillna(smoothed_prior)
            
        # 4. Compute Average
        prob_cols = [c for c in master_df.columns if c.startswith('prob_')]
        if not prob_cols:
            continue
            
        master_df['group_prob'] = master_df[prob_cols].mean(axis=1)
        master_df = master_df.sort_values('group_prob', ascending=False)
        
        print(f"\nGroup Top 10 for P(5* | {col}):")
        # Select columns to display: Feature, prob_U1, prob_U2..., group_prob
        display_cols = [col] + prob_cols + ['group_prob']
        print(master_df[display_cols].head(10).to_string(index=False))

    # 4. Feature Interactions for Group (Optional/High Effort? User asked for "feature columnsda kullandıklarımızı")
    # Doing pairwise for group might be overkill output-wise, but logic applies same way.
    # Let's skip pairwise for group unless explicitly demanded to avoid massive output.
    # User said "burda bizim feature columnsda kullandıklarımızı kullanalım" -> usually implies single features.
    
    return

def main():
    # Reconfigure stdout to use utf-8 to handle special characters
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    
    print("Part 1: Conditional Probability Modeling")
    print("========================================")
    
    # Set pandas display precision
    pd.set_option('display.float_format', '{:.10f}'.format)
    
    # 1. Load Data
    try:
        # Adjust path if running from project root or src
        data_dir = 'data'
        if not os.path.exists(data_dir) and os.path.exists('../data'):
            data_dir = '../data'
            
        merged_df, tracks, ratings = load_data(data_dir)
        print(f"Loaded {len(ratings)} ratings and {len(tracks)} tracks.")
        
        # Apply binning
        merged_df = create_binned_features(merged_df)
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 1.2 Global Rating Distribution
    print("\n--- Global Rating Distribution (All Songs) ---")
    rating_counts = ratings['rating'].value_counts().sort_index()
    total_ratings = len(ratings)
    print(f"Total Ratings: {total_ratings}")
    print("Rating  Count   Probability")
    for r in range(1, 6):
        count = rating_counts.get(r, 0)
        prob = count / total_ratings
        print(f"{r}       {count:<7} {prob:.10f}")

    # 1.5 Rating Distributions for Songs
    print("\n--- Song Rating Distributions P(k* | Song) ---")
    print("Top 10 Most Rated Songs with their Rating Probabilities:")
    song_dist = compute_rating_distribution(merged_df, 'track_name')
    print(song_dist.head(10).to_string(index=False))

    # 2. Global Conditional Probabilities
    print("\n--- Global Conditional Probabilities P(5* | Feature) ---")
    
    feature_columns = [
        'popularity_bin', 'duration_bin', 'explicit', 'available_markets_bin',
        'artist_names', 'primary_artist_name', 'album_name', 'album_release_year',
        'ab_danceability_value', 'ab_mood_acoustic_value', 'ab_mood_aggressive_value',
        'ab_mood_electronic_value', 'ab_mood_happy_value', 'ab_mood_party_value',
        'ab_mood_relaxed_value', 'ab_mood_sad_value', 'ab_gender_value',
        'ab_voice_instrumental_value', 'ab_timbre_value',
        'ab_genre_dortmund_value', 'ab_genre_rosamerica_value'
    ]

    for col in feature_columns:
        if col not in merged_df.columns:
            continue
            
        print(f"\nTop 5 for P(5* | {col}):")
        probs = compute_conditional_prob(merged_df, col)
        # Show feature name, count, and probability, hide index
        print(probs[[col, 'total_count', 'probability']].head(5).to_string(index=False))

    # 3. Feature Interactions
    print("\n--- Feature Interactions (Exhaustive Pairwise Analysis) ---")
    print("Calculating P(5* | Pair) and P(Pair | 5*) for all combinations...")
    
    # Using the same list as feature_columns to ensure no pair is missing
    interaction_features = feature_columns
    
    # Filter to ensure columns exist
    interaction_features = [col for col in interaction_features if col in merged_df.columns]
    
    # Total number of 5-star ratings for P(Pair | 5*) calculation
    total_5_star = len(merged_df[merged_df['rating'] == 5])
    
    # Iterate through all unique pairs
    for col1, col2 in itertools.combinations(interaction_features, 2):
        print(f"\nInteraction: {col1} & {col2}")
        
        # Group by the two columns
        # Count total occurrences
        pair_counts = merged_df.groupby([col1, col2]).size().reset_index(name='total_count')
        
        # Count 5-star occurrences
        pair_5_star = merged_df[merged_df['rating'] == 5].groupby([col1, col2]).size().reset_index(name='target_count')
        
        # Merge
        stats = pd.merge(pair_counts, pair_5_star, on=[col1, col2], how='left')
        stats['target_count'] = stats['target_count'].fillna(0).astype(int)
        stats['total_count'] = stats['total_count'].astype(int)
        
        # Calculate P(5* | Pair) = target_count / total_count
        # Laplace smoothing: (target + 1) / (total + 5)
        stats['P(5*|Pair)'] = (stats['target_count'] + 1) / (stats['total_count'] + 5)
        
        # Sort by P(5*|Pair) descending
        stats = stats.sort_values('P(5*|Pair)', ascending=False)
        
        # Print table
        print(stats.to_string(index=False))

    # 4. Bayesian Interpretation
    print("\n--- Bayesian Interpretation P(Feature | 5*) ---")
    
    # List of features to analyze for Bayesian interpretation
    bayes_features = [
        'primary_artist_name',
        'ab_danceability_value', 'ab_mood_acoustic_value', 'ab_mood_aggressive_value',
        'ab_mood_electronic_value', 'ab_mood_happy_value', 'ab_mood_party_value',
        'ab_mood_relaxed_value', 'ab_mood_sad_value', 'ab_gender_value',
        'ab_voice_instrumental_value', 'ab_timbre_value',
        'ab_genre_dortmund_value', 'ab_genre_rosamerica_value'
    ]

    for col in bayes_features:
        if col not in merged_df.columns:
            continue
            
        print(f"\nWhich {col} values dominate the 5-star ratings?")
        bayes_result = bayesian_interpretation(merged_df, col)
        print(bayes_result.head(10).to_string(index=False))

    # 5. Personal and Group Analysis
    # Check for 'my_session.csv' or similar in data dir
    # Using the specific file provided by the user
    analyze_group_preferences(tracks, data_dir, feature_columns)

if __name__ == "__main__":
    main()
