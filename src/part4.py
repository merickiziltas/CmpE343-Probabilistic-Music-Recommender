import pandas as pd
import numpy as np
import random
import os
from part3 import MusicRecommender
from scipy import stats

class SimulatedUser:
    """
    Full Matrix Lookup Oracle.
    
    Strategy:
    - Uses pre-computed 'full_user_song_ratings.csv' provided by the user.
    - This dataset contains the "True" (Simulated) rating for every Song x User pair.
    - No inference, no noise calculation at runtime (assumed baked into data).
    """
    def __init__(self, user_id, user_ratings_df):
        self.user_id = user_id
        # Expects DataFrame with ['song_id', 'rating']
        # Depending on file, it might be 'song_id' or 'spotify_id', handled in run_simulation
        self.ratings_map = dict(zip(user_ratings_df['song_id'], user_ratings_df['rating']))

    def rate_song(self, track_id):
        """
        Returns the pre-calculated rating for the given track_id.
        """
        return self.ratings_map.get(track_id, 3) # Default to 3 if missing (Edge case)

def compute_confidence_interval(data_a, data_b, confidence=0.95):
    """
    Computes mean difference (A - B) and 95% Confidence Interval.
    Returns: diff, (lower, upper), is_significant
    """
    a = np.array(data_a)
    b = np.array(data_b)
    
    diff = np.mean(a) - np.mean(b)
    
    # Welch's t-test principle for CI of difference
    se_a = stats.sem(a)
    se_b = stats.sem(b)
    se_diff = np.sqrt(se_a**2 + se_b**2)
    
    # Critical value (z or t). For N=1000, z=1.96 is fine.
    z_score = 1.96
    margin = z_score * se_diff
    
    lower = diff - margin
    upper = diff + margin
    
    # Significant if 0 is NOT in the interval
    is_significant = not (lower <= 0 <= upper)
    
    return diff, (lower, upper), is_significant

def run_simulation(n_users=100, rounds=20, k=5, data_dir='data', full_ratings_csv='data/full_user_song_ratings.csv'):
    """
    Runs the Monte Carlo Simulation using Pre-Calculated Full Matrix.
    """
    print(f"--- Starting Monte Carlo Simulation (N={n_users}) ---")
    print(f"Loading Simulated Truth from {full_ratings_csv}...")
    
    try:
        full_ratings_df = pd.read_csv(full_ratings_csv)
    except FileNotFoundError:
        print(f"Error: {full_ratings_csv} not found.")
        return {}

    # Check consistency of columns
    col_map = {'user_id': 'user_id', 'song_id': 'song_id', 'rating': 'rating'}
    # Try to detect if columns are different (e.g. spotify_id vs song_id)
    if 'spotify_id' in full_ratings_df.columns:
        col_map['song_id'] = 'spotify_id'
        
    full_ratings_df = full_ratings_df.rename(columns={
        col_map['user_id']: 'user_id', 
        col_map['song_id']: 'song_id', 
        col_map['rating']: 'rating'
    })

    # Pick Random Users
    all_users = full_ratings_df['user_id'].unique()
    if len(all_users) < n_users:
        print(f"Warning: Requested {n_users} users but file only has {len(all_users)}. Using all.")
        selected_users = all_users
    elif n_users > 0:
        selected_users = np.random.choice(all_users, n_users, replace=False)
    else:
        selected_users = all_users # If N=0 or None
    
    print(f"Selected {len(selected_users)} unique users for simulation.")
    
    results = {
        'Model A': {'hits_at_k': [], 'avg_ratings': [], 'time_to_5': []},
        'Model B': {'hits_at_k': [], 'avg_ratings': [], 'time_to_5': []}
    }

    # Initialize Recommender
    recommender = MusicRecommender(data_dir=data_dir)

    # --- Simulation Loop ---
    for idx, user_id in enumerate(selected_users):
        if (idx+1) % 50 == 0:
            print(f"Processing User {idx+1}/{len(selected_users)}...")
            
        # Get User's Total Truth
        user_total_data = full_ratings_df[full_ratings_df['user_id'] == user_id]
        
        # 1. Instantiate Oracle with FULL knowledge
        sim_user = SimulatedUser(user_id, user_total_data)
        
        # 2. Extract Seed Ratings for the Recommender (Cold Start State)
        # We give the recommender 5 random ratings to start with
        # 2. Extract Seed Ratings for the Recommender (Cold Start State)
        # We give the recommender 5 *CONSECUTIVE* ratings to start with.
        # This preserves the user's sequential behavior pattern (Crucial for Model B).
        if len(user_total_data) >= 5:
            # Pick a random starting point for the 5-song window
            max_start = len(user_total_data) - 5
            start_idx = np.random.randint(0, max_start + 1)
            seed_df = user_total_data.iloc[start_idx : start_idx + 5]
        else:
            seed_df = user_total_data # Edge case (keep all)
            
        seed_ratings = []
        for _, row in seed_df.iterrows():
            seed_ratings.append({
                'spotify_id': row['song_id'],
                'rating': row['rating']
            })

        # --- Run Session for Each Model ---
        for model_name in ['Model A', 'Model B']:
            current_session_ratings = seed_ratings.copy()
            
            found_5_star = False
            first_5_star_round = None
            session_satisfaction = []
            
            for r_round in range(1, rounds + 1):
                # Request recommendations
                # NOTE: Recommender is BLIND to the full_ratings_df. It only sees `current_session_ratings`.
                if model_name == 'Model A':
                    recs = recommender.recommend_model_a(current_session_ratings, k=k)
                else:
                    recs = recommender.recommend_model_b(current_session_ratings, k=k)
                
                round_has_5_star = False
                
                for track_id, track_name in recs:
                    # Oracle Lookup
                    rating = sim_user.rate_song(track_id)
                    
                    # Store rating
                    current_session_ratings.append({
                        'spotify_id': track_id,
                        'rating': rating
                    })
                    session_satisfaction.append(rating)
                    
                    if rating == 5:
                        round_has_5_star = True
                
                # Update Metrics
                if round_has_5_star and not found_5_star:
                    found_5_star = True
                    first_5_star_round = r_round
            
            # --- Metrics ---
            # 1. Hit@k (First Round Hit)
            is_hit_at_k = 1 if (first_5_star_round == 1) else 0
            results[model_name]['hits_at_k'].append(is_hit_at_k)
            
            # 2. Avg Rating
            # Calculate Average of ALL recommendations in this session
            avg_rating = np.mean(session_satisfaction) if session_satisfaction else 0
            results[model_name]['avg_ratings'].append(avg_rating)
            
            # 3. Time-to-5*
            if first_5_star_round:
                results[model_name]['time_to_5'].append(first_5_star_round)
            else:
                results[model_name]['time_to_5'].append(rounds + 1)

    return results

if __name__ == "__main__":
    n = 100 
    results = run_simulation(n_users=n, rounds=20, k=5)
    
    if results:
        # Save Report
        with open("metrics.txt", "w") as f:
            header = f"--- Monte Carlo Results (N={n}, Full Matrix) ---"
            print("\n" + header)
            f.write(header + "\n\n")
            
            for metric in ['hits_at_k', 'avg_ratings', 'time_to_5']:
                print(f"Analyzing {metric}...")
                f.write(f"Metric: {metric}\n")
                
                vals_a = results['Model A'][metric]
                vals_b = results['Model B'][metric]
                
                mean_a = np.mean(vals_a)
                mean_b = np.mean(vals_b)
                
                diff, (low, high), is_sig = compute_confidence_interval(vals_a, vals_b)
                
                summary = (
                    f"  Model A Mean: {mean_a:.4f}\n"
                    f"  Model B Mean: {mean_b:.4f}\n"
                    f"  Difference (A - B): {diff:.4f} [{low:.4f}, {high:.4f}]\n"
                    f"  Significant? {'YES' if is_sig else 'NO'}\n"
                )
                
                print(summary)
                f.write(summary + "\n")
                    
        print("\nDetailed statistical report written to metrics.txt")
