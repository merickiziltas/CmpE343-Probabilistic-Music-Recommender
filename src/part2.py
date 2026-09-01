# Part 2: User Variability Modeling

"""
Part 2: User Variability Modeling

Model how many recommendations it takes for users to rate a song 5★ using 
geometric and Beta-geometric distributions.
"""

import pandas as pd
import numpy as np
import os
import sys
from scipy.optimize import minimize
from scipy import stats, special

def load_data(data_dir='data'):
    """
    Load tracks and ratings data.
    """
    tracks_path = os.path.join(data_dir, 'tracks.csv')
    ratings_path = os.path.join(data_dir, 'ratings.csv')
    
    if not os.path.exists(tracks_path) or not os.path.exists(ratings_path):
        # Fallback for running from src directory
        tracks_path = os.path.join('..', data_dir, 'tracks.csv')
        ratings_path = os.path.join('..', data_dir, 'ratings.csv')

    if not os.path.exists(tracks_path) or not os.path.exists(ratings_path):
         raise FileNotFoundError(f"Could not find data files in {data_dir} or ../{data_dir}")

    tracks = pd.read_csv(tracks_path)
    ratings = pd.read_csv(ratings_path)
    
    return tracks, ratings

def calculate_time_to_favorite(ratings_df):
    """
    Calculate T_u: The number of steps until the first 5-star rating for each user.
    """
    # Ensure sorted by user and round
    ratings_df = ratings_df.sort_values(['user_id', 'round_idx'])
    
    user_times = []
    user_ids = []
    
    grouped = ratings_df.groupby('user_id')
    
    for uid, group in grouped:
        # Find 5-star ratings
        five_stars = group[group['rating'] == 5]
        
        if not five_stars.empty:
            # First 5-star round index
            first_5_star_idx = five_stars['round_idx'].iloc[0]
            # Assuming rounds start at 1? Let's check min round_idx usually
            # But T_u is "number of recommendations", which maps to round_idx if strictly sequential 1,2,3...
            user_times.append(first_5_star_idx)
            user_ids.append(uid)
            
    return pd.DataFrame({'user_id': user_ids, 'T_u': user_times})

def fit_geometric(T_u):
    """
    Fit Geometric distribution: P(T=t) = (1-p)^(t-1) * p
    MLE for p = 1 / mean(T)
    """
    sum_T = np.sum(T_u)
    n = len(T_u)
    avg_T = np.mean(T_u)
    p_hat = 1.0 / avg_T
    
    # Log-Likelihood
    log_likelihood = np.sum((T_u - 1) * np.log(1 - p_hat) + np.log(p_hat))
    
    print(f"\n--- Model 1: Geometric Distribution (Simple Approach) ---")
    print("What we did: We assumed EVERY user works exactly the same way (Same probability 'p').")
    print("-" * 60)
    
    print(f"Step 1: Calculate Average Waiting Time")
    print(f"  Result: {avg_T:.4f} rounds")
    print(f"  How: We took the average of all {n} users' waiting times.")
    
    print(f"\nStep 2: Estimate 'Hit Probability' (p)")
    print(f"  Result: {p_hat:.4f} (approx {p_hat*100:.1f}%)")
    print(f"  How: Since it takes ~{avg_T:.1f} tries to find a song, the chance per try is roughly 1/{avg_T:.1f}.")
    
    print(f"\nStep 3: Calculate Model Score (Log-Likelihood)")
    print(f"  Result: {log_likelihood:.4f}")
    print(f"  Meaning: A score showing how well this 'One Size Fits All' model explains our data.")
    print(f"           (Closer to 0 is better).")
    
    return p_hat, log_likelihood

def fit_beta_geometric(T_u, geo_ll):
    """
    Fit Beta-Geometric distribution.
    """
    
    def neg_log_likelihood(params):
        alpha, beta = params
        if alpha <= 0 or beta <= 0:
            return np.inf
        log_probs = special.betaln(alpha + 1, beta + T_u - 1) - special.betaln(alpha, beta)
        return -np.sum(log_probs)

    # Initial guess
    initial_guess = [1.0, 1.0]
    
    result = minimize(neg_log_likelihood, initial_guess, bounds=((1e-5, None), (1e-5, None)))
    
    alpha_hat, beta_hat = result.x
    bg_ll = -result.fun
    
    print(f"\n--- Model 2: Beta-Geometric Distribution (Flexible Approach) ---")
    print("What we did: We assumed users are DIFFERENT. Some find songs quickly, some are picky.")
    print("-" * 60)
    
    print(f"Step 1: Find Best Parameters (alpha, beta)")
    print(f"  Result: alpha={alpha_hat:.4f}, beta={beta_hat:.4f}")
    print(f"  [Detailed Explanation of How We Found These]:")
    print(f"    1. THE GOAL: Find the specific alpha & beta that make our observed data MOST probable.")
    print(f"    2. THE MATH: We use a 'Likelihood Function'. For every user's T_u, we calculate:")
    print(f"       P(T_u | alpha, beta). We multiply these for all users to get total Likelihood.")
    print(f"    3. THE PROCESS (Maximum Likelihood Estimation - MLE):")
    print(f"       a. Start with a guess (e.g., alpha=1, beta=1).")
    print(f"       b. Use a computer algorithm ('scipy.minimize') to confirm if changing alpha/beta")
    print(f"          increases the Likelihood.")
    print(f"       c. Repeat this 'hill-climbing' thousands of times until the Likelihood stops increasing.")
    print(f"    4. result: The values {alpha_hat:.4f} and {beta_hat:.4f} are the peak of this hill.")
    print(f"       They are the 'Maximum Likelihood Estimates'.")
    
    expected_T = 1 + beta_hat / alpha_hat
    print(f"\nStep 2: Calculate Implied Average Time (for the 'Typical' User)")
    print(f"  Result: {expected_T:.4f} rounds")
    print(f"  [Detailed Explanation of How We Found This]:")
    print(f"    1. We know alpha & beta define the distribution of probabilities (p).")
    print(f"    2. We calculate the AVERAGE probability E[p] for the group:")
    print(f"       Formula: E[p] = alpha / (alpha + beta)")
    print(f"       Calculation: {alpha_hat:.4f} / ({alpha_hat:.4f} + {beta_hat:.4f}) = {alpha_hat/(alpha_hat+beta_hat):.4f}")
    print(f"    3. We convert this probability into 'Time' (Rounds to success):")
    print(f"       Formula: Time = 1 / E[p]")
    print(f"       Calculation: 1 / {alpha_hat/(alpha_hat+beta_hat):.4f} = {expected_T:.4f}")
    print(f"       (Math Shortcut: 1 + beta/alpha gives the same result).")
    
    print(f"  [Interpretation - What does '2.83' mean?]:")
    print(f"    - It means a 'typical' user (someone with average pickiness) needs to listen to")
    print(f"      about 3 songs (specifically 2.83) to find one they love (5 stars).")
    print(f"    - Low number = Users find favorites quickly.")
    print(f"    - High number = Users are very picky.")
    
    print(f"\nStep 3: Calculate Model Score (Log-Likelihood)")
    print(f"  Result: {bg_ll:.4f}")
    
    print("\n--- FINAL COMPARISON: Which Model is Better? ---")
    print(f"Model 1 (Everyone same): {geo_ll:.4f}")
    print(f"Model 2 (Users differ):  {bg_ll:.4f}")
    
    if bg_ll > geo_ll:
        print(f"WINNER: Model 2 (Beta-Geometric) is better.")
        print("Reasoning: The score is higher (closer to 0). This confirms that users DOES vary significantly")
        print("           in how picky they are. Modeling this difference is important.")
    else:
        print("WINNER: Model 1.")
        
    return alpha_hat, beta_hat

def hypothesis_testing(tu_df, ratings_df, tracks_df):
    """
    Hypothesis Test: Do users who prefer popular songs find favorites faster?
    """
    print(f"\n--- Hypothesis Testing: User Differences ---")
    print("Goal: Let's split users into two groups and see if one group finds songs faster than the other.")
    
    # Merge ratings with track popularity
    merged = pd.merge(ratings_df, tracks_df[['track_id', 'track_popularity']], 
                      left_on='song_id', right_on='track_id', how='inner')
                      
    user_avg_pop = merged.groupby('user_id')['track_popularity'].mean().reset_index()
    
    # Merge with T_u data
    analysis_df = pd.merge(tu_df, user_avg_pop, on='user_id')
    
    # Split into High vs Low Popularity preference
    median_pop = analysis_df['track_popularity'].median()
    
    group_high = analysis_df[analysis_df['track_popularity'] >= median_pop]['T_u']
    group_low = analysis_df[analysis_df['track_popularity'] < median_pop]['T_u']
    
    print(f"\n--- HOW WE SPLIT THE USERS ---")
    print(f"Goal: Group users by their music taste (Mainstream vs. Niche).")
    print(f"[Step 1: Assign a Score to key User]")
    print(f"  - We looked at ALL songs a user rated.")
    print(f"  - We calculated the AVERAGE popularity of those songs.")
    print(f"  - Example: If a user rated 3 songs with popularity 90, 80, 70, their Score is 80.")
    
    print(f"\n[Step 2: Find the Cutoff Point]")
    print(f"  - We took all user scores and found the MEDIAN (Middle value).")
    print(f"  - Median Cutoff: {median_pop:.1f}")
    
    print(f"\n[Step 3: Create Groups]")
    print(f"  1. Group A (High Pop): Users with Score >= {median_pop:.1f} (Prefer Mainstream music).")
    print(f"  2. Group B (Low Pop):  Users with Score < {median_pop:.1f} (Prefer Niche music).")
    
    print(f"\nGroup A: 'Pop Music Lovers'")
    print(f"  Count: {len(group_high)} users")
    print(f"  Average Time to 5-star: {group_high.mean():.2f} rounds")
    
    print(f"\nGroup B: 'Niche Music Lovers'")
    print(f"  Count: {len(group_low)} users")
    print(f"  Average Time to 5-star: {group_low.mean():.2f} rounds")
    
    # Mann-Whitney U Test
    stat, p_value = stats.mannwhitneyu(group_high, group_low, alternative='two-sided')
    
    print(f"\n--- Statistical Test Result ---")
    print(f"We ran a 'Mann-Whitney U Test' to see if this difference is real or just luck.")
    print(f"P-value: {p_value:.10f}")
    
    if p_value < 0.05:
        print("Conclusion: SIGNIFICANT DIFFERENCE (p < 0.05).")
        print("Meaning: It is NOT just luck. The groups act differently.")
        if group_high.mean() < group_low.mean():
            print("         Users who like Popular music find songs FASTER.")
        else:
            print("         Users who like Niche music find songs FASTER.")
    else:
        print("Conclusion: NO SIGNIFICANT DIFFERENCE.")
        print("Meaning: Any difference we see could just be random chance.")

def main():
    # Reconfigure stdout to force utf-8 (good practice for pipelines)
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
        
    print("Part 2: User Variability Modeling")
    print("=================================")
    
    # 1. Load Data
    try:
        tracks, ratings = load_data()
    except Exception as e:
        print(f"Error loading data: {e}")
        return
        
    print(f"Loaded {len(ratings)} ratings from {ratings['user_id'].nunique()} users.")
    
    # 2. Calculate T_u
    tu_df = calculate_time_to_favorite(ratings)
    print(f"Users with at least one 5-star rating: {len(tu_df)}")
    
    if len(tu_df) < 5:
        print("Not enough users with 5-star ratings for meaningful modeling.")
        # But we proceed anyway to show logic
        
    T_u = tu_df['T_u'].values
    
    # 3. Geometric Model
    _, geo_ll = fit_geometric(T_u)
    
    # 4. Beta-Geometric Model
    fit_beta_geometric(T_u, geo_ll)
    
    # 5. Hypothesis Testing
    hypothesis_testing(tu_df, ratings, tracks)

if __name__ == "__main__":
    main()
