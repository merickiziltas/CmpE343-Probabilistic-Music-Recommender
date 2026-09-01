"""
Part 4: Monte Carlo Evaluation using Collaborative Filtering

Evaluates Model A and Model B using simulated user interactions.
Uses collaborative filtering to predict ratings - no need for full matrix CSV.
"""

import pandas as pd
import numpy as np
import os
from part3 import MusicRecommender
from scipy import stats


class CollaborativeFilteringUser:
    """
    Simulates user ratings using collaborative filtering.
    Finds similar users and predicts ratings based on their behavior.
    """
    def __init__(self, user_id, all_ratings_df, target_user_ratings):
        self.user_id = user_id
        self.all_ratings = all_ratings_df
        self.my_ratings = dict(zip(target_user_ratings['song_id'], 
                                   target_user_ratings['rating']))
        self._find_similar_users()
    
    def _find_similar_users(self, top_k=10):
        """Find top K most similar users using Jaccard similarity."""
        my_songs = set(self.my_ratings.keys())
        
        similarities = []
        for other_user in self.all_ratings['user_id'].unique():
            if other_user == self.user_id:
                continue
            
            other_songs = set(self.all_ratings[
                self.all_ratings['user_id'] == other_user
            ]['song_id'])
            
            # Jaccard: |intersection| / |union|
            intersection = len(my_songs & other_songs)
            union = len(my_songs | other_songs)
            
            if union > 0:
                similarity = intersection / union
                similarities.append((other_user, similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        self.similar_users = [u for u, s in similarities[:top_k]]
    
    def rate_song(self, track_id):
        """Predict rating for a song."""
        # Return actual if already rated
        if track_id in self.my_ratings:
            return self.my_ratings[track_id]
        
        # Get ratings from similar users
        similar_ratings = self.all_ratings[
            (self.all_ratings['song_id'] == track_id) &
            (self.all_ratings['user_id'].isin(self.similar_users))
        ]['rating'].tolist()
        
        if similar_ratings:
            predicted = np.mean(similar_ratings)
            noise = np.random.uniform(-0.3, 0.3)
            predicted = np.clip(predicted + noise, 1, 5)
            return int(np.round(predicted))
        else:
            return np.random.choice([3, 4], p=[0.6, 0.4])


def compute_confidence_interval(data_a, data_b):
    """Compute 95% CI for difference (A - B)."""
    a = np.array(data_a)
    b = np.array(data_b)
    
    diff = np.mean(a) - np.mean(b)
    se_diff = np.sqrt(stats.sem(a)**2 + stats.sem(b)**2)
    margin = 1.96 * se_diff
    
    lower = diff - margin
    upper = diff + margin
    is_significant = not (lower <= 0 <= upper)
    
    return diff, (lower, upper), is_significant


def run_simulation(n_users=100, rounds=20, k=5, data_dir='data', hit_k_values=[5, 10, 20]):
    """
    Run Monte Carlo simulation.
    
    Args:
        n_users: Number of users to simulate
        rounds: Recommendation rounds per user  
        k: Songs per round
        hit_k_values: k values for Hit@k metric
    """
    print("=" * 70)
    print("Part 4: Monte Carlo Evaluation")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Users: {n_users}")
    print(f"  Rounds: {rounds}")
    print(f"  Songs/round: {k}")
    print(f"  Total songs: {rounds * k}")
    print(f"  Hit@k values: {hit_k_values}\n")
    
    # Load data - handle path from src/ directory
    if not os.path.exists(data_dir):
        data_dir = os.path.join('..', data_dir)
    
    ratings_path = os.path.join(data_dir, 'ratings.csv')
    
    if not os.path.exists(ratings_path):
        print(f"ERROR: Cannot find {ratings_path}")
        print(f"Current directory: {os.getcwd()}")
        print(f"Please run from project root or src directory")
        return {}
    
    ratings_df = pd.read_csv(ratings_path)
    
    print(f"Loaded {len(ratings_df)} ratings from {ratings_df['user_id'].nunique()} users")
    
    # Select eligible users (>=10 ratings)
    user_counts = ratings_df['user_id'].value_counts()
    eligible = user_counts[user_counts >= 10].index.tolist()
    
    if len(eligible) < n_users:
        print(f"Warning: Only {len(eligible)} users available")
        selected = eligible
    else:
        selected = np.random.choice(eligible, n_users, replace=False)
    
    print(f"Testing {len(selected)} users\n")
    
    # Initialize results
    results = {
        'Model A': {
            'hits_at_k': {k_val: [] for k_val in hit_k_values},
            'avg_ratings': [],
            'time_to_5': []
        },
        'Model B': {
            'hits_at_k': {k_val: [] for k_val in hit_k_values},
            'avg_ratings': [],
            'time_to_5': []
        }
    }
    
    recommender = MusicRecommender(data_dir=data_dir)
    
    # Simulate users
    for idx, user_id in enumerate(selected):
        if (idx + 1) % 20 == 0:
            print(f"Processing {idx + 1}/{len(selected)}...")
        
        user_data = ratings_df[ratings_df['user_id'] == user_id]
        sim_user = CollaborativeFilteringUser(user_id, ratings_df, user_data)
        
        # Seed ratings
        seed_size = min(5, len(user_data))
        seed_data = user_data.sample(seed_size, random_state=idx)
        seed_ratings = [
            {'spotify_id': row['song_id'], 'rating': row['rating']}
            for _, row in seed_data.iterrows()
        ]
        
        # Test both models
        for model_name in ['Model A', 'Model B']:
            session = seed_ratings.copy()
            found_5 = False
            first_5_round = None
            all_ratings = []
            
            for r in range(1, rounds + 1):
                recs = (recommender.recommend_model_a(session, k=k) 
                       if model_name == 'Model A' 
                       else recommender.recommend_model_b(session, k=k))
                
                round_has_5 = False
                for track_id, _ in recs:
                    rating = sim_user.rate_song(track_id)
                    session.append({'spotify_id': track_id, 'rating': rating})
                    all_ratings.append(rating)
                    
                    if rating == 5:
                        round_has_5 = True
                
                if round_has_5 and not found_5:
                    found_5 = True
                    first_5_round = r
            
            # Metric 1: Hit@k
            for k_val in hit_k_values:
                if len(all_ratings) >= k_val:
                    has_5 = 1 if (5 in all_ratings[:k_val]) else 0
                    results[model_name]['hits_at_k'][k_val].append(has_5)
                else:
                    results[model_name]['hits_at_k'][k_val].append(0)
            
            # Metric 2: Avg Rating
            results[model_name]['avg_ratings'].append(np.mean(all_ratings) if all_ratings else 0)
            
            # Metric 3: Time-to-5★
            results[model_name]['time_to_5'].append(first_5_round if first_5_round else rounds + 1)
    
    print(f"\nComplete!\n")
    return results


def print_results(results, hit_k_values):
    """Print formatted results."""
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    # Hit@k
    print("\nMETRIC 1: Hit@k")
    print("-" * 70)
    for k_val in sorted(hit_k_values):
        a = results['Model A']['hits_at_k'][k_val]
        b = results['Model B']['hits_at_k'][k_val]
        
        mean_a, mean_b = np.mean(a), np.mean(b)
        diff, (low, high), sig = compute_confidence_interval(a, b)
        
        print(f"\nHit@{k_val}:")
        print(f"  Model A: {mean_a:.4f} ({mean_a*100:.1f}%)")
        print(f"  Model B: {mean_b:.4f} ({mean_b*100:.1f}%)")
        print(f"  Diff: {diff:.4f} [{low:.4f}, {high:.4f}]")
        print(f"  Significant: {'YES ✓' if sig else 'NO'}")
        if sig:
            winner = "B" if diff < 0 else "A"
            print(f"  → Model {winner} wins by {abs(diff)*100:.1f}%")
    
    # Avg Rating
    print("\n" + "-" * 70)
    print("METRIC 2: Average Rating")
    print("-" * 70)
    a = results['Model A']['avg_ratings']
    b = results['Model B']['avg_ratings']
    
    mean_a, mean_b = np.mean(a), np.mean(b)
    diff, (low, high), sig = compute_confidence_interval(a, b)
    
    print(f"\n  Model A: {mean_a:.4f}")
    print(f"  Model B: {mean_b:.4f}")
    print(f"  Diff: {diff:.4f} [{low:.4f}, {high:.4f}]")
    print(f"  Significant: {'YES ✓' if sig else 'NO'}")
    if sig:
        print(f"  → Model {'B' if diff < 0 else 'A'} has higher satisfaction")
    
    # Time-to-5★
    print("\n" + "-" * 70)
    print("METRIC 3: Time-to-5★")
    print("-" * 70)
    a = results['Model A']['time_to_5']
    b = results['Model B']['time_to_5']
    
    mean_a, mean_b = np.mean(a), np.mean(b)
    diff, (low, high), sig = compute_confidence_interval(a, b)
    
    print(f"\n  Model A: {mean_a:.2f} rounds")
    print(f"  Model B: {mean_b:.2f} rounds")
    print(f"  Diff: {diff:.2f} [{low:.2f}, {high:.2f}]")
    print(f"  Significant: {'YES ✓' if sig else 'NO'}")
    if sig:
        if diff > 0:
            print(f"  → Model B is {abs(diff):.1f} rounds faster")
        else:
            print(f"  → Model A is {abs(diff):.1f} rounds faster")
    
    print("\n" + "=" * 70)


def save_results(results, hit_k_values, filename="metrics.txt"):
    """Save to file."""
    with open(filename, "w", encoding='utf-8') as f:
        f.write("Part 4: Monte Carlo Results\n")
        f.write("=" * 70 + "\n\n")
        
        for k_val in sorted(hit_k_values):
            a = results['Model A']['hits_at_k'][k_val]
            b = results['Model B']['hits_at_k'][k_val]
            diff, (low, high), sig = compute_confidence_interval(a, b)
            
            f.write(f"Hit@{k_val}:\n")
            f.write(f"  Model A: {np.mean(a):.4f}\n")
            f.write(f"  Model B: {np.mean(b):.4f}\n")
            f.write(f"  Diff: {diff:.4f} [{low:.4f}, {high:.4f}]\n")
            f.write(f"  Significant: {'YES' if sig else 'NO'}\n\n")
        
        a = results['Model A']['avg_ratings']
        b = results['Model B']['avg_ratings']
        diff, (low, high), sig = compute_confidence_interval(a, b)
        f.write(f"Average Rating:\n")
        f.write(f"  Model A: {np.mean(a):.4f}\n")
        f.write(f"  Model B: {np.mean(b):.4f}\n")
        f.write(f"  Diff: {diff:.4f} [{low:.4f}, {high:.4f}]\n")
        f.write(f"  Significant: {'YES' if sig else 'NO'}\n\n")
        
        a = results['Model A']['time_to_5']
        b = results['Model B']['time_to_5']
        diff, (low, high), sig = compute_confidence_interval(a, b)
        f.write(f"Time-to-5★:\n")
        f.write(f"  Model A: {np.mean(a):.2f}\n")
        f.write(f"  Model B: {np.mean(b):.2f}\n")
        f.write(f"  Diff: {diff:.2f} [{low:.2f}, {high:.2f}]\n")
        f.write(f"  Significant: {'YES' if sig else 'NO'}\n")


if __name__ == "__main__":
    print("\n📊 Part 4: Monte Carlo Evaluation (Collaborative Filtering)\n")
    
    # Determine data directory - try multiple paths
    current_dir = os.getcwd()
    possible_paths = [
        'data',
        '../data',
        'proje/data',
        './proje/data',
        os.path.join(os.path.dirname(__file__), '..', 'data'),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
    ]
    
    data_dir = None
    for path in possible_paths:
        if os.path.exists(path) and os.path.exists(os.path.join(path, 'ratings.csv')):
            data_dir = path
            break
    
    if not data_dir:
        print("ERROR: Cannot find data directory with ratings.csv!")
        print(f"Current directory: {current_dir}")
        print(f"Script location: {os.path.abspath(__file__)}")
        print("\nPlease run from:")
        print("  1. Project root: python src/part4_cf.py")
        print("  2. Src directory: python part4_cf.py")
        exit(1)
    
    print(f"✓ Found data directory: {os.path.abspath(data_dir)}\n")
    
    confirm = input("Start simulation? (y/n): ").strip().lower()
    
    if confirm == 'y':
        results = run_simulation(
            n_users=100,
            rounds=20,
            k=5,
            data_dir=data_dir,
            hit_k_values=[5, 10, 20]
        )
        
        if results:  # Check if simulation succeeded
            print_results(results, [5, 10, 20])
            save_results(results, [5, 10, 20])
            
            print("\n💾 Saved to metrics.txt")
            print("✅ Done!\n")
        else:
            print("\n❌ Simulation failed!")
    else:
        print("Cancelled.")
