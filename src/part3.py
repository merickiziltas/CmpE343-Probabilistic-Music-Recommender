import pandas as pd
import numpy as np
import random
import os
import sys

# Import helpers from Part 1 and Part 2
try:
    from part1 import load_data, create_binned_features, compute_conditional_prob
    from part2 import calculate_time_to_favorite
except ImportError:
    # Handle running from src/ directory vs root
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from src.part1 import load_data, create_binned_features, compute_conditional_prob
    from src.part2 import calculate_time_to_favorite

class MusicRecommender:
    def __init__(self, data_dir='data'):
        """
        Initialize the recommender system.
        Loads data and pre-computes global probabilities (Training Phase).
        """
        print("Initializing MusicRecommender...")
        self.merged_df, self.tracks_df, self.ratings_df = load_data(data_dir)
        
        # Apply binning to tracks for feature matching
        self.tracks_df = create_binned_features(self.tracks_df)
        self.merged_df = create_binned_features(self.merged_df)
        
        # Pre-compute Global Probabilities (Part 1 Logic) for Efficiency
        # We include ALL relevant features now as requested
        self.features_to_use = [
            'primary_artist_name',
            'popularity_bin', 
            'duration_bin', 
            'explicit',
            'album_release_year',
            'ab_danceability_value', 
            'ab_mood_acoustic_value', 
            'ab_mood_aggressive_value',
            'ab_mood_electronic_value', 
            'ab_mood_happy_value', 
            'ab_mood_party_value',
            'ab_mood_relaxed_value', 
            'ab_mood_sad_value', 
            'ab_gender_value',
            'ab_voice_instrumental_value', 
            'ab_timbre_value',
            'ab_genre_dortmund_value', 
            'ab_genre_rosamerica_value'
        ]
        
        self.feature_probs = {}
        for feature in self.features_to_use:
            # Check if column exists to avoid errors
            if feature in self.merged_df.columns:
                stats = compute_conditional_prob(self.merged_df, feature, target_rating=5)
                # Convert to dictionary for O(1) lookup: {value: probability}
                self.feature_probs[feature] = pd.Series(
                    stats.probability.values, index=stats[feature]
                ).to_dict()
            
        # Global 5-star probability (Baseline) for fillna
        total_5 = len(self.ratings_df[self.ratings_df['rating'] == 5])
        self.global_p5 = total_5 / len(self.ratings_df) if len(self.ratings_df) > 0 else 0.2
        print("Initialization Complete.")

    def _get_user_history_df(self, user_ratings):
        """
        Convert user ratings list to a DataFrame merged with track features.
        We need ALL ratings (1-5) to calculate positive and negative impacts.
        """
        if not user_ratings:
            return pd.DataFrame()
            
        user_df = pd.DataFrame(user_ratings)
        
        # Merge with track info to get features
        history_df = pd.merge(
            user_df, 
            self.tracks_df, 
            left_on='spotify_id', 
            right_on='track_id', 
            how='inner'
        )
        return history_df

    def recommend_model_a(self, user_ratings, k=5, return_full_scores=False, exclude_ids=None):
        """
        Model A: Deterministic / Feature Matcher.
        Returns top-k songs based on personalized probability P(5* | features).
        
        exclude_ids: Optional set/list of track_ids to exclude (e.g., seen but unrated).
        """
        # 1. Identify Rated Songs to exclude
        rated_ids = set([r['spotify_id'] for r in user_ratings])
        if exclude_ids:
            rated_ids.update(exclude_ids)
            
        # 2. Get Candidates (All tracks - Rated/Excluded)
        # Assuming self.tracks_df contains all candidates
        # For efficiency, we can filter indices if dataset is huge, but here pandas is fine
        candidates = self.tracks_df[~self.tracks_df['track_id'].isin(rated_ids)].copy()
        
        history_df = self._get_user_history_df(user_ratings)
        
        # --- 1. CONFIGURATION ---
        
        # How much does the Rating Matter? (Direction & Intensity)
        RATING_IMPACT = {
            5:  1.5,   # Strong Love
            4:  1.0,   # Like
            3:  0.0,   # Neutral (No effect)
            2: -1.0,   # Dislike
            1: -1.5    # Hate (Strong Penalty)
        }
        
        # How important is the Feature itself? (Feature Hierarchy)
        BOOST_WEIGHTS = {
            'primary_artist_name': 1.25,       
            'ab_genre_rosamerica_value': 2.25, # Genre is Queen
            'ab_genre_dortmund_value': 1.8,
            'album_release_year': 1.6,
            
            # Moods & Tone (The Vibe)
            'ab_mood_happy_value': 1.2,
            'ab_mood_sad_value': 1.2,
            'ab_mood_party_value': 1.2,
            'ab_mood_aggressive_value': 1.2,
            'ab_mood_relaxed_value': 1.2,
            'ab_mood_acoustic_value': 1.2,
            'ab_mood_electronic_value': 1.2,
            'ab_danceability_value': 1.2,
            'ab_voice_instrumental_value': 1.2,
            'ab_timbre_value': 1.2,
            'ab_gender_value': 1.0,
            
            'popularity_bin': 1.0,
            'duration_bin': 0.75,
            'explicit': 1.0
        }

        # Start with all tracks as candidates
        candidates = self.tracks_df.copy()
        
        # Initialize Score with Base Probability (The "Quality" Baseline)
        # We sum up weighted base probabilities first so good songs float to top naturally
        candidates['score'] = 0.0
        for feature in self.features_to_use:
            if feature in candidates.columns:
                 prob_map = self.feature_probs.get(feature, {})
                 # Add small base score (0.1 weight) just to differentiate unrated territory
                 candidates['score'] += candidates[feature].map(prob_map).fillna(self.global_p5) * 0.1

        if history_df.empty:
            # Cold start: return top by base score
            recs = candidates.sort_values('score', ascending=False).head(k)
            return list(zip(recs['track_id'], recs['track_name']))

        # --- 2. CALCULATE SENTIMENT VOTES ---
        
        # Iterate through user's history to build a "Vote Map" for features
        # Structure: {feature_name: {value: total_impact_score}}
        feature_votes = {}
        
        for _, row in history_df.iterrows():
            rating = row['rating']
            impact = RATING_IMPACT.get(rating, 0.0)
            
            if impact == 0: continue # Skip neutral
            
            for feature in self.features_to_use:
                val = row.get(feature)
                if pd.isna(val): continue
                
                weight = BOOST_WEIGHTS.get(feature, 1.0)
                vote_power = impact * weight
                
                # Special Year Logic (Spread the vote to neighbors)
                if feature == 'album_release_year':
                    try:
                        y_int = int(val)
                        # Spread vote: Full power to exact year, half power to neighbors
                        year_range = list(range(y_int - 5, y_int + 6))
                        if feature not in feature_votes: feature_votes[feature] = {}
                        
                        for y in year_range:
                            decay = 1.0 if y == y_int else 0.5
                            current = feature_votes[feature].get(y, 0.0)
                            feature_votes[feature][y] = current + (vote_power * decay)
                    except: pass
                else:
                    # Standard Feature
                    if feature not in feature_votes: feature_votes[feature] = {}
                    current = feature_votes[feature].get(val, 0.0)
                    feature_votes[feature][val] = current + vote_power

        # --- 3. APPLY VOTES TO CANDIDATES ---
        
        # Now apply these votes to the candidate songs
        # Vectorized application is tricky with dicts, so we map
        for feature, vote_dict in feature_votes.items():
            if feature not in candidates.columns: continue
            
            # Create a localized series for mapping
            # This maps the song's feature value to the aggregated User Vote Score
            vote_series = candidates[feature].map(vote_dict).fillna(0.0)
            
            # Add to total score (Scale by prob to ensure quality alignment)
            # P(5*|F) acts as a confidence scaling here. 
            # If user loves "Rock" (Vote +6) AND Rock is generally high quality (0.8), score += 4.8
            prob_map = self.feature_probs.get(feature, {})
            feature_confidence = candidates[feature].map(prob_map).fillna(self.global_p5)
            
            candidates['score'] += vote_series * feature_confidence

        # Remove songs user has already rated/seen
        rated_ids = set([r['spotify_id'] for r in user_ratings])
        if exclude_ids:
            rated_ids.update(exclude_ids)
        candidates = candidates[~candidates['track_id'].isin(rated_ids)]
        
        # Sort by Score (Deterministic)
        sorted_candidates = candidates.sort_values('score', ascending=False)
        
        if return_full_scores:
            # For Model B, we want the raw dataframe with scores
            return sorted_candidates

        # --- DEBUG: Write full scores to check.txt ---
        try:
            debug_cols = ['track_name', 'primary_artist_name', 'score']
            # Ensure columns exist
            cols_to_write = [c for c in debug_cols if c in sorted_candidates.columns]
            sorted_candidates[cols_to_write].to_csv('check.txt', sep='\t', index=False, float_format='%.4f')
            print("  [Model A] Detailed scores written to 'check.txt'")
        except Exception as e:
            print(f"  [Model A] Could not write check.txt: {e}")
            
        recommendations = sorted_candidates.head(k)
        
        # Return format: (id, "Name (Id: ...)")
        return [
            (tid, f"{tname} (Id: {tid})") 
            for tid, tname in zip(recommendations['track_id'], recommendations['track_name'])
        ]

    def recommend_model_b(self, user_ratings, k=5, exclude_ids=None):
        """
        Model B: Hybrid Utility-Based Sampling (Probabilistic & Continuous Patience).
        
        Strategy:
        1. Reuse Model A to get Personalized Scores (Uu).
        2. Calculate User Patience (Tu) using Part 2 Metric.
        3. Apply Continuous Risk Adjustment:
           - Calculate 'Risk' based on Score confidence.
           - Adjust prob based on (Tu - Average_Tu).
        4. Probabilistic Sampling.
        """
        # 1. Get Personalized Utility from Model A
        # This gives us scores like +5.2, -1.0, +12.5 etc.
        # We assume recommend_model_a returns the candidates dataframe when return_full_scores=True
        candidates = self.recommend_model_a(user_ratings, k=k, return_full_scores=True, exclude_ids=exclude_ids)
        
        # 2. Calculate User Patience (Tu) - Real Time
        temp_ratings = []
        for i, r in enumerate(user_ratings):
            temp_ratings.append({
                'round_idx': i + 1,
                'rating': r['rating']
            })
        temp_df = pd.DataFrame(temp_ratings)
        
        try:
            five_stars = temp_df[temp_df['rating'] == 5]
            if not five_stars.empty:
                T_u = five_stars['round_idx'].min()
            else:
                # If no 5-star yet, assume current round + 1 (Optimistic)
                # or a high number if they rated many without 5*
                T_u = len(user_ratings) + 1
        except:
            T_u = 4.0 # Fallback near average
            
        # 3. Continuous Patience Adjustment
        # Pivot Point from Part 2 Data: 3.7
        AVG_TU_THRESHOLD = 3.7
        
        # Patience Factor: How far is user from average?
        # T_u = 1 (Impatient) -> Factor = -2.7
        # T_u = 10 (Patient)  -> Factor = +6.3
        patience_factor = T_u - AVG_TU_THRESHOLD
        
        # Normalize Base Scores to Positive Probabilities (Softmax-like or simple MinMax)
        # We use simple MinMax to keep relative distances but ensure positivity for sampling
        min_score = candidates['score'].min()
        max_score = candidates['score'].max()
        
        # Avoid zero division
        if max_score - min_score == 0:
            candidates['prob_score'] = 1.0
        else:
            candidates['prob_score'] = (candidates['score'] - min_score) / (max_score - min_score)
            
        # Apply Additive Smoothing Logic (User Request):
        # Adjusted = Score + (Factor * Constant)
        # Patient (+Factor) -> Adds base value -> Ratios flatten -> EXPLORATION
        # Impatient (-Factor) -> Subtracts value -> Low scores hit 0 -> EXPLOITATION
        
        ADDITIVE_CONSTANT = 0.05
        adjustment = patience_factor * ADDITIVE_CONSTANT
        
        # Apply and Clip (Cannot have negative probability)
        candidates['final_weight'] = (candidates['prob_score'] + adjustment).clip(lower=0.0)
        
        # --- DEBUG: Write final probabilities to checkb.txt ---
        try:
            debug_cols = ['track_name', 'primary_artist_name', 'score', 'prob_score', 'final_weight']
            # Ensure columns exist
            cols_to_write = [c for c in debug_cols if c in candidates.columns]
            
            # Sort by Final Weight (Probability) to see what Model B prefers
            debug_df = candidates.sort_values('final_weight', ascending=False)
            debug_df[cols_to_write].to_csv('checkb.txt', sep='\t', index=False, float_format='%.4f')
            print("  [Model B] Detailed probabilities written to 'checkb.txt'")
        except Exception as e:
            print(f"  [Model B] Could not write checkb.txt: {e}")

        # 4. Probabilistic Sampling
        # Normalize to probability distribution
        total_weight = candidates['final_weight'].sum()
        if total_weight == 0:
             weights = None # Uniform
        else:
             weights = candidates['final_weight'] / total_weight
        
        try:
            sample = candidates.sample(n=k, weights=weights, replace=False)
        except ValueError:
             # Fallback
             sample = candidates.head(k)
             
        # Return format: (id, "Name (Id: ...)")
        return [
            (tid, f"{tname} (Id: {tid})") 
            for tid, tname in zip(sample['track_id'], sample['track_name'])
        ]


