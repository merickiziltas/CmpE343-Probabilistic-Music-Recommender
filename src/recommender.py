"""
Music Recommendation System for Tune Duel Competition

This module implements the main recommendation function for the Tune Duel platform.
It uses the 'Patience-Aware Sampler' (Model B) designed in Part 3.
"""

from typing import List, Dict, Tuple, Any
import os
import sys

# Ensure we can import from the same directory
sys.path.append(os.path.dirname(__file__))

try:
    from part3 import MusicRecommender
except ImportError:
    # Fallback if running from root
    from src.part3 import MusicRecommender

# Global Singleton for the Recommender System
# We load it once to avoid reloading CSVs on every query
REC_SYS = None

def get_recommender():
    """Lazy loader for the recommender system."""
    global REC_SYS
    if REC_SYS is None:
        # Determine data path (handle submission vs local dev structure)
        data_dir = 'data'
        if not os.path.exists(data_dir) and os.path.exists('../data'):
            data_dir = '../data'
        elif not os.path.exists(data_dir) and os.path.exists('cmpee343miniproject/project-108-301-110/data'): # Absolute fallback
             data_dir = 'cmpee343miniproject/project-108-301-110/data'
             
        REC_SYS = MusicRecommender(data_dir)
    return REC_SYS

def query(song_ratings: List[Dict[str, Any]], topk: int = 5, model_type: str = 'b') -> List[Tuple[str, str]]:
    """
    Generate recommendations based on user's song ratings.
    
    Args:
        song_ratings: List of dicts with 'song', 'rating', and 'spotify_id' keys
        topk: Number of recommendations to return
        model_type: 'a' for Deterministic/Personalized, 'b' for Probabilistic/Patience-Aware
        
    Returns:
        List of (spotify_id, track_name) tuples
    """
    recommender = get_recommender()
    
    if model_type.lower() == 'a':
        return recommender.recommend_model_a(song_ratings, k=topk)
    else:
        # Default to Model B
        return recommender.recommend_model_b(song_ratings, k=topk)

def search_song(rec, query_str):
    """Search for a song in the tracks database."""
    # Case insensitive search
    mask = rec.tracks_df['track_name'].str.contains(query_str, case=False, na=False)
    matches = rec.tracks_df[mask]
    
    # Also search artist if no track match? Or just simple track search.
    # Let's stick to track name for simplicity first, sorting by popularity
    matches = matches.sort_values('track_popularity', ascending=False)
    return matches.head(5)

def test_recommender():
    """Interactive CLI to test the recommender."""
    print("Initializing Recommender System (Loading Data...)\n")
    try:
        rec = get_recommender()
        print("--- Tune Duel Recommender Test ---")
        print("Enter songs you like/dislike to build a profile.")
        print("Type 'done' when finished.\n")
        
        my_ratings = []
        
        while True:
            query_str = input("Search Song Name (or 'done'): ").strip()
            if query_str.lower() == 'done':
                break
            if not query_str:
                continue
                
            matches = search_song(rec, query_str)
            
            if matches.empty:
                print("  [!] Song not found. Try another.")
                continue
                
            print(f"\nSelect the correct song:")
            options = []
            for i, (_, row) in enumerate(matches.iterrows(), 1):
                artist = row['artist_names']
                print(f"  {i}. {row['track_name']} - {artist} (Pop: {row['track_popularity']})")
                options.append(row)
                
            try:
                sel = input("Enter Number (0 to cancel): ").strip()
                if sel == '0': continue
                selection_idx = int(sel) - 1
                
                if 0 <= selection_idx < len(options):
                    selected_track = options[selection_idx]
                    
                    # Get Rating
                    while True:
                        try:
                            r_str = input(f"Rate '{selected_track['track_name']}' (1-5): ").strip()
                            rating = int(r_str)
                            if 1 <= rating <= 5:
                                break
                            print("  Please enter 1-5.")
                        except ValueError:
                            pass
                            
                    # Add to profile
                    my_ratings.append({
                        'song': selected_track['track_name'],
                        'rating': rating,
                        'spotify_id': selected_track['track_id']
                    })
                    print(f"  [+] Added: {selected_track['track_name']} ({rating}*)\n")
                else:
                    print("  Invalid selection.")
            except ValueError:
                print("  Invalid input.")
                
        if not my_ratings:
            print("No ratings provided. Exiting.")
            return

        print(f"\nGeneratin Recommendations for {len(my_ratings)} songs...")
        
        # Helper to get artist name
        # We assume rec.tracks_df has 'track_id' and 'artist_names' (or 'primary_artist_name')
        def get_artist(tid):
            try:
                # Fast lookup if possible, or filtered
                # For speed in test, we can just filter
                rows = rec.tracks_df[rec.tracks_df['track_id'] == tid]
                if not rows.empty:
                    # Prefer full artist names string if available
                    if 'artist_names' in rows.columns:
                        return rows.iloc[0]['artist_names']
                    return rows.iloc[0]['primary_artist_name']
                return "Unknown Artist"
            except:
                return "Unknown"

        # 2. Test Model A
        print(f"\n[Model A] Deterministic / Feature Matcher:")
        recs_a = query(my_ratings, topk=5, model_type='a')
        for i, (tid, name) in enumerate(recs_a, 1):
            artist = get_artist(tid)
            print(f"  {i}. {name} - {artist}")
            
        # 3. Test Model B
        print(f"\n[Model B] Patience-Aware Sampler (Competition Model):")
        recs_b = query(my_ratings, topk=5, model_type='b')
        for i, (tid, name) in enumerate(recs_b, 1):
            artist = get_artist(tid)
            print(f"  {i}. {name} - {artist}")
            
    except Exception as e:
        print(f"Test Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_recommender()
