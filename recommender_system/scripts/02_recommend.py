import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
import json

class CineScoreRecommender:
    def __init__(self, model_path, data_dir=None):
        """Initialize recommender with trained model"""
        self.model_path = Path(model_path)
        self.data_dir = Path(data_dir) if data_dir else Path(__file__).resolve().parents[2] / "recommender_system/data/processed"

        # Load model
        self.model = lgb.Booster(model_file=str(self.model_path))

        # Load data
        self._load_data()

    def _load_data(self):
        """Load necessary data"""
        # Load TMDB catalog
        catalog_path = self.data_dir.parent / "tmdb_catalog.csv"
        self.catalog = pd.read_csv(catalog_path)
        self.movie_id_map = {int(mid): title for mid, title in zip(self.catalog['tmdb_id'], self.catalog['title'])}

    def recommend(self, user_id, kion_history=None, ml_history=None, top_k=10):
        """
        Generate recommendations for a user

        Args:
            user_id: User identifier
            kion_history: List of KION movie IDs watched
            ml_history: List of MovieLens movie IDs watched
            top_k: Number of recommendations to return

        Returns:
            List of (movie_id, title, score) tuples
        """
        # Combine history
        if kion_history is None:
            kion_history = []
        if ml_history is None:
            ml_history = []

        watched_movies = set(kion_history) | set(ml_history)

        # Get all movies not in history
        all_movies = set(self.catalog['tmdb_id'].values)
        candidate_movies = list(all_movies - watched_movies)

        if not candidate_movies:
            return []

        # Create feature matrix
        features = []
        for movie_id in candidate_movies:
            # Simple features: [tmdb_id, source]
            # Using source 0.5 for combined recommendations
            features.append([float(movie_id), 0.5])

        features = np.array(features, dtype=np.float32)

        # Score with model
        scores = self.model.predict(features)

        # Get top-k
        top_indices = np.argsort(scores)[-top_k:][::-1]

        recommendations = []
        for idx in top_indices:
            movie_id = candidate_movies[idx]
            title = self.movie_id_map.get(movie_id, f"Movie {movie_id}")
            score = float(scores[idx])
            recommendations.append({
                'movie_id': int(movie_id),
                'title': title,
                'score': score
            })

        return recommendations

def main():
    """Example usage"""
    ROOT = Path(__file__).resolve().parents[2]
    model_path = ROOT / "recommender_system/model/cinescore_final.txt"
    data_dir = ROOT / "recommender_system/data/processed"

    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        return

    print("="*80)
    print("CineScore Recommendation Engine")
    print("="*80)

    # Initialize recommender
    rec = CineScoreRecommender(model_path, data_dir)

    # Example recommendation
    print("\nExample: Generating recommendations for user...")
    recommendations = rec.recommend(
        user_id="example_user",
        kion_history=[1, 2, 5],
        ml_history=[10, 20],
        top_k=10
    )

    print(f"\nTop 10 Recommendations:")
    for i, rec_item in enumerate(recommendations, 1):
        print(f"{i}. {rec_item['title']} (ID: {rec_item['movie_id']}, Score: {rec_item['score']:.4f})")

if __name__ == "__main__":
    main()
