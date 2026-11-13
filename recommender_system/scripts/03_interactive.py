import numpy as np
import pandas as pd
from pathlib import Path
import re
from difflib import SequenceMatcher

class MovieResolver:
    """Resolves movie names to movie IDs"""

    def __init__(self, catalog_path):
        """Load movie catalog"""
        catalog = pd.read_csv(catalog_path)
        self.movies_by_id = {int(mid): {'title': str(t), 'year': int(y) if pd.notna(y) else 0}
                            for mid, t, y in zip(catalog['tmdb_id'], catalog['title'], catalog['release_year'])}
        self.movies_list = list(self.movies_by_id.items())

    def normalize_title(self, title):
        """Normalize movie title for comparison"""
        # Convert to lowercase
        title = title.lower()
        # Remove extra spaces
        title = re.sub(r'\s+', ' ', title).strip()
        # Remove hyphens and treat as spaces
        title = title.replace('-', ' ')
        return title

    def similarity_score(self, input_title, catalog_title):
        """Calculate similarity between two titles"""
        input_norm = self.normalize_title(input_title)
        catalog_norm = self.normalize_title(catalog_title)

        # Exact match after normalization
        if input_norm == catalog_norm:
            return 1.0

        # Substring match
        if input_norm in catalog_norm or catalog_norm in input_norm:
            return 0.9

        # Sequence matching
        return SequenceMatcher(None, input_norm, catalog_norm).ratio()

    def resolve_movie(self, movie_input):
        """
        Resolve user input to movie ID(s)

        Returns:
            - Single (movie_id, movie_info) tuple if exact match found
            - List of (movie_id, movie_info) tuples for fuzzy matches
            - None if no matches
        """
        if not movie_input or not isinstance(movie_input, str):
            return None

        # Calculate similarity scores
        scores = []
        for movie_id, movie_info in self.movies_list:
            score = self.similarity_score(movie_input, movie_info['title'])
            scores.append((score, movie_id, movie_info))

        # Sort by score descending
        scores.sort(reverse=True, key=lambda x: x[0])

        # If best score is very high, return it as primary match
        if scores and scores[0][0] > 0.85:
            return (scores[0][1], scores[0][2])

        # Otherwise return top 5 fuzzy matches
        if scores and scores[0][0] > 0.5:
            return [(sid, sinfo) for _, sid, sinfo in scores[:5]]

        return None


class CineScoreRecommender:
    """Interactive movie recommender"""

    def __init__(self):
        """Initialize recommender"""
        self.root = Path(__file__).resolve().parents[2]
        self.model_dir = self.root / "recommender_system/model"
        self.data_dir = self.root / "recommender_system/data/processed"

        print("[1/3] Loading movies...", end='', flush=True)
        catalog_path = self.data_dir / "tmdb_catalog.csv"
        self.resolver = MovieResolver(catalog_path)
        print(" Done")

        print("[2/3] Loading embeddings...", end='', flush=True)
        embeddings_path = self.model_dir / "tmdb_embeddings.npy"
        self.embeddings = np.load(embeddings_path).astype(np.float32)
        # Normalize
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.embeddings = self.embeddings / np.maximum(norms, 1e-8)
        print(" Done")

        print("[3/3] Computing similarity...", end='', flush=True)
        from sklearn.metrics.pairwise import cosine_similarity
        self.similarity_matrix = cosine_similarity(self.embeddings)
        self.id_to_idx = {mid: i for i, (mid, _) in enumerate(self.resolver.movies_list)}
        print(" Done\n")

        self.print_header()

    def print_header(self):
        """Print help"""
        print("=" * 80)
        print("CineScore - Movie Recommendations")
        print("=" * 80)
        print("\nUsage:")
        print("  rec <Movie Name>           - Get recommendations (10 default)")
        print("  rec <Movie Name> [k]       - Get k recommendations")
        print("  info <Movie Name>          - Show movie information")
        print("  search <Movie Name>        - Search for movie")
        print("  <Movie Name>               - Get recommendations (same as rec)")
        print("\nExamples:")
        print("  rec Iron Man")
        print("  info Inception")
        print("  search Batman")
        print("  The Matrix")
        print("\nCommands:")
        print("  help                       - Show this help")
        print("  quit/exit                  - Exit")
        print("=" * 80 + "\n")

    def parse_input(self, user_input):
        """
        Parse user input into (command, movie_name, parameter)

        Returns:
            (command, movie_name, parameter) or None if invalid
        """
        if not user_input or not isinstance(user_input, str):
            return None

        parts = user_input.strip().split(maxsplit=1)
        if not parts:
            return None

        first_word = parts[0].lower()
        supported_commands = ['rec', 'info', 'search', 'help', 'quit', 'exit']

        # If first word is a command
        if first_word in supported_commands:
            command = first_word
            rest = parts[1] if len(parts) > 1 else ""

            # Parse rest for movie name and optional parameter (k value)
            rest_parts = rest.rsplit(maxsplit=1)

            if not rest_parts:
                return None

            movie_name = rest_parts[0]
            parameter = None

            # Check if last part is a number (k value for rec command)
            if len(rest_parts) > 1:
                try:
                    parameter = int(rest_parts[1])
                    movie_name = rest_parts[0]
                except ValueError:
                    # Not a number, treat whole thing as movie name
                    movie_name = rest

            return (command, movie_name, parameter)

        # If first word is not a command, assume it's a movie name
        # Treat as "rec" command by default
        movie_name = user_input.strip()

        # Check if last word might be a parameter (k value)
        parts = movie_name.rsplit(maxsplit=1)
        parameter = None

        if len(parts) > 1:
            try:
                parameter = int(parts[1])
                movie_name = parts[0]
            except ValueError:
                pass

        return ('rec', movie_name, parameter)

    def get_recommendations(self, movie_id, k=10):
        """Get k similar movies"""
        if movie_id not in self.id_to_idx:
            return None

        idx = self.id_to_idx[movie_id]
        similarities = self.similarity_matrix[idx]

        # Get top k+1 (excluding movie itself)
        top_indices = np.argsort(similarities)[::-1][:k + 1]

        recommendations = []
        for top_idx in top_indices:
            if top_idx == idx:
                continue

            top_movie_id = self.resolver.movies_list[top_idx][0]
            top_movie_info = self.resolver.movies_list[top_idx][1]
            score = float(similarities[top_idx])

            recommendations.append({
                'id': top_movie_id,
                'title': top_movie_info['title'],
                'year': top_movie_info['year'],
                'similarity': score
            })

            if len(recommendations) == k:
                break

        return recommendations

    def handle_command(self, command, movie_name, parameter=None):
        """Handle user command"""

        # Help
        if command == 'help':
            self.print_header()
            return

        # Quit
        if command in ['quit', 'exit']:
            print("Goodbye!")
            return 'quit'

        # Resolve movie name
        resolved = self.resolver.resolve_movie(movie_name)

        if resolved is None:
            print(f"\nNo movies found matching '{movie_name}'")
            print("Try a different search term.\n")
            return

        # Handle multiple results (fuzzy matches)
        if isinstance(resolved, list):
            print(f"\nMultiple matches for '{movie_name}':")
            print("-" * 80)
            for i, (mid, info) in enumerate(resolved, 1):
                print(f"{i}. {info['title']:<50} ({info['year']})")
            print("\nTry searching with a more specific name.\n")
            return

        # Single match found
        movie_id, movie_info = resolved

        if command == 'rec':
            k = parameter if parameter else 10
            recs = self.get_recommendations(movie_id, k)

            if recs:
                print(f"\nRecommendations for: {movie_info['title']} ({movie_info['year']})")
                print("-" * 80)
                for i, rec in enumerate(recs, 1):
                    print(f"{i:2d}. {rec['title']:<50} ({rec['year']}) [Similarity: {rec['similarity']:.4f}]")
                print()
            else:
                print(f"Could not generate recommendations.\n")

        elif command == 'info':
            print(f"\nMovie Information:")
            print("-" * 80)
            print(f"Title: {movie_info['title']}")
            print(f"Year:  {movie_info['year']}")
            print(f"ID:    {movie_id}")
            print()

        elif command == 'search':
            print(f"\nFound: {movie_info['title']} ({movie_info['year']})")
            print(f"ID: {movie_id}\n")

    def run(self):
        """Interactive loop"""
        while True:
            try:
                user_input = input("cinescore> ").strip()

                if not user_input:
                    continue

                parsed = self.parse_input(user_input)

                if parsed is None:
                    print("Invalid input. Type 'help' for commands.\n")
                    continue

                command, movie_name, parameter = parsed
                result = self.handle_command(command, movie_name, parameter)

                if result == 'quit':
                    break

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}\n")


if __name__ == "__main__":
    try:
        print("\nInitializing CineScore Movie Recommender...\n")
        app = CineScoreRecommender()
        app.run()
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
