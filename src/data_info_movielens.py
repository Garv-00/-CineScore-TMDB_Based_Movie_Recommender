import pandas as pd
import re

# Load MovieLens data
movies = pd.read_csv("data/movielens/movies.csv")
ratings = pd.read_csv("data/movielens/ratings.csv")
tags = pd.read_csv("data/movielens/tags.csv")
genome_tags = pd.read_csv("data/movielens/genome-tags.csv")
genome_scores = pd.read_csv("data/movielens/genome-scores.csv")
links = pd.read_csv("data/movielens/links.csv")

# Load CineScore TMDB data
tmdb = pd.read_csv("data/tmdb/movies.csv")

# Audit MovieLens movies
print("\nMovies")
print("Rows:", len(movies))
print("Unique movies:", movies["movieId"].nunique())
print("Missing values:", movies.isna().sum().sum())

# Audit ratings
print("\nRatings")
print("Rows:", len(ratings))
print("Unique users:", ratings["userId"].nunique())
print("Unique movies:", ratings["movieId"].nunique())
print("Rating range:", ratings["rating"].min(), "-", ratings["rating"].max())
print("Average rating:", round(ratings["rating"].mean(), 3))
print("Duplicate user-movie pairs:",
      ratings.duplicated(["userId", "movieId"]).sum())

# Audit tags
print("\nTags")
print("Rows:", len(tags))
print("Unique users:", tags["userId"].nunique())
print("Unique movies:", tags["movieId"].nunique())
print("Unique tags:", tags["tag"].nunique())
print("Missing tags:", tags["tag"].isna().sum())

# Audit genome data
print("\nGenome")
print("Tags:", genome_tags["tagId"].nunique())
print("Score rows:", len(genome_scores))
print("Movies:", genome_scores["movieId"].nunique())
print("Tags:", genome_scores["tagId"].nunique())
print("Relevance range:",
      genome_scores["relevance"].min(),
      "-",
      genome_scores["relevance"].max())
print("Duplicate movie-tag pairs:",
      genome_scores.duplicated(["movieId", "tagId"]).sum())

# Audit MovieLens to TMDB links
print("\nMovieLens → TMDB")
print("Rows:", len(links))
print("Movies with TMDB ID:", links["tmdbId"].notna().sum())
print("Missing TMDB IDs:", links["tmdbId"].isna().sum())
print("Duplicate TMDB IDs:",
      links["tmdbId"].dropna().duplicated().sum())

print(
    "Genome coverage:",
    f"{genome_scores['movieId'].nunique() / movies['movieId'].nunique() * 100:.2f}%"
)

print(
    "TMDB link coverage:",
    f"{links['tmdbId'].notna().mean() * 100:.2f}%"
)

# Find MovieLens movies without a TMDB ID
missing_tmdb = movies.merge(
    links[["movieId", "tmdbId"]],
    on="movieId",
    how="left",
    validate="one_to_one"
)

missing_tmdb = missing_tmdb[
    missing_tmdb["tmdbId"].isna()
].copy()

print("\nMissing TMDB IDs:", len(missing_tmdb))

# Normalize titles for local TMDB matching
def normalize_title(title):
    title = str(title).lower()
    title = re.sub(r"\s*\(\d{4}\)\s*$", "", title)
    title = re.sub(r"[^a-z0-9]+", " ", title)
    return re.sub(r"\s+", " ", title).strip()

missing_tmdb["title_normalized"] = (
    missing_tmdb["title"].apply(normalize_title)
)

tmdb["title_normalized"] = (
    tmdb["title"].apply(normalize_title)
)

# Check whether missing movies exist in the CineScore TMDB dataset
tmdb_titles = set(tmdb["title_normalized"])

found = missing_tmdb[
    missing_tmdb["title_normalized"].isin(tmdb_titles)
]

print("\nLocal TMDB matching")
print("Found:", len(found))
print("Not found:", len(missing_tmdb) - len(found))

# Create MovieLens movie-level master
movies["release_year"] = (
    movies["title"]
    .str.extract(r"\((\d{4})\)\s*$")[0]
)

movielens_master = movies.merge(
    links,
    on="movieId",
    how="left",
    validate="one_to_one"
)

movielens_master = movielens_master[
    [
        "movieId",
        "tmdbId",
        "imdbId",
        "title",
        "genres",
        "release_year"
    ]
].rename(
    columns={
        "tmdbId": "tmdb_id",
        "imdbId": "imdb_id"
    }
)

movielens_master.to_csv(
    "data/movielens/movielens_master.csv",
    index=False
)

print("\nMovieLens master")
print("Rows:", len(movielens_master))
print("Movies with TMDB ID:",
      movielens_master["tmdb_id"].notna().sum())

# Create MovieLens interaction data
movielens_interactions = ratings.merge(
    movielens_master[["movieId", "tmdb_id"]],
    on="movieId",
    how="left",
    validate="many_to_one"
)

movielens_interactions["user_id"] = (
    "ML:" + movielens_interactions["userId"].astype(str)
)

movielens_interactions = movielens_interactions[
    [
        "user_id",
        "movieId",
        "tmdb_id",
        "rating",
        "timestamp"
    ]
]

movielens_interactions.to_csv(
    "data/movielens/movielens_interactions.csv",
    index=False
)

print("\nMovieLens interactions")
print("Rows:", len(movielens_interactions))
print(
    "Interactions with TMDB ID:",
    movielens_interactions["tmdb_id"].notna().sum()
)

print("\nFiles created:")
print("data/movielens/movielens_master.csv")
print("data/movielens/movielens_interactions.csv")