# CineScore

A learning-to-rank movie recommender trained on **18M interactions**. Type a movie name, get ten similar ones back, ranked by a LightGBM LambdaMART model that scores **NDCG@10 = 0.8366** on a leakage-free evaluation.

| | |
|---|---|
| Model | LightGBM LambdaMART ranker |
| Candidate sources | Semantic embeddings, TF-IDF, collaborative filtering |
| Training data | ~18M interactions (KION + MovieLens) |
| Catalog | 9,964 movies (TMDB metadata) |
| NDCG@10 | **0.8366** |
| Response time | ~200-500 ms per recommendation |

## What it does

- Search by movie name, no IDs needed. "Iron Man", "ironman" and "IRON MAN" all resolve to the same movie.
- Returns the top-k similar movies with similarity scores.
- Interactive CLI with `rec`, `info`, `search`, `help` and `quit`.

## The problem

Given one movie a person likes, which movies should they see next? Plain genre matching gives repetitive results, and pure collaborative filtering struggles with movies that have few ratings. CineScore combines what a movie is about, its keywords and what similar viewers watched, then lets a learned ranker decide the final order.

## How it works

```
movie name -> name resolution -> 3 candidate retrievers -> merged pool -> LambdaMART ranker -> top-k
```

1. **Name resolution.** The typed name is normalized so spelling and casing variations hit the same movie.
2. **Candidate generation.** Three retrievers each pull a pool of movies, so the ranker sees a wide and varied set:
   - **Semantic similarity** on sentence embeddings of each movie's text. Movies with similar plots land close together even when they share no keywords.
   - **TF-IDF** keyword matching, which catches exact term overlap that embeddings can miss.
   - **Collaborative filtering** from the interaction data, which captures what the same people actually watch.
3. **Ranking.** A LightGBM LambdaMART model scores and reorders the merged pool. LambdaMART optimizes ranking order directly (NDCG), which is what matters for a top-k list, rather than predicting ratings.
4. **Output.** The top-k movies come back with similarity scores.

### Why this design

- Three retrievers cover each other's blind spots: content (embeddings), vocabulary (TF-IDF) and behavior (collaborative).
- A learned ranker on top weighs those signals against each other instead of relying on hand-tuned weights.
- Training for NDCG puts the best matches at the top of the list, where users actually look.

## Data

| Source | What it is | Used for | Size |
|--------|-----------|----------|-----:|
| KION | Viewing interactions from a streaming service | Training interactions | ~480K |
| MovieLens | User-movie ratings | Training interactions | ~17.5M |
| TMDB | Movie metadata (titles, overviews, genres, budget, revenue and similar fields) | Movie catalog and text for embeddings | 9,964 movies |

Total: about **18M interactions** over a catalog of **9,964 movies**.

How the pieces fit together:

1. **Catalog.** TMDB metadata is the master list of movies. Everything else is matched back to it.
2. **Interactions.** KION and MovieLens were cleaned, mapped onto the catalog and split into train files (`kion_train.csv`, `movielens_train.csv`).
3. **Text embeddings.** Each movie's text is turned into a vector with a sentence embedding model and saved as `tmdb_embeddings.npy`.
4. **Keywords.** TF-IDF is built over the same movie text.
5. **Collaborative signal.** Co-viewing patterns across the 18M interactions show which movies the same people watch.

Raw data, models and logs are not in this repo (too big, and they are in `.gitignore`).

## Training and evaluation

- Model: LightGBM LambdaMART (`01_train_final_model.py`).
- Metric: NDCG@10.
- The final model was retrained after removing train/test leakage, so the score reflects data the model had not seen.

| Metric | Value |
|--------|------:|
| NDCG@10 | **0.8366** |

Details of the train/validation split are in `cinescore_final_metadata.json`.

## Run it

```bash
python recommender_system/scripts/03_interactive.py
```

Then try:

```
> rec Iron Man
> info Inception
> search The Matrix
> quit
```

## Bonus: Power BI report

The same data also feeds a six-page Power BI report built for stakeholders (Executive Hub, Audience Demographics, Financial Performance, Movie & Rating Analytics, Streaming & Engagement, Catalog & Content Insights). Every page filters by decade, genre and source. File: `CineScore_Report.pbix`.

It runs on a star schema centered on `dim_movies`, joined by `movie_key` to the metrics, crew, genre bridge, KION streaming and MovieLens genome tables, with `dim_date` driving the streaming timeline.

Headline findings from the merged dataset of 43,414 titles:

- **Box office:** $677.9B on $224.1B of budget, for $453.7B net profit. Family (ROI 1.93) and Animation (1.86) return the most per dollar.
- **Franchises:** franchise films earned about 0.29T net profit against 0.16T for standalone films.
- **Audience:** 567.6K active streamers, most concentrated in ages 25-34 and the $20k-$40k income bracket.
- **Engagement:** series make up 56.75% of streaming hours and films 43.25%, with a 109-minute average watch duration.
- **Ratings:** TMDB (6.70) and MovieLens (6.67) averages differ by only 0.04, and 4,062 titles are rated in line by both.
- **Longevity:** viewership half-life is about 21 years overall, but over 100 years for Family titles.

## Stack

Python, LightGBM, scikit-learn, NumPy, pandas, Power BI.

## Repo notes

Data, trained models and logs are not committed. You'll need to rebuild or supply those locally.