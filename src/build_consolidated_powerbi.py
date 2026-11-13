"""
build_consolidated_powerbi.py

CineScore Streamlined Star Schema Pipeline (8 Files)
Collapses 25 over-fragmented snowflake files into an elegant, high-performance
8-table Star Schema without any data loss and without column bloat:

Tables:
  1. dim_movies.csv             (~30 cols) - Master movie dimension (metadata, flags, bands)
  2. fact_movie_metrics.csv     (~25 cols) - All movie-level quantitative KPIs (TMDB, ML, KION, gaps)
  3. bridge_movie_genres.csv    (3 cols)   - Clean movie-to-genre bridge for instant slicing
  4. dim_movie_crew.csv         (5 cols)   - Directors, cast, keywords, studios (isolated text)
  5. fact_kion_streaming.csv    (9 cols)   - Granular streaming logs + user demographics (age, income)
  6. fact_movielens_genome.csv  (4 cols)   - Tag DNA & continuous relevance scores
  7. fact_lifecycle_decay.csv   (11 cols)  - Decay curve points + regression parameters (half-life, R2)
  8. dim_date.csv               (10 cols)  - Calendar dimension
"""

import argparse
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

# ------------------------------- PARAMETERS -------------------------------
MIN_ML_RATINGS = 30
MIN_KION_USERS = 5
KION_REF_YEAR = 2021
KION_WINDOW = ("2021-03-13", "2021-08-22")
DURATION_CAP_PCTL = 0.999
GAP_BAND = 0.5
GENOME_RELEVANCE_THRESHOLD = 0.25
CONSENSUS = dict(tmdb_vote_avg=7.0, tmdb_min_votes=100, ml_avg=3.5, kion_completion=0.50, kion_min_interactions=10)
CHUNK = 2_000_000

# ------------------------------- GENRE MAPPINGS -------------------------------
G = {
 "Action": ["action", "militant", "militants", "martial arts", "disaster", "disasters", "comics"],
 "Adult": ["adult", "18 +"],
 "Adventure": ["adventure", "adventures"],
 "Animation": ["animation", "animated", "animated series", "animated tv series", "anime", "cartoon", "cartoons",
               "russian cartoons", "western animation", "western cartoons"],
 "Arthouse": ["art house", "arthouse", "world classics"],
 "Biography": ["biography"],
 "Comedy": ["comedy", "comedies", "humor", "stand-up"],
 "Crime": ["crime", "detective", "detectives"],
 "Documentary": ["documentary", "documentaries", "wildlife", "living nature", "animals", "animal", "about animals",
                 "about celebrities", "around the world", "i want to know everything",
                 "i want to know everything about animals"],
 "Drama": ["drama", "dramas", "drama films", "doramas", "dorams", "doramus", "foreign dramas", "plays", "film-plays",
           "film-performance", "screenplay", "screenplays", "screen adaptations"],
 "Educational": ["education", "educational", "cognitive", "development", "informative", "popular science fiction"],
 "Family": ["family", "children", "children's", "children's films", "children's series", "children's stories",
            "children's songs", "for children", "for the youngest", "child-rearing", "parenting", "fairy tales",
            "youth", "kids"],
 "Fantasy": ["fantasy"],
 "Film Noir": ["film noir", "film-noir"],
 "History": ["history", "historical"],
 "Horror": ["horror", "horrors"],
 "Lifestyle": ["beauty and health", "cooking", "fitness", "hobbies", "hobby", "hunting and fishing", "medical", "blogger"],
 "Music": ["music", "musical", "musicals", "concert", "karaoke", "foreign songs"],
 "Mystery": ["mystery", "mystical", "mysticism", "mystics"],
 "Reality & TV": ["reality", "reality show", "reality shows", "show", "shows", "talk show", "tv magazines", "tv show",
                  "interview", "interviews", "entertainment", "screenings"],
 "Romance": ["romance", "melodrama", "melodramas"],
 "Science Fiction": ["science fiction", "fiction", "sci-fi", "scifi", "sci fi"],
 "Short": ["short", "short film", "short films", "short meter"],
 "Sport": ["sport", "sports", "football"],
 "Thriller": ["thriller", "thrillers"],
 "TV Movie": ["tv movie"],
 "War": ["war", "military"],
 "Western": ["western"],
}
DROP_GENRE = {"feature", "feature films", "film", "films", "movies", "full-length", "foreign", "foreign films",
              "indian film", "indian films", "imax", "russian", "russians", "soviet", "soviets", "soviet films",
              "series", "serials", "tv series", "we recommend", "no _ genre", "(no genres listed)", "hbo films", "hbo movies"}
GENRE_MAP = {alias: canon for canon, aliases in G.items() for alias in aliases}
PRIMARY_PRIORITY = ["Animation", "Documentary", "Horror", "Science Fiction", "Fantasy", "Western", "War", "Music", "Crime",
                    "Thriller", "Mystery", "Action", "Adventure", "Comedy", "Romance", "Family", "Biography", "History",
                    "Sport", "Film Noir", "Arthouse", "Educational", "Lifestyle", "Reality & TV", "Short", "TV Movie",
                    "Adult", "Drama"]

COUNTRY_FIX = {"usa": "United States", "uk": "United Kingdom", "ussr": "Soviet Union", "republic of korea": "South Korea",
               "russia": "Russia", "czech republic": "Czechia", "uae": "United Arab Emirates",
               "united states of america": "United States", "russian federation": "Russia", "south korea": "South Korea"}

# ------------------------------- HELPER FUNCTIONS -------------------------------
def as_int(s): return pd.to_numeric(s, errors="coerce").round().astype("Int64")
def as_dec(s, n): return pd.to_numeric(s, errors="coerce").round(n)
def as_bool(s):
    if s.dtype == bool: return s
    return s.map({True: True, False: False, "True": True, "False": False, "true": True, "false": False, 1: True, 0: False}).fillna(False).astype(bool)

def clean_text(s):
    s = s.astype("string").str.strip().str.replace(r"\s+", " ", regex=True)
    return s.mask(s.isin(["", "nan", "NaN", "None", "<NA>"]))

def add_sort(df, col, order):
    if col in df.columns:
        df[col + "_sort"] = df[col].map({v: i + 1 for i, v in enumerate(order)}).astype("Int64")

def split_list(v):
    if pd.isna(v): return []
    return [x.strip() for x in str(v).strip("[]").replace("'", "").replace('"', "").split(",") if x.strip()]

SCHEMA = []
def write(df, dst, name):
    df.to_csv(dst / f"{name}.csv", index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
    for c in df.columns:
        d = df[c].dtype
        kind = ("Date" if "datetime" in str(d) else "True/False" if str(d) == "bool" else
                "Whole number" if str(d).lower().startswith(("int", "uint")) else
                "Decimal number" if str(d).lower().startswith("float") else "Text")
        SCHEMA.append((name, c, kind, int(df[c].isna().sum()), round(100 * df[c].isna().mean(), 2)))
    print(f"  -> [OK] {name}.csv ({df.shape[0]:,} rows x {df.shape[1]} cols)")

# =================================================================================
def main():
    parser = argparse.ArgumentParser(description="Build streamlined 8-table Star Schema for CineScore Power BI.")
    parser.add_argument("--src", default="data/processed")
    parser.add_argument("--dst", default="data/powerbi")
    parser.add_argument("--tmdb_master", default="data/tmdb/tmdb_master.csv")
    parser.add_argument("--kion_items", default="data/kion/data_en/items_en.csv")
    parser.add_argument("--genome_scores", default="data/movielens/genome-scores.csv")
    parser.add_argument("--genome_tags", default="data/movielens/genome-tags.csv")
    args = parser.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    def need(n):
        p = src / n
        if not p.exists(): sys.exit(f"Missing {p}.")
        return p

    print("=================================================================")
    print(" CineScore Streamlined 8-Table Star Schema Pipeline")
    print("=================================================================")

    # 1. LOAD ANALYTICAL MART
    print("\n[1/8] Loading and enriching movie mart...")
    m = pd.read_csv(need("movie_analytics_mart.csv"), low_memory=False)

    for c in m.select_dtypes(include=["object", "string"]).columns:
        m[c] = clean_text(m[c])
    m["title"] = m["title"].fillna("Unknown title")
    m["content_type"] = m["content_type"].str.title()
    m["original_language"] = m["original_language"].str.lower()

    for c in ["movie_key", "tmdb_id", "movielens_id", "kion_item_id", "release_year", "decade", "runtime",
              "tmdb_vote_count", "age_rating", "for_kids", "genre_count", "kion_interaction_count",
              "kion_unique_users", "movielens_rating_count", "movielens_unique_users", "movielens_tag_count",
              "budget_clean", "revenue_clean", "profit", "collection_id"]:
        if c in m: m[c] = as_int(m[c])

    m.loc[m.runtime <= 0, "runtime"] = pd.NA
    m.loc[(m.release_year < 1880) | (m.release_year > 2026), "release_year"] = pd.NA
    m["decade"] = (m.release_year // 10 * 10).astype("Int64")
    m["decade_label"] = m.decade.astype("string") + "s"

    for c, n in [("tmdb_popularity", 3), ("tmdb_vote_average", 2), ("kion_avg_watched_pct", 2),
                 ("kion_avg_duration_sec", 0), ("kion_completion_rate", 4), ("kion_total_watch_hours", 1),
                 ("movielens_avg_rating", 3), ("movielens_rating_std", 3), ("movielens_pct_high", 4),
                 ("movielens_pct_low", 4), ("weighted_rating", 3), ("ml_weighted_rating", 3),
                 ("rating_gap_tmdb_minus_ml", 3), ("kion_engagement_score", 2), ("roi", 3)]:
        if c in m: m[c] = as_dec(m[c], n)
    if "kion_avg_duration_sec" in m:
        m["kion_avg_duration_sec"] = m["kion_avg_duration_sec"].round(0).astype("Int64")

    for c in ["in_tmdb_master", "in_movielens", "in_kion", "in_all_three_sources", "has_financials", "is_franchise"]:
        if c in m: m[c] = as_bool(m[c])

    ml_scope = m.in_movielens & m.in_tmdb_master
    ml_cols = [c for c in m.columns if c.startswith("movielens_") or c in ("ml_weighted_rating", "rating_gap_tmdb_minus_ml")]
    for c in ml_cols: m.loc[~ml_scope, c] = pd.NA
    for c in ["kion_interaction_count", "kion_unique_users"]: m.loc[~m.in_kion, c] = pd.NA

    nml = m.movielens_rating_count.astype(float)
    m["ml_support_status"] = np.select(
        [~m.in_movielens, ~ml_scope, nml.fillna(0) < MIN_ML_RATINGS],
        ["Not in MovieLens", "MovieLens (ratings not loaded)", f"Cold-start (<{MIN_ML_RATINGS} ratings)"],
        default=f"Supported ({MIN_ML_RATINGS}+ ratings)")

    ku = m.kion_unique_users.astype(float)
    m["kion_support_status"] = np.select(
        [~m.in_kion, ku.fillna(0) == 0, ku < MIN_KION_USERS],
        ["Not in KION", "In KION, no viewers", f"Cold-start (<{MIN_KION_USERS} viewers)"],
        default=f"Supported ({MIN_KION_USERS}+ viewers)")

    m["source_combo"] = pd.Series(
        np.where(m.in_tmdb_master, "TMDB+", "") + np.where(m.in_movielens, "MovieLens+", "") +
        np.where(m.in_kion, "KION+", "")
    ).str.rstrip("+")
    m["source_count"] = (m.in_tmdb_master.astype(int) + m.in_movielens.astype(int) + m.in_kion.astype(int)).astype("Int64")

    # Analysis bands
    m["popularity_tier"] = m.popularity_tier.replace({"nan": pd.NA})
    m["runtime_bin"] = pd.cut(m.runtime.astype(float), [0, 80, 100, 120, 150, 1000],
                              labels=["<80", "80-100", "100-120", "120-150", "150+"]).astype("string")
    m["completion_category"] = m.completion_category.replace({"nan": pd.NA})

    g = m.rating_gap_tmdb_minus_ml.astype(float)
    m["rating_gap_band"] = np.select([g.isna(), g > GAP_BAND, g < -GAP_BAND],
                                     [pd.NA, "TMDB rates higher", "MovieLens rates higher"], default="Aligned")
    m["rating_gap_band"] = m.rating_gap_band.replace({"nan": pd.NA, "<NA>": pd.NA})

    sup = (m.movielens_rating_count.astype(float) >= MIN_ML_RATINGS)
    q1, q2 = m.loc[sup, "movielens_rating_std"].quantile([1 / 3, 2 / 3])
    s = m.movielens_rating_std.astype(float)
    m["rating_consensus_band"] = np.select([~sup | s.isna(), s <= q1, s <= q2], [pd.NA, "Consensus", "Mixed"], default="Divisive")
    m["rating_consensus_band"] = m.rating_consensus_band.replace({"nan": pd.NA, "<NA>": pd.NA})

    w = m.weighted_rating.astype(float)
    m["weighted_rating_tier"] = pd.qcut(w.rank(method="first"), 4, labels=["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"]).astype("string")

    if "budget_clean" in m and m.budget_clean.notna().sum() >= 40:
        m["budget_tier"] = pd.qcut(m.budget_clean.astype(float).rank(method="first"), 4,
                                   labels=["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"]).astype("string")
    if "profit" in m and m.profit.notna().sum() >= 40:
        m["profit_tier"] = pd.qcut(m.profit.astype(float).rank(method="first"), 4,
                                   labels=["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"]).astype("string")

    ay = (KION_REF_YEAR - m.release_year).clip(lower=0)
    m["title_age_at_kion"] = ay.astype("Int64")
    m["age_bucket"] = pd.cut(ay.astype(float), [-1, 1, 5, 10, 20, 50, 500],
                             labels=["0-1 yrs", "2-5 yrs", "6-10 yrs", "11-20 yrs", "21-50 yrs", "50+ yrs"]).astype("string")

    c = CONSENSUS
    t_ok = (m.tmdb_vote_count.astype(float) >= c["tmdb_min_votes"]); m_ok = sup; k_ok = (m.kion_interaction_count.astype(float) >= c["kion_min_interactions"])
    all_ok = t_ok & m_ok & k_ok
    hits = ((m.tmdb_vote_average.astype(float) >= c["tmdb_vote_avg"]).astype(int) +
            (m.movielens_avg_rating.astype(float) >= c["ml_avg"]).astype(int) +
            (m.kion_completion_rate.astype(float) >= c["kion_completion"]).astype(int))
    m["consensus_level"] = np.select([~all_ok, hits == 3, hits == 2],
                                     ["Insufficient data", "Strong consensus (3 of 3)", "Partial (2 of 3)"],
                                     default="Split (0-1 of 3)")

    ks = m.kion_unique_users.astype(float); kion_sup = ks >= MIN_KION_USERS
    p75_u = ks[kion_sup].quantile(0.75); p25_u = ks[kion_sup].quantile(0.25)
    r_weak = (~m.in_movielens) | (ml_scope & ~sup)
    m["signal_quadrant"] = np.select(
        [kion_sup & (ks >= p75_u) & sup, kion_sup & (ks >= p75_u) & r_weak, kion_sup & (ks < p75_u) & sup, kion_sup & (ks < p75_u) & r_weak],
        ["Proven hit", "Hidden favourite", "Rated, rarely watched", "Low signal"], default="n/a")
    m["signal_quadrant"] = m.signal_quadrant.replace({"n/a": pd.NA})

    both = kion_sup & sup
    pa25, pa75 = m.loc[both, "movielens_avg_rating"].astype(float).quantile([0.25, 0.75])
    avg = m.movielens_avg_rating.astype(float)
    m["kion_vs_rating_mismatch"] = np.select(
        [~both, (ks >= p75_u) & (avg <= pa25), (ks <= p25_u) & (avg >= pa75)],
        [pd.NA, "Watched more than rated", "Rated more than watched"], default="Aligned")
    m["kion_vs_rating_mismatch"] = m.kion_vs_rating_mismatch.replace({"nan": pd.NA, "<NA>": pd.NA})

    # Add sort columns
    add_sort(m, "popularity_tier", ["Low", "Medium", "High", "Very high"])
    add_sort(m, "completion_category", ["Low (<25%)", "Medium (25-50%)", "High (>50%)"])
    add_sort(m, "runtime_bin", ["<80", "80-100", "100-120", "120-150", "150+"])
    add_sort(m, "age_bucket", ["0-1 yrs", "2-5 yrs", "6-10 yrs", "11-20 yrs", "21-50 yrs", "50+ yrs"])
    add_sort(m, "rating_gap_band", ["MovieLens rates higher", "Aligned", "TMDB rates higher"])
    add_sort(m, "rating_consensus_band", ["Consensus", "Mixed", "Divisive"])
    add_sort(m, "weighted_rating_tier", ["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"])
    add_sort(m, "consensus_level", ["Strong consensus (3 of 3)", "Partial (2 of 3)", "Split (0-1 of 3)", "Insufficient data"])
    add_sort(m, "signal_quadrant", ["Proven hit", "Hidden favourite", "Rated, rarely watched", "Low signal"])
    add_sort(m, "kion_vs_rating_mismatch", ["Watched more than rated", "Aligned", "Rated more than watched"])
    for _c in ("budget_tier", "profit_tier"):
        if _c in m: add_sort(m, _c, ["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"])

    # 2. CANONICAL GENRES & BRIDGE
    print("\n[2/8] Canonicalizing Genres & Building Multi-Genre Bridge...")
    dg_old = pd.read_csv(need("dim_genre.csv"))
    bg_old = pd.read_csv(need("bridge_movie_genre.csv"))
    dg_old["canon"] = dg_old.genre_name.str.lower().map(lambda x: GENRE_MAP.get(x, "__DROP__" if x in DROP_GENRE else "__UNMAPPED__"))
    bg = bg_old.merge(dg_old[["genre_key", "canon"]], on="genre_key")
    bg = bg[~bg.canon.isin(["__DROP__", "__UNMAPPED__"])].drop(columns="genre_key").drop_duplicates(["movie_key", "canon"])
    bg = bg.rename(columns={"canon": "genre_name"})

    pr = {g: i for i, g in enumerate(PRIMARY_PRIORITY)}
    bg["_p"] = bg.genre_name.map(pr).fillna(99)
    prim = bg.sort_values(["movie_key", "_p"]).groupby("movie_key").genre_name.first()
    m["primary_genre"] = m.movie_key.map(prim).fillna("Unknown")
    bg["is_primary"] = bg["genre_name"] == bg["movie_key"].map(prim)

    bridge_genres = bg[["movie_key", "genre_name", "is_primary"]].sort_values(["movie_key", "genre_name"])

    # Extract Primary Country
    col = "production_countries" if ("production_countries" in m and m.production_countries.notna().any()) else "k_countries"
    c_list = []
    for mk, cs in zip(m.movie_key, m[col]):
        spl = split_list(cs)
        if spl:
            c_name = COUNTRY_FIX.get(spl[0].lower(), spl[0].title() if spl[0].islower() else spl[0])
            c_list.append((mk, c_name))
    c_df = pd.DataFrame(c_list, columns=["movie_key", "primary_country"]).drop_duplicates("movie_key")
    m = m.merge(c_df, on="movie_key", how="left")
    m["primary_country"] = m["primary_country"].fillna("Unknown")

    # 3. SPLIT DIM_MOVIES AND FACT_MOVIE_METRICS
    print("\n[3/8] Building dim_movies and fact_movie_metrics...")
    dim_movie_cols = [
        "movie_key", "tmdb_id", "movielens_id", "kion_item_id", "imdb_id",
        "title", "original_title", "original_language", "content_type",
        "release_year", "decade", "decade_label", "runtime", "runtime_bin", "runtime_bin_sort",
        "primary_genre", "primary_country", "age_rating", "for_kids", "is_franchise",
        "in_tmdb_master", "in_movielens", "in_kion", "in_all_three_sources", "source_combo", "source_count",
        "consensus_level", "consensus_level_sort", "signal_quadrant", "signal_quadrant_sort",
        "rating_gap_band", "rating_gap_band_sort", "rating_consensus_band", "rating_consensus_band_sort",
        "kion_vs_rating_mismatch", "kion_vs_rating_mismatch_sort", "age_bucket", "age_bucket_sort"
    ]
    dim_movies = m[[c for c in dim_movie_cols if c in m]].copy()

    fact_metric_cols = [
        "movie_key",
        "budget_clean", "revenue_clean", "profit", "roi", "has_financials",
        "budget_tier", "budget_tier_sort", "profit_tier", "profit_tier_sort",
        "tmdb_popularity", "tmdb_vote_average", "tmdb_vote_count", "weighted_rating", "weighted_rating_tier", "weighted_rating_tier_sort", "popularity_tier", "popularity_tier_sort",
        "movielens_avg_rating", "movielens_rating_count", "movielens_rating_std", "movielens_pct_high", "movielens_pct_low", "ml_weighted_rating", "ml_support_status",
        "kion_unique_users", "kion_interaction_count", "kion_avg_watched_pct", "kion_avg_duration_sec",
        "kion_completion_rate", "kion_total_watch_hours", "kion_engagement_score", "completion_category", "completion_category_sort", "kion_support_status",
        "rating_gap_tmdb_minus_ml", "title_age_at_kion"
    ]
    fact_movie_metrics = m[[c for c in fact_metric_cols if c in m]].copy()

    # 4. DIM_MOVIE_CREW
    print("\n[4/8] Building dim_movie_crew (directors, cast, keywords)...")
    crew_cols = ["movie_key", "director", "cast_top", "keywords", "production_companies"]
    dim_movie_crew = m[[c for c in crew_cols if c in m]].copy()
    dim_movie_crew = dim_movie_crew.dropna(subset=["director", "cast_top", "keywords", "production_companies"], how="all")

    # 5. KION STREAMING (Granular Events + User Demographics)
    print("\n[5/8] Building fact_kion_streaming (with embedded user demographics)...")
    kfile = src / "fact_kion_interactions.csv"
    ku_file = src / "dim_kion_user.csv"
    
    k = pd.read_csv(kfile, usecols=["user_key", "movie_key", "date_key", "total_dur_clean", "watched_pct_clean", "is_completed"])
    k = k[k.movie_key.notna()]
    cap = k.total_dur_clean.quantile(DURATION_CAP_PCTL)
    k["total_dur_sec"] = k.total_dur_clean.clip(upper=cap)
    wp = pd.to_numeric(k.watched_pct_clean, errors="coerce")
    wp_int = bool((wp.dropna() % 1 == 0).all())
    
    fact_kion_streaming = pd.DataFrame({
        "user_key": as_int(k.user_key),
        "movie_key": as_int(k.movie_key),
        "date_key": as_int(k.date_key),
        "watched_pct": as_int(wp) if wp_int else wp.round(2),
        "total_dur_sec": as_int(k.total_dur_sec),
        "is_completed": as_int(k.is_completed).astype("Int8")
    })
    del k

    if ku_file.exists():
        u = pd.read_csv(ku_file, low_memory=False)
        def bucket(s): return s.astype("string").str.replace(r"^(age|income)_", "", regex=True).str.replace("_inf", "+", regex=False).str.replace("_", "-", regex=False)
        ag = u["age_group"] if "age_group" in u else bucket(u["age"])
        ig = u["income_group"] if "income_group" in u else bucket(u["income"])
        sex = u["sex"].map({"F": "Female", "M": "Male"}).fillna("Unknown") if "sex" in u else "Unknown"
        user_demo = pd.DataFrame({
            "user_key": as_int(u.user_key),
            "age_group": ag.fillna("Unknown").astype("string"),
            "income_group": ig.fillna("Unknown").astype("string"),
            "sex": sex
        }).drop_duplicates("user_key")
        del u
        fact_kion_streaming = fact_kion_streaming.merge(user_demo, on="user_key", how="left")
        fact_kion_streaming["age_group"] = fact_kion_streaming["age_group"].fillna("Unknown")
        fact_kion_streaming["income_group"] = fact_kion_streaming["income_group"].fillna("Unknown")
        fact_kion_streaming["sex"] = fact_kion_streaming["sex"].fillna("Unknown")

    # 6. MOVIELENS GENOME SCORES + TAG NAMES
    print("\n[6/8] Building fact_movielens_genome (scores merged with tag names)...")
    ml_to_movie_key = m.dropna(subset=["movielens_id"]).set_index("movielens_id")["movie_key"].to_dict()
    genome_tags_path = Path(args.genome_tags)
    genome_scores_path = Path(args.genome_scores)
    
    tag_map = {}
    if genome_tags_path.exists():
        gt_raw = pd.read_csv(genome_tags_path)
        tag_map = dict(zip(gt_raw["tagId"], clean_text(gt_raw["tag"])))

    fact_movielens_genome = pd.DataFrame()
    if genome_scores_path.exists() and tag_map:
        gs_chunks = []
        for ch in pd.read_csv(genome_scores_path, chunksize=CHUNK, dtype={"movieId": "int64", "tagId": "int64", "relevance": "float32"}):
            ch_sub = ch[ch["movieId"].isin(ml_to_movie_key) & (ch["relevance"] >= GENOME_RELEVANCE_THRESHOLD)]
            if len(ch_sub):
                gs_chunks.append(ch_sub)
        if gs_chunks:
            gs_df = pd.concat(gs_chunks, ignore_index=True)
            fact_movielens_genome = pd.DataFrame({
                "movie_key": as_int(gs_df["movieId"].map(ml_to_movie_key)),
                "tag_id": as_int(gs_df["tagId"]),
                "tag_name": gs_df["tagId"].map(tag_map),
                "relevance": as_dec(gs_df["relevance"], 3)
            }).dropna(subset=["movie_key", "tag_name"])

    # 7. CONSOLIDATED LIFECYCLE DECAY
    print("\n[7/8] Building fact_lifecycle_decay (curve + fitted model parameters)...")
    base = m.loc[m.kion_unique_users.astype(float).fillna(0) > 0,
                 ["movie_key", "content_type", "title_age_at_kion", "kion_unique_users", "kion_interaction_count", "kion_completion_rate"]]
    base = base[base.title_age_at_kion.notna() & (base.title_age_at_kion.astype(float) <= 60)].copy()
    base["age"] = base.title_age_at_kion.astype(int)
    cur = pd.concat([base.assign(genre_name="All genres"), base.merge(bg[["movie_key", "genre_name"]], on="movie_key")])
    dec = cur.groupby(["content_type", "genre_name", "age"]).agg(
        titles=("movie_key", "size"), avg_viewers=("kion_unique_users", "mean"),
        total_views=("kion_interaction_count", "sum"), avg_completion=("kion_completion_rate", "mean")).reset_index()
    fits = []
    for (ct, gn), grp in dec.groupby(["content_type", "genre_name"]):
        g1 = grp[(grp.age >= 1) & (grp.titles >= 3)]
        if len(g1) < 8: continue
        x, y, w = g1.age.values.astype(float), np.log(g1.avg_viewers.values), np.sqrt(g1.titles.values)
        b, a = np.polyfit(x, y, 1, w=w)
        yhat = a + b * x
        ybar = np.average(y, weights=w ** 2)
        r2 = 1 - np.sum(w ** 2 * (y - yhat) ** 2) / np.sum(w ** 2 * (y - ybar) ** 2)
        a0 = grp.loc[grp.age == 0, "avg_viewers"]
        a1 = grp.loc[grp.age == 1, "avg_viewers"]
        fits.append(dict(content_type=ct, genre_name=gn, halflife_years=round(np.log(2) / -b, 1) if b < 0 else np.nan,
                         fit_r2=round(r2, 3), new_release_multiple=round(a0.iloc[0] / a1.iloc[0], 1) if len(a0) and len(a1) else np.nan,
                         fit_a=a, fit_b=b))
    fit_df = pd.DataFrame(fits)
    dec = dec.merge(fit_df, on=["content_type", "genre_name"], how="left")
    dec["fitted_viewers"] = np.where(dec.age >= 1, np.exp(dec.fit_a + dec.fit_b * dec.age), np.nan).round(1)
    fact_lifecycle_decay = pd.DataFrame({
        "content_type": dec.content_type, "genre_name": dec.genre_name, "title_age_years": as_int(dec.age),
        "titles": as_int(dec.titles), "avg_viewers": dec.avg_viewers.round(1), "total_views": as_int(dec.total_views),
        "avg_completion": dec.avg_completion.round(4), "fitted_viewers": dec.fitted_viewers,
        "halflife_years": dec.halflife_years, "fit_r2": dec.fit_r2, "new_release_multiple": dec.new_release_multiple
    })

    # 8. DATE DIMENSION
    print("\n[8/8] Building dim_date...")
    d = pd.read_csv(need("dim_date.csv"), parse_dates=["date"])
    dim_date = pd.DataFrame({
        "date": d.date,
        "date_key": as_int(d.date_key),
        "year": as_int(d.year),
        "quarter": d.quarter,
        "month_number": as_int(d.month_number),
        "month_name": d.month_name,
        "month_short": d.date.dt.strftime("%b"),
        "year_month": d.year_month,
        "is_weekend": as_bool(d.is_weekend),
        "in_kion_window": (d.date >= KION_WINDOW[0]) & (d.date <= KION_WINDOW[1])
    })

    # =========================================================================
    # WRITE THE 8 STREAMLINED FILES
    # =========================================================================
    print("\nWriting streamlined 8-table star schema to destination...")
    write(dim_movies, dst, "dim_movies")
    write(fact_movie_metrics, dst, "fact_movie_metrics")
    write(bridge_genres, dst, "bridge_movie_genres")
    write(dim_movie_crew, dst, "dim_movie_crew")
    write(fact_kion_streaming, dst, "fact_kion_streaming")
    write(fact_movielens_genome, dst, "fact_movielens_genome")
    write(fact_lifecycle_decay, dst, "fact_lifecycle_decay")
    write(dim_date, dst, "dim_date")

    # CLEAN UP ALL DEPRECATED OVER-FRAGMENTED FILES IN data/powerbi
    deprecated = [
        "agg_age_decay.csv", "agg_decay_fit.csv", "bridge_movie_country.csv", "bridge_movie_genre.csv",
        "dim_country.csv", "dim_genre.csv", "dim_kion_enrichment_credits.csv", "dim_kion_items.csv",
        "dim_kion_user.csv", "dim_movie_crosswalk.csv", "dim_movielens_genome_tag.csv", "dim_movielens_movies.csv",
        "dim_movielens_user.csv", "dim_tmdb_enrichment_credits.csv", "dim_tmdb_movies.csv",
        "fact_cross_source_analytics.csv", "fact_kion_interactions.csv", "fact_kion_movie_metrics.csv",
        "fact_movielens_genome_scores.csv", "fact_movielens_movie_metrics.csv", "fact_movielens_ratings_yearly.csv",
        "fact_movielens_tags.csv", "fact_tmdb_financials.csv", "fact_tmdb_ratings.csv"
    ]
    removed_count = 0
    for dep in deprecated:
        p = dst / dep
        if p.exists():
            p.unlink()
            removed_count += 1
    print(f"\n[Clean Up] Removed {removed_count} redundant/over-fragmented files from {dst}.")

    # Schema report
    schema_df = pd.DataFrame(SCHEMA, columns=["table", "column", "power_bi_type", "nulls", "null_pct"])
    schema_df.to_csv(dst / "schema_report.csv", index=False)
    print(f"  -> Generated schema_report.csv ({len(schema_df)} column definitions across 8 tables)")

    elapsed = round(time.time() - t0, 1)
    print(f"\nPipeline successfully completed in {elapsed}s!")

if __name__ == "__main__":
    main()
