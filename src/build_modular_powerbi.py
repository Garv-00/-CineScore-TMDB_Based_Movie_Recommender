"""
build_modular_powerbi.py

End-to-end pipeline to produce a normalized, modular Snowflake schema for CineScore in Power BI.
Includes:
  1. Central Crosswalk & Hub (dim_movie_crosswalk, fact_cross_source_analytics)
  2. TMDB Domain (dim_tmdb_movies, fact_tmdb_ratings, fact_tmdb_financials, dim_tmdb_enrichment_credits)
  3. KION Domain (dim_kion_items, dim_kion_enrichment_credits, fact_kion_movie_metrics, dim_kion_user, fact_kion_interactions)
  4. MovieLens Domain (dim_movielens_movies, fact_movielens_movie_metrics, dim_movielens_user,
                       fact_movielens_ratings_yearly, fact_movielens_tags,
                       dim_movielens_genome_tag, fact_movielens_genome_scores)
  5. Shared Conformed Dimensions (dim_date, dim_genre, bridge_movie_genre, dim_country, bridge_movie_country)
  6. Statistical Aggregates (agg_age_decay, agg_decay_fit)
  7. Power Query M-scripts, DAX measures, and schema reports.
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

LOG = []
def log(table, action, detail=""):
    LOG.append((table, action, str(detail)))
    print(f"  [{table}] {action} {detail}")

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
    print(f"  -> {name}.csv  {df.shape[0]:,} rows x {df.shape[1]} cols")

# =================================================================================
def main():
    parser = argparse.ArgumentParser(description="Build clean modular snowflake schema for CineScore Power BI.")
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
    print(" CineScore Modular Snowflake Data Pipeline")
    print("=================================================================")

    # 1. LOAD MART & FEATURE MASTER
    print("\n[1/9] Loading analytical mart and TMDB feature engineering...")
    m = pd.read_csv(need("movie_analytics_mart.csv"), low_memory=False)
    n0 = len(m)
    log("mart", f"Loaded mart with {n0:,} entities and {len(m.columns)} raw columns")

    # Load normalized feature columns from tmdb_master if available
    tmdb_features = None
    tmdb_master_path = Path(args.tmdb_master)
    if tmdb_master_path.exists():
        t_master = pd.read_csv(tmdb_master_path, low_memory=False)
        feat_cols = ["id", "year_normalized", "popularity_normalized", "vote_count_log",
                     "vote_count_normalized", "vote_average_normalized"]
        available_feat = [c for c in feat_cols if c in t_master.columns]
        tmdb_features = t_master[available_feat].rename(columns={"id": "tmdb_id"}).drop_duplicates(subset=["tmdb_id"])
        log("tmdb_master", f"Loaded feature engineering metrics for {len(tmdb_features):,} TMDB movies", available_feat)

    # Text cleaning
    for c in m.select_dtypes(include=["object", "string"]).columns:
        m[c] = clean_text(m[c])
    m["title"] = m["title"].fillna("Unknown title")
    m["content_type"] = m["content_type"].str.title()
    m["original_language"] = m["original_language"].str.lower()

    # Whole numbers
    for c in ["movie_key", "tmdb_id", "movielens_id", "kion_item_id", "release_year", "decade", "runtime",
              "tmdb_vote_count", "age_rating", "for_kids", "genre_count", "kion_interaction_count",
              "kion_unique_users", "movielens_rating_count", "movielens_unique_users", "movielens_tag_count",
              "budget_clean", "revenue_clean", "profit", "collection_id"]:
        if c in m: m[c] = as_int(m[c])

    # Runtime and release_year sanity
    m.loc[m.runtime <= 0, "runtime"] = pd.NA
    m.loc[(m.release_year < 1880) | (m.release_year > 2026), "release_year"] = pd.NA
    m["decade"] = (m.release_year // 10 * 10).astype("Int64")
    m["decade_label"] = m.decade.astype("string") + "s"

    # Decimals
    for c, n in [("tmdb_popularity", 3), ("tmdb_vote_average", 2), ("kion_avg_watched_pct", 2),
                 ("kion_avg_duration_sec", 0), ("kion_completion_rate", 4), ("kion_total_watch_hours", 1),
                 ("movielens_avg_rating", 3), ("movielens_rating_std", 3), ("movielens_pct_high", 4),
                 ("movielens_pct_low", 4), ("weighted_rating", 3), ("ml_weighted_rating", 3),
                 ("rating_gap_tmdb_minus_ml", 3), ("kion_engagement_score", 2), ("roi", 3)]:
        if c in m: m[c] = as_dec(m[c], n)
    if "kion_avg_duration_sec" in m:
        m["kion_avg_duration_sec"] = m["kion_avg_duration_sec"].round(0).astype("Int64")

    # Flags
    for c in ["in_tmdb_master", "in_movielens", "in_kion", "in_all_three_sources", "has_financials", "is_franchise"]:
        if c in m: m[c] = as_bool(m[c])

    # Scope-aware MovieLens blanks
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

    m["source_combo"] = (np.where(m.in_tmdb_master, "TMDB+", "") + np.where(m.in_movielens, "MovieLens+", "") +
                         np.where(m.in_kion, "KION+", ""))
    m["source_combo"] = m.source_combo.str.rstrip("+")
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
    m["engagement_decile"] = pd.qcut(m.kion_engagement_score.astype(float).rank(method="first"), 10, labels=range(1, 11)).astype("Int64")

    # Add sort columns
    add_sort(m, "popularity_tier", ["Low", "Medium", "High", "Very high"])
    add_sort(m, "completion_category", ["Low (<25%)", "Medium (25-50%)", "High (>50%)"])
    add_sort(m, "runtime_bin", ["<80", "80-100", "100-120", "120-150", "150+"])
    add_sort(m, "age_bucket", ["0-1 yrs", "2-5 yrs", "6-10 yrs", "11-20 yrs", "21-50 yrs", "50+ yrs"])
    add_sort(m, "rating_gap_band", ["MovieLens rates higher", "Aligned", "TMDB rates higher"])
    add_sort(m, "rating_consensus_band", ["Consensus", "Mixed", "Divisive"])
    add_sort(m, "weighted_rating_tier", ["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"])
    add_sort(m, "consensus_level", ["Strong consensus (3 of 3)", "Partial (2 of 3)", "Split (0-1 of 3)", "Insufficient data"])
    add_sort(m, "ml_support_status", ["Not in MovieLens", "MovieLens (ratings not loaded)", f"Cold-start (<{MIN_ML_RATINGS} ratings)", f"Supported ({MIN_ML_RATINGS}+ ratings)"])
    add_sort(m, "kion_support_status", ["Not in KION", "In KION, no viewers", f"Cold-start (<{MIN_KION_USERS} viewers)", f"Supported ({MIN_KION_USERS}+ viewers)"])
    add_sort(m, "signal_quadrant", ["Proven hit", "Hidden favourite", "Rated, rarely watched", "Low signal"])
    add_sort(m, "kion_vs_rating_mismatch", ["Watched more than rated", "Aligned", "Rated more than watched"])
    for _c in ("budget_tier", "profit_tier"):
        if _c in m: add_sort(m, _c, ["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"])

    # 2. CANONICAL GENRES & COUNTRIES
    print("\n[2/9] Canonicalizing Genres & Countries...")
    dg_old = pd.read_csv(need("dim_genre.csv"))
    bg_old = pd.read_csv(need("bridge_movie_genre.csv"))
    dg_old["canon"] = dg_old.genre_name.str.lower().map(lambda x: GENRE_MAP.get(x, "__DROP__" if x in DROP_GENRE else "__UNMAPPED__"))
    bg = bg_old.merge(dg_old[["genre_key", "canon"]], on="genre_key")
    bg = bg[~bg.canon.isin(["__DROP__", "__UNMAPPED__"])].drop(columns="genre_key").drop_duplicates(["movie_key", "canon"])
    dg = pd.DataFrame({"genre_name": sorted(bg.canon.unique())})
    dg["genre_key"] = range(1, len(dg) + 1)
    bg = bg.merge(dg, left_on="canon", right_on="genre_name")[["movie_key", "genre_key", "genre_name"]]
    pr = {g: i for i, g in enumerate(PRIMARY_PRIORITY)}
    bg["_p"] = bg.genre_name.map(pr).fillna(99)
    prim = bg.sort_values(["movie_key", "_p"]).groupby("movie_key").genre_name.first()
    m["primary_genre"] = m.movie_key.map(prim).fillna("Unknown")
    m["genre_count"] = m.movie_key.map(bg.groupby("movie_key").size()).fillna(0).astype(int).astype("Int64")
    bridge_genre = bg[["movie_key", "genre_key"]].astype("int64")
    dg = dg[["genre_key", "genre_name"]].astype({"genre_key": "int64"})

    # Countries
    col = "production_countries" if ("production_countries" in m and m.production_countries.notna().any()) else "k_countries"
    rows = []
    for mk, cs in zip(m.movie_key, m[col]):
        for x in split_list(cs):
            rows.append((mk, COUNTRY_FIX.get(x.lower(), x.title() if x.islower() else x)))
    bc = pd.DataFrame(rows, columns=["movie_key", "country_name"]).drop_duplicates()
    dc = pd.DataFrame({"country_name": sorted(bc.country_name.unique())})
    dc["country_key"] = range(1, len(dc) + 1)
    bc = bc.merge(dc, on="country_name")
    first_c = bc.drop_duplicates("movie_key").set_index("movie_key").country_name
    m["primary_country"] = m.movie_key.map(first_c)
    bridge_country = bc[["movie_key", "country_key"]].astype("int64")
    dc = dc[["country_key", "country_name"]]

    # 3. KION FACT INTERACTIONS
    print("\n[3/9] Processing KION Interactions Fact...")
    kfile = src / "fact_kion_interactions.csv"
    kion_fact = None
    if kfile.exists():
        k = pd.read_csv(kfile, usecols=["user_key", "movie_key", "date_key", "total_dur_clean", "watched_pct_clean", "is_completed"])
        k = k[k.movie_key.notna()]
        cap = k.total_dur_clean.quantile(DURATION_CAP_PCTL)
        k["total_dur_sec"] = k.total_dur_clean.clip(upper=cap)
        wp = pd.to_numeric(k.watched_pct_clean, errors="coerce")
        wp_int = bool((wp.dropna() % 1 == 0).all())
        kion_fact = pd.DataFrame({
            "user_key": as_int(k.user_key),
            "movie_key": as_int(k.movie_key),
            "date_key": as_int(k.date_key),
            "watched_pct": as_int(wp) if wp_int else wp.round(2),
            "total_dur_sec": as_int(k.total_dur_sec),
            "is_completed": as_int(k.is_completed).astype("Int8")
        })
        agg = k.groupby("movie_key").total_dur_sec.agg(["mean", "sum"])
        m["kion_avg_duration_sec"] = m.movie_key.map(agg["mean"]).round(0).astype("Int64")
        m["kion_total_watch_hours"] = m.movie_key.map(agg["sum"] / 3600).round(1)
        del k

    # 4. MOVIELENS RATINGS & TAGS
    print("\n[4/9] Processing MovieLens Yearly Ratings & Tags...")
    rfile = src / "fact_movielens_ratings.csv"
    ratings_y = None
    if rfile.exists():
        parts = []
        for ch in pd.read_csv(rfile, usecols=["movie_key", "date_key", "rating"], chunksize=CHUNK,
                              dtype={"movie_key": "float64", "date_key": "int64", "rating": "float32"}):
            ch = ch[ch.movie_key.notna()]
            ch["rating_year"] = (ch.date_key // 10000).astype("int32")
            r = ch.rating.astype("float64")
            ch["r2"] = r * r
            ch["hi"] = (r >= 4).astype("int32")
            ch["lo"] = (r <= 2).astype("int32")
            ch["r"] = r
            parts.append(ch.groupby(["movie_key", "rating_year"]).agg(
                n_ratings=("r", "size"), sum_rating=("r", "sum"), sum_rating_sq=("r2", "sum"),
                n_high=("hi", "sum"), n_low=("lo", "sum")).reset_index())
        ry = pd.concat(parts).groupby(["movie_key", "rating_year"], as_index=False).sum()
        ratings_y = pd.DataFrame({
            "movie_key": as_int(ry.movie_key),
            "rating_date": pd.to_datetime(ry.rating_year.astype(str) + "-01-01"),
            "n_ratings": as_int(ry.n_ratings),
            "sum_rating": ry.sum_rating.round(1),
            "sum_rating_sq": ry.sum_rating_sq.round(2),
            "n_high": as_int(ry.n_high),
            "n_low": as_int(ry.n_low)
        })

    tags = None
    tfile = src / "fact_movielens_tags.csv"
    if tfile.exists():
        t = pd.read_csv(tfile, usecols=["tag_key", "user_key", "movie_key", "date_key", "tag_clean"])
        t["tag_clean"] = clean_text(t.tag_clean).str.lower()
        t = t[t.tag_clean.notna() & t.movie_key.notna()]
        tags = pd.DataFrame({
            "tag_key": as_int(t.tag_key),
            "user_key": as_int(t.user_key),
            "movie_key": as_int(t.movie_key),
            "date_key": as_int(t.date_key),
            "tag": t.tag_clean
        })

    # Users
    mu = pd.read_csv(need("dim_movielens_user.csv"))
    mu_o = pd.DataFrame({
        "user_key": as_int(mu.user_key),
        "n_ratings": as_int(mu.n_ratings),
        "avg_rating_given": as_dec(mu.avg_rating_given, 3),
        "rating_stddev": as_dec(mu.rating_stddev, 3),
        "first_rating_date": pd.to_datetime(mu.first_rating_date),
        "last_rating_date": pd.to_datetime(mu.last_rating_date),
        "activity_segment": clean_text(mu.activity_segment)
    })
    add_sort(mu_o, "activity_segment", ["<=20", "21-50", "51-200", "201-1000", "1000+"])

    ku_file = src / "dim_kion_user.csv"
    kion_user = None
    if ku_file.exists():
        u = pd.read_csv(ku_file, low_memory=False)
        def bucket(s): return s.astype("string").str.replace(r"^(age|income)_", "", regex=True).str.replace("_inf", "+", regex=False).str.replace("_", "-", regex=False)
        ag = u["age_group"] if "age_group" in u else bucket(u["age"])
        ig = u["income_group"] if "income_group" in u else bucket(u["income"])
        sex = u["sex"].map({"F": "Female", "M": "Male"}).fillna("Unknown") if "sex" in u else "Unknown"
        kids = u["has_kids"] if "has_kids" in u else u["kids_flg"].map({1: "Yes", 0: "No"}).fillna("Unknown")
        kion_user = pd.DataFrame({
            "user_key": as_int(u.user_key),
            "age_group": ag.fillna("Unknown").astype("string"),
            "income_group": ig.fillna("Unknown").astype("string"),
            "sex": sex,
            "has_kids": kids.fillna("Unknown").astype("string")
        })
        kion_user["profile_known"] = kion_user.has_kids != "Unknown"
        add_sort(kion_user, "age_group", ["18-24", "25-34", "35-44", "45-54", "55-64", "65+", "Unknown"])
        add_sort(kion_user, "income_group", ["0-20", "20-40", "40-60", "60-90", "90-150", "150+", "Unknown"])

    # Dates
    d = pd.read_csv(need("dim_date.csv"), parse_dates=["date"])
    dd = pd.DataFrame({
        "date": d.date,
        "date_key": as_int(d.date_key),
        "year": as_int(d.year),
        "quarter": d.quarter,
        "month_number": as_int(d.month_number),
        "month_name": d.month_name,
        "month_short": d.date.dt.strftime("%b"),
        "year_month": d.year_month,
        "month_start": d.date.dt.to_period("M").dt.to_timestamp(),
        "iso_week": as_int(d.week),
        "week_start": (d.date - pd.to_timedelta(d.date.dt.dayofweek, unit="D")),
        "day": as_int(d.day),
        "day_name": d.day_name,
        "day_of_week": as_int(d.day_of_week),
        "is_weekend": as_bool(d.is_weekend),
        "in_kion_window": (d.date >= KION_WINDOW[0]) & (d.date <= KION_WINDOW[1])
    })

    # Decay
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
        fits.append(dict(content_type=ct, genre_name=gn, titles=int(grp.titles.sum()),
                         halflife_years=round(np.log(2) / -b, 1) if b < 0 else np.nan,
                         fit_r2=round(r2, 3), new_release_multiple=round(a0.iloc[0] / a1.iloc[0], 1) if len(a0) and len(a1) else np.nan,
                         fit_a=a, fit_b=b))
    fit_df = pd.DataFrame(fits)
    dec = dec.merge(fit_df[["content_type", "genre_name", "fit_a", "fit_b"]], on=["content_type", "genre_name"], how="left")
    dec["fitted_viewers"] = np.where(dec.age >= 1, np.exp(dec.fit_a + dec.fit_b * dec.age), np.nan).round(1)
    agg_decay = pd.DataFrame({
        "content_type": dec.content_type, "genre_name": dec.genre_name, "title_age_years": as_int(dec.age),
        "titles": as_int(dec.titles), "avg_viewers": dec.avg_viewers.round(1), "total_views": as_int(dec.total_views),
        "avg_completion": dec.avg_completion.round(4), "fitted_viewers": dec.fitted_viewers
    })
    agg_fit = fit_df.drop(columns=["fit_a", "fit_b"])
    agg_fit["titles"] = as_int(agg_fit.titles)

    # -----------------------------------------------------------------------------
    # 5. MOVIELENS GENOME TAGS & SCORES PROCESSING (NEW)
    # -----------------------------------------------------------------------------
    print("\n[5/9] Processing MovieLens Genome Tags and Relevance Scores...")
    # Crosswalk dict for fast lookup
    ml_to_movie_key = m.dropna(subset=["movielens_id"]).set_index("movielens_id")["movie_key"].to_dict()

    # Genome tags dimension
    genome_tags_path = Path(args.genome_tags)
    dim_genome_tag = None
    if genome_tags_path.exists():
        gt_raw = pd.read_csv(genome_tags_path)
        dim_genome_tag = pd.DataFrame({
            "tag_id": as_int(gt_raw["tagId"]),
            "tag": clean_text(gt_raw["tag"])
        })
        log("dim_movielens_genome_tag", f"Loaded {len(dim_genome_tag):,} genome tags")

    # Genome scores fact
    genome_scores_path = Path(args.genome_scores)
    fact_genome_scores = None
    if genome_scores_path.exists() and dim_genome_tag is not None:
        gs_chunks = []
        for ch in pd.read_csv(genome_scores_path, chunksize=CHUNK, dtype={"movieId": "int64", "tagId": "int64", "relevance": "float32"}):
            # Keep rows matching CineScore movies and meeting analytical relevance threshold
            ch_sub = ch[ch["movieId"].isin(ml_to_movie_key) & (ch["relevance"] >= GENOME_RELEVANCE_THRESHOLD)]
            if len(ch_sub):
                gs_chunks.append(ch_sub)
        if gs_chunks:
            gs_df = pd.concat(gs_chunks, ignore_index=True)
            fact_genome_scores = pd.DataFrame({
                "movie_key": as_int(gs_df["movieId"].map(ml_to_movie_key)),
                "movielens_id": as_int(gs_df["movieId"]),
                "tag_id": as_int(gs_df["tagId"]),
                "relevance": as_dec(gs_df["relevance"], 3)
            })
            log("fact_movielens_genome_scores", f"Processed {len(fact_genome_scores):,} scores across {fact_genome_scores.movie_key.nunique():,} movies (relevance >= {GENOME_RELEVANCE_THRESHOLD})")

    # -----------------------------------------------------------------------------
    # 6. KION ENRICHMENT / METADATA PROCESSING (NEW)
    # -----------------------------------------------------------------------------
    print("\n[6/9] Processing KION Extended Credits & Metadata...")
    kion_items_path = Path(args.kion_items)
    dim_kion_credits = None
    kion_to_movie_key = m.dropna(subset=["kion_item_id"]).set_index("kion_item_id")["movie_key"].to_dict()

    if kion_items_path.exists():
        k_items = pd.read_csv(kion_items_path, low_memory=False)
        k_items_sub = k_items[k_items["item_id"].isin(kion_to_movie_key)].copy()
        dim_kion_credits = pd.DataFrame({
            "kion_item_id": as_int(k_items_sub["item_id"]),
            "movie_key": as_int(k_items_sub["item_id"].map(kion_to_movie_key)),
            "directors": clean_text(k_items_sub["directors_translated"]),
            "actors": clean_text(k_items_sub["actors_translated"]),
            "studios": clean_text(k_items_sub["studios"]),
            "keywords": clean_text(k_items_sub["keywords"]),
            "description": clean_text(k_items_sub["description"])
        })
        log("dim_kion_enrichment_credits", f"Loaded {len(dim_kion_credits):,} KION enriched titles")

    # -----------------------------------------------------------------------------
    # 7. ASSEMBLE ALL DOMAIN TABLES
    # -----------------------------------------------------------------------------
    print("\n[7/9] Assembling Normalized Snowflake Schema Tables...")

    # A. Central Crosswalk & Hub
    crosswalk_cols = [
        "movie_key", "tmdb_id", "movielens_id", "kion_item_id", "imdb_id",
        "title", "content_type", "release_year", "decade", "decade_label",
        "primary_genre", "primary_country", "match_confidence",
        "in_tmdb_master", "in_movielens", "in_kion", "in_all_three_sources",
        "source_combo", "source_count"
    ]
    dim_movie_crosswalk = m[[c for c in crosswalk_cols if c in m]].copy()

    # B. Cross-Source Comparative Analytics
    cross_analytics_cols = [
        "movie_key", "rating_gap_tmdb_minus_ml", "rating_gap_band", "rating_gap_band_sort",
        "rating_consensus_band", "rating_consensus_band_sort", "consensus_level", "consensus_level_sort",
        "signal_quadrant", "signal_quadrant_sort", "kion_vs_rating_mismatch", "kion_vs_rating_mismatch_sort",
        "title_age_at_kion", "age_bucket", "age_bucket_sort"
    ]
    fact_cross_source_analytics = m[[c for c in cross_analytics_cols if c in m]].copy()

    # C. TMDB Domain
    tmdb_mask = m.tmdb_id.notna() & m.in_tmdb_master
    m_tmdb = m[tmdb_mask].copy()

    dim_tmdb_movies = m_tmdb[[c for c in [
        "tmdb_id", "movie_key", "title", "original_title", "original_language",
        "release_date", "release_year", "decade", "runtime", "runtime_bin",
        "runtime_bin_sort", "status", "tagline"
    ] if c in m_tmdb]].copy()

    fact_tmdb_ratings = m_tmdb[[c for c in [
        "tmdb_id", "movie_key", "tmdb_popularity", "tmdb_vote_average", "tmdb_vote_count",
        "weighted_rating", "weighted_rating_tier", "weighted_rating_tier_sort",
        "popularity_tier", "popularity_tier_sort"
    ] if c in m_tmdb]].copy()
    if tmdb_features is not None:
        fact_tmdb_ratings = fact_tmdb_ratings.merge(
            tmdb_features[["tmdb_id", "year_normalized", "popularity_normalized", "vote_count_log",
                           "vote_count_normalized", "vote_average_normalized"]],
            on="tmdb_id", how="left"
        )
        for c in ["year_normalized", "popularity_normalized", "vote_count_log", "vote_count_normalized", "vote_average_normalized"]:
            if c in fact_tmdb_ratings:
                fact_tmdb_ratings[c] = as_dec(fact_tmdb_ratings[c], 4)

    fact_tmdb_financials = m_tmdb[[c for c in [
        "tmdb_id", "movie_key", "budget_clean", "revenue_clean", "profit", "roi",
        "has_financials", "budget_tier", "budget_tier_sort", "profit_tier", "profit_tier_sort",
        "collection_id", "collection_name", "is_franchise"
    ] if c in m_tmdb]].copy()

    dim_tmdb_enrichment_credits = m_tmdb[[c for c in [
        "tmdb_id", "movie_key", "director", "cast_top", "keywords",
        "production_companies", "production_countries", "spoken_languages", "overview"
    ] if c in m_tmdb]].copy()

    # D. KION Domain
    kion_mask = m.kion_item_id.notna() & m.in_kion
    m_kion = m[kion_mask].copy()

    dim_kion_items = m_kion[[c for c in [
        "kion_item_id", "movie_key", "title", "original_title", "content_type",
        "release_year", "age_rating", "for_kids"
    ] if c in m_kion]].copy()

    fact_kion_movie_metrics = m_kion[[c for c in [
        "kion_item_id", "movie_key", "kion_interaction_count", "kion_unique_users",
        "kion_avg_watched_pct", "kion_avg_duration_sec", "kion_completion_rate",
        "kion_total_watch_hours", "kion_engagement_score", "completion_category",
        "completion_category_sort", "engagement_decile", "kion_support_status",
        "kion_support_status_sort"
    ] if c in m_kion]].copy()

    # E. MovieLens Domain
    ml_mask = m.movielens_id.notna() & m.in_movielens
    m_ml = m[ml_mask].copy()

    dim_movielens_movies = m_ml[[c for c in ["movielens_id", "movie_key", "title", "release_year"] if c in m_ml]].copy()

    fact_movielens_movie_metrics = m_ml[[c for c in [
        "movielens_id", "movie_key", "movielens_rating_count", "movielens_unique_users",
        "movielens_avg_rating", "movielens_rating_std", "movielens_pct_high", "movielens_pct_low",
        "movielens_tag_count", "ml_weighted_rating", "ml_support_status", "ml_support_status_sort"
    ] if c in m_ml]].copy()

    # -----------------------------------------------------------------------------
    # 8. WRITE CLEAN MODULAR FILES
    # -----------------------------------------------------------------------------
    print("\n[8/9] Writing All Snowflake Tables to destination...")
    # Hub & Cross-Source
    write(dim_movie_crosswalk, dst, "dim_movie_crosswalk")
    write(fact_cross_source_analytics, dst, "fact_cross_source_analytics")

    # TMDB Domain
    write(dim_tmdb_movies, dst, "dim_tmdb_movies")
    write(fact_tmdb_ratings, dst, "fact_tmdb_ratings")
    write(fact_tmdb_financials, dst, "fact_tmdb_financials")
    write(dim_tmdb_enrichment_credits, dst, "dim_tmdb_enrichment_credits")

    # KION Domain
    write(dim_kion_items, dst, "dim_kion_items")
    if dim_kion_credits is not None: write(dim_kion_credits, dst, "dim_kion_enrichment_credits")
    write(fact_kion_movie_metrics, dst, "fact_kion_movie_metrics")
    if kion_user is not None: write(kion_user, dst, "dim_kion_user")
    if kion_fact is not None: write(kion_fact, dst, "fact_kion_interactions")

    # MovieLens Domain
    write(dim_movielens_movies, dst, "dim_movielens_movies")
    write(fact_movielens_movie_metrics, dst, "fact_movielens_movie_metrics")
    write(mu_o, dst, "dim_movielens_user")
    if ratings_y is not None: write(ratings_y, dst, "fact_movielens_ratings_yearly")
    if tags is not None: write(tags, dst, "fact_movielens_tags")
    if dim_genome_tag is not None: write(dim_genome_tag, dst, "dim_movielens_genome_tag")
    if fact_genome_scores is not None: write(fact_genome_scores, dst, "fact_movielens_genome_scores")

    # Shared Dimensions & Aggregates
    write(dd, dst, "dim_date")
    write(dg, dst, "dim_genre")
    write(bridge_genre, dst, "bridge_movie_genre")
    write(dc, dst, "dim_country")
    write(bridge_country, dst, "bridge_movie_country")
    write(agg_decay, dst, "agg_age_decay")
    write(agg_fit, dst, "agg_decay_fit")

    # -----------------------------------------------------------------------------
    # 9. VALIDATION, POWER QUERY & DAX MEASURES
    # -----------------------------------------------------------------------------
    print("\n[9/9] Validating schema integrity, relationships, and data types...")
    V = []
    def chk(name, ok, detail=""):
        V.append((name, "PASS" if ok else "FAIL", detail))
        print(f"  {'PASS' if ok else 'FAIL'}  {name} {detail}")

    chk("dim_movie_crosswalk PK unique & non-null", dim_movie_crosswalk.movie_key.is_unique and dim_movie_crosswalk.movie_key.notna().all(), f"{len(dim_movie_crosswalk):,} rows")
    chk("dim_tmdb_movies FK movie_key in crosswalk", dim_tmdb_movies.movie_key.isin(dim_movie_crosswalk.movie_key).all(), f"{len(dim_tmdb_movies):,} rows")
    chk("fact_tmdb_ratings FK movie_key in crosswalk", fact_tmdb_ratings.movie_key.isin(dim_movie_crosswalk.movie_key).all())
    chk("fact_tmdb_financials FK movie_key in crosswalk", fact_tmdb_financials.movie_key.isin(dim_movie_crosswalk.movie_key).all())
    chk("dim_kion_items FK movie_key in crosswalk", dim_kion_items.movie_key.isin(dim_movie_crosswalk.movie_key).all(), f"{len(dim_kion_items):,} rows")
    chk("fact_kion_movie_metrics FK movie_key in crosswalk", fact_kion_movie_metrics.movie_key.isin(dim_movie_crosswalk.movie_key).all())
    chk("dim_movielens_movies FK movie_key in crosswalk", dim_movielens_movies.movie_key.isin(dim_movie_crosswalk.movie_key).all(), f"{len(dim_movielens_movies):,} rows")
    chk("fact_movielens_movie_metrics FK movie_key in crosswalk", fact_movielens_movie_metrics.movie_key.isin(dim_movie_crosswalk.movie_key).all())
    chk("fact_cross_source_analytics FK movie_key in crosswalk", fact_cross_source_analytics.movie_key.isin(dim_movie_crosswalk.movie_key).all())
    chk("bridge_genre FK movie_key in crosswalk", bridge_genre.movie_key.isin(dim_movie_crosswalk.movie_key).all())
    chk("bridge_country FK movie_key in crosswalk", bridge_country.movie_key.isin(dim_movie_crosswalk.movie_key).all())
    if dim_genome_tag is not None:
        chk("dim_movielens_genome_tag PK unique", dim_genome_tag.tag_id.is_unique, f"{len(dim_genome_tag)} tags")
    if fact_genome_scores is not None:
        chk("fact_genome_scores FK movie_key in crosswalk", fact_genome_scores.movie_key.isin(dim_movie_crosswalk.movie_key).all())
        chk("fact_genome_scores FK tag_id in dim_genome_tag", fact_genome_scores.tag_id.isin(dim_genome_tag.tag_id).all(), f"{len(fact_genome_scores):,} rows")

    # Raw CSV data type audit
    bad = []
    for f in sorted(dst.glob("*.csv")):
        if f.name in ("cleaning_log.csv", "schema_report.csv", "cleaning_validation.csv"): continue
        df_check = pd.read_csv(f, dtype=str, encoding="utf-8-sig", keep_default_na=False)
        for c in df_check.columns:
            col = df_check[c]
            is_dec = (f.stem, c, "Decimal number") in {(a, b, k) for a, b, k, *_ in SCHEMA}
            nz = col[col != ""]
            hit = nz.str.fullmatch(r"-?\d+\.0+")
            if (is_dec and len(nz) and hit.all()) or (not is_dec and hit.any()):
                bad.append(f"{f.stem}.{c} has '.0' values")
            if col.isin(["nan", "NaN", "None", "NaT", "<NA>"]).any():
                bad.append(f"{f.stem}.{c} has literal nan text")
    chk("Typing audit: no '.0' strings or literal 'nan' in CSVs", not bad, str(bad[:5]))

    # Generate Power Query M scripts
    pq = dst / "power_query"
    pq.mkdir(exist_ok=True)
    MT = {"Whole number": "Int64.Type", "Decimal number": "type number", "True/False": "type logical", "Date": "type date", "Text": "type text"}
    sch = pd.DataFrame(SCHEMA, columns=["table", "column", "kind", "nulls", "null_pct"])
    for tname, grp in sch.groupby("table"):
        types = ", ".join(f'{{"{r.column}", {MT[r.kind]}}}' for r in grp.itertuples())
        (pq / f"{tname}.m").write_text(
            "let\n"
            f'    Source = Csv.Document(File.Contents(DataFolder & "{tname}.csv"), [Delimiter = ",", Encoding = 65001, QuoteStyle = QuoteStyle.Csv]),\n'
            "    Promoted = Table.PromoteHeaders(Source, [PromoteAllScalars = true]),\n"
            f"    Typed = Table.TransformColumnTypes(Promoted, {{{types}}}, \"en-US\")\n"
            "in\n    Typed\n", encoding="utf-8")

    # Generate complete DAX Measures
    dax_code = """// ============================================================
//  CineMetrics DAX Measures — Complete Snowflake Schema
//  Create: Home > Enter Data > name table "_Measures" > Load
//  Measures directly reference clean domain tables.
// ============================================================

// ---- PORTFOLIO & RECONCILIATION ----

Titles = COUNTROWS ( dim_movie_crosswalk )

Titles in All 3 = CALCULATE ( COUNTROWS ( dim_movie_crosswalk ), dim_movie_crosswalk[in_all_three_sources] = TRUE() )

% All 3 Sources = DIVIDE ( [Titles in All 3], [Titles] )

Strong Consensus = CALCULATE ( COUNTROWS ( fact_cross_source_analytics ), fact_cross_source_analytics[consensus_level_sort] = 1 )


// ---- FINANCIAL METRICS (TMDB) ----

Total Revenue = CALCULATE ( SUM ( fact_tmdb_financials[revenue_clean] ), fact_tmdb_financials[has_financials] = TRUE() )

Total Budget = CALCULATE ( SUM ( fact_tmdb_financials[budget_clean] ), fact_tmdb_financials[has_financials] = TRUE() )

Total Profit = [Total Revenue] - [Total Budget]

Median ROI = MEDIAN ( fact_tmdb_financials[roi] )

ROI Franchise = CALCULATE ( [Median ROI], fact_tmdb_financials[is_franchise] = TRUE() )

ROI Standalone = CALCULATE ( [Median ROI], fact_tmdb_financials[is_franchise] = FALSE() )


// ---- RATINGS & COMPARISON ----

TMDB Avg Vote = AVERAGE ( fact_tmdb_ratings[tmdb_vote_average] )

ML Avg Rating = AVERAGE ( fact_movielens_movie_metrics[movielens_avg_rating] )

ML Avg 10pt = [ML Avg Rating] * 2

Rating Gap = AVERAGE ( fact_cross_source_analytics[rating_gap_tmdb_minus_ml] )

Correlation TMDB vs ML =
    VAR T = FILTER (
        ADDCOLUMNS (
            VALUES ( dim_movie_crosswalk[movie_key] ),
            "TMDB_Vote", CALCULATE ( SELECTEDVALUE ( fact_tmdb_ratings[tmdb_vote_average] ) ),
            "ML_Rating", CALCULATE ( SELECTEDVALUE ( fact_movielens_movie_metrics[movielens_avg_rating] ) ),
            "ML_Count", CALCULATE ( SELECTEDVALUE ( fact_movielens_movie_metrics[movielens_rating_count] ) )
        ),
        NOT ISBLANK ( [TMDB_Vote] ) && [ML_Count] >= 30
    )
    VAR n = COUNTROWS ( T )
    VAR sx = SUMX ( T, [TMDB_Vote] )
    VAR sy = SUMX ( T, [ML_Rating] )
    VAR sxy = SUMX ( T, [TMDB_Vote] * [ML_Rating] )
    VAR sxx = SUMX ( T, [TMDB_Vote] ^ 2 )
    VAR syy = SUMX ( T, [ML_Rating] ^ 2 )
    RETURN DIVIDE ( n * sxy - sx * sy, SQRT ( ( n * sxx - sx ^ 2 ) * ( n * syy - sy ^ 2 ) ) )


// ---- STREAMING ENGAGEMENT (KION) ----

Avg Completion = CALCULATE ( AVERAGE ( fact_kion_movie_metrics[kion_completion_rate] ), fact_kion_movie_metrics[kion_interaction_count] >= 10 )

Avg Viewers per Title = CALCULATE ( AVERAGE ( fact_kion_movie_metrics[kion_unique_users] ), fact_kion_movie_metrics[kion_unique_users] > 0 )

Engagement Score = AVERAGE ( fact_kion_movie_metrics[kion_engagement_score] )


// ---- GENOME TAG ANALYTICS (MOVIELENS) ----

Top Genome Tag Relevance = MAX ( fact_movielens_genome_scores[relevance] )

Avg Genome Relevance = AVERAGE ( fact_movielens_genome_scores[relevance] )

Distinct Genome Tags = DISTINCTCOUNT ( fact_movielens_genome_scores[tag_id] )


// ---- AGE DECAY CURVE ----

Decay Viewers = AVERAGE ( agg_age_decay[avg_viewers] )

Decay Fitted = AVERAGE ( agg_age_decay[fitted_viewers] )

Half-Life Yrs = SELECTEDVALUE ( agg_decay_fit[halflife_years] )
"""
    (dst / "MovieAnalytics_Measures.dax").write_text(dax_code, encoding="utf-8")

    # Reports
    pd.DataFrame(LOG, columns=["table", "action", "detail"]).to_csv(dst / "cleaning_log.csv", index=False)
    pd.DataFrame(SCHEMA, columns=["table", "column", "power_bi_type", "nulls", "null_pct"]).to_csv(dst / "schema_report.csv", index=False)
    pd.DataFrame(V, columns=["check", "status", "detail"]).to_csv(dst / "cleaning_validation.csv", index=False)

    print(f"\nPipeline completed in {time.time()-t0:.1f}s!")
    print(f"Results written to: {dst.resolve()}")

if __name__ == "__main__":
    main()
