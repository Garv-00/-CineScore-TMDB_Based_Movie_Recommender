import os
import tempfile
import time
import requests
import pandas as pd

from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv


load_dotenv()

INPUT_PATH = "data/tmdb/movies_with_genres.csv"
OUTPUT_PATH = "data/tmdb/tmdb_master.csv"

TMDB_API_URL = "https://api.themoviedb.org/3/movie"

MAX_WORKERS = 100
MAX_RETRIES = 5
REQUEST_TIMEOUT = 30

TMDB_API_KEY = os.getenv("TMDB_API_KEY")

if not TMDB_API_KEY:
    raise ValueError(
        "TMDB_API_KEY not found in environment variables."
    )

HEADERS = {
    "Authorization": f"Bearer {TMDB_API_KEY}",
    "accept": "application/json",
}


def fetch_tmdb_metadata(tmdb_id):

    url = f"{TMDB_API_URL}/{int(tmdb_id)}"

    params = {
        "language": "en-US",
        "append_to_response": "credits,keywords"
    }

    for attempt in range(1, MAX_RETRIES + 1):

        try:

            response = requests.get(
                url,
                headers=HEADERS,
                params=params,
                timeout=REQUEST_TIMEOUT
            )

            if response.status_code == 404:

                return {
                    "id": tmdb_id,
                    "tmdb_enriched": False
                }

            if response.status_code == 429:

                retry_after = response.headers.get(
                    "Retry-After",
                    "2"
                )

                try:
                    wait_time = float(retry_after)
                except ValueError:
                    wait_time = 2

                time.sleep(
                    min(wait_time * attempt, 30)
                )

                continue

            response.raise_for_status()

            data = response.json()

            credits = data.get(
                "credits",
                {}
            )

            cast = credits.get(
                "cast",
                []
            )

            crew = credits.get(
                "crew",
                []
            )

            companies = data.get(
                "production_companies",
                []
            )

            keywords = (
                data.get("keywords", {})
                .get("keywords", [])
            )

            directors = [
                person.get("name")
                for person in crew
                if person.get("job") == "Director"
                and person.get("name")
            ]

            cast_names = [
                person.get("name")
                for person in cast[:20]
                if person.get("name")
            ]

            company_names = [
                company.get("name")
                for company in companies
                if company.get("name")
            ]

            keyword_names = [
                keyword.get("name")
                for keyword in keywords
                if keyword.get("name")
            ]

            return {
                "id": tmdb_id,

                "tmdb_keywords": (
                    " | ".join(
                        dict.fromkeys(keyword_names)
                    )
                    if keyword_names
                    else pd.NA
                ),

                "tmdb_director": (
                    " | ".join(
                        dict.fromkeys(directors)
                    )
                    if directors
                    else pd.NA
                ),

                "tmdb_cast": (
                    " | ".join(
                        dict.fromkeys(cast_names)
                    )
                    if cast_names
                    else pd.NA
                ),

                "tmdb_production_companies": (
                    " | ".join(
                        dict.fromkeys(company_names)
                    )
                    if company_names
                    else pd.NA
                ),

                "tmdb_runtime": (
                    data.get("runtime")
                    if data.get("runtime") is not None
                    else pd.NA
                ),

                "tmdb_original_language": (
                    data.get("original_language")
                    if data.get("original_language")
                    else pd.NA
                ),

                "tmdb_enriched": True
            }

        except requests.exceptions.RequestException:

            if attempt == MAX_RETRIES:

                return {
                    "id": tmdb_id,
                    "tmdb_enriched": False
                }

            time.sleep(
                min(2 ** (attempt - 1), 30)
            )

        except Exception:

            return {
                "id": tmdb_id,
                "tmdb_enriched": False
            }

    return {
        "id": tmdb_id,
        "tmdb_enriched": False
    }


def clean_base(df):

    # Remove unwanted index columns
    df = df.drop(
        columns=[
            column
            for column in df.columns
            if column.startswith("Unnamed:")
        ],
        errors="ignore"
    )

    # Convert TMDB ID
    df["id"] = pd.to_numeric(
        df["id"],
        errors="coerce"
    ).astype("Int64")

    # Convert numeric columns
    for column in [
        "popularity",
        "vote_average"
    ]:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            ).astype("Float64")

    if "vote_count" in df.columns:

        df["vote_count"] = pd.to_numeric(
            df["vote_count"],
            errors="coerce"
        ).astype("Int64")

    # Convert dates
    if "release_date" in df.columns:

        df["release_date"] = pd.to_datetime(
            df["release_date"],
            errors="coerce"
        )

    # Create release year only if missing
    if "release_year" not in df.columns:

        df["release_year"] = (
            df["release_date"]
            .dt.year
            .astype("Int64")
        )

    else:

        df["release_year"] = pd.to_numeric(
            df["release_year"],
            errors="coerce"
        ).astype("Int64")

    # Convert text columns
    for column in [
        "title",
        "overview",
        "original_language",
        "genre_ids",
        "genres"
    ]:

        if column in df.columns:

            df[column] = df[column].astype(
                "string"
            )

    # Remove exact duplicate rows
    df = df.drop_duplicates()

    # Remove rows without TMDB IDs
    df = df.dropna(
        subset=["id"]
    )

    # Resolve duplicate TMDB IDs
    df = (
        df
        .sort_values(
            ["id", "vote_count"],
            ascending=[True, False],
            na_position="last"
        )
        .drop_duplicates(
            subset="id",
            keep="first"
        )
        .reset_index(drop=True)
    )

    return df


def enrich_movies(df):

    movie_ids = (
        df["id"]
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    results = []

    with ThreadPoolExecutor(
        max_workers=MAX_WORKERS
    ) as executor:

        futures = {
            executor.submit(
                fetch_tmdb_metadata,
                tmdb_id
            ): tmdb_id
            for tmdb_id in movie_ids
        }

        completed = 0

        for future in as_completed(futures):

            results.append(
                future.result()
            )

            completed += 1

            if completed % 500 == 0:

                print(
                    f"TMDB enrichment: "
                    f"{completed}/{len(movie_ids)}"
                )

    enrichment = pd.DataFrame(
        results
    )

    enrichment["id"] = pd.to_numeric(
        enrichment["id"],
        errors="coerce"
    ).astype("Int64")

    enrichment = (
        enrichment
        .drop_duplicates(
            subset="id",
            keep="last"
        )
    )

    # Ensure one-to-one merge
    if enrichment["id"].duplicated().any():

        raise ValueError(
            "Duplicate TMDB IDs found in API results."
        )

    master = df.merge(
        enrichment,
        on="id",
        how="left",
        validate="one_to_one"
    )

    return master


def standardize_types(df):

    # Final datatype cleanup
    df["id"] = pd.to_numeric(
        df["id"],
        errors="coerce"
    ).astype("Int64")

    for column in [
        "release_year",
        "vote_count",
        "tmdb_runtime"
    ]:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            ).astype("Int64")

    for column in [
        "popularity",
        "vote_average"
    ]:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            ).astype("Float64")

    if "release_date" in df.columns:

        df["release_date"] = pd.to_datetime(
            df["release_date"],
            errors="coerce"
        )

    for column in [
        "title",
        "overview",
        "original_language",
        "genre_ids",
        "genres",
        "tmdb_keywords",
        "tmdb_director",
        "tmdb_cast",
        "tmdb_production_companies",
        "tmdb_original_language"
    ]:

        if column in df.columns:

            df[column] = df[column].astype(
                "string"
            )

    if "tmdb_enriched" in df.columns:

        df["tmdb_enriched"] = (
            df["tmdb_enriched"]
            .fillna(False)
            .astype(bool)
        )

    return df


def validate(df):

    # Validate final master
    if df.empty:

        raise ValueError(
            "TMDB master is empty."
        )

    if df["id"].isna().any():

        raise ValueError(
            "TMDB master contains missing IDs."
        )

    if df["id"].duplicated().any():

        raise ValueError(
            "TMDB master contains duplicate IDs."
        )

    if df.duplicated().any():

        raise ValueError(
            "TMDB master contains duplicate rows."
        )

    unwanted_columns = [
        column
        for column in df.columns
        if column.startswith("Unnamed:")
    ]

    if unwanted_columns:

        raise ValueError(
            f"Unwanted columns found: "
            f"{unwanted_columns}"
        )

    if "vote_average" in df.columns:

        invalid = (
            (df["vote_average"] < 0)
            |
            (df["vote_average"] > 10)
        )

        if invalid.fillna(False).any():

            raise ValueError(
                "Invalid vote_average values."
            )

    if "vote_count" in df.columns:

        invalid = (
            df["vote_count"] < 0
        )

        if invalid.fillna(False).any():

            raise ValueError(
                "Invalid vote_count values."
            )

    if "popularity" in df.columns:

        invalid = (
            df["popularity"] < 0
        )

        if invalid.fillna(False).any():

            raise ValueError(
                "Invalid popularity values."
            )


def save_master(df):

    # Save safely
    folder = os.path.dirname(
        OUTPUT_PATH
    )

    os.makedirs(
        folder,
        exist_ok=True
    )

    fd, temp_path = tempfile.mkstemp(
        suffix=".csv",
        dir=folder
    )

    os.close(fd)

    try:

        df.to_csv(
            temp_path,
            index=False
        )

        check = pd.read_csv(
            temp_path,
            low_memory=False
        )

        if len(check) != len(df):

            raise ValueError(
                "Saved row count does not match."
            )

        if check["id"].duplicated().any():

            raise ValueError(
                "Saved file contains duplicate IDs."
            )

        os.replace(
            temp_path,
            OUTPUT_PATH
        )

    except Exception:

        if os.path.exists(temp_path):
            os.remove(temp_path)

        raise


def main():

    # Load the existing 9,964-movie catalog
    df = pd.read_csv(
        INPUT_PATH,
        low_memory=False
    )

    original_rows = len(df)

    df = clean_base(df)

    if len(df) != original_rows:

        print(
            f"Rows after cleaning: "
            f"{len(df)} "
            f"(from {original_rows})"
        )

    master = enrich_movies(df)

    master = standardize_types(
        master
    )

    validate(master)

    save_master(master)

    print(
        f"TMDB master created: "
        f"{OUTPUT_PATH}"
    )

    print(
        f"Rows: {len(master)} | "
        f"Columns: {len(master.columns)} | "
        f"Unique TMDB IDs: "
        f"{master['id'].nunique()}"
    )


if __name__ == "__main__":
    main()