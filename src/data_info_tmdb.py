import pandas as pd 
import numpy as np 


df = pd.read_csv("data/tmdb/movies.csv")
# print(df.info())
# print(df.isnull().sum())

# once we have the idea of what is missing by info() function we can try to fetch that data from the TMBD data base itself 
missing_overview = df[df["overview"].isnull()]
print(missing_overview)
missing_release_date = df[df["release_date"].isnull()]
print(missing_release_date)


print(df.dtypes) # We can see that genre_ids are str data type , release_date are str data type 
import ast

# df["release_date"] = df["release_date"].apply(ast.literal_eval) # ERROR As  leading zeros in decimal integer literals are not permitted  1994-09-23
df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce") # Using this
# df["genre_ids"] = df["genre_ids"].apply(ast.literal_eval) # This will make the genre id into an obj as they are in str data type but stored as [1,2] format which is a list format we have to explode() them then save then as elements in list  genre_ids ----> object
# print(df.dtypes)


import requests
from dotenv import load_dotenv
import os

load_dotenv()

movie_ids = [670355, 1045781, 733668, 1198870] # All movie_id to check for missing_overview & missing_release_date 

API_KEY = os.getenv("TMDB_API_KEY")

if not API_KEY:
    raise ValueError("TMDB API_KEY not found in .env file")

# for movie_id in movie_ids:
#     url = f"https://api.themoviedb.org/3/movie/{movie_id}"

#     response = requests.get(
#         url,
#         params={
#             "api_key" : API_KEY
#         }
#     )

#     response.raise_for_status()

#     data = response.json()

#     overview = data.get("overview") or "Unknown"
#     release_date = data.get("release_date") or "Unknown"

#     print(movie_id,overview,release_date ) # Output says 3 missing dates remain not found also the 2 missing overviews 

df["genre_ids"] = df["genre_ids"].apply(
    lambda x: ast.literal_eval(x) if pd.notna(x) else []
)
url = "https://api.themoviedb.org/3/genre/movie/list"

response = requests.get(
    url,
    params={
        "api_key": API_KEY,
        "language": "en"
    }
)

response.raise_for_status()

genre_data = response.json()

genre_map = {}

for genre in genre_data["genres"]:
    genre_map[genre["id"]] = genre["name"]

print(genre_map)

df["genres"] = df["genre_ids"].apply(
    lambda ids: [
        genre_map.get(genre_id, "Unknown")
        for genre_id in ids
    ]
)
# Adding release year column here 

df["release_year"] = (
    pd.to_datetime(
        df["release_date"],
        errors="coerce"
    )
    .dt.year
    .astype("Int64")
)


df.to_csv(
    "data/tmdb/movies_with_genres.csv",
    index=False
)

# import matplotlib.pyplot as plt

# genre_count = df["genres"].explode().value_counts()

# genre_count.plot(
#     kind="bar",
#     figsize=(12,6)
# )

# plt.xlabel("Genres")
# plt.ylabel("Number of Movies")
# plt.title("Number of Movies by Genre")
# plt.tight_layout
# plt.xticks(rotation = 45)
# plt.show()


















