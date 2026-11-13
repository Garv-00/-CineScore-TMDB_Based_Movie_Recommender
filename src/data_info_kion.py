import pandas as pd

interactions = pd.read_csv("data/kion/interactions.csv")
users_en = pd.read_csv("data/kion/data_en/users_en.csv")
items_en = pd.read_csv("data/kion/data_en/items_en.csv")

# print(df.isnull().sum())
# print(users_en.isnull().sum())
# print(title_en.isnull().sum())

# Trying to recover the missing data

users_original = pd.read_csv("data/kion/data_original/users.csv")
items_original = pd.read_csv("data/kion/data_original/items.csv")

# print("Interactions:", interactions.columns)
# print("Users EN:", users_en.columns)
# print("Users Original:", users_original.columns)
# print("Items EN:", items_en.columns)
# print("Items Original:", items_original.columns)

# print ("Users EN unique :",users_en["user_id"].nunique())
# print ("Users Original; unique :",users_original["user_id"].nunique())
# print ("Users Interaction unique :",interactions["user_id"].nunique())  # We lose around 840,197 - 567,588 = 272,609 users due to no interaction


# print("Items EN unique:",items_en["item_id"].nunique())
# print("Items Original unique:",items_original["item_id"].nunique())
# print("Interaction items:",interactions["item_id"].nunique())             # We lose around 15,963 - 12,693 = 3,270 items here 


# # Compare User files 
# users_compare = users_en.merge(
#     users_original,
#     on="user_id",
#     how="left",
#     suffixes=("_en", "_original")
# )

# for column in ["age", "income", "sex", "kids_flg"]:

#     en_column = f"{column}_en"
#     original_column = f"{column}_original"

#     if (
#         en_column in users_compare.columns
#         and original_column in users_compare.columns
#     ):

#         recoverable = (
#             users_compare[en_column].isna()
#             & users_compare[original_column].notna()
#         ).sum()

#         print(
#             f"{column}: {recoverable:,} "
#             "missing EN values recoverable from original"
#         )
# # Nothing is recoverable here 

# items_compare = items_en.merge(
#     items_original,
#     on="item_id",
#     how="left",
#     suffixes=("_en", "_original")
# )

# item_columns = [
#     "title",
#     "title_orig",
#     "release_year",
#     "genres",
#     "countries",
#     "for_kids",
#     "age_rating",
#     "studios",
#     "description",
#     "keywords",
#     "actors_translated",
#     "actors_transliterated",
#     "directors_translated",
#     "transliterated",
# ]

# for column in item_columns:

#     en_column = f"{column}_en"
#     original_column = f"{column}_original"

#     if (
#         en_column in items_compare.columns
#         and original_column in items_compare.columns
#     ):

#         recoverable = (
#             items_compare[en_column].isna()        
#             & items_compare[original_column].notna()
#         ).sum()

#         print(
#             f"{column}: {recoverable:,} "
#             "missing EN values recoverable from original"
#         )

# We can recover 2 misiing titles , 6 missing keywords


# recovering vakues and also merging the files in one as kion_master_data.py

items_original_lookup = items_original[
    [
        "item_id",
        "title",
        "keywords"
    ]
].copy()


items_en = items_en.merge(
    items_original_lookup,
    on="item_id",
    how="left",
    suffixes=("", "_original")
)

missing_titles = items_en["title"].isna()

items_en.loc[
    missing_titles,
    "title"
] = items_en.loc[
    missing_titles,
    "title_original"
]

missing_keywords = items_en["keywords"].isna()

items_en.loc[
    missing_keywords,
    "keywords"
] = items_en.loc[
    missing_keywords,
    "keywords_original"
]

items_en = items_en.drop(
    columns=[
        "title_original",
        "keywords_original"
    ]
)

if not users_en["user_id"].is_unique:
    raise ValueError(
        "user_id is not unique in the users dataset."
    )

if not items_en["item_id"].is_unique:
    raise ValueError(
        "item_id is not unique in the items dataset."
    )

master = interactions.merge(
    users_en,
    on="user_id",
    how="left",
    validate="many_to_one"
)

master = master.merge(
    items_en,
    on="item_id",
    how="left",
    validate="many_to_one"
)

print("Interactions:", len(interactions))
print("Master:", len(master))

if len(master) != len(interactions):
    raise ValueError(
        "Master row count does not match interactions."
    )

print("\nMissing values in master dataset:")
print(
    master.isna().sum()[
        master.isna().sum() > 0
    ].sort_values(
        ascending=False
    )
)

master.to_csv(
    "data/kion/kion_master_data.csv",
    index=False
)
