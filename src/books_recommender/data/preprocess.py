"""
Preprocessing for Books data.

Normalizes keys, removes duplicates, filters sparse users/books,
and returns a clean (user_id, title, rating) table plus book metadata.
"""

import pandas as pd
from loguru import logger

from books_recommender.config import MIN_BOOK_RATINGS, MIN_USER_RATINGS


def preprocess(
    books: pd.DataFrame,
    ratings: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Preprocess raw tables into a clean ratings table and metadata.

    Args:
        books: Raw books table.
        ratings: Raw ratings table.

    Returns:
        A tuple (ratings_clean, book_meta):
          - ratings_clean: DataFrame with columns
            ['user_id', 'title', 'rating'].
          - book_meta: DataFrame indexed by title with columns
            ['author', 'image_url'].
    """
    books = books.copy()
    ratings = ratings.copy()

    books['ISBN'] = books['ISBN'].astype(str).str.strip().str.upper()
    ratings['ISBN'] = ratings['ISBN'].astype(str).str.strip().str.upper()

    books.drop_duplicates(subset='ISBN', inplace=True)
    ratings.drop_duplicates(subset=['User-ID', 'ISBN'], inplace=True)
    logger.info("After dedup: {} books, {} ratings", len(books), len(ratings))

    books.rename(
        columns={
            'Book-Title': 'title',
            'Book-Author': 'author',
            'Year-Of-Publication': 'year_of_publication',
            'Publisher': 'publisher',
            'Image-URL-M': 'image_url',
        },
        inplace=True,
    )

    ratings.rename(
        columns={
            'User-ID': 'user_id',
            'Book-Rating': 'rating',
        },
        inplace=True,
    )

    ratings = ratings[ratings['rating'] > 0]
    logger.info("Ratings after removing zeros: {}", len(ratings))

    active_users = ratings['user_id'].value_counts()
    active_users = active_users[active_users > MIN_USER_RATINGS].index
    ratings = ratings[ratings['user_id'].isin(active_users)]
    logger.info("Ratings after active-user filter (>{} ratings): {}", MIN_USER_RATINGS, len(ratings))

    ratings = ratings.merge(books, on='ISBN', how='inner')
    logger.info("Ratings after merge with books: {}", len(ratings))

    rating_counts = (
        ratings.groupby('ISBN', as_index=False)['rating']
        .count()
        .rename(columns={'rating': 'num_of_rating'})
    )

    ratings = ratings.merge(rating_counts, on='ISBN', how='inner')
    ratings = ratings[ratings['num_of_rating'] >= MIN_BOOK_RATINGS]
    ratings.drop_duplicates(subset=['user_id', 'ISBN'], inplace=True)
    logger.info("Ratings after min-book-ratings filter (>={}) and final dedup: {}", MIN_BOOK_RATINGS, len(ratings))

    # --- Normalize ratings (user-centering) ---
    user_mean = ratings.groupby('user_id')['rating'].transform('mean')
    ratings['rating'] = ratings['rating'] - user_mean

    book_meta = (
        ratings[['title', 'author', 'image_url']]
        .drop_duplicates(subset='title')
        .set_index('title')
    )

    ratings_clean = ratings[['user_id', 'title', 'rating']]
    return ratings_clean, book_meta
