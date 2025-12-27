"""
Data loading utilities for the Books dataset.
"""

import pandas as pd

from books_recommender.config import RAW_DIR


def load_users() -> pd.DataFrame:
    """Load users CSV."""
    return pd.read_csv(
        RAW_DIR / 'BX-Users.csv',
        sep=';',
        encoding='latin-1',
        on_bad_lines='skip',
    )


def load_books() -> pd.DataFrame:
    """Load books CSV."""
    return pd.read_csv(
        RAW_DIR / 'BX-Books.csv',
        sep=';',
        encoding='latin-1',
        on_bad_lines='skip',
        low_memory=False,
    )


def load_ratings() -> pd.DataFrame:
    """Load ratings CSV."""
    return pd.read_csv(
        RAW_DIR / 'BX-Book-Ratings.csv',
        sep=';',
        encoding='latin-1',
        on_bad_lines='skip',
    )
