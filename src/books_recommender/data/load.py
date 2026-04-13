"""
Data loading utilities for the Books dataset.
"""

import pandas as pd
from loguru import logger

from books_recommender.config import RAW_DIR


def load_books() -> pd.DataFrame:
    """Load books CSV."""
    path = RAW_DIR / "BX-Books.csv"
    if not path.exists():
        raise FileNotFoundError(f"Required data file not found: {path}")
    df = pd.read_csv(
        path,
        sep=";",
        encoding="latin-1",
        on_bad_lines="skip",
        low_memory=False,
        usecols=["ISBN", "Book-Title", "Book-Author", "Image-URL-M"],
    )
    logger.info("Loaded {} rows from {}", len(df), path.name)
    return df


def load_ratings() -> pd.DataFrame:
    """Load ratings CSV."""
    path = RAW_DIR / "BX-Book-Ratings.csv"
    if not path.exists():
        raise FileNotFoundError(f"Required data file not found: {path}")
    df = pd.read_csv(
        path,
        sep=";",
        encoding="latin-1",
        on_bad_lines="skip",
    )
    logger.info("Loaded {} rows from {}", len(df), path.name)
    return df
