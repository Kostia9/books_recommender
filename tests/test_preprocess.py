"""Tests for ratings/book preprocessing."""

import pandas as pd
import pytest

from books_recommender.data.preprocess import preprocess


def test_preprocess_produces_clean_ratings_and_meta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("books_recommender.data.preprocess.MIN_USER_RATINGS", 1)
    monkeypatch.setattr("books_recommender.data.preprocess.MIN_BOOK_RATINGS", 1)
    books = pd.DataFrame(
        {
            "ISBN": ["X1", "Y1"],
            "Book-Title": ["Book X", "Book Y"],
            "Book-Author": ["Ax", "Ay"],
            "Year-Of-Publication": [2000, 2001],
            "Publisher": ["P", "P"],
            "Image-URL-M": ["http://x", "http://y"],
        }
    )
    ratings = pd.DataFrame(
        {
            "User-ID": [1, 1, 2, 2],
            "ISBN": ["X1", "Y1", "X1", "Y1"],
            "Book-Rating": [5, 5, 4, 4],
        }
    )
    ratings_clean, book_meta = preprocess(books, ratings)

    assert list(ratings_clean.columns) == ["user_id", "title", "rating"]
    assert list(book_meta.columns) == ["author", "image_url"]
    assert set(book_meta.index) == {"Book X", "Book Y"}
    assert len(ratings_clean) == 4
