"""Tests for leave-one-out evaluation helpers."""

import numpy as np
import pandas as pd

from books_recommender.evaluation import (
    leave_one_out_split,
    recommend_from_history,
)


def test_leave_one_out_holds_one_row_per_eligible_user() -> None:
    ratings = pd.DataFrame(
        {
            "user_id": [1, 1, 1, 2, 2, 2],
            "title": ["a", "b", "c", "x", "y", "z"],
            "rating": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )
    rng = np.random.default_rng(0)
    train, test = leave_one_out_split(ratings, rng)
    assert len(test) == 2
    assert len(train) == len(ratings) - 2
    assert len(ratings) == len(train) + len(test)


def test_leave_one_out_skips_users_with_single_rating() -> None:
    ratings = pd.DataFrame(
        {
            "user_id": [1, 2, 2],
            "title": ["only", "a", "b"],
            "rating": [1.0, 1.0, 1.0],
        }
    )
    rng = np.random.default_rng(0)
    _train, test = leave_one_out_split(ratings, rng)
    assert len(test) == 1
    assert test.iloc[0]["user_id"] == 2


def test_recommend_from_history_empty_when_no_known_titles() -> None:
    ratings = pd.DataFrame(
        {
            "user_id": [1, 2],
            "title": ["X", "Y"],
            "rating": [1.0, 1.0],
        }
    )
    from books_recommender.models.knn import build_knn

    model, book_sparse, book_mapper, title_to_idx = build_knn(ratings)
    out = recommend_from_history(
        history_titles=["missing"],
        model=model,
        book_sparse=book_sparse,
        title_to_idx=title_to_idx,
        book_mapper=book_mapper,
        k=5,
        neighbors_per_item=5,
    )
    assert out == []


def test_recommend_from_history_returns_other_titles() -> None:
    ratings = pd.DataFrame(
        {
            "user_id": [1, 1, 1, 2, 2, 2, 3, 3],
            "title": ["A", "B", "C", "A", "B", "C", "A", "B"],
            "rating": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )
    from books_recommender.models.knn import build_knn

    model, book_sparse, book_mapper, title_to_idx = build_knn(ratings)
    out = recommend_from_history(
        history_titles=["A"],
        model=model,
        book_sparse=book_sparse,
        title_to_idx=title_to_idx,
        book_mapper=book_mapper,
        k=5,
        neighbors_per_item=10,
    )
    assert "A" not in out
    assert len(out) >= 1
