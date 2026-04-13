"""Tests for kNN training and recommendation."""

import pandas as pd
import pytest

from books_recommender.models.knn import (
    build_knn,
    load_artifacts,
    recommend_by_title,
    save_artifacts,
)


@pytest.fixture
def tiny_ratings() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": [1, 1, 1, 2, 2, 2, 3, 3],
            "title": ["A", "B", "C", "A", "B", "C", "A", "B"],
            "rating": [1.0, 2.0, 0.5, -0.5, 0.0, 1.0, 0.0, 1.0],
        }
    )


@pytest.fixture
def tiny_book_meta(tiny_ratings: pd.DataFrame) -> pd.DataFrame:
    titles = tiny_ratings["title"].unique()
    return pd.DataFrame(
        {
            "author": [f"Author {t}" for t in titles],
            "image_url": ["" for _ in titles],
        },
        index=titles,
    )


def test_build_knn_mappers_are_inverse(tiny_ratings: pd.DataFrame) -> None:
    _, _, book_mapper, title_to_idx = build_knn(tiny_ratings)
    for idx, title in book_mapper.items():
        assert title_to_idx[title] == idx


def test_recommend_unknown_title_returns_empty_columns(
    tiny_ratings: pd.DataFrame,
    tiny_book_meta: pd.DataFrame,
) -> None:
    model, book_sparse, book_mapper, title_to_idx = build_knn(tiny_ratings)
    out = recommend_by_title(
        book_title="not-a-real-title",
        model=model,
        book_sparse=book_sparse,
        title_to_idx=title_to_idx,
        book_mapper=book_mapper,
        book_meta=tiny_book_meta,
        n=3,
    )
    assert out.empty
    assert list(out.columns) == ["title", "author", "image_url", "distance"]


def test_recommend_known_title_returns_neighbors(
    tiny_ratings: pd.DataFrame,
    tiny_book_meta: pd.DataFrame,
) -> None:
    model, book_sparse, book_mapper, title_to_idx = build_knn(tiny_ratings)
    out = recommend_by_title(
        book_title="A",
        model=model,
        book_sparse=book_sparse,
        title_to_idx=title_to_idx,
        book_mapper=book_mapper,
        book_meta=tiny_book_meta,
        n=2,
    )
    assert not out.empty
    assert list(out.columns) == ["title", "author", "image_url", "distance"]
    assert "A" not in out["title"].tolist()


def test_save_and_load_artifacts_roundtrip(
    tiny_ratings: pd.DataFrame,
    tiny_book_meta: pd.DataFrame,
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "books_recommender.models.knn.ARTIFACTS_PATH",
        tmp_path / "recommender_system.joblib",
    )
    model, book_sparse, book_mapper, title_to_idx = build_knn(tiny_ratings)
    save_artifacts(
        model=model,
        book_sparse=book_sparse,
        book_mapper=book_mapper,
        title_to_idx=title_to_idx,
        book_meta=tiny_book_meta,
    )
    loaded = load_artifacts()
    assert loaded["title_to_idx"] == title_to_idx
    recs = recommend_by_title(
        book_title="B",
        model=loaded["model"],
        book_sparse=loaded["book_sparse"],
        title_to_idx=loaded["title_to_idx"],
        book_mapper=loaded["book_mapper"],
        book_meta=loaded["book_meta"],
        n=2,
    )
    assert not recs.empty
