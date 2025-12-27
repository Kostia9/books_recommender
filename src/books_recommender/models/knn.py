"""
kNN model utilities for the Books Recommender System.

Provides:
- build_knn: train NearestNeighbors over a title×user sparse matrix
- save_artifacts / load_artifacts: persist and restore training outputs
- recommend_by_title: query neighbors for a given book title
"""

import pickle
from typing import Any

import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

from books_recommender.config import ARTIFACTS_DIR, KNN_ALGO, KNN_METRIC

Artifacts = dict[str, Any]


def build_knn(
    ratings: pd.DataFrame,
) -> tuple[NearestNeighbors, csr_matrix, dict[int, str], dict[str, int]]:
    """
    Train an item-based kNN recommender.

    Args:
        ratings: DataFrame with columns ['user_id', 'title', 'rating'].

    Returns:
        (model, book_sparse, book_mapper, title_to_idx)
    """
    ratings_agg = (
        ratings.groupby(['user_id', 'title'], as_index=False)['rating']
        .mean()
    )

    ratings_agg['title_cat'] = ratings_agg['title'].astype('category')
    ratings_agg['user_cat'] = ratings_agg['user_id'].astype('category')

    book_mapper: dict[int, str] = dict(
        enumerate(ratings_agg['title_cat'].cat.categories)
    )
    user_mapper: dict[int, int] = dict(
        enumerate(ratings_agg['user_cat'].cat.categories)
    )
    title_to_idx: dict[str, int] = {
        title: idx for idx, title in book_mapper.items()
    }

    book_sparse = csr_matrix(
        (
            ratings_agg['rating'],
            (
                ratings_agg['title_cat'].cat.codes,
                ratings_agg['user_cat'].cat.codes,
            ),
        ),
        shape=(len(book_mapper), len(user_mapper)),
    )

    model = NearestNeighbors(metric=KNN_METRIC, algorithm=KNN_ALGO)
    model.fit(book_sparse)

    return model, book_sparse, book_mapper, title_to_idx


def save_artifacts(
    *,
    model: NearestNeighbors,
    book_sparse: csr_matrix,
    book_mapper: dict[int, str],
    title_to_idx: dict[str, int],
    book_meta: pd.DataFrame,
) -> None:
    """
    Save model artifacts to a single pickle.

    Args:
        model: Trained NearestNeighbors model.
        book_sparse: Item×user sparse matrix.
        book_mapper: Maps row index -> title.
        title_to_idx: Maps title -> row index.
        book_meta: Metadata indexed by title.
    """
    artifacts: Artifacts = {
        'model': model,
        'book_sparse': book_sparse,
        'book_mapper': book_mapper,
        'title_to_idx': title_to_idx,
        'book_meta': book_meta,
    }

    path = ARTIFACTS_DIR / 'recommender_system.pkl'
    with open(path, 'wb') as f:
        pickle.dump(artifacts, f)


def load_artifacts() -> Artifacts:
    """
    Load saved artifacts from disk.

    Raises:
        FileNotFoundError: If artifacts pickle does not exist.
    """
    path = ARTIFACTS_DIR / 'recommender_system.pkl'
    if not path.exists():
        raise FileNotFoundError(
            f'Artifacts not found at {path}. Run pipeline training first.'
        )

    with open(path, 'rb') as f:
        artifacts: Artifacts = pickle.load(f)

    return artifacts


def recommend_by_title(
    *,
    book_title: str,
    model: NearestNeighbors,
    book_sparse: csr_matrix,
    title_to_idx: dict[str, int],
    book_mapper: dict[int, str],
    book_meta: pd.DataFrame,
    n: int = 5,
) -> pd.DataFrame:
    """
    Recommend nearest neighbor titles for a given title.

    Args:
        book_title: Query title.
        model: Trained NearestNeighbors model.
        book_sparse: Item×user sparse matrix.
        title_to_idx: Title -> row index.
        book_mapper: Row index -> title.
        book_meta: Metadata indexed by title.
        n: Number of recommendations.

    Returns:
        DataFrame with columns: ['title', 'author', 'image_url', 'distance'].
        Empty DataFrame if title is not known.
    """
    if book_title not in title_to_idx:
        return pd.DataFrame(
            columns=['title', 'author', 'image_url', 'distance']
        )

    idx = title_to_idx[book_title]
    book_vector = book_sparse[idx, :].reshape(1, -1)

    distances, indices = model.kneighbors(book_vector, n_neighbors=n + 1)

    rows: list[dict[str, Any]] = []
    for neighbor_pos in range(1, len(indices[0])):
        neighbor_idx = int(indices[0][neighbor_pos])
        rec_title = book_mapper[neighbor_idx]

        author = 'Unknown'
        image_url = ''
        if rec_title in book_meta.index:
            author = str(book_meta.at[rec_title, 'author'])
            image_url = str(book_meta.at[rec_title, 'image_url'])

        rows.append(
            {
                'title': rec_title,
                'author': author,
                'image_url': image_url,
                'distance': float(distances[0][neighbor_pos]),
            }
        )

    return pd.DataFrame(rows)
