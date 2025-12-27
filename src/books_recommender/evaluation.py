"""
Offline evaluation utilities for the Books Recommender System.

Implements leave-one-out cross-validation (per user) for item-based kNN.
"""

from __future__ import annotations

import argparse
import logging
from typing import Iterable

import numpy as np
import pandas as pd

from books_recommender.data.load import load_books, load_ratings, load_users
from books_recommender.data.preprocess import preprocess
from books_recommender.models.knn import build_knn

logger = logging.getLogger(__name__)


def leave_one_out_split(
    ratings: pd.DataFrame,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split ratings into train/test by holding out one item per user.

    Args:
        ratings: DataFrame with columns ['user_id', 'title', 'rating'].
        rng: NumPy random generator.

    Returns:
        (train, test) DataFrames.
    """
    holdout_indices: list[int] = []
    for _, group in ratings.groupby('user_id'):
        if len(group) < 2:
            continue
        holdout_idx = int(rng.choice(group.index.values))
        holdout_indices.append(holdout_idx)

    test = ratings.loc[holdout_indices].copy()
    train = ratings.drop(index=holdout_indices).copy()
    return train, test


def recommend_from_history(
    *,
    history_titles: Iterable[str],
    model,
    book_sparse,
    title_to_idx: dict[str, int],
    book_mapper: dict[int, str],
    k: int,
    neighbors_per_item: int,
) -> list[str]:
    """
    Recommend items by aggregating kNN neighbors from a user's history.

    Args:
        history_titles: Titles the user interacted with in training.
        model: Trained NearestNeighbors model.
        book_sparse: Item×user sparse matrix.
        title_to_idx: Title -> row index.
        book_mapper: Row index -> title.
        k: Number of recommendations to return.
        neighbors_per_item: Neighbors to retrieve per seed item.
    """
    history_set = {title for title in history_titles if title in title_to_idx}
    if not history_set:
        return []

    n_items = book_sparse.shape[0]
    n_neighbors = min(neighbors_per_item, n_items)
    scores: dict[str, float] = {}

    for title in history_set:
        idx = title_to_idx[title]
        distances, indices = model.kneighbors(
            book_sparse[idx].reshape(1, -1),
            n_neighbors=n_neighbors,
        )
        for dist, neighbor_idx in zip(distances[0], indices[0], strict=False):
            neighbor_title = book_mapper[int(neighbor_idx)]
            if neighbor_title in history_set:
                continue
            score = 1.0 - float(dist)
            scores[neighbor_title] = scores.get(neighbor_title, 0.0) + score

    if not scores:
        return []

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [title for title, _ in ranked[:k]]


def evaluate_leave_one_out(
    ratings: pd.DataFrame,
    *,
    k: int = 10,
    neighbors_per_item: int = 50,
    random_state: int = 42,
    max_users: int | None = None,
) -> dict[str, float]:
    """
    Evaluate the model using leave-one-out cross-validation.

    Args:
        ratings: Clean ratings DataFrame.
        k: Recommendation list size.
        neighbors_per_item: Neighbors retrieved per seed item.
        random_state: RNG seed for deterministic splits.
        max_users: Optional cap on number of users evaluated.

    Returns:
        Dict with evaluation metrics.
    """
    rng = np.random.default_rng(random_state)
    train, test = leave_one_out_split(ratings, rng)

    model, book_sparse, book_mapper, title_to_idx = build_knn(train)
    history_by_user = train.groupby('user_id')['title'].apply(list)

    test_users = test['user_id'].unique()
    if max_users is not None and max_users < len(test_users):
        sampled_users = set(
            rng.choice(test_users, size=max_users, replace=False).tolist()
        )
    else:
        sampled_users = None

    users_evaluated = 0
    hits = 0
    mrr_total = 0.0
    skipped_missing_items = 0
    skipped_empty_history = 0

    for row in test.itertuples(index=False):
        user_id = row.user_id
        true_title = row.title

        if sampled_users is not None and user_id not in sampled_users:
            continue

        history_titles = history_by_user.get(user_id, [])
        if not history_titles:
            skipped_empty_history += 1
            continue

        if true_title not in title_to_idx:
            skipped_missing_items += 1
            continue

        recs = recommend_from_history(
            history_titles=history_titles,
            model=model,
            book_sparse=book_sparse,
            title_to_idx=title_to_idx,
            book_mapper=book_mapper,
            k=k,
            neighbors_per_item=neighbors_per_item,
        )

        users_evaluated += 1
        if true_title in recs:
            hits += 1
            rank = recs.index(true_title) + 1
            mrr_total += 1.0 / rank
        if users_evaluated % 100 == 0:
            logger.info("Evaluated %d users...", users_evaluated)

    hit_rate = hits / users_evaluated if users_evaluated else 0.0
    mrr = mrr_total / users_evaluated if users_evaluated else 0.0

    return {
        'users_evaluated': users_evaluated,
        'hit_rate': hit_rate,
        'mrr': mrr,
        'k': k,
        'neighbors_per_item': neighbors_per_item,
        'skipped_empty_history': skipped_empty_history,
        'skipped_missing_items': skipped_missing_items,
    }


def main() -> None:
    """Run leave-one-out evaluation and print summary metrics."""
    parser = argparse.ArgumentParser(
        description='Evaluate item-based kNN with leave-one-out CV.'
    )
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--neighbors-per-item', type=int, default=50)
    parser.add_argument('--random-state', type=int, default=42)
    parser.add_argument('--max-users', type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)s] %(message)s',
    )

    logger.info('Loading data...')
    users = load_users()
    books = load_books()
    ratings = load_ratings()

    logger.info('Preprocessing...')
    ratings_clean, _ = preprocess(users, books, ratings)

    logger.info('Evaluating...')
    results = evaluate_leave_one_out(
        ratings_clean,
        k=args.k,
        neighbors_per_item=args.neighbors_per_item,
        random_state=args.random_state,
        max_users=args.max_users,
    )

    logger.info(
        'Users evaluated: %d',
        int(results['users_evaluated']),
    )
    logger.info('Hit Rate@%d: %.4f', args.k, results['hit_rate'])
    logger.info('MRR@%d: %.4f', args.k, results['mrr'])
    logger.info(
        'Skipped (no history): %d',
        int(results['skipped_empty_history']),
    )
    logger.info(
        'Skipped (missing item in train): %d',
        int(results['skipped_missing_items']),
    )


if __name__ == '__main__':
    main()
