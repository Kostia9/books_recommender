"""
Training pipeline for the Books Recommender System.

Loads raw CSVs, preprocesses ratings, trains an item-based kNN model,
and saves artifacts to data/artifacts/recommender_system.pkl.
"""

import logging

from books_recommender.data.load import load_books, load_ratings, load_users
from books_recommender.data.preprocess import preprocess
from books_recommender.models.knn import build_knn, save_artifacts

logger = logging.getLogger(__name__)


def run_training_pipeline() -> None:
    """Run the full training pipeline and persist artifacts to disk."""
    logger.info('Loading data...')
    users = load_users()
    books = load_books()
    ratings = load_ratings()

    logger.info('Preprocessing...')
    ratings_clean, book_meta = preprocess(users, books, ratings)

    logger.info('Building model...')
    model, book_sparse, book_mapper, title_to_idx = build_knn(ratings_clean)

    logger.info('Saving artifacts...')
    save_artifacts(
        model=model,
        book_sparse=book_sparse,
        book_mapper=book_mapper,
        title_to_idx=title_to_idx,
        book_meta=book_meta,
    )

    logger.info('Done.')


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)s] %(message)s'
    )
    run_training_pipeline()
