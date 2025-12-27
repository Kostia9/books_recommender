"""
Project-wide configuration constants and paths.
"""

from pathlib import Path
from typing import Final

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]

DATA_DIR: Final[Path] = PROJECT_ROOT / 'data'
RAW_DIR: Final[Path] = DATA_DIR / 'raw'
ARTIFACTS_DIR: Final[Path] = DATA_DIR / 'artifacts'

ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

MIN_USER_RATINGS: Final[int] = 25
MIN_BOOK_RATINGS: Final[int] = 15

KNN_METRIC: Final[str] = 'cosine'
KNN_ALGO: Final[str] = 'brute'
