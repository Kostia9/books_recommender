"""Pure helpers for matching book titles in the UI search."""

from collections.abc import Sequence


def filter_titles_by_substring(
    all_titles: Sequence[str],
    query: str,
    *,
    default_limit: int = 100,
    max_limit: int = 2000,
) -> list[str]:
    """
    Return titles whose full string contains ``query`` (case-insensitive).

    If ``query`` is empty or whitespace-only, returns the first ``default_limit``
    titles (same as showing an unfiltered short list in the UI).
    """
    q = query.strip()
    if not q:
        return list(all_titles[:default_limit])
    term = q.lower()
    matched = [t for t in all_titles if term in t.lower()]
    return matched[:max_limit]
