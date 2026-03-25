"""Tests for UI title substring search."""

from books_recommender.title_filter import filter_titles_by_substring


def test_empty_query_returns_prefix_of_titles() -> None:
    titles = [f"Book {i}" for i in range(150)]
    out = filter_titles_by_substring(titles, "   ")
    assert len(out) == 100
    assert out[0] == "Book 0"


def test_substring_match_is_case_insensitive() -> None:
    titles = ["Harry Potter", "other"]
    assert filter_titles_by_substring(titles, "HARRY") == ["Harry Potter"]
    assert filter_titles_by_substring(titles, "potter") == ["Harry Potter"]


def test_no_match_returns_empty_list() -> None:
    assert filter_titles_by_substring(["Alpha", "Beta"], "zzz") == []


def test_respects_max_limit() -> None:
    titles = [f"T{i}" for i in range(3000)]
    out = filter_titles_by_substring(titles, "T", max_limit=50)
    assert len(out) == 50
