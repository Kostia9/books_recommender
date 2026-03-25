"""
Streamlit UI for the Books Recommender System.

The app loads pre-trained artifacts (kNN model + sparse matrix + mappings)
and provides title search + item-based recommendations.
"""

from typing import Any

import streamlit as st
from loguru import logger

from books_recommender.config import ARTIFACTS_DIR
from books_recommender.models.knn import load_artifacts, recommend_by_title
from books_recommender.pipeline import run_training_pipeline
from books_recommender.title_filter import filter_titles_by_substring

ARTIFACTS_PATH = ARTIFACTS_DIR / "recommender_system.joblib"
NO_TITLE_MATCH_PLACEHOLDER = "— No matching titles — change «Title contains» above"


@st.cache_resource
def load_system() -> dict[str, Any]:
    """Load trained recommender artifacts from disk (cached by Streamlit)."""
    return load_artifacts()


def ensure_artifacts() -> None:
    """Train the model if the artifacts file is missing."""
    if ARTIFACTS_PATH.exists():
        return

    with st.spinner("Training model (first run)..."):
        run_training_pipeline()


def main() -> None:
    """Run the Streamlit application."""
    st.set_page_config(page_title="Books Recommender", layout="wide")
    st.title("End to End Books Recommender System")
    st.caption("Collaborative filtering: item-based kNN over Title x Users matrix.")

    try:
        ensure_artifacts()
        artifacts = load_system()
    except FileNotFoundError as exc:
        logger.exception("Artifact file not found")
        st.error(f"Artifact file missing: {exc}")
        return
    except ValueError as exc:
        logger.exception("Invalid artifact data")
        st.error(f"Invalid artifact data: {exc}")
        return
    except Exception as exc:
        logger.exception("Unexpected error loading artifacts")
        st.error(f"Failed to prepare model artifacts: {exc}")
        return

    model = artifacts["model"]
    book_sparse = artifacts["book_sparse"]
    title_to_idx = artifacts["title_to_idx"]
    book_mapper = artifacts["book_mapper"]
    book_meta = artifacts["book_meta"]

    all_titles = list(title_to_idx.keys())

    st.sidebar.header("Search")
    query = st.sidebar.text_input("Title contains", value="Harry Potter")

    filtered_titles = filter_titles_by_substring(all_titles, query)
    if not filtered_titles:
        st.warning("No titles match your search. Adjust «Title contains».")
        options = [NO_TITLE_MATCH_PLACEHOLDER]
    else:
        options = []
        for title in filtered_titles:
            author = "Unknown"
            if title in book_meta.index:
                # book_meta is indexed by title; value is a Series
                author = str(book_meta.at[title, "author"])
            options.append(f"{title} — {author}")

    choice = st.sidebar.selectbox(
        "Type or select a book from the dropdown",
        options=options,
        disabled=not filtered_titles,
    )

    separator = " — "
    selected_title = choice.split(separator, maxsplit=1)[0] if filtered_titles else None

    k = st.sidebar.slider(
        "Number of recommendations",
        min_value=3,
        max_value=20,
        value=5,
    )

    if st.button("Show Recommendation"):
        if not filtered_titles or selected_title is None:
            st.error("No book selected. Search until the dropdown lists titles.")
            return
        if selected_title not in title_to_idx:
            st.error("Selected title is not in the model. Pick another book.")
            return
        with st.spinner("Computing neighbors..."):
            recs = recommend_by_title(
                book_title=selected_title,
                model=model,
                book_sparse=book_sparse,
                title_to_idx=title_to_idx,
                book_mapper=book_mapper,
                book_meta=book_meta,
                n=k,
            )

        if recs.empty:
            st.error("Book not found in pivot table.")
            return

        st.subheader(f"Recommendations for: {selected_title}")

        n_cols = min(5, len(recs))
        cols = st.columns(n_cols)

        for i, (_, row) in enumerate(recs.iterrows()):
            col = cols[i % n_cols]
            with col:
                display_title = str(row["title"])
                if len(display_title) > 50:
                    display_title = display_title[:47] + "..."

                st.write(f"**{display_title}**")

                image_url = str(row.get("image_url", "") or "")
                if image_url:
                    st.image(image_url, width="stretch")

                st.caption(str(row.get("author", "Unknown")))
                st.write(f"*Dist: {float(row['distance']):.3f}*")


if __name__ == "__main__":
    main()
