Books Recommender
=================

Item-based collaborative filtering demo built with Streamlit and scikit-learn. It trains a kNN model on the Books dataset and serves interactive book-to-book recommendations.

Data
----
- Raw CSVs from the Books dataset live in `data/raw/` (`BX-Books.csv`, `BX-Book-Ratings.csv`).
- Trained artifacts are saved to `data/artifacts/recommender_system.joblib` (created by the pipeline below).

Setup (uv)
----------
Install runtime dependencies only:

```bash
uv sync
```

With test tooling (`pytest`):

```bash
uv sync --extra dev
```

Run the Streamlit app
---------------------
The UI will train the model on first run if `data/artifacts/recommender_system.joblib`
is missing. Use **Title contains** in the sidebar to filter the book list; pick a book and
click **Show Recommendation** for nearest-neighbor titles.

```bash
uv run streamlit run app.py
```

Tests
-----
From the project root (after `uv sync --extra dev`):

```bash
uv run pytest
```

Project layout
--------------
```
.
├── .gitignore
├── app.py
├── uv.lock
├── pyproject.toml
├── README.md
├── data/
│   ├── artifacts/
│   │   └── recommender_system.joblib
│   └── raw/
│       ├── BX-Book-Ratings.csv
│       └── BX-Books.csv
├── tests/
│   ├── test_evaluation.py
│   ├── test_knn.py
│   ├── test_preprocess.py
│   └── test_title_filter.py
└── src/
    └── books_recommender/
        ├── config.py
        ├── evaluation.py
        ├── pipeline.py
        ├── title_filter.py
        ├── data/
        │   ├── load.py
        │   └── preprocess.py
        └── models/
            └── knn.py
```

Offline evaluation (optional)
-----------------------------
Leave-one-out metrics on the preprocessed dataset (requires raw CSVs under `data/raw/`):

```bash
uv run python -m books_recommender.evaluation --k 10 --max-users 500
```

Use `--help` for more options.

Run with Docker on EC2
---------------------

Create an EC2 instance (Ubuntu) and add a **Custom TCP inbound rule on port 8501** in the Security Group.

In the instance terminal:

```bash
sudo apt-get update -y

sudo apt-get upgrade -y

curl -fsSL https://get.docker.com -o get-docker.sh

sudo sh get-docker.sh

sudo usermod -aG docker $USER

newgrp docker

git clone https://github.com/Kostia9/books_recommender

cd books_recommender

docker build -t books-app .

docker run -d -p 8501:8501 --name recommender-container books-app
```

Open in browser:
```
http://<public-ip>:8501
```
