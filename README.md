Books Recommender
=================

Item-based collaborative filtering demo built with Streamlit and scikit-learn. It trains a kNN model on the Books dataset and serves interactive book-to-book recommendations.

Data
----
- Raw CSVs from the Books dataset live in `data/raw/` (`BX-Users.csv`, `BX-Books.csv`, `BX-Book-Ratings.csv`).
- Trained artifacts are saved to `data/artifacts/recommender_system.pkl` (created by the pipeline below).

Setup (Poetry)
--------------
```bash
poetry install
```

Run the Streamlit app
---------------------
The UI will train the model on first run if `data/artifacts/recommender_system.pkl`
is missing. Launch the app and search for a title to get nearest-neighbor recommendations.
```bash
poetry run streamlit run app.py
```

Project layout
--------------
```
.
├── .gitignore
├── app.py
├── poetry.lock
├── pyproject.toml
├── README.md
├── data/
│   ├── artifacts/
│   │   └── recommender_system.pkl
│   └── raw/
│       ├── BX-Book-Ratings.csv
│       ├── BX-Books.csv
│       └── BX-Users.csv
└── src/
    └── books_recommender/
        ├── __init__.py
        ├── config.py
        ├── pipeline.py
        ├── data/
        │   ├── load.py
        │   └── preprocess.py
        └── models/
            └── knn.py
```

Run with Docker on EC2
---------------------

Create an EC2 instance (Ubuntu) and add a **Custom TCP inbound rule on port 8501** in the Security Group.

In the instance terminal:

```bash
sudo apt-get update -y

sudo apt-get upgrade -y

curl -fsSL https://get.docker.com -o get-docker.sh

sudo sh get-docker.sh

sudo usermod -aG docker ubuntu

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