# IMDb Movie Review Sentiment Analysis

A Streamlit web app that predicts whether a movie review is **Positive** or **Negative** using multiple Machine Learning models trained on the IMDb dataset.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue).
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)

## Live Project
- **Streamlit App:** [moviesreviewmodel.streamlit.app](https://moviesreviewmodel.streamlit.app/)

## Connect With Me
- **LinkedIn:** [Muhammad Hassaan](https://www.linkedin.com/in/muhammadhassaan026)

## Project Overview
This project includes:
- A trained TF-IDF text vectorizer
- Four ML sentiment classifiers
- A Streamlit UI to test predictions in real time
- A notebook (`IMDb_reviews.ipynb`) showing EDA, training, and evaluation

The app lets users select a model and enter any movie review text to get instant sentiment output.

## Features
- Clean Streamlit interface
- Model switcher in UI:
  - Logistic Regression
  - Naive Bayes
  - Decision Tree
  - Random Forest
- Real-time sentiment prediction
- Reusable saved `.pkl` model artifacts

## Dataset
- **File:** `IMDB Dataset.csv`
- **Size:** 50,000 reviews
- **Classes:** `positive` / `negative` (balanced: 25,000 each)
- **Train/Test split:** 80/20 (`random_state=42`)

## Text Preprocessing
- TF-IDF Vectorization with:
  - `stop_words='english'`
  - `max_features=5000`

Saved vectorizer:
- `tfidf_vectorizer.pkl`

## Models Trained
The following models were trained in `IMDb_reviews.ipynb` and exported with `joblib`:
- `logistic_regression_imdb_model.pkl`
- `naive_bayes_imdb_model.pkl`
- `decision_tree_imdb_model.pkl`
- `random_forest_imdb_model.pkl`

## Model Performance (from notebook output)
| Model | Accuracy |
|------|----------|
| Logistic Regression | **0.8889** |
| Naive Bayes | **0.8508** |
| Decision Tree | **0.7250** |
| Random Forest | **0.7250** |

## Project Structure
```text
.
|-- app.py
|-- requirements.txt
|-- IMDB Dataset.csv
|-- IMDb_reviews.ipynb
|-- tfidf_vectorizer.pkl
|-- logistic_regression_imdb_model.pkl
|-- naive_bayes_imdb_model.pkl
|-- decision_tree_imdb_model.pkl
|-- random_forest_imdb_model.pkl
`-- ScreenShorts/
```

## How `app.py` Works
`app.py` does the following:
1. Loads vectorizer and all trained models using `joblib`.
2. Builds a Streamlit page for user input.
3. Lets user choose one model from dropdown.
4. Converts review text into TF-IDF vector.
5. Runs selected model prediction.
6. Displays sentiment as Positive or Negative.

## Installation
```bash
pip install -r requirements.txt
```

## Run Locally
```bash
streamlit run app.py
```

If you want to run on a custom port:
```bash
streamlit run app.py --server.port 8502
```

## Screenshots
Add your UI screenshots here:
- `ScreenShorts/Screenshot 2026-02-13 015803.png`
- `ScreenShorts/Screenshot 2026-02-13 015841.png`

## Notes
- In the notebook, the Random Forest export line currently saves `dt_model` to `random_forest_imdb_model.pkl`. Update that line to `joblib.dump(rf_model, ...)` when retraining to avoid artifact mismatch.
- In the current repository, `decision_tree_imdb_model.pkl` and `random_forest_imdb_model.pkl` have the same file size, which may indicate that export issue.

## Future Improvements
- Add text cleaning pipeline (HTML removal, stemming/lemmatization)
- Hyperparameter tuning for tree-based models
- Add probability/confidence score in UI
- Deploy on Streamlit Community Cloud

---

## GitHub Short Description (Suggested)
**Streamlit-based IMDb movie review sentiment analysis app using TF-IDF + Logistic Regression, Naive Bayes, Decision Tree, and Random Forest models.**


