# TripAdvisor Hotel Review Sentiment

This project implements the CS 273 final-project proposal: a reproducible binary sentiment-classification comparison for TripAdvisor hotel reviews. It evaluates five feature representations (`basic10`, `extra5`, `all15`, `doc2vec`, and `all15_doc2vec`) with Logistic Regression and a Linear SVM.

## Requirements

- Python 3.10 or newer
- The Kaggle TripAdvisor Hotel Reviews CSV, with columns `Review` and `Rating`

Create an environment and install the pinned dependencies:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Place the downloaded CSV at:

```text
data/raw/tripadvisor_hotel_reviews.csv
```

The first run may need the NLTK resources used by the feature extractor:

```bash
python -c "import nltk; [nltk.download(x) for x in ['punkt', 'punkt_tab', 'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng', 'wordnet', 'omw-1.4', 'vader_lexicon']]"
```

The implementation includes conservative fallbacks for tokenization, tagging, and VADER when an optional NLTK resource is unavailable, but downloading the resources is recommended for the intended experiment.

## Run the project

From the repository root:

```bash
python -m src.data
python -m src.run_experiments
pytest
python app.py
```

`python -m src.data` cleans the raw file, removes 3-star reviews, creates the reproducible balanced 6,000-review sample, and writes stratified train/validation/test CSV files under `data/processed/`.

`python -m src.run_experiments` trains Doc2Vec only on the model-selection training reviews, evaluates all validation candidates, selects `C` using validation F1, refits the ten final models on train plus validation, and evaluates the held-out test set once.

## Generated outputs

- `artifacts/features/` — NumPy feature matrices
- `artifacts/doc2vec/doc2vec.model` — final Doc2Vec model trained on train plus validation
- `artifacts/models/` — ten loadable scikit-learn pipelines
- `artifacts/results/validation_results.csv` — all 30 validation candidates
- `artifacts/results/test_results.csv` — one row for each of the ten final models
- `artifacts/results/errors.csv` — incorrectly classified test reviews
- `artifacts/results/linguistic_coefficients.csv` — coefficients for named linguistic dimensions
- `artifacts/results/confusion_matrices/` — one PNG per final model

The GUI loads the saved artifacts and routes every review through the same linguistic-feature and Doc2Vec code used during training.
