# Hotel Review Sentiment Classification

This project implements the CS 273 final-project design: binary sentiment
classification of TripAdvisor hotel reviews using 15 linguistic features,
Doc2Vec, Logistic Regression, and linear SVM. It evaluates exactly five
feature setups with both classifiers, producing ten held-out-test results.

## Setup

Use Python 3.9 or later. From the project folder, create and activate a virtual
environment if desired, then install the dependencies:

```powershell
python -m pip install -r requirements.txt
```

Place the Kaggle TripAdvisor Hotel Reviews CSV at:

```text
data/tripadvisor_hotel_reviews.csv
```

The loader automatically recognizes common review and rating column names,
including the usual `Review` and `Rating` columns. The input must contain the
review text and star rating; it is not included in this repository.

The project uses NLTK for sentence/word tokenization, POS tagging, and VADER.
On the first run, download the needed NLTK data automatically:

```powershell
python main.py --download-nltk
```

Alternatively, download it manually:

```powershell
python -m nltk.downloader punkt punkt_tab averaged_perceptron_tagger averaged_perceptron_tagger_eng vader_lexicon
```

## Train and evaluate

Run the complete workflow:

```powershell
python main.py
```

The workflow:

1. Removes blank/missing reviews, missing ratings, and duplicate reviews.
2. Labels 1--2 stars as negative and 4--5 stars as positive; it removes 3-star reviews.
3. Samples exactly 3,000 reviews from each class and creates stratified 60/20/20 train/validation/test splits.
4. Extracts the 10 basic and 5 additional linguistic features.
5. Trains Doc2Vec on training reviews only.
6. Tests Logistic Regression and linear SVM with all five required representations.
7. Selects each model's `C` value on validation F1, then evaluates the selected model once on the held-out test set.

Useful optional arguments:

```powershell
python main.py --data-path path\to\reviews.csv --random-state 42
python main.py --doc2vec-vector-size 100 --doc2vec-epochs 40
```

## Generated files

After a successful training run, these artifacts are created:

```text
models/doc2vec.model
models/classifiers.joblib
results/metrics.csv
results/confusion_matrices/<one CSV per model>.csv
results/error_analysis.csv
results/feature_contributions.csv
```

`metrics.csv` contains accuracy, precision, recall, F1, validation F1, and
selected hyperparameters for all ten combinations. `error_analysis.csv`
contains every incorrect held-out prediction and its original review.
`feature_contributions.csv` contains the standardized linear-model
coefficients that can support discussion of useful linguistic features.

## GUI

Train the models first, then start the demonstration interface:

```powershell
python src/gui.py
```

Enter a hotel review, choose one of the ten trained classifier/feature-set
combinations, and click **Predict sentiment**. The result is shown as
**Positive** or **Negative**. The GUI reuses exactly the same preprocessing,
linguistic feature order, and Doc2Vec inference used during evaluation.

## Project layout

```text
data/                         # Place tripadvisor_hotel_reviews.csv here
models/                       # Generated Doc2Vec and classifier artifacts
results/                      # Generated metrics and analysis files
src/data_preparation.py       # Cleaning, labeling, balancing, tokenization, split
src/feature_extraction.py     # 15 linguistic features
src/doc2vec_model.py          # Training/inference/persistence for Doc2Vec
src/train_models.py           # Five feature sets and ten model pipelines
src/evaluate_models.py        # Metrics, matrices, errors, coefficients
src/gui.py                    # Tkinter prediction interface
main.py                       # Full training and evaluation entry point
```

