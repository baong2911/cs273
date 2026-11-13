# CS 273 Final Project — Implementation Design

## 1. Project Goal

Build a reproducible binary sentiment-classification pipeline for TripAdvisor hotel reviews that compares:

1. **10 basic linguistic features**
2. **5 additional linguistic features**
3. **All 15 linguistic features**
4. **Doc2Vec embeddings**
5. **All 15 linguistic features + Doc2Vec**

Each representation will be evaluated with:

- Logistic Regression
- Linear SVM

This produces **10 model/feature-set combinations**. Model selection will use the validation set; the test set will be used only for final evaluation.

The final project will also include a small Python GUI where a user can enter a hotel review, choose a trained model, and receive a positive/negative prediction.

---

## 2. Fixed Experimental Specification

### Dataset

Use the Kaggle **TripAdvisor Hotel Reviews** dataset.

Expected raw columns:

- `Review` — review text
- `Rating` — integer rating from 1–5

### Labeling

Convert ratings to binary sentiment labels:

- `1` or `2` → `negative` / `0`
- `3` → exclude
- `4` or `5` → `positive` / `1`

### Cleaning

Before sampling:

1. Drop rows with missing review text or missing rating.
2. Strip surrounding whitespace.
3. Drop reviews that become empty after stripping.
4. Remove duplicate reviews using normalized review text.
5. Remove 3-star reviews.
6. Validate that remaining ratings are in `{1, 2, 4, 5}`.

Do **not** aggressively clean punctuation, capitalization, negations, or misspellings because several linguistic features depend on them.

### Sample

Create a reproducible balanced working dataset of exactly **6,000 reviews**:

- 3,000 negative
- 3,000 positive

Use a single project-wide random seed.

### Split

Split the 6,000 reviews using stratification:

- 60% train = 3,600
- 20% validation = 1,200
- 20% test = 1,200

Because the source sample is balanced, each split should remain approximately 50/50 positive and negative.

**Important:** split before training Doc2Vec. The Doc2Vec model must be fit only on training reviews. Validation/test vectors must be inferred using the trained Doc2Vec model.

---

## 3. Feature Definitions

All feature definitions must live in one implementation module so that training and GUI inference use exactly the same logic.

### Basic 10 linguistic features

| Feature | Definition |
|---|---|
| `avg_word_length` | Mean number of alphabetic characters per word token. |
| `avg_sentence_length` | Mean number of word tokens per sentence. |
| `word_count` | Number of word tokens. |
| `verb_count` | Count of tokens POS-tagged as verbs. |
| `adjective_count` | Count of tokens POS-tagged as adjectives. |
| `passive_voice_count` | Number of sentence-level passive-voice patterns detected using a documented heuristic based on forms of *be* followed by a past participle (`VBN`). |
| `sentence_count` | Number of sentence tokens. |
| `content_word_diversity` | Number of unique lowercase content-word lemmas divided by total content-word tokens. Content words are nouns, verbs, adjectives, and adverbs. Return `0` if there are no content words. |
| `typo_count` | Count of alphabetic tokens not recognized by the selected spell-check dictionary after lowercasing. |
| `typo_ratio` | `typo_count / word_count`; return `0` if `word_count == 0`. |

### Additional 5 linguistic features

| Feature | Definition |
|---|---|
| `negation_count` | Count of explicit negation tokens such as `no`, `not`, `never`, `none`, `neither`, `nor`, plus contracted `n't`. |
| `first_person_pronoun_ratio` | Count of first-person pronouns (`I`, `me`, `my`, `mine`, `myself`, `we`, `us`, `our`, `ours`, `ourselves`) divided by word count. |
| `adverb_count` | Count of tokens POS-tagged as adverbs. |
| `exclamation_count` | Number of `!` characters in the original review text. |
| `sentiment_lexicon_score` | NLTK VADER compound sentiment score in `[-1, 1]`. |

### Doc2Vec features

Train one Gensim Doc2Vec model on the **training reviews only**.

Initial fixed configuration:

- `vector_size = 100`
- `window = 5`
- `min_count = 2`
- `workers = 1` for reproducibility
- `epochs = 40`
- project random seed

Training reviews receive vectors from the trained document model. Validation/test/user-entered reviews receive vectors using `infer_vector`.

These parameters may be changed during validation experiments, but every tested configuration must be recorded.

---

## 4. Feature Sets

Use exactly these five named feature sets throughout the codebase:

- `basic10` — first 10 linguistic features
- `extra5` — additional 5 linguistic features
- `all15` — all linguistic features
- `doc2vec` — Doc2Vec vector only
- `all15_doc2vec` — concatenation of all 15 linguistic features and Doc2Vec vector

Feature-set names should be stable because they will be used in output filenames, metric tables, model selection, and the GUI.

---

## 5. Models and Preprocessing

### Logistic Regression

Use scikit-learn `LogisticRegression`.

Pipeline:

1. Feature matrix
2. `StandardScaler`
3. Logistic Regression

### Linear SVM

Use scikit-learn `LinearSVC`.

Pipeline:

1. Feature matrix
2. `StandardScaler`
3. Linear SVM

For sparse/non-sparse compatibility, keep all experiment feature matrices as dense NumPy arrays.

### Validation

Use the training set to fit candidate configurations and the validation set to choose model hyperparameters.

Recommended small search:

**Logistic Regression**
- `C ∈ {0.1, 1.0, 10.0}`

**Linear SVM**
- `C ∈ {0.1, 1.0, 10.0}`

Primary validation selection metric: **F1 score**.

Tie-breaker:
1. higher accuracy
2. simpler/default `C` closest to `1.0`

After choosing settings, refit the selected model on **train + validation** and evaluate it once on the held-out test set.

Do not use test metrics to choose hyperparameters.

---

## 6. Evaluation Outputs

For every final feature-set/classifier pair, compute:

- Accuracy
- Precision
- Recall
- F1 score
- Confusion matrix

Produce one summary table with one row per final experiment.

Also produce:

- confusion-matrix image for each final model
- validation results table containing all tested hyperparameter settings
- final test results table
- error-analysis CSV containing incorrectly classified test reviews
- linguistic feature coefficient table for interpretable linear models

For the combined Doc2Vec + linguistic model, coefficient interpretation should focus only on the 15 named linguistic dimensions; individual Doc2Vec dimensions should not be assigned semantic meaning.

---

## 7. Repository Structure

```text
hotel-sentiment-project/
├── proposal.md
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/
│   │   └── tripadvisor_hotel_reviews.csv
│   └── processed/
│       ├── balanced_6000.csv
│       ├── train.csv
│       ├── validation.csv
│       └── test.csv
├── artifacts/
│   ├── doc2vec/
│   │   └── doc2vec.model
│   ├── features/
│   │   ├── train_basic10.npy
│   │   ├── validation_basic10.npy
│   │   ├── test_basic10.npy
│   │   ├── train_extra5.npy
│   │   ├── validation_extra5.npy
│   │   ├── test_extra5.npy
│   │   ├── train_all15.npy
│   │   ├── validation_all15.npy
│   │   ├── test_all15.npy
│   │   ├── train_doc2vec.npy
│   │   ├── validation_doc2vec.npy
│   │   ├── test_doc2vec.npy
│   │   ├── train_all15_doc2vec.npy
│   │   ├── validation_all15_doc2vec.npy
│   │   └── test_all15_doc2vec.npy
│   ├── models/
│   │   ├── logistic_basic10.joblib
│   │   ├── logistic_extra5.joblib
│   │   ├── logistic_all15.joblib
│   │   ├── logistic_doc2vec.joblib
│   │   ├── logistic_all15_doc2vec.joblib
│   │   ├── svm_basic10.joblib
│   │   ├── svm_extra5.joblib
│   │   ├── svm_all15.joblib
│   │   ├── svm_doc2vec.joblib
│   │   └── svm_all15_doc2vec.joblib
│   └── results/
│       ├── validation_results.csv
│       ├── test_results.csv
│       ├── errors.csv
│       ├── linguistic_coefficients.csv
│       └── confusion_matrices/
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── text_processing.py
│   ├── linguistic_features.py
│   ├── doc2vec_features.py
│   ├── feature_builder.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── run_experiments.py
├── app.py
└── tests/
    ├── test_data.py
    ├── test_linguistic_features.py
    ├── test_feature_builder.py
    └── test_predict.py
```

Generated `data/processed/` and `artifacts/` files may be excluded from Git if desired, but the directory layout and generation commands must remain documented.

---

## 8. File-by-File Responsibilities

### Root files

#### `proposal.md`

This document. It is the implementation contract for the project.

#### `README.md`

Short user-facing instructions only:

- project purpose
- Python version
- environment setup
- where to place the Kaggle CSV
- required NLTK resource downloads
- commands to prepare data, run experiments, run tests, and launch the GUI
- brief explanation of generated outputs

Do not duplicate the full design specification from `proposal.md`.

#### `requirements.txt`

Pin the libraries needed to reproduce the project. Expected packages:

- `pandas`
- `numpy`
- `scikit-learn`
- `nltk`
- `gensim`
- `joblib`
- `matplotlib`
- `pyspellchecker`

Tkinter is normally supplied with Python and does not need a pip dependency.

#### `.gitignore`

Ignore:

- virtual environments
- Python cache files
- IDE files
- raw dataset if licensing/size makes committing it undesirable
- generated processed data
- trained model artifacts
- generated plots/results, if they are intended to be reproducible rather than versioned

---

### Data layer

#### `src/config.py`

Single source of truth for project constants.

Define:

- random seed
- repository-relative paths
- raw column names
- label mapping
- sample size per class
- train/validation/test proportions
- all 15 linguistic feature names
- feature-set names
- Doc2Vec defaults
- classifier hyperparameter grids

No data processing logic should live here.

#### `src/data.py`

Own the complete dataset preparation workflow.

Functions should include approximately:

```python
load_raw_data(path) -> DataFrame
clean_and_label(df) -> DataFrame
create_balanced_sample(df, per_class, seed) -> DataFrame
split_dataset(df, seed) -> tuple[DataFrame, DataFrame, DataFrame]
save_splits(...)
```

Required behavior:

- validate required columns
- clean missing/blank/duplicate rows
- remove 3-star reviews
- create numeric binary labels
- sample exactly 3,000 examples per class
- perform stratified 60/20/20 split
- write `balanced_6000.csv`, `train.csv`, `validation.csv`, and `test.csv`

The saved split CSVs should contain only the information needed later, preferably:

- `review_id`
- `review`
- `rating`
- `label`

`review_id` should be stable within the processed dataset so prediction errors can be traced back to examples.

---

### Text and linguistic features

#### `src/text_processing.py`

Centralize reusable NLP preprocessing.

Responsibilities:

- sentence tokenization
- word tokenization
- POS tagging
- optional lemmatization
- helper for identifying word tokens
- helper for mapping Penn Treebank POS tags to content-word categories

This prevents slightly different preprocessing logic from appearing in multiple files.

Do not remove punctuation or normalize the original review before features that depend on punctuation/case are calculated.

#### `src/linguistic_features.py`

Implement all 15 linguistic features.

Primary API:

```python
extract_linguistic_features(text: str) -> dict[str, float]
extract_linguistic_matrix(texts: Sequence[str]) -> DataFrame
```

Requirements:

- output columns must always appear in the order defined in `config.py`
- handle empty/very short text safely
- avoid `NaN` or infinite values
- keep feature definitions deterministic
- include comments for the passive-voice and typo heuristics because these are the least self-explanatory features

The GUI must call this same module rather than reimplementing feature extraction.

---

### Doc2Vec

#### `src/doc2vec_features.py`

Own Doc2Vec training, saving, loading, and inference.

Functions should include approximately:

```python
train_doc2vec(train_texts, config) -> Doc2Vec
infer_vectors(model, texts) -> np.ndarray
save_doc2vec(model, path) -> None
load_doc2vec(path) -> Doc2Vec
```

Rules:

- train only on training reviews during model-selection experiments
- use tokenized review text
- seed both training and inference where supported
- validation/test vectors are inferred, never used to update the Doc2Vec model
- final production Doc2Vec model may be retrained on train + validation only after hyperparameters are fixed

---

### Feature assembly

#### `src/feature_builder.py`

Build the five canonical feature matrices.

Primary API:

```python
build_feature_sets(
    texts,
    linguistic_features,
    doc2vec_vectors,
) -> dict[str, np.ndarray]
```

Output keys must be:

- `basic10`
- `extra5`
- `all15`
- `doc2vec`
- `all15_doc2vec`

Also handle saving/loading generated matrices under `artifacts/features/`.

This module is responsible for column slicing and concatenation; training code should not manually reconstruct feature sets.

---

### Training and evaluation

#### `src/train.py`

Contain model-construction and fitting utilities.

Functions should include approximately:

```python
make_pipeline(classifier_name, C)
fit_model(X, y, classifier_name, C)
save_model(model, path)
load_model(path)
```

Responsibilities:

- build the scaler + classifier pipeline
- avoid duplicated classifier setup
- save final fitted scikit-learn pipelines with `joblib`

Do not perform dataset loading or feature extraction here.

#### `src/evaluate.py`

Contain metric and analysis utilities.

Functions should include approximately:

```python
compute_metrics(y_true, y_pred) -> dict
save_confusion_matrix(...)
build_error_table(...)
extract_linguistic_coefficients(...)
```

Metric behavior:

- use positive sentiment (`1`) as the positive class
- use consistent zero-division behavior
- produce machine-readable result rows and human-readable confusion-matrix plots

#### `src/run_experiments.py`

Main orchestration script for the complete experiment.

It should:

1. Load prepared train/validation/test splits.
2. Extract linguistic features.
3. Train Doc2Vec on train reviews.
4. Infer train/validation/test Doc2Vec vectors.
5. Build all five feature sets.
6. For each of 5 feature sets:
   - train Logistic Regression candidates
   - train Linear SVM candidates
   - score candidates on validation
   - select the best `C` for each classifier
7. Save `validation_results.csv`.
8. Freeze all selected hyperparameters.
9. Rebuild final representations using train + validation where appropriate.
10. Fit each of the 10 final models on train + validation.
11. Evaluate each model exactly once on test.
12. Save all trained models.
13. Save `test_results.csv`.
14. Save confusion matrices.
15. Save misclassified-review rows.
16. Save interpretable linguistic coefficients.

The script should print a compact ranked summary at the end, sorted by test F1.

This file is the only file that should coordinate the full experiment. Lower-level modules should remain reusable.

---

### Prediction and GUI

#### `src/predict.py`

Provide the inference API used by the GUI.

Responsibilities:

- load a selected trained classifier
- load Doc2Vec when the selected model requires it
- determine which feature representation the selected model expects
- extract the review's linguistic features
- infer its Doc2Vec vector if needed
- assemble a single-row feature matrix
- return a normalized prediction result

Suggested return structure:

```python
{
    "label": 1,
    "sentiment": "positive"
}
```

If a score is shown:
- Logistic Regression may use `predict_proba`
- Linear SVM may use `decision_function`

Do **not** label an SVM decision-function value as a probability.

#### `app.py`

Implement a minimal Tkinter GUI.

Required UI:

- multiline text box for review input
- model dropdown containing the 10 trained model choices
- `Predict` button
- result label showing `Positive` or `Negative`
- validation message for blank input

Optional:
- show Logistic Regression probability or SVM decision score with the correct label

The GUI should contain no ML logic beyond calling `src.predict`.

---

### Tests

#### `tests/test_data.py`

Test:

- 3-star reviews are removed
- 1/2 map to negative
- 4/5 map to positive
- duplicate/blank reviews are removed
- balanced sampling gives equal class counts
- split sizes and labels are correct

Use tiny synthetic DataFrames; do not require the Kaggle dataset.

#### `tests/test_linguistic_features.py`

Test features using hand-written sentences with predictable properties.

Examples should verify:

- exclamation count
- negation count
- first-person pronoun ratio
- sentence count
- word count
- adjective/adverb/verb identification
- no division-by-zero or `NaN`
- passive-voice heuristic on at least one clearly passive sentence

Avoid brittle tests that assume every ambiguous word receives one exact POS tag.

#### `tests/test_feature_builder.py`

Verify:

- each feature set has the expected number of columns
- `basic10` has 10 columns
- `extra5` has 5
- `all15` has 15
- `doc2vec` matches configured vector size
- `all15_doc2vec` has `15 + vector_size`
- row order is unchanged

#### `tests/test_predict.py`

Smoke-test end-to-end inference using small temporary/mock artifacts.

Verify:

- blank/valid input handling
- expected feature-set routing
- output sentiment is one of `{positive, negative}`
- linguistic-only models do not require Doc2Vec
- Doc2Vec-based models do require the Doc2Vec artifact

---

## 9. Reproducibility Rules

The project must be runnable from a fresh environment with the raw CSV plus the repository.

Use the same random seed for:

- balanced sampling
- dataset split
- Doc2Vec initialization where supported
- model settings where relevant

Record package versions in `requirements.txt`.

Store experiment configuration in result rows, including:

- feature set
- classifier
- `C`
- Doc2Vec parameters
- random seed

Do not rely on manually edited intermediate files.

---

## 10. Expected Commands

The final implementation should support a workflow similar to:

```bash
python -m src.data
python -m src.run_experiments
pytest
python app.py
```

`python -m src.data` should create the processed 6,000-review dataset and splits.

`python -m src.run_experiments` should generate feature artifacts, train/select/finalize models, and create all evaluation outputs.

---

## 11. Result Tables

### `validation_results.csv`

One row per candidate configuration:

```text
feature_set,classifier,C,accuracy,precision,recall,f1
```

With 5 feature sets × 2 classifiers × 3 `C` values, the expected minimum is **30 rows**.

### `test_results.csv`

Exactly one row per final model:

```text
feature_set,classifier,C,accuracy,precision,recall,f1
```

Expected: **10 rows**.

### `errors.csv`

Recommended columns:

```text
review_id,review,rating,true_label,feature_set,classifier,predicted_label
```

This enables analysis of common classifier mistakes.

### `linguistic_coefficients.csv`

Recommended columns:

```text
feature_set,classifier,feature,coefficient,abs_coefficient
```

Only include named linguistic features.

---

## 12. Implementation Order

### Phase 1 — Repository and data

1. Create directory structure.
2. Add dependencies/config.
3. Implement `src/data.py`.
4. Generate and verify the balanced dataset and split sizes.

### Phase 2 — Linguistic features

1. Implement shared tokenization/POS helpers.
2. Implement all 15 features.
3. Add unit tests.
4. Run extraction on a small sample and manually inspect values.

### Phase 3 — Doc2Vec and feature matrices

1. Implement Doc2Vec training/inference.
2. Implement feature-set assembly.
3. Verify matrix dimensions.
4. Save generated feature artifacts.

### Phase 4 — Classifiers and validation

1. Implement training pipeline.
2. Implement evaluation metrics.
3. Run all validation candidates.
4. Freeze one hyperparameter configuration per classifier/feature-set pair.

### Phase 5 — Final test evaluation

1. Refit selected models on train + validation.
2. Evaluate once on test.
3. Generate final comparison table.
4. Generate confusion matrices.
5. Generate error analysis and coefficient tables.

### Phase 6 — GUI

1. Implement inference wrapper.
2. Build Tkinter UI.
3. Test at least one linguistic-only and one Doc2Vec-based model manually.

### Phase 7 — Final report

Use generated artifacts to answer:

- Do the additional 5 linguistic features improve performance?
- How does Doc2Vec compare with hand-engineered linguistic features?
- Does combining Doc2Vec with linguistic features improve results?
- Does the preferred representation depend on classifier?
- What types of reviews are commonly misclassified?
- Which linguistic features have the strongest linear-model coefficients?
- What limitations come from using star ratings as sentiment labels?

---

## 13. Acceptance Criteria

The implementation is complete when all of the following are true:

- The raw Kaggle CSV can be converted into a cleaned, labeled, balanced 6,000-review dataset.
- The data split is reproducible and stratified at 60/20/20.
- All 15 linguistic features are implemented and tested.
- Doc2Vec is trained without validation/test leakage.
- All five feature sets can be generated with correct dimensions.
- Logistic Regression and Linear SVM are evaluated for every feature set.
- Validation is used for hyperparameter selection.
- The test set is used only after choices are frozen.
- `validation_results.csv` contains the complete model-selection record.
- `test_results.csv` contains exactly 10 final model results.
- Confusion matrices and error-analysis outputs are generated.
- Final trained model artifacts can be loaded without retraining.
- The GUI can classify a newly entered review using any of the saved model choices.
- `pytest` passes.
- `README.md` contains enough instructions for another person to reproduce the project.

---

## 14. Scope Boundaries

To keep the project aligned with the proposal, do **not** add unrelated model families such as neural networks, transformers, random forests, or Naive Bayes unless they are explicitly introduced later as optional extensions.

The required scientific comparison is:

**linguistic features vs. Doc2Vec vs. their combination, using the same two linear classifiers under the same train/validation/test split.**

That comparison should remain the center of the implementation.
