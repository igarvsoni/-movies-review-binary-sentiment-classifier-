# Movies Review Binary Sentiment Classifier

This project performs NLP-based binary sentiment classification of movie reviews as **positive** or **negative**.

## What it implements

- Text preprocessing (lowercasing, HTML/noise cleanup, whitespace normalization)
- Exploratory data analysis (class balance and review-length summaries)
- TF-IDF feature extraction with **unigram + bigram** features
- Baseline model comparison:
  - Naive Bayes
  - Logistic Regression
  - Linear SVM
- Evaluation with:
  - Accuracy
  - Classification report
  - Confusion matrix
  - 5-fold cross-validation

In typical runs on standard movie-review datasets, **Linear SVM** is expected to provide the best baseline performance (around **86% accuracy**).

## Usage

1. Install dependencies:

```bash
pip install pandas scikit-learn
```

2. Prepare a CSV file with:
   - a review text column (default: `review`)
   - a sentiment label column (default: `sentiment`), where labels are either positive/negative (or 1/0)

3. Run:

```bash
python sentiment_pipeline.py --csv /absolute/path/to/reviews.csv
```

Optional:

```bash
python sentiment_pipeline.py --csv /absolute/path/to/reviews.csv --text-column review --label-column sentiment
```
