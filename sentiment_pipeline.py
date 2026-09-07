from __future__ import annotations

import argparse
import re
from dataclasses import dataclass

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


@dataclass(frozen=True)
class DatasetEDA:
    total_reviews: int
    positive_reviews: int
    negative_reviews: int
    avg_char_length: float
    avg_word_length: float


def preprocess_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_label(label: object) -> int:
    if isinstance(label, str):
        value = label.strip().lower()
        if value in {"positive", "pos", "1", "true"}:
            return 1
        if value in {"negative", "neg", "0", "false"}:
            return 0
    if label in {1, True}:
        return 1
    if label in {0, False}:
        return 0
    raise ValueError(f"Unsupported label value: {label!r}")


def load_reviews(csv_path: str, text_column: str, label_column: str) -> tuple[pd.Series, pd.Series]:
    df = pd.read_csv(csv_path)
    if text_column not in df.columns or label_column not in df.columns:
        raise ValueError(
            f"Input data must contain '{text_column}' and '{label_column}' columns. "
            f"Found columns: {list(df.columns)}"
        )

    texts = df[text_column].fillna("").map(preprocess_text)
    labels = df[label_column].map(normalize_label)
    return texts, labels


def summarize_eda(texts: pd.Series, labels: pd.Series) -> DatasetEDA:
    review_lengths = texts.map(len)
    word_lengths = texts.map(lambda t: len(t.split()))
    return DatasetEDA(
        total_reviews=len(texts),
        positive_reviews=int((labels == 1).sum()),
        negative_reviews=int((labels == 0).sum()),
        avg_char_length=float(review_lengths.mean()),
        avg_word_length=float(word_lengths.mean()),
    )


def build_model(model_name: str) -> Pipeline:
    estimators = {
        "naive_bayes": MultinomialNB(),
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
        "linear_svm": LinearSVC(random_state=42),
    }
    if model_name not in estimators:
        raise ValueError(f"Unsupported model '{model_name}'")

    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    stop_words="english",
                    sublinear_tf=True,
                ),
            ),
            ("model", estimators[model_name]),
        ]
    )


def run_experiment(csv_path: str, text_column: str, label_column: str) -> dict[str, float]:
    texts, labels = load_reviews(csv_path, text_column, label_column)
    eda = summarize_eda(texts, labels)

    print("=== Exploratory Data Analysis ===")
    print(f"Total reviews: {eda.total_reviews}")
    print(f"Positive reviews: {eda.positive_reviews}")
    print(f"Negative reviews: {eda.negative_reviews}")
    print(f"Average review length (chars): {eda.avg_char_length:.2f}")
    print(f"Average review length (words): {eda.avg_word_length:.2f}")

    x_train, x_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    model_names = ["naive_bayes", "logistic_regression", "linear_svm"]
    accuracies: dict[str, float] = {}
    cv_folds = min(5, int(labels.value_counts().min()))

    print("\n=== Model Comparison ===")
    for model_name in model_names:
        clf = build_model(model_name)
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)

        accuracy = accuracy_score(y_test, y_pred)
        accuracies[model_name] = accuracy

        print(f"\nModel: {model_name}")
        print(f"Test accuracy: {accuracy:.4f}")
        print("Classification report:")
        print(
            classification_report(
                y_test,
                y_pred,
                target_names=["negative", "positive"],
                zero_division=0,
            )
        )
        print("Confusion matrix:")
        print(confusion_matrix(y_test, y_pred))

        if cv_folds >= 2:
            cv_scores = cross_val_score(clf, texts, labels, cv=cv_folds, scoring="accuracy")
            print(
                f"{cv_folds}-fold CV accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}"
            )
        else:
            print("Cross-validation skipped: at least 2 examples per class are required.")

    best_model = max(accuracies, key=accuracies.get)
    print(f"\nBest test-set model: {best_model} ({accuracies[best_model]:.4f})")

    return accuracies


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Binary sentiment classifier for movie reviews."
    )
    parser.add_argument("--csv", required=True, help="Path to CSV dataset")
    parser.add_argument(
        "--text-column",
        default="review",
        help="Name of text column in the CSV",
    )
    parser.add_argument(
        "--label-column",
        default="sentiment",
        help="Name of binary label column in the CSV",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_experiment(args.csv, args.text_column, args.label_column)


if __name__ == "__main__":
    main()
