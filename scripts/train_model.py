import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.classifier import preprocess_text  # noqa: E402

RANDOM_STATE = 42
TEST_SIZE = 0.20
SPAM_SAMPLE_SIZE = 5_000


def evaluate(model: Pipeline, x_test: pd.Series, y_test: pd.Series) -> dict:
    predictions = model.predict(x_test)
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    return {
        "test_samples": len(y_test),
        "accuracy": accuracy_score(y_test, predictions),
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
        "macro_f1": report["macro avg"]["f1-score"],
        "labels": list(model.classes_),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=model.classes_).tolist(),
        "classification_report": report,
    }


def train_spam_detector() -> tuple[Pipeline, dict]:
    raw_data = pd.read_csv(ROOT / "spam_ham.txt", sep="\t")
    data = raw_data.drop_duplicates(subset=["Msg"])
    unique_rows = len(data)
    if len(data) < SPAM_SAMPLE_SIZE:
        raise ValueError(f"Need {SPAM_SAMPLE_SIZE} unique spam/ham messages, found {len(data)}")

    data = data.sample(n=SPAM_SAMPLE_SIZE, random_state=RANDOM_STATE)
    texts = data["Msg"].map(preprocess_text)
    x_train, x_test, y_train, y_test = train_test_split(
        texts,
        data["Target"],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=data["Target"],
    )
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)),
    ])
    pipeline.fit(x_train, y_train)
    metrics = evaluate(pipeline, x_test, y_test)
    metrics.update({
        "source_rows": len(raw_data),
        "unique_rows": unique_rows,
        "sample_size": SPAM_SAMPLE_SIZE,
        "train_samples": len(x_train),
        "split": "80/20 stratified",
        "sample_label_counts": data["Target"].value_counts().to_dict(),
    })
    return pipeline, metrics


def train_business_intent_classifier() -> tuple[Pipeline, dict]:
    data = pd.read_csv(ROOT / "data" / "emails.csv")
    data = data[data["intent"] != "spam"]
    texts = (data["subject"] + " " + data["body"]).map(preprocess_text)
    x_train, x_test, y_train, y_test = train_test_split(
        texts,
        data["intent"],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=data["intent"],
    )
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)),
        ("classifier", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    ])
    pipeline.fit(x_train, y_train)
    metrics = evaluate(pipeline, x_test, y_test)
    metrics.update({"sample_size": len(data), "train_samples": len(x_train), "split": "80/20 stratified"})
    return pipeline, metrics


def train() -> dict:
    spam_pipeline, spam_metrics = train_spam_detector()
    intent_pipeline, intent_metrics = train_business_intent_classifier()
    metrics = {
        "spam_detector": spam_metrics,
        "business_intent_classifier": intent_metrics,
    }

    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump(spam_pipeline, ROOT / "models" / "spam_classifier.joblib")
    joblib.dump(intent_pipeline, ROOT / "models" / "intent_classifier.joblib")
    (ROOT / "models" / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    train()