import re
from pathlib import Path

import joblib

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "intent_classifier.joblib"
SPAM_MODEL_PATH = BASE_DIR / "models" / "spam_classifier.joblib"


def preprocess_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


class IntentClassifier:
    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        spam_model_path: Path = SPAM_MODEL_PATH,
    ) -> None:
        if not model_path.exists() or not spam_model_path.exists():
            raise RuntimeError("Models not found. Run: python scripts/train_model.py")
        self.pipeline = joblib.load(model_path)
        self.spam_pipeline = joblib.load(spam_model_path)

    def predict(self, subject: str, body: str) -> tuple[str, float]:
        text = preprocess_text(f"{subject} {body}")
        spam_probabilities = self.spam_pipeline.predict_proba([text])[0]
        spam_index = list(self.spam_pipeline.classes_).index("spam")
        if self.spam_pipeline.predict([text])[0] == "spam":
            return "spam", float(spam_probabilities[spam_index])

        probabilities = self.pipeline.predict_proba([text])[0]
        best_index = int(probabilities.argmax())
        return str(self.pipeline.classes_[best_index]), float(probabilities[best_index])


HIGH_PRIORITY_TERMS = {"urgent", "immediately", "payment failed", "payment failure", "account blocked", "locked out", "service unavailable", "not working", "refund not received", "refund overdue", "charged twice", "security", "fraud"}
LOW_PRIORITY_TERMS = {"newsletter", "partnership", "demo", "pricing", "brochure", "offer"}


def detect_priority(subject: str, body: str, intent: str) -> str:
    text = preprocess_text(f"{subject} {body}")
    if intent == "spam":
        return "low"
    if any(term in text for term in HIGH_PRIORITY_TERMS):
        return "high"
    if intent == "sales" or any(term in text for term in LOW_PRIORITY_TERMS):
        return "low"
    return "normal"