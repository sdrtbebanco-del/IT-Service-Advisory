"""
IT Helpdesk ML Model Training Script
- Preprocessing: lowercasing, punctuation removal
- TF-IDF with ngram_range=(1,2)
- Class-balanced Logistic Regression
- Metrics + confusion matrix heatmap
"""
import os
import re
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


def preprocess_text(text):
    """Lowercase and remove punctuation for robust inference."""
    if not isinstance(text, str) or not text.strip():
        return ""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def build_dataset():
    """Build or load helpdesk dataset."""
    csv_path = os.path.join(DATA_DIR, "helpdesk_tickets.csv")
    if os.path.isfile(csv_path):
        df = pd.read_csv(csv_path)
        df["text"] = df["text"].astype(str).apply(preprocess_text)
        return df

    categories = {
        "network": [
            "Cannot connect to WiFi",
            "Internet is very slow",
            "Network cable unplugged",
            "VPN not connecting",
            "DNS server not responding",
            "Frequent network disconnection",
            "IP address conflict detected",
            "Router keeps restarting",
        ],
        "hardware": [
            "Laptop is overheating",
            "Keyboard not working",
            "Mouse is not detected",
            "Screen flickering",
            "Battery not charging",
            "Hard drive failure",
            "Computer not turning on",
            "USB port not working",
        ],
        "software": [
            "MS Word keeps crashing",
            "Application not responding",
            "Unable to install software",
            "System update failed",
            "Blue screen error",
            "Antivirus not updating",
            "Software license expired",
            "File cannot be opened",
        ],
        "account": [
            "Forgot my email password",
            "Account locked after login attempts",
            "Unable to access shared drive",
            "Permission denied error",
            "Need password reset",
            "Two-factor authentication not working",
            "Cannot login to system",
            "Email not syncing",
        ],
        "printer": [
            "Printer not responding",
            "Paper jam error",
            "Printer offline",
            "Cannot print document",
            "Low ink warning",
            "Printer driver missing",
            "Printer printing blank pages",
            "Printer not detected",
        ],
    }
    actions = {
        "network": "Restart router and check cables",
        "hardware": "Check physical connections or repair hardware",
        "software": "Reinstall or update the software",
        "account": "Reset password or verify account settings",
        "printer": "Restart printer and check paper/ink",
    }
    data = []
    np.random.seed(42)
    for category, texts in categories.items():
        for _ in range(50):
            text = np.random.choice(texts)
            data.append({"text": preprocess_text(text), "category": category, "action": actions[category]})
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)
    return df


def main():
    df = build_dataset()
    X = df["text"].values
    y = df["category"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=2000,
        min_df=1,
        max_df=0.95,
        sublinear_tf=True,
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
        C=1.0,
        solver="lbfgs",
    )
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    y_proba = model.predict_proba(X_test_vec)

    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )

    print("=" * 50)
    print("MODEL METRICS")
    print("=" * 50)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print()
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=model.classes_,
        yticklabels=model.classes_,
        cmap="Blues",
        cbar_kws={"label": "Count"},
    )
    plt.title("Confusion Matrix Heatmap")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "confusion_matrix.png"), dpi=120)
    plt.close()

    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
    }
    joblib.dump(metrics, os.path.join(MODEL_DIR, "metrics.pkl"))
    joblib.dump(model, os.path.join(MODEL_DIR, "helpdesk_model.pkl"))
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
    joblib.dump(model.classes_.tolist(), os.path.join(MODEL_DIR, "classes.pkl"))

    print("\nModel and artifacts saved to:", MODEL_DIR)


if __name__ == "__main__":
    main()
