"""
Train a classifier on collected gesture sequences and save it.

Usage:
    python src/train.py
"""

import glob
import os

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from sequence_utils import SEQUENCE_LENGTH, sequence_to_features

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "sequences")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "asl_classifier.pkl")


def load_dataset():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.npy")))
    X, y = [], []
    for f in files:
        label = os.path.basename(f).split("__")[0]
        seq = np.load(f)
        X.append(sequence_to_features(seq))
        y.append(label)
    return np.array(X), np.array(y)


def main():
    if not os.path.isdir(DATA_DIR) or not glob.glob(os.path.join(DATA_DIR, "*.npy")):
        raise FileNotFoundError(
            f"No sequence data found in {DATA_DIR}. Run collect_data.py first to record samples."
        )

    X, y = load_dataset()
    if len(set(y)) < 2:
        raise ValueError("Need at least 2 distinct signs to train a classifier. Collect more.")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=300, random_state=42)
    clf.fit(X_train, y_train)

    val_preds = clf.predict(X_val)
    acc = accuracy_score(y_val, val_preds)
    print(f"Validation accuracy: {acc:.3f}")
    print(classification_report(y_val, val_preds))

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump({"model": clf, "sequence_length": SEQUENCE_LENGTH}, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
