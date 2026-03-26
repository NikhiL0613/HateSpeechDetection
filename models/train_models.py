"""
Model Training - Hate Speech Detection
Trains 6 models, evaluates them, picks the best one.
"""

import os
import json
import pickle
import time
import warnings
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix,
    roc_auc_score,
)

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
PROCESSED_DIR = "data/processed"
MODEL_DIR = "models"


def load_data():
    data = {}
    for name in ["X_train", "X_val", "X_test",
                 "y_train", "y_val", "y_test"]:
        with open(f"{PROCESSED_DIR}/{name}.pkl", "rb") as f:
            data[name] = pickle.load(f)
    return data


def evaluate(name, y_true, y_pred, y_prob=None):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob) if y_prob is not None else 0
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    metrics = {
        "model": name,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "auc_roc": round(auc, 4),
        "false_positive_rate": round(fpr, 4),
        "confusion_matrix": cm.tolist(),
    }
    print(f"  {name}: Acc={acc:.4f}  F1={f1:.4f}  AUC={auc:.4f}  FPR={fpr:.4f}")
    return metrics


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    print("=" * 60)
    print("  HATE SPEECH DETECTION - MODEL TRAINING")
    print("=" * 60)

    data = load_data()
    X_train = data["X_train"]
    X_val = data["X_val"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_val = data["y_val"]
    y_test = data["y_test"]

    models = {
        "LogisticRegression": LogisticRegression(
            C=1.0, max_iter=500, solver="lbfgs",
            random_state=RANDOM_STATE, n_jobs=-1
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=100, max_depth=30,
            random_state=RANDOM_STATE, n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=50, learning_rate=0.2, max_depth=4,
            random_state=RANDOM_STATE
        ),
        "LinearSVC": CalibratedClassifierCV(
            LinearSVC(C=1.0, max_iter=500, random_state=RANDOM_STATE),
            cv=2
        ),
        "MLP_NeuralNet": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            batch_size=512, max_iter=10,
            early_stopping=True, random_state=RANDOM_STATE
        ),
    }

    all_results = {}
    trained_models = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        start = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - start
        print(f"  Trained in {elapsed:.1f}s")

        y_pred = model.predict(X_val)
        y_prob = None
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_val)[:, 1]
        metrics = evaluate(name, y_val, y_pred, y_prob)
        metrics["train_time_sec"] = round(elapsed, 2)
        all_results[name] = metrics
        trained_models[name] = model

    print("\nTraining Ensemble (combines top 3 models)...")
    top3 = sorted(all_results.keys(),
                  key=lambda k: all_results[k]["accuracy"],
                  reverse=True)[:3]
    estimators = [(n, trained_models[n]) for n in top3
                  if hasattr(trained_models[n], "predict_proba")]
    ensemble = VotingClassifier(
        estimators=estimators, voting="soft", n_jobs=-1
    )
    ensemble.fit(X_train, y_train)
    y_pred = ensemble.predict(X_val)
    y_prob = ensemble.predict_proba(X_val)[:, 1]
    metrics = evaluate("Ensemble", y_val, y_pred, y_prob)
    all_results["Ensemble"] = metrics
    trained_models["Ensemble"] = ensemble

    best = max(all_results, key=lambda k: all_results[k]["accuracy"])
    print(f"\n{'=' * 60}")
    print(f"  BEST MODEL: {best} -> {all_results[best]['accuracy']}")
    print(f"{'=' * 60}")

    best_model = trained_models[best]
    y_pred_test = best_model.predict(X_test)
    y_prob_test = best_model.predict_proba(X_test)[:, 1]
    test_metrics = evaluate(f"{best} [TEST]", y_test, y_pred_test, y_prob_test)
    all_results["TEST_FINAL"] = test_metrics
    print(f"\n{classification_report(y_test, y_pred_test, target_names=['Non-Hate', 'Hate'])}")

    for name, model in trained_models.items():
        with open(f"{MODEL_DIR}/{name}.pkl", "wb") as f:
            pickle.dump(model, f)
    with open(f"{MODEL_DIR}/all_metrics.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    with open(f"{MODEL_DIR}/best_model.json", "w") as f:
        json.dump({"best_model": best,
                   "test_accuracy": test_metrics["accuracy"]}, f, indent=2)
    print(f"\nAll models saved to {MODEL_DIR}/")


if __name__ == "__main__":
    main()