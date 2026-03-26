"""
Deep Learning Model for Hate Speech Detection
Multi-layer Neural Network with advanced architecture
"""

import os
import json
import pickle
import time
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix,
    roc_auc_score
)

PROCESSED_DIR = "data/processed"
MODEL_DIR = "models"

print("=" * 60)
print("  DEEP LEARNING MODEL - MULTI-LAYER NEURAL NETWORK")
print("=" * 60)

# Load data
print("\nLoading data...")
with open(f"{PROCESSED_DIR}/X_train.pkl", "rb") as f:
    X_train = pickle.load(f)
with open(f"{PROCESSED_DIR}/X_val.pkl", "rb") as f:
    X_val = pickle.load(f)
with open(f"{PROCESSED_DIR}/X_test.pkl", "rb") as f:
    X_test = pickle.load(f)
with open(f"{PROCESSED_DIR}/y_train.pkl", "rb") as f:
    y_train = pickle.load(f)
with open(f"{PROCESSED_DIR}/y_val.pkl", "rb") as f:
    y_val = pickle.load(f)
with open(f"{PROCESSED_DIR}/y_test.pkl", "rb") as f:
    y_test = pickle.load(f)

print(f"  Train: {X_train.shape}")
print(f"  Val:   {X_val.shape}")
print(f"  Test:  {X_test.shape}")

# Build deep neural network
# Architecture: 512 -> 256 -> 128 -> 64 -> 32 (5 hidden layers)
print("\nBuilding 5-layer Deep Neural Network...")
print("  Architecture: Input -> 512 -> 256 -> 128 -> 64 -> 32 -> Output")
print("  Activation: ReLU")
print("  Optimizer: Adam")
print("  Regularization: Early Stopping + Adaptive Learning Rate")

model = MLPClassifier(
    hidden_layer_sizes=(512, 256, 128, 64, 32),
    activation="relu",
    solver="adam",
    alpha=0.0001,
    batch_size=256,
    learning_rate="adaptive",
    learning_rate_init=0.001,
    max_iter=100,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=10,
    random_state=42,
    verbose=True,
)

# Train
print("\nTraining (this takes 2-5 minutes)...")
start = time.time()
model.fit(X_train, y_train)
elapsed = time.time() - start
print(f"\nTraining complete in {elapsed:.1f}s")
print(f"  Iterations: {model.n_iter_}")
print(f"  Final loss: {model.loss_:.4f}")

# Evaluate on validation
print("\n--- Validation Results ---")
y_val_pred = model.predict(X_val)
y_val_prob = model.predict_proba(X_val)[:, 1]
val_acc = accuracy_score(y_val, y_val_pred)
print(f"  Val Accuracy: {val_acc:.4f}")

# Evaluate on test
print("\n" + "=" * 60)
print("  TEST SET EVALUATION")
print("=" * 60)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
fpr = fp / (fp + tn)

print(f"\n  Accuracy:  {acc:.4f}")
print(f"  Precision: {prec:.4f}")
print(f"  Recall:    {rec:.4f}")
print(f"  F1 Score:  {f1:.4f}")
print(f"  AUC-ROC:   {auc:.4f}")
print(f"  FPR:       {fpr:.4f}")
print(f"\n{classification_report(y_test, y_pred, target_names=['Non-Hate', 'Hate'])}")

# Save model
with open(f"{MODEL_DIR}/DeepNeuralNet.pkl", "wb") as f:
    pickle.dump(model, f)
print(f"Model saved to {MODEL_DIR}/DeepNeuralNet.pkl")

# Update metrics
dl_metrics = {
    "model": "DeepNeuralNet",
    "accuracy": round(acc, 4),
    "precision": round(prec, 4),
    "recall": round(rec, 4),
    "f1_score": round(f1, 4),
    "auc_roc": round(auc, 4),
    "false_positive_rate": round(fpr, 4),
    "confusion_matrix": cm.tolist(),
    "train_time_sec": round(elapsed, 2),
    "architecture": "512-256-128-64-32",
    "iterations": model.n_iter_,
}

with open(f"{MODEL_DIR}/all_metrics.json") as f:
    all_metrics = json.load(f)
all_metrics["DeepNeuralNet"] = dl_metrics
with open(f"{MODEL_DIR}/all_metrics.json", "w") as f:
    json.dump(all_metrics, f, indent=2, default=str)

# Find best model
best_acc = 0
best_name = ""
for k, v in all_metrics.items():
    if isinstance(v, dict) and "accuracy" in v and k != "TEST_FINAL":
        if v["accuracy"] > best_acc:
            best_acc = v["accuracy"]
            best_name = k

with open(f"{MODEL_DIR}/best_model.json", "w") as f:
    json.dump({"best_model": best_name, "test_accuracy": best_acc}, f, indent=2)

print(f"\nBest overall model: {best_name} ({best_acc:.4f})")
print("=" * 60)
