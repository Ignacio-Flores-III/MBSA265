"""
Logistic Regression from Scores to Decisions
Project 02: Classification
Author: Ignacio Flores III
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, log_loss

# Resolve paths relative to script location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

def sigmoid(z):
    # Numerically stable sigmoid using exp(-logaddexp(0, -z))
    return np.exp(-np.logaddexp(0, -z))

def binary_cross_entropy(z, y):
    # Numerically stable BCE using mean(logaddexp(0, z) - y*z)
    return np.mean(np.logaddexp(0, z) - y * z)

def main():
    print("="*70)
    print("PART 1: INTERACTIVE EXPLORER CASE CHECKS")
    print("="*70)
    cases = [
        {"case": "A", "b0": -1.0, "b1": 0.8, "x": 2.0, "thresh": 0.50},
        {"case": "B", "b0": -1.0, "b1": 0.8, "x": 2.0, "thresh": 0.70},
        {"case": "C", "b0": -1.0, "b1": -0.8, "x": 2.0, "thresh": 0.50},
        {"case": "D", "b0": 1.0, "b1": 0.0, "x": 2.0, "thresh": 0.50},
    ]
    for c in cases:
        z = c["b0"] + c["b1"] * c["x"]
        p = sigmoid(z)
        pred = 1 if p >= c["thresh"] else 0
        print(f"Case {c['case']}: z = {z:7.4f}, p = {p:6.4f}, Threshold = {c['thresh']:.2f} -> Class {pred}")

    print("\n" + "="*70)
    print("PART 2: LEAKAGE-FREE EXPERIMENT PREPARATION")
    print("="*70)
    data = load_breast_cancer(as_frame=True)
    df = data.frame
    
    # Select exact features and recode target: 1 = malignant, 0 = benign
    feature_cols = ["mean radius", "mean texture"]
    X_raw = df[feature_cols].values
    y = (data.target == 0).astype(int).values

    print(f"Features: {feature_cols}")
    print(f"Missing values: {df[feature_cols].isna().sum().to_dict()}")
    print(f"Overall Class Distribution: Benign (0)={np.sum(y == 0)}, Malignant (1)={np.sum(y == 1)}")

    # Split row indices exactly per instructions
    indices = np.arange(len(y))
    dev_idx, test_idx = train_test_split(
        indices, test_size=0.20, stratify=y, random_state=42
    )
    train_idx, val_idx = train_test_split(
        dev_idx, test_size=0.25, stratify=y[dev_idx], random_state=42
    )

    # Verification of disjoint indices
    all_combined = set(train_idx) | set(val_idx) | set(test_idx)
    assert len(train_idx) + len(val_idx) + len(test_idx) == len(y), "Overlap detected in splits!"
    assert len(all_combined) == len(y), "Splits do not cover all observations!"

    print(f"Train partition: {len(train_idx)} rows (Benign: {np.sum(y[train_idx]==0)}, Malignant: {np.sum(y[train_idx]==1)})")
    print(f"Val partition:   {len(val_idx)} rows (Benign: {np.sum(y[val_idx]==0)}, Malignant: {np.sum(y[val_idx]==1)})")
    print(f"Test partition:  {len(test_idx)} rows (Benign: {np.sum(y[test_idx]==0)}, Malignant: {np.sum(y[test_idx]==1)})")

    # Fit scaler on training set ONLY
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_raw[train_idx])
    X_val = scaler.transform(X_raw[val_idx])
    X_test = scaler.transform(X_raw[test_idx])
    y_train = y[train_idx]
    y_val = y[val_idx]
    y_test = y[test_idx]

    print("\n" + "="*70)
    print("PART 3: NUMPY LOGISTIC REGRESSION VIA GRADIENT DESCENT")
    print("="*70)
    # Augment feature matrices with leading column of ones
    X_train_aug = np.hstack([np.ones((len(X_train), 1)), X_train])
    X_val_aug = np.hstack([np.ones((len(X_val), 1)), X_val])
    X_test_aug = np.hstack([np.ones((len(X_test), 1)), X_test])

    beta = np.zeros(3)
    learning_rate = 0.1
    epochs = 20000

    # Initial loss
    initial_z = X_train_aug @ beta
    loss_history = [binary_cross_entropy(initial_z, y_train)]
    print(f"Initial BCE Loss: {loss_history[0]:.6f}")

    n_train = len(X_train)
    for epoch in range(1, epochs + 1):
        z = X_train_aug @ beta
        p = sigmoid(z)
        gradient = (X_train_aug.T @ (p - y_train)) / n_train
        beta -= learning_rate * gradient
        
        # Loss after update
        z_new = X_train_aug @ beta
        loss_history.append(binary_cross_entropy(z_new, y_train))

    print(f"Final BCE Loss after {epochs} epochs: {loss_history[-1]:.6f}")
    print(f"Learned NumPy beta (standardized units):")
    print(f"  beta_0 (intercept):    {beta[0]:.6f}")
    print(f"  beta_1 (mean radius):  {beta[1]:.6f}")
    print(f"  beta_2 (mean texture): {beta[2]:.6f}")

    # Save training loss plot
    plt.figure(figsize=(8, 5))
    plt.plot(range(len(loss_history)), loss_history, color="#d35400", lw=2)
    plt.title("NumPy Logistic Regression: Training Loss vs. Epoch", fontsize=14, pad=12)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Binary Cross-Entropy Loss", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    loss_plot_path = os.path.join(REPORTS_DIR, "training_loss.png")
    plt.savefig(loss_plot_path, dpi=300)
    plt.close()
    print(f"Saved: {loss_plot_path}")

    print("\n" + "="*70)
    print("PART 4: SCIKIT-LEARN REFERENCE & BASELINE COMPARISON")
    print("="*70)
    # Unregularized Scikit-learn model
    sklearn_model = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=10000, tol=1e-9)
    sklearn_model.fit(X_train, y_train)

    # Dummy majority baseline
    baseline_model = DummyClassifier(strategy="prior")
    baseline_model.fit(X_train, y_train)

    b0_sk = sklearn_model.intercept_[0]
    b1_sk, b2_sk = sklearn_model.coef_[0]
    print(f"Scikit-learn coefficients (standardized units):")
    print(f"  beta_0 (intercept):    {b0_sk:.6f}")
    print(f"  beta_1 (mean radius):  {b1_sk:.6f}")
    print(f"  beta_2 (mean texture): {b2_sk:.6f}")

    # Compare validation probabilities
    val_probs_np = sigmoid(X_val_aug @ beta)
    val_probs_sk = sklearn_model.predict_proba(X_val)[:, 1]
    mean_abs_diff = np.mean(np.abs(val_probs_np - val_probs_sk))
    print(f"Mean Absolute Difference (Validation Probabilities): {mean_abs_diff:.6e}")

    print("\n" + "="*70)
    print("PART 5: THRESHOLD SELECTION & FINAL EVALUATION")
    print("="*70)
    candidate_thresholds = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
    thresh_records = []

    for t in candidate_thresholds:
        preds = (val_probs_np >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, preds, labels=[0, 1]).ravel()
        acc = accuracy_score(y_val, preds)
        prec = precision_score(y_val, preds, zero_division=0)
        rec = recall_score(y_val, preds, zero_division=0)
        f1 = f1_score(y_val, preds, zero_division=0)
        thresh_records.append({
            "threshold": t,
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1
        })

    thresh_df = pd.DataFrame(thresh_records)
    thresh_csv_path = os.path.join(REPORTS_DIR, "threshold_results.csv")
    thresh_df.to_csv(thresh_csv_path, index=False)
    print(f"Exported threshold results to: {thresh_csv_path}")
    print(thresh_df.to_string(index=False))

    # Pick threshold with highest validation F1 (smallest on tie)
    max_f1 = thresh_df["f1"].max()
    best_threshold = thresh_df[thresh_df["f1"] == max_f1]["threshold"].min()
    print(f"\nOptimal Validation Threshold Selected: {best_threshold:.2f} (F1 = {max_f1:.6f})")

    # Evaluate 5 configurations on Validation and Test
    test_probs_np = sigmoid(X_test_aug @ beta)
    test_probs_sk = sklearn_model.predict_proba(X_test)[:, 1]
    val_probs_base = baseline_model.predict_proba(X_val)[:, 1]
    test_probs_base = baseline_model.predict_proba(X_test)[:, 1]

    eval_configs = [
        {"model": "baseline", "policy": "default", "thresh": 0.50, "val_p": val_probs_base, "test_p": test_probs_base},
        {"model": "numpy",    "policy": "default", "thresh": 0.50, "val_p": val_probs_np,   "test_p": test_probs_np},
        {"model": "numpy",    "policy": "selected","thresh": best_threshold, "val_p": val_probs_np, "test_p": test_probs_np},
        {"model": "sklearn",  "policy": "default", "thresh": 0.50, "val_p": val_probs_sk,   "test_p": test_probs_sk},
        {"model": "sklearn",  "policy": "selected","thresh": best_threshold, "val_p": val_probs_sk, "test_p": test_probs_sk},
    ]

    results_records = []
    for split_name, y_true in [("validation", y_val), ("test", y_test)]:
        for cfg in eval_configs:
            probs = cfg["val_p"] if split_name == "validation" else cfg["test_p"]
            preds = (probs >= cfg["thresh"]).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
            acc = accuracy_score(y_true, preds)
            prec = precision_score(y_true, preds, zero_division=0)
            rec = recall_score(y_true, preds, zero_division=0)
            f1 = f1_score(y_true, preds, zero_division=0)
            ll = log_loss(y_true, probs)
            results_records.append({
                "split": split_name,
                "model": cfg["model"],
                "policy": cfg["policy"],
                "threshold": cfg["thresh"],
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "log_loss": ll
            })

    results_df = pd.DataFrame(results_records)
    results_csv_path = os.path.join(REPORTS_DIR, "assignment_results.csv")
    results_df.to_csv(results_csv_path, index=False)
    print(f"\nExported 10 assignment configurations to: {results_csv_path}")
    print(results_df.to_string(index=False))
    print("\nRun complete. All outputs generated successfully.")

if __name__ == "__main__":
    main()