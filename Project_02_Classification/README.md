# Breast Mass Classification: Logistic Regression & Decision Thresholds
**Author:** Ignacio Flores III  
**Course:** MBSA 265: Machine Learning Pathway  

This project implements binary logistic regression using manual NumPy gradient descent and compares it against an unregularized Scikit-learn estimator and a majority baseline on the Wisconsin Diagnostic Breast Cancer dataset.

---

## 1. Project Overview & Findings
* **Objective:** Predict breast mass malignancy (**1 = Malignant**, **0 = Benign**) using two standardized features: `mean radius` and `mean texture`.
* **Equivalence:** NumPy gradient descent converged to within $1.19 \times 10^{-10}$ mean absolute probability difference of Scikit-learn (`C=inf`).
* **Thresholding:** Validation tuning selected an optimal decision threshold of $\tau = 0.60$, prioritizing precision ($90.32\%$) with an $F_1$ of $0.7671$ on the test set.
* **Scope:** Educational modeling prototype only; non-clinical.

---

## 2. Directory Structure
```text
Project_02_Classification/
├── logistic_regression_analysis.py  # End-to-end training and evaluation script
├── memo.md                          # Detailed research memo and report
├── assignment_requirements.txt      # Dependency specification
├── reports/
│   ├── training_loss.png            # NumPy BCE loss vs. epoch curve
│   ├── threshold_results.csv        # Validation metric scan across 7 thresholds
│   └── assignment_results.csv       # 10 configuration rows (Validation & Test)
└── README.md                        # Reproduction guide