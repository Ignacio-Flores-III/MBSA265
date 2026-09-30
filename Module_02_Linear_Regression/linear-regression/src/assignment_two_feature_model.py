from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Project Paths & Settings
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Robust search for insurance.csv
CANDIDATE_PATHS = [
    PROJECT_ROOT / "data" / "insurance-premium-prediction" / "insurance.csv",
    PROJECT_ROOT / "data" / "insurance.csv",
    PROJECT_ROOT.parent / "linear-regression" / "data" / "insurance.csv",
]
CSV_PATH = next((p for p in CANDIDATE_PATHS if p.exists()), CANDIDATE_PATHS[0])

REPORTS_DIR = PROJECT_ROOT / "reports"
OUTPUT_CSV = REPORTS_DIR / "assignment_results.csv"

# Gradient Descent Hyperparameters
LEARNING_RATE = 0.05
EPOCHS = 10000


# ---------------------------------------------------------------------------
# 1. Data Ingestion & Preprocessing
# ---------------------------------------------------------------------------
def load_dataset(csv_path: Path) -> Tuple[List[float], List[float], List[float]]:
    """Loads bmi, age, and expenses from the CSV dataset."""
    bmi: List[float] = []
    age: List[float] = []
    expenses: List[float] = []

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            bmi.append(float(row["bmi"]))
            age.append(float(row["age"]))
            expenses.append(float(row["expenses"]))

    return bmi, age, expenses


class StandardScaler:
    """Standardizes features to zero-mean and unit-variance."""

    def __init__(self) -> None:
        self.mean: float = 0.0
        self.std: float = 1.0

    def fit_transform(self, values: List[float]) -> List[float]:
        n = len(values)
        self.mean = sum(values) / n
        variance = sum((v - self.mean) ** 2 for v in values) / n
        self.std = math.sqrt(variance) if variance > 0 else 1.0
        return [(v - self.mean) / self.std for v in values]


# ---------------------------------------------------------------------------
# 2. Evaluation Metrics Helpers
# ---------------------------------------------------------------------------
def compute_metrics(y_true: List[float], y_pred: List[float]) -> Dict[str, float]:
    """Calculates MSE, RMSE, MAE, and R^2 for model predictions."""
    n = len(y_true)
    residuals = [yt - yp for yt, yp in zip(y_true, y_pred)]

    mse = sum(r**2 for r in residuals) / n
    rmse = math.sqrt(mse)
    mae = sum(abs(r) for r in residuals) / n

    y_mean = sum(y_true) / n
    ss_tot = sum((yt - y_mean) ** 2 for yt in y_true)
    ss_res = sum(r**2 for r in residuals)
    r2 = 1.0 - (ss_res / ss_tot if ss_tot != 0 else 0.0)

    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}


# ---------------------------------------------------------------------------
# 3. Model 1: Baseline OLS (1-Feature BMI)
# ---------------------------------------------------------------------------
class BaselineBMIModel:
    """Analytical single-feature Ordinary Least Squares: expenses = w0 + w1*bmi."""

    def __init__(self) -> None:
        self.w0: float = 0.0
        self.w1: float = 0.0

    def fit(self, x: List[float], y: List[float]) -> None:
        n = len(x)
        x_mean = sum(x) / n
        y_mean = sum(y) / n
        covariance = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
        variance = sum((xi - x_mean) ** 2 for xi in x)
        self.w1 = covariance / variance
        self.w0 = y_mean - (self.w1 * x_mean)

    def predict(self, x: List[float]) -> List[float]:
        return [self.w0 + self.w1 * xi for xi in x]


# ---------------------------------------------------------------------------
# 4. Model 2: Two-Feature Normal Equation (General Matrix Solve)
# ---------------------------------------------------------------------------
def _gauss_jordan_solve(A: List[List[float]], b: List[float]) -> List[float]:
    """Solves A * w = b using Gauss-Jordan elimination with partial pivoting."""
    n = len(b)
    # Form augmented matrix [A | b]
    aug = [row[:] + [bi] for row, bi in zip(A, b)]

    for col in range(n):
        # Pivot selection for numerical stability
        max_row = max(range(col, n), key=lambda r: abs(aug[r][col]))
        aug[col], aug[max_row] = aug[max_row], aug[col]

        pivot = aug[col][col]
        if abs(pivot) < 1e-12:
            raise ValueError("System matrix is singular or near-singular.")

        # Scale pivot row to leading 1
        for j in range(col, n + 1):
            aug[col][j] /= pivot

        # Zero out other rows
        for r in range(n):
            if r != col:
                factor = aug[r][col]
                for j in range(col, n + 1):
                    aug[r][j] -= factor * aug[col][j]

    return [aug[i][n] for i in range(n)]


class NormalEquationTwoFeatureModel:
    """Closed-form two-feature regression: expenses = w0 + w1*bmi + w2*age."""

    def __init__(self) -> None:
        self.w0: float = 0.0
        self.w1: float = 0.0
        self.w2: float = 0.0

    def fit(self, bmi: List[float], age: List[float], y: List[float]) -> None:
        n = len(y)
        # Compute components of X^T X
        sum_1 = float(n)
        sum_bmi = sum(bmi)
        sum_age = sum(age)
        sum_bmi2 = sum(b**2 for b in bmi)
        sum_age2 = sum(a**2 for a in age)
        sum_bmi_age = sum(b * a for b, a in zip(bmi, age))

        XtX = [
            [sum_1, sum_bmi, sum_age],
            [sum_bmi, sum_bmi2, sum_bmi_age],
            [sum_age, sum_bmi_age, sum_age2],
        ]

        # Compute components of X^T y
        sum_y = sum(y)
        sum_bmi_y = sum(b * yi for b, yi in zip(bmi, y))
        sum_age_y = sum(a * yi for a, yi in zip(age, y))
        Xty = [sum_y, sum_bmi_y, sum_age_y]

        weights = _gauss_jordan_solve(XtX, Xty)
        self.w0, self.w1, self.w2 = weights[0], weights[1], weights[2]

    def predict(self, bmi: List[float], age: List[float]) -> List[float]:
        return [self.w0 + self.w1 * b + self.w2 * a for b, a in zip(bmi, age)]


# ---------------------------------------------------------------------------
# 5. Model 3: Two-Feature Gradient Descent (with Feature Scaling)
# ---------------------------------------------------------------------------
class GradientDescentTwoFeatureModel:
    """Iterative two-feature regression with standardizing transform."""

    def __init__(self, lr: float = 0.05, epochs: int = 10000) -> None:
        self.lr = lr
        self.epochs = epochs
        self.scaler_bmi = StandardScaler()
        self.scaler_age = StandardScaler()
        self.w0: float = 0.0
        self.w1: float = 0.0
        self.w2: float = 0.0

    def fit(self, bmi: List[float], age: List[float], y: List[float]) -> None:
        z_bmi = self.scaler_bmi.fit_transform(bmi)
        z_age = self.scaler_age.fit_transform(age)
        n = len(y)

        # Initialize scaled weights
        w0_scaled = 0.0
        w1_scaled = 0.0
        w2_scaled = 0.0

        for _ in range(self.epochs):
            # Compute vectorized predictions & residuals on scaled inputs
            errors = [
                (w0_scaled + w1_scaled * zb + w2_scaled * za) - yi
                for zb, za, yi in zip(z_bmi, z_age, y)
            ]

            grad_w0 = (2.0 / n) * sum(errors)
            grad_w1 = (2.0 / n) * sum(err * zb for err, zb in zip(errors, z_bmi))
            grad_w2 = (2.0 / n) * sum(err * za for err, za in zip(errors, z_age))

            w0_scaled -= self.lr * grad_w0
            w1_scaled -= self.lr * grad_w1
            w2_scaled -= self.lr * grad_w2

        # Convert back to raw feature space
        self.w1 = w1_scaled / self.scaler_bmi.std
        self.w2 = w2_scaled / self.scaler_age.std
        self.w0 = w0_scaled - (self.w1 * self.scaler_bmi.mean) - (self.w2 * self.scaler_age.mean)

    def predict(self, bmi: List[float], age: List[float]) -> List[float]:
        return [self.w0 + self.w1 * b + self.w2 * a for b, a in zip(bmi, age)]


# ---------------------------------------------------------------------------
# Main Execution Pipeline
# ---------------------------------------------------------------------------
def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Could not locate dataset at: {CSV_PATH}")

    bmi, age, expenses = load_dataset(CSV_PATH)
    print(f"\n=======================================================")
    print(f" Loaded {len(expenses)} observations from: {CSV_PATH.name}")
    print(f"=======================================================\n")

    # 1. Train Baseline
    baseline = BaselineBMIModel()
    baseline.fit(bmi, expenses)
    m_base = compute_metrics(expenses, baseline.predict(bmi))

    # 2. Train Two-Feature Normal Equation
    ne_model = NormalEquationTwoFeatureModel()
    ne_model.fit(bmi, age, expenses)
    m_ne = compute_metrics(expenses, ne_model.predict(bmi, age))

    # 3. Train Two-Feature Gradient Descent
    gd_model = GradientDescentTwoFeatureModel(lr=LEARNING_RATE, epochs=EPOCHS)
    gd_model.fit(bmi, age, expenses)
    m_gd = compute_metrics(expenses, gd_model.predict(bmi, age))

    # Format Results
    table_rows = [
        {
            "Model": "Baseline (BMI only, OLS)",
            "w0": f"{baseline.w0:.4f}",
            "w1 (bmi)": f"{baseline.w1:.4f}",
            "w2 (age)": "-",
            "MSE": f"{m_base['MSE']:.2f}",
            "RMSE": f"{m_base['RMSE']:.2f}",
            "MAE": f"{m_base['MAE']:.2f}",
            "R2": f"{m_base['R2']:.4f}",
        },
        {
            "Model": "Two-Feature (Normal Equation)",
            "w0": f"{ne_model.w0:.4f}",
            "w1 (bmi)": f"{ne_model.w1:.4f}",
            "w2 (age)": f"{ne_model.w2:.4f}",
            "MSE": f"{m_ne['MSE']:.2f}",
            "RMSE": f"{m_ne['RMSE']:.2f}",
            "MAE": f"{m_ne['MAE']:.2f}",
            "R2": f"{m_ne['R2']:.4f}",
        },
        {
            "Model": "Two-Feature (Gradient Descent)",
            "w0": f"{gd_model.w0:.4f}",
            "w1 (bmi)": f"{gd_model.w1:.4f}",
            "w2 (age)": f"{gd_model.w2:.4f}",
            "MSE": f"{m_gd['MSE']:.2f}",
            "RMSE": f"{m_gd['RMSE']:.2f}",
            "MAE": f"{m_gd['MAE']:.2f}",
            "R2": f"{m_gd['R2']:.4f}",
        },
    ]

    # Display Terminal Table
    headers = ["Model", "w0", "w1 (bmi)", "w2 (age)", "MSE", "RMSE", "MAE", "R2"]
    row_fmt = "{:<32} | {:>12} | {:>10} | {:>10} | {:>14} | {:>10} | {:>10} | {:>8}"
    print(row_fmt.format(*headers))
    print("-" * 118)
    for r in table_rows:
        print(row_fmt.format(*(r[h] for h in headers)))

    # Export to CSV
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(table_rows)

    print(f"\n[+] Results successfully exported to: {OUTPUT_CSV.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()