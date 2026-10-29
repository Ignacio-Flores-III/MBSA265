import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# 1. Load Data
# ----------------------------------------------------------------------
# Update path if insurance.csv is in a data/ subfolder
csv_path = "data/insurance-premium-prediction/insurance.csv"
df = pd.read_csv(csv_path)

print(f"Loaded {len(df)} rows.")
print(df[['bmi', 'age', 'expenses']].head())

# ----------------------------------------------------------------------
# 2. Extract Features and Target
# ----------------------------------------------------------------------
# Features: bmi (x1), age (x2)
# Target: expenses (y)
bmi = df['bmi'].to_numpy(dtype=float)
age = df['age'].to_numpy(dtype=float)
y = df['expenses'].to_numpy(dtype=float)

n_samples = len(y)

# ----------------------------------------------------------------------
# 3. Build Design Matrix X with Bias Column (Intercept)
# ----------------------------------------------------------------------
# Matrix shape: (N, 3) where columns are [1, bmi, age]
ones = np.ones((n_samples, 1))
features = np.column_stack((bmi, age))
X = np.hstack((ones, features))

# ----------------------------------------------------------------------
# 4. Normal Equation: w = (X^T * X)^(-1) * X^T * y
# ----------------------------------------------------------------------
# General matrix solve using numpy.linalg.solve (more stable than explicit inverse)
# X_transpose_X * w = X_transpose_y
Xt = X.T
Xt_X = Xt @ X
Xt_y = Xt @ y

# Solve for weights [w0, w1, w2]
weights = np.linalg.solve(Xt_X, Xt_y)

w0, w1, w2 = weights[0], weights[1], weights[2]

print("\n--- Model Parameters ---")
print(f"Intercept (w0): {w0:.4f}")
print(f"BMI Coefficient (w1): {w1:.4f}")
print(f"Age Coefficient (w2): {w2:.4f}")
print(f"\nModel Equation: expenses = {w0:.2f} + {w1:.2f}*bmi + {w2:.2f}*age")

# ----------------------------------------------------------------------
# 5. Predictions and Model Evaluation
# ----------------------------------------------------------------------
y_pred = X @ weights

# Metrics: MSE, RMSE, R^2
residuals = y - y_pred
mse = np.mean(residuals ** 2)
rmse = np.sqrt(mse)

ss_tot = np.sum((y - np.mean(y)) ** 2)
ss_res = np.sum(residuals ** 2)
r2 = 1.0 - (ss_res / ss_tot)

print("\n--- Model Performance ---")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R^2): {r2:.4f}")

# ----------------------------------------------------------------------
# 6. Visualizations
# ----------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Actual vs. Predicted Expenses
axes[0].scatter(y, y_pred, alpha=0.5, color='teal', edgecolor='k')
axes[0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label='Perfect Fit')
axes[0].set_xlabel('Actual Expenses ($)')
axes[0].set_ylabel('Predicted Expenses ($)')
axes[0].set_title('Actual vs. Predicted Expenses')
axes[0].legend()
axes[0].grid(True, linestyle='--', alpha=0.5)

# Plot 2: Residuals Distribution
axes[1].hist(residuals, bins=30, color='darkorange', edgecolor='k', alpha=0.7)
axes[1].axvline(0, color='red', linestyle='--', linewidth=1.5, label='Zero Error')
axes[1].set_xlabel('Residual ($)')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Residuals Distribution')
axes[1].legend()
axes[1].grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig("two_feature_regression_plots.png", dpi=300)
print("\nSaved diagnostic plots to 'two_feature_regression_plots.png'.")
plt.show()