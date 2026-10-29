# Executive Memo: Logistic Regression from Scores to Decisions
**Author:** Ignacio Flores III  
**Course:** MBSA 265 — Machine Learning Pathway  
**Dataset:** Wisconsin Diagnostic Breast Cancer (scikit-learn bundled)  

---

## 1. Mathematical Foundations & Interactive Explorer (Part 1)

In binary logistic regression, an unconstrained linear score $z = \beta_0 + \beta_1 x_1 + \beta_2 x_2$ is mapped onto an open probability interval $(0, 1)$ using the standard sigmoid function:
$$p = \sigma(z) = \frac{1}{1 + e^{-z}}$$

Predictions are generated via a decision policy: $\hat{y} = 1$ if $p \ge \tau$, else $0$, where $\tau$ denotes the classification threshold. The four evaluated benchmark configurations show:

| Case | Intercept $\beta_0$ | Coefficient $\beta_1$ | Input $x$ | Threshold $\tau$ | Score $z$ | Probability $p$ | Predicted Class |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A** | -1.0 | 0.8 | 2.0 | 0.50 | 0.6000 | 0.6457 | 1 |
| **B** | -1.0 | 0.8 | 2.0 | 0.70 | 0.6000 | 0.6457 | 0 |
| **C** | -1.0 | -0.8 | 2.0 | 0.50 | -2.6000 | 0.0691 | 0 |
| **D** | 1.0 | 0.0 | 2.0 | 0.50 | 1.0000 | 0.7311 | 1 |

* **Case A Calculation:** $z = -1.0 + 0.8(2.0) = 0.6000$. Probability $p = \frac{1}{1 + e^{-0.6000}} = \frac{1}{1 + 0.54881} \approx 0.6457$. Because $0.6457 \ge 0.50$, the predicted class is 1.
* **A vs. B Dynamics:** Both instances possess identical input features, scores ($z = 0.6000$), and continuous probabilities ($p = 0.6457$). Adjusting the decision threshold from $0.50$ to $0.70$ alters only the hard decision boundary, reclassifying the observation from positive to negative without modifying underlying model confidence.
* **Coefficient Inversion (A vs. C):** Negating $\beta_1$ from $+0.8$ to $-0.8$ inverses the relationship: larger values of $x$ systematically penalize the log-odds ($z = -2.6000$), causing $p$ to collapse to $0.0691$.
* **Zero Slope Invariance (Case D):** With $\beta_1 = 0$, $z = \beta_0 + 0(x) = 1.0000$ for all real $x$. The sigmoid curve is flat across feature space, maintaining a fixed probability of $p \approx 0.7311$.

---

## 2. Leakage-Free Experimental Protocol (Part 2)

* **Source Attribution:** Derived from the digitized cell nuclei characteristics of fine needle aspirates (FNA) of breast masses (Wolberg, Street, & Mangasarian, 1995; UCI Machine Learning Repository / Scikit-learn).
* **Target Encoding:** Recoded to ensure clinical alignment: **1 = malignant (positive event)** and **0 = benign (negative event)**. The full dataset contains 569 instances (357 benign, 212 malignant).
* **Predictor Space:** Restricted to two features: `mean radius` and `mean texture`. No missing values exist.
* **Data Partitioning:** To guarantee strict separation, row indices were partitioned sequentially with stratified sampling (`random_state=42`):
  * Development set: 80% (455 rows) vs. Test set: 20% (114 rows: 72 benign, 42 malignant).
  * Development set further partitioned into Train: 75% (341 rows: 214 benign, 127 malignant) and Validation: 25% (114 rows: 71 benign, 43 malignant).
  * Verification confirmed zero set overlap ($\text{Train} \cap \text{Val} \cap \text{Test} = \emptyset$) and complete coverage ($\sum = 569$).
* **Standardization:** `StandardScaler` was fitted exclusively on the 341 training rows and applied identically across validation and test partitions, preventing statistical leakage.

---

## 3. Gradient Descent Optimization & Model Equivalence (Parts 3 & 4)

Binary cross-entropy loss (negative log-likelihood) serves as the objective function:
$$\mathcal{L}(\beta) = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \log(\hat{p}_i) + (1 - y_i) \log(1 - \hat{p}_i) \right]$$

Minimizing cross-entropy directly maximizes the joint likelihood of observed independent Bernoulli trials. Optimization utilized full-batch gradient descent for 20,000 epochs at $\alpha = 0.1$:

| Parameter | Manual NumPy (GD) | Scikit-learn (`C=inf`) | Absolute Delta |
| :--- | :---: | :---: | :---: |
| **Intercept ($\beta_0$)** | -0.727172 | -0.727172 | $< 10^{-6}$ |
| **Coefficient $\beta_1$ (`mean radius`)** | +3.656794 | +3.656794 | $< 10^{-6}$ |
| **Coefficient $\beta_2$ (`mean texture`)** | +1.009872 | +1.009872 | $< 10^{-6}$ |

* **Validation Probability Discrepancy:** The mean absolute difference across all validation observations was $1.186417 \times 10^{-10}$, proving mathematical equivalence between the NumPy gradient implementation and Scikit-learn's unregularized L-BFGS solver.
* **Log-Odds Interpretation:** For $\beta_1$ (`mean radius`), holding texture constant, each 1-standard-deviation increase in mean radius increases the log-odds of malignancy by $3.6568$ units ($\text{Odds Ratio} = e^{3.6568} \approx 38.74$).

---

## 4. Threshold Selection & Comparative Evaluation (Part 5)

Validation evaluation evaluated candidate thresholds across $\{0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80\}$:

* **Selected Policy:** The validation set achieved its maximum $F_1$ score ($0.831169$) at **$\tau = 0.60$**.
* **Test Set Performance Breakdown (114 samples: 72 Benign, 42 Malignant):**
  * **Baseline (`strategy="prior"`):** Predicts class 0 for all samples at $\tau = 0.50$. Accuracy is $63.16\%$ (72/114), but Recall is $0.0000$ and $F_1$ is $0.0000$. It completely fails to detect malignant cases despite deceptively moderate accuracy.
  * **Default Policy ($\tau = 0.50$):** $\text{TN}=68, \text{FP}=4, \text{FN}=11, \text{TP}=31$. Accuracy: $86.84\%$, Precision: $88.57\%$, Recall: $73.81\%$, $F_1$: $0.8052$.
  * **Selected Policy ($\tau = 0.60$):** $\text{TN}=69, \text{FP}=3, \text{FN}=14, \text{TP}=28$. Accuracy: $85.09\%$, Precision: $90.32\%$, Recall: $66.67\%$, $F_1$: $0.7671$.
* **Analysis:** Shifting the threshold to $0.60$ reduced false positives (from 4 to 3) and raised precision to $90.32\%$, but incurred 3 additional false negatives on the test set, reducing recall. The log loss ($0.255657$) remained completely unchanged because threshold shifts reallocate binary labels without modifying underlying continuous probabilities.

---

## 5. Clinical Safety & Scope Limitations (Part 6)

This model is strictly an academic demonstration and is **not clinically viable**:
1. **False-Negative Risk:** In oncology screening, false negatives (misclassifying malignant tumors as benign) carry catastrophic risk, delaying life-saving intervention. Clinical thresholds prioritize sensitivity/recall over precision.
2. **Omitted Variable Bias & Feature Scarcity:** Utilizing only two cell nucleus measurements ignores critical prognostic dimensions (e.g., vascularity, concavity, margins, patient medical history).
3. **Generalization Constraints:** The 569-case dataset lacks external demographic, institutional, and geographic validation across varied imaging hardware.

---

## 6. Research & Engineering Process

* **Numerical Stability:** To avoid overflow/underflow when evaluating exponents and log losses, `scipy`/`numpy` stable expansions (`np.logaddexp`) were implemented.
* **Disjoint Index Integrity:** Stratification was executed in two explicit steps (`train_test_split` on row indices), and validated with assert statements to confirm zero data leakage.

Unregularized vs. Regularized Formulation:
In Scikit-learn, C=np.inf disables the penalty. Testing $L_2$-penalty strengths (e.g., $C \in \{0.01, 0.1, 1.0, 10.0\}$) shows how ridge regularization shrinks $\beta_1$ and $\beta_2$ toward zero.   Feature Collinearity & Shrinkage:
Because mean radius has a strong predictive signal ($\beta_1 \approx 3.66$), imposing an $L_2$ penalty reduces parameter variance and limits potential overfitting to extreme train-partition boundaries.