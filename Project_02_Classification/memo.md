# Executive Memo: Logistic Regression from Scores to Decisions
**Author:** Ignacio Flores III  
**Course:** MSBA 265: Machine Learning Pathway  
**Project:** Project Assignment 2 – Binary Classification & Decision Thresholds  
**Scope:** Educational Modeling Prototype Only; Strictly Non-Clinical  

---

## 1. Score Calculation, Sigmoid Mapping, and Thresholding (Part 1)

Binary logistic regression maps a linear combination of predictors—the log-odds score $z$—into a posterior probability $p \in [0, 1]$ using the standard logistic sigmoid function:
$$z = \beta_0 + \beta_1 x, \quad p = \sigma(z) = \frac{1}{1 + e^{-z}}$$
Class assignments follow a decision policy: $\hat{y} = 1$ if $p \ge \tau$, else $0$, where $\tau$ denotes the classification threshold. The sigmoid curve is point-symmetric around its inflection point $(z=0, p=0.50)$, where the gradient $\frac{\partial p}{\partial z} = p(1-p)$ reaches its maximum of $0.25$.

### Four-Case Calculation Table
For Case A: $z = -1.0 + 0.8(2.0) = 0.6000$. Probability: $p = \frac{1}{1 + e^{-0.6000}} \approx 0.6457$. Since $0.6457 \ge 0.50$, predicted class is 1.

| Case | Intercept ($\beta_0$) | Slope ($\beta_1$) | Input ($x$) | Score ($z$) | Probability ($p$) | Threshold ($\tau$) | Predicted Class |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A** | -1.0 | 0.8 | 2.0 | 0.6000 | 0.6457 | 0.50 | 1 |
| **B** | -1.0 | 0.8 | 2.0 | 0.6000 | 0.6457 | 0.70 | 0 |
| **C** | -1.0 | -0.8 | 2.0 | -2.6000 | 0.0691 | 0.50 | 0 |
| **D** | 1.0 | 0.0 | 2.0 | 1.0000 | 0.7311 | 0.50 | 1 |

* **Case A vs. Case B:** The underlying probability remains identical ($p = 0.6457$). Raising $\tau$ from $0.50$ to $0.70$ alters only the decision boundary, flipping $\hat{y}$ from 1 to 0 without changing the model's confidence.
* **Coefficient Reversal (Case C):** Negating $\beta_1$ inverts the slope: higher $x$ lowers log-odds, driving $p \to 0$.
* **Zero Slope (Case D):** With $\beta_1 = 0$, $\frac{\partial z}{\partial x} = 0$. The feature provides zero signal; the model outputs a static baseline probability $p = \sigma(1.0) \approx 0.7311$ across all $x$.
* **Manual vs. Learned Parameters:** These four cases illustrate structural properties using hard-coded parameters. In empirical modeling, parameters are learned by optimizing an objective function over training data.

---

## 2. Dataset Provenance, Coding, and Leakage Prevention (Part 2)

* **Source Acknowledgement:** Data originates from the Wisconsin Diagnostic Breast Cancer (WDBC) dataset bundled in Scikit-learn (`sklearn.datasets.load_breast_cancer`), compiled by Wolberg, Street, and Mangasarian. Attributes describe cell nucleus morphology derived from digitized fine needle aspirate images.
* **Positive-Class Coding & Features:** Target labels were recoded as $y = (\text{data.target} == 0)\text{.astype}(\text{int})$, setting $1 = \text{Malignant}$ and $0 = \text{Benign}$. Predictors were strictly restricted to `mean radius` followed by `mean texture`.
* **Class Balance:** The sample contains 569 cases: 357 benign ($62.74\%$) and 212 malignant ($37.26\%$) with no missing entries.
* **Leakage-Free Partitioning:** A two-stage stratified split was performed with `random_state=42`: first isolating 114 test rows ($20\%$), then splitting development rows into 341 training ($75\%$) and 114 validation ($25\%$) rows. Stratification preserved class ratios across partitions. Sets are strictly disjoint ($\text{Train} \cap \text{Val} \cap \text{Test} = \emptyset$).
* **Scaling Hygiene:** `StandardScaler` was fitted exclusively on the 341 training rows. Validation and test vectors were transformed using training parameters $(\mu_{\text{train}}, \sigma_{\text{train}})$. Fitting on pooled data would leak distributional moments and produce optimistically biased validation scores.

---

## 3. Training Dynamics, Likelihood Connection, and Parameter Interpretability (Parts 3 & 4)

Minimizing mean binary cross-entropy (BCE):
$$J(\beta) = -\frac{1}{N} \sum_{i=1}^N \left[ y^{(i)} \ln(p^{(i)}) + (1 - y^{(i)}) \ln(1 - p^{(i)}) \right]$$
is mathematically equivalent to maximizing the log-likelihood of independent Bernoulli outcomes $\sum \ln P(Y = y^{(i)} \mid x^{(i)}; \beta)$. Full-batch gradient descent was executed with learning rate $\eta = 0.1$ for 20,000 epochs from $\beta = [0, 0, 0]^T$. The loss decreased smoothly from $\ln(2) \approx 0.693147$ to $0.250495$, confirming numerical convergence.

### Standardized Coefficient Equivalence

| Parameter | Feature | NumPy Manual Fit | Scikit-learn (`C=inf`) | Absolute Difference |
| :--- | :--- | :---: | :---: | :---: |
| $\beta_0$ | Intercept | -0.669527 | -0.669528 | $1.02 \times 10^{-6}$ |
| $\beta_1$ | Mean Radius | 3.664406 | 3.664408 | $2.14 \times 10^{-6}$ |
| $\beta_2$ | Mean Texture | 1.455246 | 1.455247 | $8.79 \times 10^{-7}$ |

The mean absolute difference across validation probabilities is $1.19 \times 10^{-10}$, proving mathematical parity.

* **Log-Odds Interpretation:** For $\beta_1 = 3.6644$, holding `mean texture` constant, a one-standard-deviation increase in `mean radius` ($3.52\text{ mm}$ on training data) increases the log-odds of malignancy by $3.6644$, multiplying the odds by $e^{3.6644} \approx 39.03$. This reflects a standardized statistical association, not a constant percentage increase in probability or a causal mechanism.

---

## 4. Benchmark Comparison and Baseline Limitations (Part 4)

The majority baseline (`DummyClassifier(strategy="prior")`) predicts the training base rate ($p = \frac{127}{341} \approx 0.3724$). At $\tau = 0.50$, it assigns all cases to benign ($\hat{y} = 0$). On the test set (71 Benign, 43 Malignant), it scores $62.28\%$ accuracy while identifying zero malignant samples ($\text{TP}=0, \text{FN}=43$). With a zero numerator and denominator, precision and recall default to $0.0000$ (`zero_division=0`). Accuracy alone masks total clinical failure in imbalanced cohorts.

* **Plain-Language Confusion Matrix (NumPy Default, Test Set):**
  * **True Negatives ($\text{TN}=68$):** Correctly classified 68 benign masses.
  * **False Positives ($\text{FP}=4$):** Flagged 4 healthy samples as malignant (prompting unnecessary diagnostic follow-ups).
  * **False Negatives ($\text{FN}=11$):** Failed to identify 11 malignant tumors, classifying them as benign.
  * **True Positives ($\text{TP}=31$):** Successfully detected 31 malignant tumors.

---

## 5. Threshold Optimization and Evaluation (Part 5)

Candidate thresholds $\{0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80\}$ were scanned on validation probabilities. Validation $F_1$ peaked at $\tau = 0.60$ ($F_1 = 0.8571$), driven by reducing validation false positives from 6 to 3.

### Test Partition Performance ($N=114$)

| Configuration | Policy ($\tau$) | TN | FP | FN | TP | Accuracy | Precision | Recall | $F_1$ | Log Loss |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline | Default (0.50) | 71 | 0 | 43 | 0 | 0.6228 | 0.0000 | 0.0000 | 0.0000 | 0.6609 |
| NumPy Fit | Default (0.50) | 68 | 4 | 11 | 31 | **0.8684** | 0.8857 | **0.7381** | **0.8052** | **0.2557** |
| NumPy Fit | Selected (0.60) | 69 | 3 | 14 | 28 | 0.8509 | **0.9032** | 0.6667 | 0.7671 | **0.2557** |

* **Trade-off Analysis:** Raising the threshold to $\tau = 0.60$ reduced test false positives from 4 to 3 (raising precision from $88.57\%$ to $90.32\%$), but increased false negatives from 11 to 14 (lowering recall from $73.81\%$ to $66.67\%$). 
* **Log Loss Invariance:** Shifting $\tau$ moves the decision plane across the probability landscape without altering posterior estimates $p^{(i)}$. Because log loss evaluates probability calibration rather than discrete labels, it remains invariant at $0.2557$.

---

## 6. Clinical Safety, Risk Asymmetry, and Limitations (Part 6)

1. **False-Negative Severity:** In clinical triage, false negatives can lead to delayed intervention for malignant disease, whereas false positives result in non-invasive imaging or secondary biopsies. Unweighted $F_1$ weights precision and recall symmetrically, which does not account for this asymmetric risk profile.
2. **Omitted Diagnostic Variables:** Using only two features ignores diagnostic dimensions like perimeter, concavity, and patient demographic history.
3. **Generalization Constraints:** The 569-patient historical cohort lacks external multi-institutional validation across modern mammographic and biopsy hardware. This educational prototype is not validated for clinical use.

---

## 7. Research and Engineering Process (Exempt from Word Count)

* **Numerical Stability:** Direct sigmoid computation ($\frac{1}{1 + e^{-z}}$) caused floating-point overflow during early training. Reformulating probability evaluation and binary cross-entropy using `np.logaddexp` resolved numerical instability across 20,000 iterations.
* **Regularization Comparisons:** Experiments with Scikit-learn's default $L_2$ penalty showed shrinkage of $\beta_1$ from $3.66$ to $0.42$ at high penalty settings. Unregularized optimization (`C=inf`) was retained to ensure direct mathematical alignment with the manual gradient descent model.
* **Pipeline Integrity:** Vectorized assertions verified that partitions were disjoint and that scaling statistics were derived strictly from training data.