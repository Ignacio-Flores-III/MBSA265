# Breast Mass Classification - Logistic Regression

A binary logistic regression model predicting breast cancer diagnosis (malignant vs. benign) using standardized cell nucleus features.

---

## Push-Button Reproduction Steps

Follow these exact steps from a clean terminal to clone, set up the virtual environment, install dependencies, and execute the analysis.

### Windows (PowerShell)
```powershell
git clone https://github.com/Ignacio-Flores-III/MBSA265.git
cd MBSA265\Project_02_Classification
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r assignment_requirements.txt
python logistic_regression_analysis.py
```

### macOS / Linux (Terminal / zsh)
```bash
git clone https://github.com/Ignacio-Flores-III/MBSA265.git
cd MBSA265/Project_02_Classification
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r assignment_requirements.txt
python3 logistic_regression_analysis.py
```

---

## Expected Output

Running `python logistic_regression_analysis.py` will:
1. Load and preprocess the breast cancer biopsy dataset.
2. Standardize features and train the logistic regression classifier.
3. Output model performance metrics (Accuracy, ROC-AUC, Precision, Recall, Confusion Matrix) to the console.
4. Save analytical charts/reports into the `reports/` folder.
