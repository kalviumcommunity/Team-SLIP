# High-Level Design (HLD) — Loan Default Predictor

## 1. System Pipeline Diagram

```
┌──────────────┐     ┌─────────────────┐     ┌──────────────────┐
│  RAW DATA    │────▶│  DATA CLEANING   │────▶│  FEATURE         │
│  Loan_default│     │  - Drop LoanID   │     │  SELECTION       │
│  .csv        │     │  - Check types   │     │  - 9 numerical   │
│  255K rows   │     │  - Verify target │     │  - 7 categorical │
└──────────────┘     └─────────────────┘     └────────┬─────────┘
                                                       │
                                                       ▼
┌──────────────┐     ┌─────────────────┐     ┌──────────────────┐
│  PICKLE SAVE │◀────│  MODEL TRAINING  │◀────│  TRAIN-TEST      │
│  pipeline.pkl│     │  + SMOTE option  │     │  SPLIT           │
│  (full pipe) │     │  + GridSearchCV  │     │  - 80/20         │
└──────┬───────┘     └─────────────────┘     │  - stratify=y    │
       │                                      └──────────────────┘
       ▼
┌──────────────┐     ┌─────────────────┐
│  EVALUATION  │────▶│  STREAMLIT APP  │
│  - F1 (cls 1)│     │  - Input form   │
│  - Confusion │     │  - Risk display │
└──────────────┘     └─────────────────┘
```

## 2. Stage Input/Output Table

| Stage | Input | Output |
|-------|-------|--------|
| Data Loading | `Loan_default.csv` (255,347 rows, 18 cols) | Raw DataFrame |
| Data Cleaning | Raw DataFrame | Clean DataFrame (255,347 rows, 17 cols — LoanID dropped) |
| Feature Selection | Clean DataFrame | 16 features + 1 target |
| Train-Test Split | Clean DataFrame | X_train, X_test, y_train, y_test (80/20 stratified) |
| Preprocessing | X_train (raw features) | X_train_transformed (scaled + encoded) |
| Model Training | Transformed X_train + y_train | Fitted Pipeline object |
| Evaluation | Pipeline + X_test, y_test | Metrics dict (F1, precision, recall, confusion matrix) |
| Pickle Save | Fitted Pipeline | `models/final_model.pkl` file |
| Streamlit Inference | User inputs (16 features) | Default probability + risk label |

## 3. Tech Stack Justification

| Tool | Purpose | Why This Tool |
|------|---------|---------------|
| Python 3.10+ | Language | Industry standard for ML, richest ecosystem |
| pandas | Data manipulation | Best for tabular data, reads CSV natively |
| scikit-learn | ML pipeline + models | Industry standard; Pipeline prevents leakage |
| imbalanced-learn | SMOTE | Integrates with sklearn Pipeline; purpose-built for imbalanced datasets |
| Streamlit | Deployment | Fastest path to interactive ML app; Python-native, no frontend needed |
| matplotlib + seaborn | Visualization | Statistical plots (heatmaps, distributions); publication quality |
| pickle | Serialization | Built-in Python; saves entire pipeline as single file |
| Jupyter | Notebooks | Interactive exploration; visible outputs for review |

## 4. Evaluation Metrics — Justification

**Primary metric: F1-score on class 1 (Default)**

### Why NOT accuracy?

- Dataset is ~88% No Default, ~12% Default
- A model that predicts "No Default" for EVERY loan achieves ~88% accuracy
- But it catches ZERO defaulters — completely useless for risk assessment
- Accuracy is therefore **banned** as a primary metric in this project

### Why F1 on class 1?

- F1 = harmonic mean of Precision and Recall
- Precision (class 1): Of loans flagged as risky, how many actually defaulted?
- Recall (class 1): Of loans that actually defaulted, how many did we catch?
- F1 balances both — we want to catch defaulters (recall) without flagging too many good loans (precision)

### Secondary priority: Recall > Precision for class 1

- Missing a real defaulter (False Negative) is worse than a false alarm (False Positive)
- A missed default = real financial loss to the lender
- A false alarm = extra manual review, but no money lost
