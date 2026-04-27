# ML Orientation — Loan Default Predictor

## 1. What is Machine Learning?

Machine Learning is a method of programming where instead of writing explicit rules,
we let the computer learn patterns from historical data.

### Traditional Programming vs ML for Loan Approval

**Traditional (rule-based):**
```
IF credit_score > 700 AND dti_ratio < 0.3 AND months_employed > 24:
    APPROVE
ELSE:
    REJECT
```
Problem: Who decides the thresholds? What if there are 16 interacting features?
A human cannot write rules for every combination.

**Machine Learning approach:**
- Give the model 255,347 past loan outcomes (Default vs No Default)
- The model learns which feature combinations predict default
- It finds patterns like: "borrowers with CreditScore < 400, DTI > 0.7,
  and Unemployed status default 5x more often"
- These patterns are too complex for a human to manually code

## 2. End-to-End ML Workflow for This Project

```
Loan_default.csv (255K rows, 18 columns)
    │
    ▼
┌─────────────────────┐
│  DATA CLEANING       │  Drop LoanID
│  (01_data_cleaning)  │  Inspect for nulls, types
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  EDA                 │  Histograms, class imbalance chart,
│  (02_eda)           │  correlation heatmap, outlier detection
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  TRAIN-TEST SPLIT    │  80/20 stratified split
│  (03_split)         │  Preserves ~88/12 class ratio in both sets
└────────┬────────────┘
         │
    ┌────┴────┐
    ▼         ▼
 TRAIN       TEST (locked — never touched until evaluation)
    │
    ▼
┌─────────────────────┐
│  PREPROCESSING       │  StandardScaler for numerical
│  PIPELINE            │  OneHotEncoder for categorical
│  (ColumnTransformer) │  SimpleImputer for safety
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  MODEL TRAINING      │  LogisticRegression, KNN, DecisionTree
│  + SMOTE (optional)  │  GridSearchCV / RandomizedSearchCV
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  EVALUATION          │  F1 (class 1), Precision, Recall
│                      │  Confusion Matrix, classification_report
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  PICKLE SAVE         │  Entire pipeline saved as .pkl
│  (preprocessor+model)│
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  STREAMLIT APP       │  User inputs 16 features
│  (app/app.py)        │  Model returns default probability
└─────────────────────┘
```

## 3. Key Decisions Made in This Project

| Decision | Choice | Why |
|----------|--------|-----|
| Metric | F1 on class 1 | Accuracy is misleading with 88/12 class split |
| Split strategy | Stratified | Preserves class ratio in train and test |
| Pipeline | sklearn Pipeline | Prevents data leakage (scaler fit on train only) |
| SMOTE | Applied on train only | Synthetic samples must never contaminate test set |
| Serialization | Pickle full pipeline | Ensures preprocessor + model are always in sync |
