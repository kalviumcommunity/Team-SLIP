# Low-Level Design (LLD) — Loan Default Predictor

## 1. Constants (src/config.py)

| Constant | Type | Value |
|----------|------|-------|
| `NUMERICAL_FEATURES` | `list[str]` | `['Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed', 'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio']` |
| `CATEGORICAL_FEATURES` | `list[str]` | `['Education', 'EmploymentType', 'MaritalStatus', 'HasMortgage', 'HasDependents', 'LoanPurpose', 'HasCoSigner']` |
| `TARGET` | `str` | `'Default'` |
| `ID_COLUMN` | `str` | `'LoanID'` |
| `MODEL_PATH` | `str` | `'models/final_model.pkl'` |
| `DATA_RAW_PATH` | `str` | `'data/raw/Loan_default.csv'` |
| `DATA_PROCESSED_PATH` | `str` | `'data/processed/clean_df.csv'` |
| `RESULTS_PATH` | `str` | `'docs/model_results.csv'` |
| `RANDOM_STATE` | `int` | `42` |
| `TEST_SIZE` | `float` | `0.2` |

## 2. Function Specifications

### src/data_loader.py

| Function | Parameters | Returns | Description |
|----------|-----------|---------|-------------|
| `load_data` | `filepath: str` | `pd.DataFrame` | Reads CSV file, returns raw DataFrame. Prints shape on load. |

### src/preprocess.py

| Function | Parameters | Returns | Description |
|----------|-----------|---------|-------------|
| `build_preprocessor` | None | `ColumnTransformer` | Returns unfitted ColumnTransformer with imputation + scaling + encoding. Does NOT fit — fitting happens inside the Pipeline during training. |

### src/train.py

| Function | Parameters | Returns | Description |
|----------|-----------|---------|-------------|
| `train_model` | `X_train: pd.DataFrame`, `y_train: pd.Series`, `model: BaseEstimator`, `preprocessor: ColumnTransformer` | `Pipeline` | Wraps preprocessor + model in a Pipeline, fits on training data, returns fitted Pipeline. |

### src/inference.py

| Function | Parameters | Returns | Description |
|----------|-----------|---------|-------------|
| `load_model` | `model_path: str` | `Pipeline` | Loads a pickled Pipeline from disk. |
| `predict` | `pipeline: Pipeline`, `input_df: pd.DataFrame` | `dict` | Returns `{'prediction': int, 'probability': float}` for single-row input. |

## 3. Preprocessing Pipeline Structure (Exact Order)

```python
ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), NUMERICAL_FEATURES),

        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), CATEGORICAL_FEATURES)
    ]
)
```

Order: Impute → Scale/Encode → ColumnTransformer combines both → Pipeline wraps with Model

**Note:** `handle_unknown='ignore'` is critical. At inference time, a new loan might have a
category not seen during training. Without this, the model crashes.

**Note:** `sparse_output=False` requires sklearn ≥ 1.2. Our pinned version (1.5.1) supports this.

## 4. File Paths for All Artifacts

| Artifact | Path | Created By |
|----------|------|------------|
| Raw dataset | `data/raw/Loan_default.csv` | Manual download (gitignored) |
| Cleaned CSV | `data/processed/clean_df.csv` | `notebooks/01_data_cleaning.ipynb` |
| X_train | `data/processed/X_train.csv` | `notebooks/03_train_test_split.ipynb` |
| X_test | `data/processed/X_test.csv` | `notebooks/03_train_test_split.ipynb` |
| y_train | `data/processed/y_train.csv` | `notebooks/03_train_test_split.ipynb` |
| y_test | `data/processed/y_test.csv` | `notebooks/03_train_test_split.ipynb` |
| EDA histograms | `docs/eda_plots/hist_numerical.png` | `notebooks/02_eda.ipynb` |
| Class imbalance | `docs/eda_plots/class_imbalance.png` | `notebooks/02_eda.ipynb` |
| Boxplot income | `docs/eda_plots/boxplot_income.png` | `notebooks/02_eda.ipynb` |
| Correlation heatmap | `docs/eda_plots/correlation_heatmap.png` | `notebooks/02_eda.ipynb` |
| Categorical dists | `docs/eda_plots/categorical_distributions.png` | `notebooks/02_eda.ipynb` |
| Feature importance | `docs/eda_plots/feature_importance.png` | `notebooks/08_knn_and_tree.ipynb` |
| Model comparison chart | `docs/eda_plots/model_comparison.png` | `notebooks/10_model_comparison.ipynb` |
| Model results CSV | `docs/model_results.csv` | `notebooks/10_model_comparison.ipynb` |
| Final model | `models/final_model.pkl` | `notebooks/11_save_and_load.ipynb` |

## 5. Evaluation Matrix

| Model | Accuracy | Precision (1) | Recall (1) | F1 (1) | Confusion Matrix |
|-------|----------|---------------|------------|--------|-----------------|
| Dummy Baseline | ✓ | ✓ | ✓ | ✓ | ✓ |
| Logistic Regression | ✓ | ✓ | ✓ | ✓ | ✓ |
| LR + class_weight | ✓ | ✓ | ✓ | ✓ | ✓ |
| LR + SMOTE | ✓ | ✓ | ✓ | ✓ | ✓ |
| KNN (best k) | ✓ | ✓ | ✓ | ✓ | ✓ |
| Decision Tree (GridSearch) | ✓ | ✓ | ✓ | ✓ | ✓ |
| DT (RandomizedSearch) | ✓ | ✓ | ✓ | ✓ | ✓ |
