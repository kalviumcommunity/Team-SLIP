# Project Plan — Loan Default Predictor

## MVP Definition

### In Scope (MVP)
- Binary classification: Default (1) vs No Default (0)
- 16 features (9 numerical, 7 categorical) from nikhil1e9/loan-default dataset
- Models: Logistic Regression, KNN, Decision Tree
- Handling class imbalance: class_weight='balanced' and SMOTE
- Evaluation: F1 on class 1 as primary metric
- Deployment: Streamlit web app with input form and risk prediction
- Serialization: pickle entire pipeline (preprocessor + model)

### Out of Scope
- Deep learning / neural networks
- Real-time data ingestion from any API
- Model monitoring / drift detection in production
- Ensemble methods (Random Forest, XGBoost) — future enhancement
- Feature engineering beyond the 16 existing columns
- Multi-class classification

## 10-Day Timeline

| Day | Owner | Goal | Deliverable |
|-----|-------|------|-------------|
| 1 | Both | Milestones 1–3: Orientation, repo analysis, project plan | `docs/ml_orientation.md`, `docs/repo_analysis.md`, `docs/project_plan.md` |
| 2 | Both | Milestones 4–5: HLD and LLD | `docs/HLD.md`, `docs/LLD.md` |
| 3 | Uday | Milestones 6–7: src/ modules + venv setup | `src/*.py`, `setup.py`, `requirements.txt` |
| 4 | Vaibhav | Milestone 8: Data cleaning notebook | `notebooks/01_data_cleaning.ipynb` |
| 5 | Vaibhav | Milestone 9: EDA notebook | `notebooks/02_eda.ipynb` |
| 6 | Uday | Milestones 10–11: Train-test split + preprocessing pipeline | `notebooks/03_train_test_split.ipynb`, `notebooks/04_preprocessing_pipeline.ipynb` |
| 7 | Both | Milestones 12–13: Baseline + Logistic Regression | `notebooks/05_baseline.ipynb`, `notebooks/06_logistic_regression.ipynb` |
| 8 | Both | Milestones 14–15: Imbalance + KNN/Tree | `notebooks/07_imbalance.ipynb`, `notebooks/08_knn_and_tree.ipynb` |
| 9 | Both | Milestones 16–18: Tuning + comparison + pickle | `notebooks/09–11`, `docs/model_results.csv` |
| 10 | Both | Milestones 19–20: Streamlit polish + final documentation | `app/app.py`, `docs/model_documentation.md`, final `README.md` |

## Risk Log

| # | Risk | Impact | Likelihood | Mitigation |
|---|------|--------|------------|------------|
| 1 | Class imbalance (88/12) causes misleading accuracy | Model appears 88% accurate but catches 0% of defaulters | High | Use F1 on class 1 as primary metric. Never report only accuracy. |
| 2 | SMOTE applied on full data instead of train-only | Data leakage — synthetic test samples inflate metrics | High | Use `imblearn.pipeline.Pipeline` so SMOTE runs only inside `.fit()`. Triple-check test set shape doesn't change. |
| 3 | Pickle file doesn't include preprocessor | Inference fails because new data isn't transformed | Medium | Always pickle the full Pipeline object, never `model` alone. Verify by loading pickle and predicting on raw input. |
| 4 | Import errors between notebooks and src/ | Wastes debugging time during demo | Medium | Use `pip install -e .` via setup.py instead of fragile sys.path hacks. |
| 5 | KNN too slow on 255K rows | Training takes hours, blocks progress | Medium | Sample 30K rows for KNN only. Document this limitation explicitly. |
