# 🏦 Loan Default Predictor

> Binary classification model that predicts whether a borrower will default on a loan.

**Team:** Uday & Vaibhav (Team-SLIP)  
**Dataset:** [nikhil1e9/loan-default](https://www.kaggle.com/datasets/nikhil1e9/loan-default) (Kaggle)  
**Curriculum:** Sections 5.1–5.21, 5.25–5.46 (skipping 5.22–5.24 regression metrics)

## 🎯 What This Does

Takes 16 borrower features (age, income, credit score, employment, etc.) and predicts:
- **0 = No Default** (safe to lend)
- **1 = Default** (high risk)

Primary metric: **F1-score on class 1** (defaulters). Accuracy is banned as primary metric due to 88/12 class imbalance.

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.9+ | Language |
| pandas, numpy | Data manipulation |
| scikit-learn | ML pipeline, models, evaluation |
| imbalanced-learn | SMOTE for class imbalance |
| Streamlit | Web app deployment |
| matplotlib, seaborn | Visualization |

## 📂 Project Structure

```
├── app/
│   └── app.py              # Streamlit web app
├── data/
│   ├── raw/                 # Raw dataset (gitignored)
│   └── processed/           # Train/test splits (gitignored)
├── docs/
│   ├── ml_orientation.md    # ML concepts and workflow
│   ├── repo_analysis.md     # Sample repo analysis
│   ├── project_plan.md      # MVP definition and timeline
│   ├── HLD.md               # High-level design
│   ├── LLD.md               # Low-level design
│   ├── environment_setup.md # Setup instructions
│   ├── model_documentation.md # Model card with limitations
│   ├── model_results.csv    # Model comparison results
│   └── eda_plots/           # EDA visualizations
├── models/                  # Trained model pickle (gitignored)
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_train_test_split.ipynb
│   ├── 04_preprocessing_pipeline.ipynb
│   ├── 05_baseline.ipynb
│   ├── 06_logistic_regression.ipynb
│   ├── 07_imbalance.ipynb
│   ├── 08_knn_and_tree.ipynb
│   ├── 09_hyperparameter_tuning.ipynb
│   ├── 10_model_comparison.ipynb
│   └── 11_save_and_load.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py            # Feature lists, paths, constants
│   ├── data_loader.py       # CSV loading
│   ├── preprocess.py        # ColumnTransformer builder
│   ├── train.py             # Pipeline training
│   └── inference.py         # Model loading and prediction
├── setup.py                 # Editable install for src/
├── requirements.txt         # Pinned dependencies
└── .gitignore
```

## 🚀 Quick Start

```bash
# Clone
git clone https://github.com/kalviumcommunity/Team-SLIP.git
cd Team-SLIP

# Setup
pip install -r requirements.txt
pip install -e .

# Download dataset from Kaggle and place in data/raw/
# Run the master pipeline notebook to train the model:
# notebooks/master_pipeline.ipynb
# Then launch the app:
streamlit run app/app.py
```

## 📊 Models Compared

| Model | Description |
|-------|-------------|
| Dummy Baseline | Always predicts majority class |
| Logistic Regression | Linear classifier |
| LR + class_weight | Weighted for imbalance |
| LR + SMOTE | Synthetic minority oversampling |
| KNN (GridSearch) | K-nearest neighbors |
| Decision Tree (Grid) | GridSearchCV over max_depth |
| Decision Tree (Randomized) | RandomizedSearchCV |

See `docs/model_results.csv` for full comparison.

## ⚠️ Key Design Decisions

1. **F1 on class 1** as primary metric (accuracy banned)
2. **SMOTE on training data only** — test set is never touched
3. **Entire pipeline pickled** — preprocessor + model always in sync
4. **Stratified splits** — class ratio preserved in train/test
5. **Editable install** — `pip install -e .` for reliable imports

## 📄 Documentation

- [Model Documentation](docs/model_documentation.md) — Full model card with limitations
- [HLD](docs/HLD.md) — System architecture
- [LLD](docs/LLD.md) — Function specs and preprocessing details
- [Environment Setup](docs/environment_setup.md) — Installation guide
