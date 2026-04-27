# Repository Analysis — handson-ml3

## Repo: ageron/handson-ml3

**URL:** https://github.com/ageron/handson-ml3

This is a companion repository for the book "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow" by Aurélien Géron — one of the most widely-used ML learning resources. It uses Python and scikit-learn throughout, making it directly relevant to our project.

## Structure Overview

| Path | Purpose |
|------|---------|
| `notebooks/` | Jupyter notebooks for each chapter (classification, regression, etc.) |
| `datasets/` | Helper functions for downloading and preparing datasets |
| `images/` | Generated plots and figures used in notebooks |
| `requirements.txt` | Pinned Python dependencies |
| `README.md` | Setup instructions and chapter index |

## How Training and Inference are Separated

- Training is done entirely within Jupyter notebooks
- Models are trained, evaluated, and compared within the same notebook
- No separate inference module or deployment code
- No model serialization for production use
- This is a **limitation** we will address in our project

## What We Will Copy vs. Do Differently

| Aspect | Their Approach | Our Approach | Why Different |
|--------|---------------|--------------|---------------|
| Folder structure | Flat notebooks per chapter | `src/` module with `__init__.py` | Better imports, testability, modularity |
| Preprocessing | Manual steps inline in notebook | sklearn ColumnTransformer + Pipeline | Reproducible, prevents data leakage |
| Model saving | Not covered in repo | Pickle entire pipeline | Ensures preprocessor travels with model |
| Deployment | None | Streamlit app | Demonstrates end-to-end ML system |
| Imbalance handling | Discussed conceptually | SMOTE + class_weight applied | Critical for our 88/12 class split |
| Config | Hardcoded values in cells | `src/config.py` | Single source of truth for feature lists, paths |
| Imports | Standard notebook style | `pip install -e .` editable install | Works from any directory, no path hacks |

## Key Takeaway

A good ML repo separates concerns: data loading, preprocessing, training, evaluation,
and inference should live in different files. Constants (feature lists, paths) should be
centralized in a config file. The handson-ml3 repo excels at education but lacks
production-readiness — we will build our project with both goals in mind.
