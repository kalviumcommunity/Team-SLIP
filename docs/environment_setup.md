# Environment Setup Guide

## Prerequisites
- Python 3.9 or higher
- pip (comes with Python)
- git

## Step 1 — Clone the repository

```bash
git clone https://github.com/kalviumcommunity/Team-SLIP.git
cd Team-SLIP
```

## Step 2 — Create a virtual environment

```bash
python3 -m venv venv
```

## Step 3 — Activate the virtual environment

**macOS / Linux:**
```bash
source venv/bin/activate
```

**Windows (PowerShell):**
```powershell
.\venv\Scripts\Activate.ps1
```

**Windows (cmd):**
```cmd
venv\Scripts\activate.bat
```

You should see `(venv)` in your terminal prompt.

## Step 4 — Install dependencies

```bash
pip install -r requirements.txt
pip install -e .  # Install src/ as editable package for reliable imports
```

## Step 5 — Verify installation

Run this one-liner to confirm all packages are installed correctly:

```bash
python -c "
import pandas; print(f'pandas: {pandas.__version__}')
import numpy; print(f'numpy: {numpy.__version__}')
import sklearn; print(f'scikit-learn: {sklearn.__version__}')
import imblearn; print(f'imbalanced-learn: {imblearn.__version__}')
import streamlit; print(f'streamlit: {streamlit.__version__}')
import matplotlib; print(f'matplotlib: {matplotlib.__version__}')
import seaborn; print(f'seaborn: {seaborn.__version__}')
from src.config import TARGET; print(f'src module working: TARGET={TARGET}')
print('All packages installed successfully!')
"
```

## Step 6 — Download the dataset

1. Go to https://www.kaggle.com/datasets/nikhil1e9/loan-default
2. Download `Loan_default.csv`
3. Place it in `data/raw/`

Note: `data/raw/` is gitignored — the dataset will NOT be pushed to GitHub.

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `pip install` fails on Windows | Try `python -m pip install -r requirements.txt` |
| `import imblearn` fails | Run `pip install imbalanced-learn` (NOT `pip install imblearn`) |
| `from src.config import ...` fails | Run `pip install -e .` from the project root |
| Jupyter not found | Run `pip install jupyter notebook` separately |
| `sparse_output=False` warning | Requires sklearn ≥ 1.2. Our pinned version (1.5.1) supports this. |
