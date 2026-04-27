"""
Configuration constants for the Loan Default Predictor.

All feature lists, file paths, and hyperparameters are centralized here.
No other file should hardcode these values.
"""

# === FEATURES ===
# 9 numerical features
NUMERICAL_FEATURES = [
    'Age',              # Borrower's age (18–69)
    'Income',           # Annual income
    'LoanAmount',       # Requested loan amount
    'CreditScore',      # Credit score (300–850)
    'MonthsEmployed',   # Months at current job
    'NumCreditLines',   # Number of open credit lines
    'InterestRate',     # Loan interest rate (%)
    'LoanTerm',         # Loan term in months (12, 24, 36, 48, 60)
    'DTIRatio',         # Debt-to-income ratio (0.0–1.0)
]

# 7 categorical features (includes binary Yes/No columns)
CATEGORICAL_FEATURES = [
    'Education',        # Bachelor's, Master's, High School, PhD
    'EmploymentType',   # Full-time, Part-time, Self-employed, Unemployed
    'MaritalStatus',    # Single, Married, Divorced
    'HasMortgage',      # Yes, No
    'HasDependents',    # Yes, No
    'LoanPurpose',      # Home, Auto, Education, Business, Other
    'HasCoSigner',      # Yes, No
]

TARGET = 'Default'
ID_COLUMN = 'LoanID'

# === FILE PATHS ===
MODEL_PATH = 'models/final_model.pkl'
DATA_RAW_PATH = 'data/raw/Loan_default.csv'
DATA_PROCESSED_PATH = 'data/processed/clean_df.csv'
RESULTS_PATH = 'docs/model_results.csv'

# === MODEL PARAMETERS ===
RANDOM_STATE = 42
TEST_SIZE = 0.2
