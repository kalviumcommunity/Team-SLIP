"""
Loan Default Risk Predictor — Professional Edition
Features:
- Single Prediction with SHAP Explanations
- Batch Processing (Upload CSV)
- Decision History / Audit Log
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import time
from datetime import datetime

from src.inference import load_model, predict
from src.config import (
    MODEL_PATH, RESULTS_PATH, TARGET,
    NUMERICAL_FEATURES, CATEGORICAL_FEATURES
)

# Constants
LOG_FILE = "logs/decision_history.csv"

@st.cache_resource
def get_model():
    """Load the trained pipeline once and cache it."""
    return load_model(MODEL_PATH)

@st.cache_resource
def get_config():
    """Load threshold and config."""
    import pickle
    config_path = 'models/model_config.pkl'
    if os.path.exists(config_path):
        with open(config_path, 'rb') as f:
            return pickle.load(f)
    return {'threshold': 0.6, 'feature_engineering': True}

def add_features(df):
    """Add engineered features to match training."""
    df = df.copy()
    df['Loan_to_Income'] = df['LoanAmount'] / (df['Income'] + 1)
    df['Interest_x_Loan'] = df['InterestRate'] * df['LoanAmount']
    df['DTI_x_Loan'] = df['DTIRatio'] * df['LoanAmount']
    df['Credit_per_Line'] = df['CreditScore'] / (df['NumCreditLines'] + 1)
    df['Income_per_Month_Employed'] = df['Income'] / (df['MonthsEmployed'] + 1)
    return df

@st.cache_resource
def get_shap_explainer():
    """Create a SHAP TreeExplainer for the model's classifier."""
    try:
        import shap
        pipeline = get_model()
        classifier = pipeline.named_steps['classifier']
        explainer = shap.TreeExplainer(classifier)
        return explainer
    except Exception:
        return None

def log_decision(input_df, probability, threshold):
    """Log the decision to a CSV file for auditing."""
    log_df = input_df.copy()
    log_df['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_df['probability'] = round(probability, 4)
    log_df['decision'] = 'High Risk' if probability >= threshold else 'Low Risk'
    
    if os.path.exists(LOG_FILE):
        log_df.to_csv(LOG_FILE, mode='a', header=False, index=False)
    else:
        os.makedirs("logs", exist_ok=True)
        log_df.to_csv(LOG_FILE, index=False)

def main():
    st.set_page_config(page_title="Loan Default Risk Predictor", page_icon="None", layout="wide")
    st.title("Loan Default Risk Predictor")
    st.markdown("Advanced risk assessment for individual and batch loan applications.")

    try:
        model = get_model()
        config = get_config()
        threshold = config.get('threshold', 0.6)
    except FileNotFoundError:
        st.error(f"Model file not found at `{MODEL_PATH}`. Run the training pipeline first.")
        return

    tab1, tab2, tab3, tab4 = st.tabs(["Predict", "Batch Processing", "Model Performance", "Decision History"])

    # ===== TAB 1: INDIVIDUAL PREDICTION =====
    with tab1:
        st.header("Individual Borrower Assessment")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Loan Details")
            loan_amount = st.slider("Loan Amount ($)", 5000, 250000, 50000, step=5000)
            interest_rate = st.slider("Interest Rate (%)", 2.0, 25.0, 12.0, step=0.5)
            loan_term = st.selectbox("Loan Term (months)", [12, 24, 36, 48, 60])
            loan_purpose = st.selectbox("Loan Purpose", ['Home', 'Auto', 'Education', 'Business', 'Other'])
            has_cosigner = st.selectbox("Has Co-Signer?", ['Yes', 'No'])

        with col2:
            st.subheader("Borrower Profile")
            age = st.slider("Age", 18, 69, 35)
            income = st.slider("Annual Income ($)", 15000, 150000, 65000, step=5000)
            education = st.selectbox("Education", ["Bachelor's", "Master's", "High School", "PhD"])
            employment_type = st.selectbox("Employment Type", ['Full-time', 'Part-time', 'Self-employed', 'Unemployed'])
            marital_status = st.selectbox("Marital Status", ['Single', 'Married', 'Divorced'])

        with col3:
            st.subheader("Credit Profile")
            credit_score = st.slider("Credit Score", 300, 850, 650)
            months_employed = st.slider("Months Employed", 0, 120, 36)
            num_credit_lines = st.slider("Number of Credit Lines", 1, 4, 2)
            dti_ratio = st.slider("DTI Ratio", 0.0, 1.0, 0.3, step=0.01)
            has_mortgage = st.selectbox("Has Mortgage?", ['Yes', 'No'])
            has_dependents = st.selectbox("Has Dependents?", ['Yes', 'No'])

        st.markdown("---")

        if st.button("Assess Risk", type="primary", use_container_width=True):
            input_data = {
                'Age': [age], 'Income': [income], 'LoanAmount': [loan_amount],
                'CreditScore': [credit_score], 'MonthsEmployed': [months_employed],
                'NumCreditLines': [num_credit_lines], 'InterestRate': [interest_rate],
                'LoanTerm': [loan_term], 'DTIRatio': [dti_ratio], 'Education': [education],
                'EmploymentType': [employment_type], 'MaritalStatus': [marital_status],
                'HasMortgage': [has_mortgage], 'HasDependents': [has_dependents],
                'LoanPurpose': [loan_purpose], 'HasCoSigner': [has_cosigner],
            }
            input_df = pd.DataFrame(input_data)
            
            # Feature Engineering and Prediction
            input_proc = add_features(input_df) if config.get('feature_engineering') else input_df
            result = predict(model, input_proc)
            probability = result['probability']
            
            # Log decision
            log_decision(input_df, probability, threshold)

            # Result Section
            st.markdown("### Assessment Result")
            col_r1, col_r2 = st.columns([2, 1])
            with col_r1:
                if probability >= threshold:
                    st.error(f"**HIGH RISK OF DEFAULT** — Probability: {probability*100:.1f}%")
                    st.warning(f"Confidence is above threshold ({threshold*100:.0f}%). Rejected.")
                else:
                    st.success(f"**LOW RISK** — Default Probability: {probability*100:.1f}%")
                    st.info("Confidence is within safe limits. Approved.")
            with col_r2:
                st.metric("Risk Score", f"{probability*100:.1f}%")
                st.progress(probability)

            # SHAP Section
            st.markdown("---")
            st.markdown("### Decisions Explained (SHAP)")
            explainer = get_shap_explainer()
            if explainer:
                try:
                    import shap
                    import matplotlib.pyplot as plt
                    preprocessor = model.named_steps['preprocessor']
                    input_trans = preprocessor.transform(input_proc)
                    
                    # Robust Feature Name extraction
                    try:
                        feature_names = list(preprocessor.get_feature_names_out())
                    except Exception:
                        # Fallback for older sklearn versions
                        num_names = NUMERICAL_FEATURES + (['Loan_to_Income', 'Interest_x_Loan', 'DTI_x_Loan', 'Credit_per_Line', 'Income_per_Month_Employed'] if config.get('feature_engineering') else [])
                        cat_names = list(preprocessor.named_transformers_['cat'].named_steps['encoder'].get_feature_names_out(CATEGORICAL_FEATURES))
                        feature_names = num_names + cat_names

                    vals = explainer.shap_values(input_trans)
                    
                    # Handle XGBoost vs RF format
                    if isinstance(vals, list): 
                        val = vals[1][0] # RF returns list [neg, pos]
                    elif vals.ndim == 3: 
                        val = vals[0, :, 1] # Some tree models return (samples, features, classes)
                    else: 
                        val = vals[0] # XGBoost returns array of samples
                    
                    # Safety check: Trim or pad names to match data shape
                    if len(feature_names) != len(val):
                        feature_names = [f"Feature_{i}" for i in range(len(val))]

                    contrib = pd.DataFrame({'Feature': feature_names, 'Impact': val})
                    contrib['Abs'] = contrib['Impact'].abs()
                    top = contrib.sort_values('Abs', ascending=False).head(8)
                    
                    fig, ax = plt.subplots(figsize=(8, 4))
                    colors = ['#e74c3c' if x > 0 else '#2ecc71' for x in top['Impact']]
                    ax.barh(top['Feature'], top['Impact'], color=colors)
                    ax.set_title("Contribution to Decision")
                    st.pyplot(fig)
                except Exception as e:
                    st.info(f"SHAP chart loading... {e}")

    # ===== TAB 2: BATCH PROCESSING =====
    with tab2:
        st.header("Batch Loan Assessment")
        st.markdown("Upload a CSV file containing applicant data for bulk processing.")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file:
            batch_df = pd.read_csv(uploaded_file)
            st.write(f"Loaded {len(batch_df)} applicants.")
            
            if st.button("Process Batch"):
                try:
                    batch_proc = add_features(batch_df) if config.get('feature_engineering') else batch_df
                    results = model.predict_proba(batch_proc)[:, 1]
                    
                    batch_df['Default_Probability'] = np.round(results, 4)
                    batch_df['Risk_Category'] = ['High Risk' if p >= threshold else 'Low Risk' for p in results]
                    
                    st.success("Batch processing complete!")
                    st.dataframe(batch_df[['Default_Probability', 'Risk_Category'] + list(batch_df.columns[:-2])], use_container_width=True)
                    
                    csv = batch_df.to_csv(index=False).encode('utf-8')
                    st.download_button("Download Results", data=csv, file_name="loan_risk_report.csv", mime="text/csv")
                except Exception as e:
                    st.error(f"Error in batch processing: {e}. Ensure columns match the expected schema.")

    # ===== TAB 3: PERFORMANCE =====
    with tab3:
        st.header("Model Performance Summary")
        if os.path.exists(RESULTS_PATH):
            res_df = pd.read_csv(RESULTS_PATH).sort_values('Test F1 (class 1)', ascending=False)
            st.dataframe(res_df, use_container_width=True)
            
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.barh(res_df['Model'], res_df['Test F1 (class 1)'], color='cornflowerblue')
            ax.set_xlabel("F1-Score (catching defaults)")
            st.pyplot(fig)
        else:
            st.warning("Performance results not found.")

    # ===== TAB 4: HISTORY =====
    with tab4:
        st.header("Decision History Log")
        st.markdown("Review of all past individual risk assessments.")
        
        if os.path.exists(LOG_FILE):
            history_df = pd.read_csv(LOG_FILE).sort_values('timestamp', ascending=False)
            st.dataframe(history_df, use_container_width=True)
            
            # Small dashboard for history
            st.divider()
            col_h1, col_h2 = st.columns(2)
            with col_h1:
                st.metric("Total Assessments", len(history_df))
            with col_h2:
                risk_counts = history_df['decision'].value_counts()
                high_count = risk_counts.get('High Risk', 0)
                st.metric("High Risk Identified", f"{high_count} ({(high_count/len(history_df))*100:.1f}%)")
        else:
            st.info("No assessment history yet. Try the 'Predict' tab.")

if __name__ == "__main__":
    main()
