"""
Loan Default Risk Predictor — Streamlit App
Run with: streamlit run app/app.py
"""

import streamlit as st
import pandas as pd
import os

from src.inference import load_model, predict
from src.config import MODEL_PATH, RESULTS_PATH


@st.cache_resource
def get_model():
    """Load the trained pipeline once and cache it."""
    return load_model(MODEL_PATH)


def main():
    st.set_page_config(page_title="Loan Default Risk Predictor", page_icon="🏦", layout="wide")
    st.title("🏦 Loan Default Risk Predictor")
    st.markdown("Predict whether a borrower is likely to default on their loan.")

    try:
        model = get_model()
    except FileNotFoundError:
        st.error(f"Model file not found at `{MODEL_PATH}`. Run the training pipeline first.")
        return

    tab1, tab2 = st.tabs(["🔍 Predict", "📊 Model Performance"])

    # ===== TAB 1: PREDICTION =====
    with tab1:
        st.header("Borrower Information")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("💰 Loan Details")
            loan_amount = st.slider("Loan Amount ($)", 5000, 250000, 50000, step=5000)
            interest_rate = st.slider("Interest Rate (%)", 2.0, 25.0, 12.0, step=0.5)
            loan_term = st.selectbox("Loan Term (months)", [12, 24, 36, 48, 60])
            loan_purpose = st.selectbox("Loan Purpose", ['Home', 'Auto', 'Education', 'Business', 'Other'])
            has_cosigner = st.selectbox("Has Co-Signer?", ['Yes', 'No'])

        with col2:
            st.subheader("👤 Borrower Profile")
            age = st.slider("Age", 18, 69, 35)
            income = st.slider("Annual Income ($)", 15000, 150000, 65000, step=5000)
            education = st.selectbox("Education", ["Bachelor's", "Master's", "High School", "PhD"])
            employment_type = st.selectbox("Employment Type", ['Full-time', 'Part-time', 'Self-employed', 'Unemployed'])
            marital_status = st.selectbox("Marital Status", ['Single', 'Married', 'Divorced'])

        with col3:
            st.subheader("📋 Credit Profile")
            credit_score = st.slider("Credit Score", 300, 850, 650)
            months_employed = st.slider("Months Employed", 0, 120, 36)
            num_credit_lines = st.slider("Number of Credit Lines", 1, 4, 2)
            dti_ratio = st.slider("DTI Ratio", 0.0, 1.0, 0.3, step=0.01)
            has_mortgage = st.selectbox("Has Mortgage?", ['Yes', 'No'])
            has_dependents = st.selectbox("Has Dependents?", ['Yes', 'No'])

        st.markdown("---")

        if st.button("🔎 Assess Risk", type="primary", use_container_width=True):
            input_data = {
                'Age': [age],
                'Income': [income],
                'LoanAmount': [loan_amount],
                'CreditScore': [credit_score],
                'MonthsEmployed': [months_employed],
                'NumCreditLines': [num_credit_lines],
                'InterestRate': [interest_rate],
                'LoanTerm': [loan_term],
                'DTIRatio': [dti_ratio],
                'Education': [education],
                'EmploymentType': [employment_type],
                'MaritalStatus': [marital_status],
                'HasMortgage': [has_mortgage],
                'HasDependents': [has_dependents],
                'LoanPurpose': [loan_purpose],
                'HasCoSigner': [has_cosigner],
            }
            input_df = pd.DataFrame(input_data)
            result = predict(model, input_df)
            probability = result['probability']

            st.markdown("### Assessment Result")
            col_r1, col_r2 = st.columns([2, 1])
            with col_r1:
                if probability >= 0.5:
                    st.error(f"⚠️ **HIGH RISK OF DEFAULT** — Probability: {probability*100:.1f}%")
                else:
                    st.success(f"✅ **LOW RISK** — Default Probability: {probability*100:.1f}%")
            with col_r2:
                st.metric("Default Probability", f"{probability*100:.1f}%")
                st.progress(probability)

    # ===== TAB 2: MODEL PERFORMANCE =====
    with tab2:
        st.header("Model Performance Summary")

        # Load actual results from Milestone 17
        if os.path.exists(RESULTS_PATH):
            results_df = pd.read_csv(RESULTS_PATH)
            st.subheader("Model Comparison Table")
            st.dataframe(results_df, use_container_width=True)
        else:
            st.warning(f"Results file not found at `{RESULTS_PATH}`. Run Milestone 17 (notebook 10) first.")

        # Feature importance plot
        feat_imp_path = 'docs/eda_plots/feature_importance.png'
        if os.path.exists(feat_imp_path):
            st.subheader("Feature Importance")
            st.image(feat_imp_path, caption="Top features (Decision Tree)")

        # Model comparison bar chart
        comparison_path = 'docs/eda_plots/model_comparison.png'
        if os.path.exists(comparison_path):
            st.subheader("Model Comparison")
            st.image(comparison_path, caption="F1 Score on Default Class")


if __name__ == "__main__":
    main()
