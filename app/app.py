"""
Loan Default Risk Predictor — Streamlit App
Run with: streamlit run app/app.py

Features:
- Tab 1: Interactive risk prediction with SHAP explanation
- Tab 2: Model performance comparison
"""

import streamlit as st
import pandas as pd
import numpy as np
import os

from src.inference import load_model, predict
from src.config import (
    MODEL_PATH, RESULTS_PATH,
    NUMERICAL_FEATURES, CATEGORICAL_FEATURES
)


@st.cache_resource
def get_model():
    """Load the trained pipeline once and cache it."""
    return load_model(MODEL_PATH)


@st.cache_resource
def get_shap_explainer():
    """Create a SHAP TreeExplainer for the model's classifier."""
    try:
        import shap
        pipeline = get_model()
        classifier = pipeline.named_steps['classifier']
        explainer = shap.TreeExplainer(classifier)
        return explainer
    except Exception as e:
        return None


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

            # ---- Risk Assessment ----
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

            # ---- SHAP Explanation ----
            st.markdown("---")
            st.markdown("### 🧠 Why did the model make this prediction?")
            st.caption("SHAP (SHapley Additive exPlanations) shows how each feature pushed the prediction toward Default or No Default.")

            explainer = get_shap_explainer()
            if explainer is not None:
                try:
                    import shap
                    import matplotlib.pyplot as plt

                    preprocessor = model.named_steps['preprocessor']
                    input_transformed = preprocessor.transform(input_df)

                    # Get feature names after encoding
                    num_feat_names = NUMERICAL_FEATURES
                    try:
                        cat_feat_names = list(
                            preprocessor.named_transformers_['cat']
                            .named_steps['encoder']
                            .get_feature_names_out(CATEGORICAL_FEATURES)
                        )
                    except Exception:
                        cat_feat_names = [f"cat_{i}" for i in range(input_transformed.shape[1] - len(num_feat_names))]
                    feature_names = num_feat_names + cat_feat_names

                    shap_values = explainer.shap_values(input_transformed)

                    # Handle binary classification (list of 2) vs single array
                    if isinstance(shap_values, list):
                        sv = shap_values[1][0]  # class 1 (Default)
                        base_val = explainer.expected_value[1]
                    else:
                        sv = shap_values[0]
                        base_val = explainer.expected_value

                    # Build a DataFrame of top contributing features
                    contributions = pd.DataFrame({
                        'Feature': feature_names,
                        'SHAP Value': sv,
                        'Direction': ['→ Default' if v > 0 else '→ Safe' for v in sv]
                    })
                    contributions['|SHAP|'] = contributions['SHAP Value'].abs()
                    contributions = contributions.sort_values('|SHAP|', ascending=False).head(10)

                    col_s1, col_s2 = st.columns([1, 1])

                    with col_s1:
                        st.markdown("#### Top 10 Feature Contributions")
                        st.dataframe(
                            contributions[['Feature', 'SHAP Value', 'Direction']].reset_index(drop=True),
                            use_container_width=True
                        )

                    with col_s2:
                        # Bar chart of top contributions
                        fig, ax = plt.subplots(figsize=(8, 5))
                        top = contributions.sort_values('SHAP Value')
                        colors = ['#e74c3c' if v > 0 else '#2ecc71' for v in top['SHAP Value']]
                        ax.barh(top['Feature'], top['SHAP Value'], color=colors, edgecolor='black', alpha=0.85)
                        ax.set_xlabel('SHAP Value (impact on default prediction)')
                        ax.set_title('Feature Contributions', fontsize=14, fontweight='bold')
                        ax.axvline(x=0, color='black', linewidth=0.8)
                        ax.grid(True, axis='x', alpha=0.3)
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()

                except Exception as e:
                    st.info(f"SHAP explanation unavailable: {e}")
            else:
                st.info("SHAP explainer not available. Install `shap` and use a tree-based model.")

    # ===== TAB 2: MODEL PERFORMANCE =====
    with tab2:
        st.header("Model Performance Summary")

        # Load actual results
        if os.path.exists(RESULTS_PATH):
            results_df = pd.read_csv(RESULTS_PATH)
            st.subheader("Model Comparison Table")
            st.dataframe(results_df, use_container_width=True)

            # Bar chart
            if 'Test F1 (class 1)' in results_df.columns and len(results_df) > 1:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(10, 5))
                sorted_df = results_df.sort_values('Test F1 (class 1)', ascending=True)
                colors = ['#e74c3c' if i == len(sorted_df)-1 else '#3498db'
                          for i in range(len(sorted_df))]
                ax.barh(sorted_df['Model'], sorted_df['Test F1 (class 1)'],
                        color=colors, edgecolor='black')
                ax.set_xlabel('F1 Score (class 1 — Default)')
                ax.set_title('Model Comparison — F1 on Default Class',
                             fontsize=14, fontweight='bold')
                ax.grid(True, axis='x', alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
        else:
            st.warning(f"Results file not found at `{RESULTS_PATH}`. Run notebook 10 or 12 first.")

        # SHAP summary plot
        shap_path = 'docs/eda_plots/shap_summary.png'
        if os.path.exists(shap_path):
            st.subheader("🧠 SHAP Global Feature Importance")
            st.image(shap_path, caption="Which features matter most across all predictions?")

        # Feature importance plot
        feat_imp_path = 'docs/eda_plots/feature_importance.png'
        if os.path.exists(feat_imp_path):
            st.subheader("Feature Importance (Decision Tree)")
            st.image(feat_imp_path, caption="Top features (Decision Tree)")


if __name__ == "__main__":
    main()
