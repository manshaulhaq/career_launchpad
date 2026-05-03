"""
╔══════════════════════════════════════════════════════════════════════╗
║  PHASE 5: Customer Churn Prediction Dashboard                      ║
║  Streamlit Web Application                                         ║
╚══════════════════════════════════════════════════════════════════════╝

Run with:  streamlit run week2/app.py
           (from the career_launchpad directory)

    OR:    cd week2 && streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
WEEK2_DIR = Path(__file__).resolve().parent
MODELS_DIR = WEEK2_DIR / "models"

# ── Page Configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Telecom Churn Predictor",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS for Premium Look ──────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }

    .main-header h1 {
        color: #ffffff;
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }

    .main-header p {
        color: #a8a5c8;
        font-size: 1rem;
        margin: 0;
    }

    .risk-card {
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
        margin-bottom: 1rem;
    }

    .high-risk {
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
        color: white;
    }

    .low-risk {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
    }

    .risk-card h2 {
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    .risk-card .probability {
        font-size: 3.5rem;
        font-weight: 800;
        margin: 0.5rem 0;
    }

    .risk-card .label {
        font-size: 1.2rem;
        font-weight: 500;
        opacity: 0.9;
    }

    .metric-box {
        background: #f8f9fc;
        border: 1px solid #e2e4ea;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        margin: 0.5rem 0;
    }

    .metric-box .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #302b63;
    }

    .metric-box .metric-label {
        font-size: 0.85rem;
        color: #6b6b8d;
        margin-top: 0.3rem;
    }

    .sidebar .stSelectbox label,
    .sidebar .stSlider label {
        font-weight: 500;
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8f9fc 0%, #eef0f5 100%);
    }

    .driver-item {
        display: flex;
        align-items: center;
        padding: 0.6rem 0;
        border-bottom: 1px solid #eee;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }

    .stTabs [data-baseweb="tab"] {
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)


# ── Load Model ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    """Load the tuned Random Forest pipeline."""
    model_path = MODELS_DIR / "tuned_rf_pipeline.joblib"
    if not model_path.exists():
        # Fallback to best_model_pipeline
        model_path = MODELS_DIR / "best_model_pipeline.joblib"
    return joblib.load(model_path)


@st.cache_resource
def load_feature_config():
    """Load feature configuration."""
    return joblib.load(MODELS_DIR / "feature_config.joblib")


model = load_model()
feature_config = load_feature_config()

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>📡 Telecom Customer Churn Predictor</h1>
    <p>Powered by a Tuned Random Forest model · Recall: 81.3% · Enter customer details to predict churn risk</p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR: Customer Input Form
# ══════════════════════════════════════════════════════════════════════════════
st.sidebar.markdown("## 🧑‍💼 Customer Profile")
st.sidebar.markdown("Enter the customer's details below:")

st.sidebar.markdown("---")
st.sidebar.markdown("### 👤 Demographics")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"], key="gender")
senior_citizen = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"], key="senior")
partner = st.sidebar.selectbox("Partner", ["Yes", "No"], key="partner")
dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"], key="dependents")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📞 Services")

phone_service = st.sidebar.selectbox("Phone Service", ["Yes", "No"], key="phone")

if phone_service == "Yes":
    multiple_lines = st.sidebar.selectbox("Multiple Lines", ["Yes", "No"], key="multilines")
else:
    multiple_lines = "No phone service"

internet_service = st.sidebar.selectbox(
    "Internet Service", ["DSL", "Fiber optic", "No"], key="internet"
)

if internet_service != "No":
    online_security = st.sidebar.selectbox("Online Security", ["Yes", "No"], key="security")
    online_backup = st.sidebar.selectbox("Online Backup", ["Yes", "No"], key="backup")
    device_protection = st.sidebar.selectbox("Device Protection", ["Yes", "No"], key="device")
    tech_support = st.sidebar.selectbox("Tech Support", ["Yes", "No"], key="tech")
    streaming_tv = st.sidebar.selectbox("Streaming TV", ["Yes", "No"], key="tv")
    streaming_movies = st.sidebar.selectbox("Streaming Movies", ["Yes", "No"], key="movies")
else:
    online_security = "No internet service"
    online_backup = "No internet service"
    device_protection = "No internet service"
    tech_support = "No internet service"
    streaming_tv = "No internet service"
    streaming_movies = "No internet service"

st.sidebar.markdown("---")
st.sidebar.markdown("### 📋 Account")

contract = st.sidebar.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"],
    key="contract"
)
paperless_billing = st.sidebar.selectbox(
    "Paperless Billing", ["Yes", "No"], key="paperless"
)
payment_method = st.sidebar.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check",
     "Bank transfer (automatic)", "Credit card (automatic)"],
    key="payment"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 💰 Financials")

tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12, key="tenure")
monthly_charges = st.sidebar.slider("Monthly Charges ($)", 18.0, 120.0, 65.0, step=0.5, key="monthly")
total_charges = st.sidebar.slider(
    "Total Charges ($)", 0.0, 9000.0,
    float(round(tenure * monthly_charges * 0.98, 2)),  # Reasonable default
    step=10.0, key="total"
)

# ══════════════════════════════════════════════════════════════════════════════
# Build Feature Vector (must match Phase 2 preprocessing pipeline input)
# ══════════════════════════════════════════════════════════════════════════════

# Derive engineered features
# tenure_group
bins = [0, 12, 24, 36, 48, 60, 72]
labels = ["0-12", "13-24", "25-36", "37-48", "49-60", "61-72"]
tenure_group = pd.cut([tenure], bins=bins, labels=labels, include_lowest=True, right=True)[0]

# LTV_segment
ltv_score = total_charges + (monthly_charges * 12)
# Use approximate quartile boundaries from training data
if ltv_score < 950:
    ltv_segment = "Low"
elif ltv_score < 2200:
    ltv_segment = "Medium"
elif ltv_score < 4800:
    ltv_segment = "High"
else:
    ltv_segment = "Premium"

# Convert SeniorCitizen to int
senior_citizen_val = 1 if senior_citizen == "Yes" else 0

# Build input DataFrame matching the exact column order the model expects
input_data = pd.DataFrame([{
    "gender":           gender,
    "SeniorCitizen":    senior_citizen_val,
    "Partner":          partner,
    "Dependents":       dependents,
    "tenure":           tenure,
    "PhoneService":     phone_service,
    "MultipleLines":    multiple_lines,
    "InternetService":  internet_service,
    "OnlineSecurity":   online_security,
    "OnlineBackup":     online_backup,
    "DeviceProtection": device_protection,
    "TechSupport":      tech_support,
    "StreamingTV":      streaming_tv,
    "StreamingMovies":  streaming_movies,
    "Contract":         contract,
    "PaperlessBilling": paperless_billing,
    "PaymentMethod":    payment_method,
    "MonthlyCharges":   monthly_charges,
    "TotalCharges":     total_charges,
    "tenure_group":     str(tenure_group),
    "LTV_segment":      ltv_segment,
}])

# ══════════════════════════════════════════════════════════════════════════════
# PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
churn_probability = model.predict_proba(input_data)[0][1]
churn_prediction = 1 if churn_probability >= 0.50 else 0
is_high_risk = churn_probability >= 0.50

# ══════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT: Results
# ══════════════════════════════════════════════════════════════════════════════

# Top row: Prediction result
col1, col2, col3 = st.columns([1.5, 2, 1.5])

with col2:
    risk_class = "high-risk" if is_high_risk else "low-risk"
    risk_label = "⚠️ HIGH RISK" if is_high_risk else "✅ LOW RISK"
    risk_icon = "🔴" if is_high_risk else "🟢"

    st.markdown(f"""
    <div class="risk-card {risk_class}">
        <h2>Churn Prediction</h2>
        <div class="probability">{churn_probability:.1%}</div>
        <div class="label">{risk_label}</div>
    </div>
    """, unsafe_allow_html=True)

# Metrics row
st.markdown("---")
st.markdown("### 📊 Customer Snapshot")

m1, m2, m3, m4, m5 = st.columns(5)

with m1:
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-value">{tenure}</div>
        <div class="metric-label">Tenure (months)</div>
    </div>
    """, unsafe_allow_html=True)

with m2:
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-value">${monthly_charges:.0f}</div>
        <div class="metric-label">Monthly Charges</div>
    </div>
    """, unsafe_allow_html=True)

with m3:
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-value">${total_charges:,.0f}</div>
        <div class="metric-label">Total Charges</div>
    </div>
    """, unsafe_allow_html=True)

with m4:
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-value">{contract}</div>
        <div class="metric-label">Contract Type</div>
    </div>
    """, unsafe_allow_html=True)

with m5:
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-value">{ltv_segment}</div>
        <div class="metric-label">LTV Segment</div>
    </div>
    """, unsafe_allow_html=True)

# ── Tabs: Details & Recommendations ─────────────────────────────────────────
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["🔍 Risk Analysis", "💡 Retention Actions", "📋 Customer Summary"])

with tab1:
    st.markdown("#### Risk Factor Breakdown")

    risk_factors = []

    # Contract risk
    if contract == "Month-to-month":
        risk_factors.append(("🔴", "Month-to-month contract", "Highest churn risk — no long-term commitment"))
    elif contract == "One year":
        risk_factors.append(("🟡", "One-year contract", "Moderate protection against churn"))
    else:
        risk_factors.append(("🟢", "Two-year contract", "Strong churn protection"))

    # Tenure risk
    if tenure <= 12:
        risk_factors.append(("🔴", f"New customer ({tenure} months)", "New customers churn at ~47%"))
    elif tenure <= 24:
        risk_factors.append(("🟡", f"Developing relationship ({tenure} months)", "Still in early phase"))
    else:
        risk_factors.append(("🟢", f"Established customer ({tenure} months)", "Lower churn probability"))

    # Internet risk
    if internet_service == "Fiber optic":
        risk_factors.append(("🔴", "Fiber optic internet", "Higher churn — possible price/quality issues"))
    elif internet_service == "DSL":
        risk_factors.append(("🟢", "DSL internet", "Lower churn rate"))
    else:
        risk_factors.append(("⚪", "No internet service", "Neutral impact"))

    # Payment risk
    if payment_method == "Electronic check":
        risk_factors.append(("🔴", "Electronic check payment", "Correlated with higher churn"))
    else:
        risk_factors.append(("🟢", f"{payment_method}", "Lower churn payment method"))

    # Support services
    if internet_service != "No":
        if tech_support == "No":
            risk_factors.append(("🟡", "No tech support", "Customers without support churn more"))
        if online_security == "No":
            risk_factors.append(("🟡", "No online security", "Security add-on reduces churn"))

    # Paperless billing
    if paperless_billing == "Yes":
        risk_factors.append(("🟡", "Paperless billing", "Slightly associated with churn"))

    for icon, factor, detail in risk_factors:
        st.markdown(f"{icon} **{factor}** — {detail}")

with tab2:
    st.markdown("#### 💡 Recommended Retention Actions")

    if is_high_risk:
        st.error("⚡ **Immediate action required** — this customer is at high risk of churning!")

        actions = []
        if contract == "Month-to-month":
            actions.append("📋 **Offer a discounted annual contract** (15-20% off) to lock in commitment")
        if tenure <= 12:
            actions.append("🎁 **Deploy welcome program** — onboarding call + loyalty bonus at 6 months")
        if internet_service == "Fiber optic":
            actions.append("📶 **Proactive quality check** — send a technician to optimize connection")
        if tech_support == "No" and internet_service != "No":
            actions.append("🛠️ **Bundle free tech support** for 3 months as a value-add")
        if online_security == "No" and internet_service != "No":
            actions.append("🔒 **Offer free online security trial** — adds perceived value")
        if payment_method == "Electronic check":
            actions.append("💳 **Incentivize auto-pay switch** — offer a $5/month discount")
        if monthly_charges > 80:
            actions.append("💰 **Review pricing** — consider a personalized discount or plan downgrade")

        if not actions:
            actions.append("📞 **Schedule a personal retention call** to understand concerns")

        for action in actions:
            st.markdown(f"- {action}")
    else:
        st.success("✅ This customer is at low risk. Focus on relationship building:")
        st.markdown("- 🌟 **Upsell opportunities** — consider premium add-ons")
        st.markdown("- 🎯 **Loyalty program enrollment** — reward their commitment")
        st.markdown("- 📧 **Regular engagement** — monthly value summaries")

with tab3:
    st.markdown("#### 📋 Full Customer Data Sent to Model")
    st.dataframe(input_data.T.rename(columns={0: "Value"}), use_container_width=True)

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #888; font-size: 0.85rem;'>"
    "📡 Telecom Churn Predictor · Tuned Random Forest · "
    "Built with Streamlit · Week 2 Project"
    "</div>",
    unsafe_allow_html=True,
)
