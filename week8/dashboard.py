import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Diabetes Risk ML - Ethical Audit", layout="wide")

# --- Model Definition ---
class ClinicalMLP(nn.Module):
    def __init__(self, input_dim):
        super(ClinicalMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.network(x)

# --- Data Loading and Model Training (Cached) ---
@st.cache_resource(show_spinner="Loading data and training model...")
def load_and_train():
    # Load the dataset
    df = pd.read_csv('dataset/diabetes.csv')
    
    # Audit for physiological anomalies (non-biological zeros)
    anomalous_features = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df_imputed = df.copy()
    df_imputed[anomalous_features] = df_imputed[anomalous_features].replace(0, np.nan)
    
    # Impute missing values with the median
    imputer = SimpleImputer(strategy='median')
    X = pd.DataFrame(imputer.fit_transform(df_imputed.drop('Outcome', axis=1)), columns=df.columns[:-1])
    y = df_imputed['Outcome']
    
    # Perform Stratified Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # Apply Standardization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Prepare PyTorch Tensors
    X_train_t = torch.FloatTensor(X_train_scaled)
    y_train_t = torch.FloatTensor(y_train.values).unsqueeze(1)
    X_test_t = torch.FloatTensor(X_test_scaled)
    y_test_t = torch.FloatTensor(y_test.values).unsqueeze(1)
    
    # Training loop
    mlp_model = ClinicalMLP(X_train_scaled.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(mlp_model.parameters(), lr=0.005)
    
    epochs = 150
    for epoch in range(epochs):
        mlp_model.train()
        optimizer.zero_grad()
        outputs = mlp_model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()
        
    mlp_model.eval()
    with torch.no_grad():
        mlp_probs = mlp_model(X_test_t).squeeze().numpy()
        mlp_preds = (mlp_probs >= 0.5).astype(int)
        
    # Prepare final dataframe for slicing
    test_indices = X_test.index
    df_test_original = df.loc[test_indices].copy()
    df_test_original['Prediction'] = mlp_preds
    df_test_original['Probability'] = mlp_probs
    
    return df_test_original, mlp_model

df_test, model = load_and_train()

# --- Dashboard UI ---
st.title("🩺 Diabetes Risk ML: Ethical Audit & Slicing Analysis")

st.markdown("""
### Algorithmic Slicing Analysis
In medical applications, **False Negatives** (classifying a high-risk patient as healthy) are critical risks that delay intervention. 
Here, we perform a *Demographic Slicing Analysis* to ensure our Neural Network (MLP) does not exhibit significant bias in False Negative Rates across different age groups.
""")

# --- Sidebar Controls ---
st.sidebar.header("Cohort Definition")
age_threshold = st.sidebar.slider(
    "Select Age Threshold for Slicing",
    min_value=20,
    max_value=80,
    value=40,
    step=1,
    help="Define the cutoff age between 'Young' and 'Senior' cohorts."
)

# Slicing Condition
young_mask = df_test['Age'] < age_threshold
senior_mask = df_test['Age'] >= age_threshold

young_cohort = df_test[young_mask]
senior_cohort = df_test[senior_mask]

def calculate_metrics(cohort):
    if len(cohort) == 0:
        return {"Accuracy": 0.0, "FN Rate": 0.0, "Count": 0}
        
    acc = (cohort['Prediction'] == cohort['Outcome']).mean()
    
    actual_positives = (cohort['Outcome'] == 1).sum()
    if actual_positives == 0:
        fn_rate = 0.0
    else:
        fn_rate = ((cohort['Prediction'] == 0) & (cohort['Outcome'] == 1)).sum() / actual_positives
        
    return {
        "Accuracy": acc,
        "FN Rate": fn_rate,
        "Count": len(cohort)
    }

young_metrics = calculate_metrics(young_cohort)
senior_metrics = calculate_metrics(senior_cohort)

# --- Top Metrics ---
col1, col2 = st.columns(2)

with col1:
    st.subheader(f"Cohort 1: Younger than {age_threshold}")
    m1, m2, m3 = st.columns(3)
    m1.metric("Patients", f"{young_metrics['Count']}")
    m2.metric("Accuracy", f"{young_metrics['Accuracy']:.1%}")
    m3.metric("False Negative Rate", f"{young_metrics['FN Rate']:.1%}", help="Lower is better in clinical settings")
    
with col2:
    st.subheader(f"Cohort 2: {age_threshold} and Older")
    m1, m2, m3 = st.columns(3)
    m1.metric("Patients", f"{senior_metrics['Count']}")
    m2.metric("Accuracy", f"{senior_metrics['Accuracy']:.1%}")
    
    # Highlight if FNR is significantly worse
    diff = senior_metrics['FN Rate'] - young_metrics['FN Rate']
    delta_color = "inverse" # High FNR is bad
    m3.metric("False Negative Rate", f"{senior_metrics['FN Rate']:.1%}", delta=f"{diff:+.1%}" if diff != 0 else None, delta_color=delta_color)

st.divider()

# --- Visualizations ---
st.subheader("Bias Visualizations")

chart_col1, chart_col2 = st.columns([1, 2])

with chart_col1:
    # Bar Chart for FN Rate comparison
    st.markdown("**False Negative Rate Comparison**")
    chart_data = pd.DataFrame({
        "Cohort": [f"< {age_threshold}", f">= {age_threshold}"],
        "False Negative Rate": [young_metrics['FN Rate'], senior_metrics['FN Rate']]
    })
    
    # We use Altair via st.altair_chart for better customization or just st.bar_chart
    st.bar_chart(data=chart_data.set_index("Cohort"), y="False Negative Rate", use_container_width=True)
    st.caption("A larger discrepancy indicates potential age bias in the model.")

with chart_col2:
    st.markdown("**Confusion Matrices**")
    from sklearn.metrics import confusion_matrix
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    if len(young_cohort) > 0:
        cm_y = confusion_matrix(young_cohort['Outcome'], young_cohort['Prediction'])
        sns.heatmap(cm_y, annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar=False)
    axes[0].set_title(f'Young (< {age_threshold})')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')
    
    if len(senior_cohort) > 0:
        cm_s = confusion_matrix(senior_cohort['Outcome'], senior_cohort['Prediction'])
        sns.heatmap(cm_s, annot=True, fmt='d', cmap='Oranges', ax=axes[1], cbar=False)
    axes[1].set_title(f'Senior (>= {age_threshold})')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')
    
    st.pyplot(fig)

st.divider()

# --- Raw Data Expander ---
with st.expander("🔍 View Raw Prediction Data (Test Set)"):
    st.markdown("Use this to manually audit specific patient predictions.")
    
    # Formatting the dataframe
    display_df = df_test.copy()
    display_df['Correct'] = display_df['Outcome'] == display_df['Prediction']
    display_df['Error Type'] = "None"
    display_df.loc[(display_df['Outcome'] == 1) & (display_df['Prediction'] == 0), 'Error Type'] = "False Negative"
    display_df.loc[(display_df['Outcome'] == 0) & (display_df['Prediction'] == 1), 'Error Type'] = "False Positive"
    
    st.dataframe(display_df.style.apply(lambda x: ['background: #ffe6e6' if v == 'False Negative' else '' for v in x], subset=['Error Type']))
