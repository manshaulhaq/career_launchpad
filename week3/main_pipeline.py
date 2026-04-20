import logging
import time
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    RocCurveDisplay, accuracy_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

# --- Environment Configuration ---
warnings.filterwarnings("ignore", category=FutureWarning)
plt.style.use("seaborn-v0_8-muted")  # Updated for modern Matplotlib
plt.rcParams["figure.dpi"] = 150

# --- Directory Setup ---
WEEK3_DIR = Path(__file__).resolve().parent
DATASET_DIR = WEEK3_DIR / "dataset"
PLOTS_DIR = WEEK3_DIR / "plots"
MODELS_DIR = WEEK3_DIR / "models"

for folder in [PLOTS_DIR, MODELS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# --- PHASE 1: Data Loading & Quality Assessment ---

def load_and_summarize(file_path: Path) -> pd.DataFrame:
    """
    Loads dataset and performs initial structural analysis.
    
    Returns:
        pd.DataFrame: Loaded dataset.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Missing required dataset at {file_path}")
        
    df = pd.read_csv(file_path)
    print(f"Dataset Dimensions: {df.shape[0]} rows, {df.shape[1]} features.")
    
    # Analyze Class Balance
    target_dist = df["loan_status"].value_counts(normalize=True)
    print(f"Target Distribution: Non-Default: {target_dist[0]:.1%}, Default: {target_dist[1]:.1%}")
    return df

df = load_and_summarize(DATASET_DIR / "credit_risk.csv")

# --- PHASE 2: Feature Engineering & Outlier Management ---

"""
Outlier Strategy:
    Rather than pure statistical removal (IQR), we apply domain-driven capping 
    to preserve as much valid data as possible while neutralizing extreme noise 
    caused by data entry errors.
"""

def process_outliers(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()
    # Age: Biological limit for loan applications
    df.loc[df["person_age"] > 100, "person_age"] = 100
    # Employment: Professional career limit
    df.loc[df["person_emp_length"] > 60, "person_emp_length"] = 60
    # Income: Noise reduction at 99.5th percentile
    income_cap = df["person_income"].quantile(0.995)
    df.loc[df["person_income"] > income_cap, "person_income"] = income_cap
    return df

df = process_outliers(df)

# --- PHASE 3: Feature Pipeline Construction ---

"""
Preprocessing Pipeline:
    Numerical: Median Imputation (robust to skew) -> Standard Scaling (Z-score).
    Categorical: One-Hot Encoding (OHE) with handling for unknown categories.
"""

X = df.drop("loan_status", axis=1)
y = df["loan_status"]

num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_pipe = Pipeline([
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer([
    ("num", num_pipe, num_cols),
    ("cat", cat_pipe, cat_cols)
])

# --- PHASE 4: Model Training (XGBoost & CatBoost) ---

"""
Modeling Logic:
    We implement SMOTE (Synthetic Minority Over-sampling Technique) to 
    address class imbalance. By generating synthetic samples along the 
    feature space lines of the minority class, we improve the model's 
    ability to recognize complex default patterns.
"""

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Shared SMOTE configuration
smote_step = SMOTE(random_state=42, k_neighbors=5)

# XGBoost Pipeline
xgb_pipe = ImbPipeline([
    ("preprocessor", preprocessor),
    ("smote", smote_step),
    ("clf", XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, 
                           reg_alpha=0.1, reg_lambda=1.0, random_state=42))
])

# CatBoost Pipeline
cb_pipe = ImbPipeline([
    ("preprocessor", preprocessor),
    ("smote", smote_step),
    ("clf", CatBoostClassifier(iterations=300, depth=6, learning_rate=0.1, 
                                verbose=0, random_seed=42))
])

# Execution
print("Training Gradient Boosted Tree Ensembles...")
xgb_pipe.fit(X_train, y_train)
cb_pipe.fit(X_train, y_train)

# --- PHASE 5: Evaluation & Artifact Persistence ---

def get_metrics(model, X, y, name):
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    
    return {
        "Model": name,
        "ROC-AUC": roc_auc_score(y, y_prob),
        "Recall": recall_score(y, y_pred),
        "Precision": precision_score(y, y_pred),
        "F1": f1_score(y, y_pred)
    }

xgb_res = get_metrics(xgb_pipe, X_test, y_test, "XGBoost")
cb_res = get_metrics(cb_pipe, X_test, y_test, "CatBoost")

# Model Selection Logic
best_pipe = cb_pipe if cb_res["ROC-AUC"] > xgb_res["ROC-AUC"] else xgb_pipe

# Save final artifact for API deployment
joblib.dump(best_pipe, WEEK3_DIR / "credit_risk_model.joblib")
print(f"Pipeline complete. Best model saved to {WEEK3_DIR / 'credit_risk_model.joblib'}")