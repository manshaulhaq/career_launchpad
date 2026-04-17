"""
╔══════════════════════════════════════════════════════════════════════╗
║  PHASE 2: Feature Engineering & Preprocessing                      ║
║  Customer Churn Prediction — Telecom Industry                      ║
╚══════════════════════════════════════════════════════════════════════╝

This phase transforms raw data into ML-ready features:
  1. Fix data quality issues (TotalCharges)
  2. Engineer new features  (tenure_group, LTV_segment)
  3. Encode categoricals    (One-Hot Encoding)
  4. Scale numericals       (StandardScaler)
  5. Save preprocessed data + fitted transformers for reuse
"""

# ── Imports ──────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ── Paths ────────────────────────────────────────────────────────────────────
WEEK2_DIR   = Path(__file__).resolve().parent
DATASET_DIR = WEEK2_DIR / "dataset"
PLOTS_DIR   = WEEK2_DIR / "plots"
MODELS_DIR  = WEEK2_DIR / "models"

PLOTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

print("=" * 70)
print("  PHASE 2: Feature Engineering & Preprocessing")
print("=" * 70)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: Load Data & Fix TotalCharges
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("🔧 STEP 1: Load Data & Fix Data Quality Issues")
print("─" * 70)

df = pd.read_csv(DATASET_DIR / "telecom_custom_churn.csv")
print(f"   Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")

# Convert TotalCharges from string to numeric
# Blank strings become NaN via errors='coerce'
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Count how many NaN values we created
nan_count = df["TotalCharges"].isna().sum()
print(f"\n   TotalCharges: converted to numeric, found {nan_count} NaN values")

# These are all tenure=0 customers (brand new), so TotalCharges should be 0
print("   Verifying: all NaN rows have tenure = 0?", 
      df[df["TotalCharges"].isna()]["tenure"].unique())
df["TotalCharges"] = df["TotalCharges"].fillna(0.0)
print("   ✅ Filled NaN TotalCharges with 0.0 (brand-new customers)")

# Drop customerID — it's a unique identifier, not a feature
df = df.drop(columns=["customerID"])
print("   ✅ Dropped 'customerID' (unique identifier, not predictive)")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: Feature Engineering — tenure_group
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("⚙️  STEP 2: Feature Engineering — tenure_group")
print("─" * 70)

"""
WHY: Raw tenure (0–72 months) is continuous. Binning it creates meaningful
customer lifecycle stages that are easier for models to learn from and 
for business users to interpret:
  • 0–12 months:  New customers (highest churn risk)
  • 13–24 months: Developing relationship
  • 25–36 months: Established
  • 37–48 months: Loyal
  • 49–60 months: Long-term
  • 61–72 months: Very loyal (lowest churn risk)
"""

bins = [0, 12, 24, 36, 48, 60, 72]
labels = ["0-12", "13-24", "25-36", "37-48", "49-60", "61-72"]

df["tenure_group"] = pd.cut(df["tenure"], bins=bins, labels=labels, 
                            include_lowest=True, right=True)

print("\n   tenure_group distribution:")
tg_dist = df["tenure_group"].value_counts().sort_index()
for group, count in tg_dist.items():
    churn_rate = df[df["tenure_group"] == group]["Churn"].value_counts(normalize=True).get("Yes", 0) * 100
    bar = "█" * int(churn_rate)
    print(f"   {group:>8} months: {count:>5} customers  │ Churn: {churn_rate:5.1f}% {bar}")

print("\n   ✅ Created 'tenure_group' feature with 6 bins")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: Feature Engineering — LTV_segment
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("⚙️  STEP 3: Feature Engineering — LTV_segment (Lifetime Value)")
print("─" * 70)

"""
WHY: Customer Lifetime Value (LTV) combines spending history with tenure
to estimate total customer value. This helps the business prioritize
retention efforts on high-value customers.

Formula:  LTV_score = TotalCharges + (MonthlyCharges × remaining_tenure_estimate)
Simplified: We'll use TotalCharges directly as our LTV proxy, segmented
into quartile-based tiers.

Alternative formula to capture forward-looking value:
  LTV_score = TotalCharges + (MonthlyCharges × 12)
  This estimates value if the customer stays another year.
"""

# Create a forward-looking LTV score
df["LTV_score"] = df["TotalCharges"] + (df["MonthlyCharges"] * 12)

# Segment into 4 tiers using quartiles
df["LTV_segment"] = pd.qcut(df["LTV_score"], q=4,
                             labels=["Low", "Medium", "High", "Premium"])

print("\n   LTV_segment distribution:")
ltv_dist = df.groupby("LTV_segment", observed=True).agg(
    count=("LTV_score", "size"),
    avg_ltv=("LTV_score", "mean"),
    churn_rate=("Churn", lambda x: (x == "Yes").mean() * 100)
).round(1)
print(ltv_dist.to_string())

print("\n   ✅ Created 'LTV_score' (continuous) and 'LTV_segment' (4 tiers)")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: Visualize New Features
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("📊 STEP 4: Visualize Engineered Features")
print("─" * 70)

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
colors = ["#2ecc71", "#e74c3c"]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Churn rate by tenure_group
churn_by_tg = df.groupby("tenure_group", observed=True)["Churn"].apply(
    lambda x: (x == "Yes").mean() * 100
)
churn_by_tg.plot(kind="bar", ax=axes[0], color="#e74c3c", edgecolor="black", linewidth=0.8)
axes[0].set_title("Churn Rate by Tenure Group", fontsize=13, fontweight="bold")
axes[0].set_xlabel("Tenure Group (months)")
axes[0].set_ylabel("Churn Rate (%)")
axes[0].tick_params(axis="x", rotation=45)
for i, v in enumerate(churn_by_tg.values):
    axes[0].text(i, v + 0.5, f"{v:.1f}%", ha="center", fontsize=10, fontweight="bold")

# Plot 2: Churn rate by LTV_segment
churn_by_ltv = df.groupby("LTV_segment", observed=True)["Churn"].apply(
    lambda x: (x == "Yes").mean() * 100
)
churn_by_ltv.plot(kind="bar", ax=axes[1], color="#3498db", edgecolor="black", linewidth=0.8)
axes[1].set_title("Churn Rate by LTV Segment", fontsize=13, fontweight="bold")
axes[1].set_xlabel("LTV Segment")
axes[1].set_ylabel("Churn Rate (%)")
axes[1].tick_params(axis="x", rotation=0)
for i, v in enumerate(churn_by_ltv.values):
    axes[1].text(i, v + 0.5, f"{v:.1f}%", ha="center", fontsize=10, fontweight="bold")

plt.suptitle("Engineered Feature Analysis", fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
plot_path = PLOTS_DIR / "05_engineered_features.png"
plt.savefig(plot_path, bbox_inches="tight")
plt.close()
print(f"   ✅ Saved: {plot_path}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: Encode Target Variable
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("🏷️  STEP 5: Encode Target Variable")
print("─" * 70)

df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})
print(f"   Churn → 0 (No), 1 (Yes)")
print(f"   Distribution: {df['Churn'].value_counts().to_dict()}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 6: Separate Features & Target, Identify Column Types
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("📋 STEP 6: Prepare Feature Matrix & Column Classification")
print("─" * 70)

# Separate features and target
X = df.drop(columns=["Churn"])
y = df["Churn"]

# Drop intermediate columns not needed for modeling
# (LTV_score was used to create LTV_segment; tenure_group is categorical version of tenure)
cols_to_drop_from_X = ["LTV_score"]  # Keep tenure_group, drop raw score
X = X.drop(columns=cols_to_drop_from_X, errors="ignore")

# Classify columns
numerical_features = ["tenure", "MonthlyCharges", "TotalCharges"]    # SeniorCitizen is really binary categorical

categorical_features = [col for col in X.columns if col not in numerical_features]

print(f"\n   📊 Numerical features ({len(numerical_features)}):")
for col in numerical_features:
    print(f"      • {col}: range [{X[col].min():.1f} – {X[col].max():.1f}], mean={X[col].mean():.1f}")

print(f"\n   📝 Categorical features ({len(categorical_features)}):")
for col in categorical_features:
    n_unique = X[col].nunique()
    print(f"      • {col}: {n_unique} unique values → {list(X[col].unique()[:5])}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 7: Build Preprocessing Pipeline (OneHotEncoder + StandardScaler)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("🔄 STEP 7: Build Preprocessing Pipeline")
print("─" * 70)

"""
WHY these choices:
  • StandardScaler for numerical features:
    - Centers data (mean=0, std=1) — required for Logistic Regression
    - Tree-based models (Random Forest) don't need scaling, but it doesn't hurt
    
  • OneHotEncoder for categorical features:
    - Creates binary columns for each category
    - handle_unknown='ignore' prevents errors on unseen categories at prediction time
    - drop='first' avoids multicollinearity (dummy variable trap) for Logistic Regression
"""

# Numerical pipeline: StandardScaler
numerical_transformer = Pipeline(steps=[
    ("scaler", StandardScaler())
])

# Categorical pipeline: OneHotEncoder
categorical_transformer = Pipeline(steps=[
    ("onehot", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"))
])

# Combined preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features),
    ],
    remainder="drop"
)

# Fit and transform
print("\n   Fitting preprocessor on full dataset...")
X_processed = preprocessor.fit_transform(X)

# Get feature names after encoding
num_feature_names = numerical_features
cat_feature_names = list(
    preprocessor.named_transformers_["cat"]
    .named_steps["onehot"]
    .get_feature_names_out(categorical_features)
)
all_feature_names = num_feature_names + cat_feature_names

print(f"\n   ✅ Preprocessing complete!")
print(f"   Original features:    {X.shape[1]}")
print(f"   After One-Hot:        {X_processed.shape[1]}")
print(f"\n   Encoded feature names ({len(all_feature_names)}):")
for i, name in enumerate(all_feature_names):
    print(f"     {i+1:>3}. {name}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 8: Save Preprocessed Data & Preprocessor
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("💾 STEP 8: Save Outputs for Phase 3")
print("─" * 70)

# Save preprocessed feature matrix and target
processed_df = pd.DataFrame(X_processed, columns=all_feature_names)
processed_df["Churn"] = y.values
processed_df.to_csv(DATASET_DIR / "processed_data.csv", index=False)
print(f"   ✅ Saved processed data:  {DATASET_DIR / 'processed_data.csv'}")

# Save the fitted preprocessor pipeline (for the Streamlit dashboard later)
joblib.dump(preprocessor, MODELS_DIR / "preprocessor.joblib")
print(f"   ✅ Saved preprocessor:    {MODELS_DIR / 'preprocessor.joblib'}")

# Save feature name lists (needed for dashboard input mapping)
feature_config = {
    "numerical_features": numerical_features,
    "categorical_features": categorical_features,
    "all_feature_names": all_feature_names,
    "original_columns": list(X.columns),
}
joblib.dump(feature_config, MODELS_DIR / "feature_config.joblib")
print(f"   ✅ Saved feature config:  {MODELS_DIR / 'feature_config.joblib'}")

# Also save the raw (pre-encoded) X and y for Phase 3 (train/test split on raw data)
joblib.dump(X, MODELS_DIR / "X_raw.joblib")
joblib.dump(y, MODELS_DIR / "y.joblib")
print(f"   ✅ Saved raw X:           {MODELS_DIR / 'X_raw.joblib'}")
print(f"   ✅ Saved y:               {MODELS_DIR / 'y.joblib'}")

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  📋 PHASE 2 — SUMMARY")
print("=" * 70)
print(f"""
  🔧 Data Cleaning:
     • TotalCharges: string → float64, 11 blanks → 0.0
     • Dropped customerID (non-predictive)

  ⚙️  New Features Created:
     • tenure_group:  6 bins (0-12, 13-24, ..., 61-72 months)
     • LTV_segment:   4 tiers (Low, Medium, High, Premium)
     • LTV_score:     TotalCharges + MonthlyCharges × 12

  🔄 Preprocessing Pipeline:
     • Numerical:   StandardScaler (mean=0, std=1)
     • Categorical: OneHotEncoder (drop='first', handle_unknown='ignore')
     • Features:    {X.shape[1]} original → {X_processed.shape[1]} after encoding

  💾 Saved Artifacts:
     • {DATASET_DIR / 'processed_data.csv'}
     • {MODELS_DIR / 'preprocessor.joblib'}
     • {MODELS_DIR / 'feature_config.joblib'}
     • {MODELS_DIR / 'X_raw.joblib'} + y.joblib

  ✅ Phase 2 complete! Ready for Phase 3: Model Training.
""")
print("=" * 70)
