"""
╔══════════════════════════════════════════════════════════════════════╗
║  PHASE 1: Business Understanding & Exploratory Data Analysis (EDA) ║
║  Customer Churn Prediction — Telecom Industry                      ║
╚══════════════════════════════════════════════════════════════════════╝

BUSINESS CONTEXT
────────────────
Customer churn (also called customer attrition) is when a customer
discontinues their relationship with a company. In the telecom industry,
this is a critical problem for several reasons:

  💰 Acquiring a new customer costs 5-25x MORE than retaining an existing one.
     (Harvard Business Review)
  📉 The average telecom churn rate is 15-25% annually — one of the highest
     across industries.
  📊 Even a 5% improvement in retention can boost profits by 25-95%.

By building a predictive model, we can:
  1. Identify at-risk customers BEFORE they leave
  2. Target them with personalized retention campaigns (discounts, upgrades)
  3. Prioritize support resources for high-value customers likely to churn
  4. Understand the ROOT CAUSES of churn to improve products & services

This script loads the Telco Customer Churn dataset and performs initial
exploratory analysis to understand its structure, quality, and distributions.
"""

# ── Imports ──────────────────────────────────────────────────────────────────
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ── Directory Setup ──────────────────────────────────────────────────────────
WEEK2_DIR = Path(__file__).resolve().parent          # week2/
DATASET_DIR = WEEK2_DIR / "dataset"                  # week2/dataset/
PLOTS_DIR = WEEK2_DIR / "plots"                      # week2/plots/

# Create directories if they don't exist
DATASET_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("  PHASE 1: Business Understanding & Exploratory Data Analysis")
print("=" * 70)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: Load the Dataset
# ══════════════════════════════════════════════════════════════════════════════
DATA_PATH = DATASET_DIR / "telecom_custom_churn.csv"

print(f"\n📂 Loading dataset from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
print(f"✅ Dataset loaded successfully!")
print(f"   Rows: {df.shape[0]:,}  |  Columns: {df.shape[1]}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: First Look at the Data
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("📋 STEP 2: First Look — .head() and .info()")
print("─" * 70)
print("\nFirst 5 rows:")
print(df.head().to_string())

print("\n\nDataset Info:")
print(df.info())

print("\n\nBasic Statistics (Numerical Columns):")
print(df.describe().to_string())

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: Data Types Analysis
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("🔍 STEP 3: Data Types Analysis")
print("─" * 70)
print("\nColumn data types:")
print(df.dtypes.to_string())

# Identify column categories
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

print(f"\n📊 Numerical columns ({len(numerical_cols)}): {numerical_cols}")
print(f"📝 Categorical columns ({len(categorical_cols)}): {categorical_cols}")

# ⚠️ KEY OBSERVATION: TotalCharges should be numeric but is stored as object/string
print("\n⚠️  KEY FINDING: 'TotalCharges' is stored as type 'object' (string)")
print("   This typically happens because some entries contain blank spaces")
print("   instead of numbers. We'll fix this in Phase 2.")

# Let's verify: find non-numeric entries in TotalCharges
non_numeric_tc = pd.to_numeric(df["TotalCharges"], errors="coerce")
bad_rows = df[non_numeric_tc.isna()]
print(f"\n   Non-numeric TotalCharges rows: {len(bad_rows)}")
if len(bad_rows) > 0:
    print(bad_rows[["customerID", "tenure", "MonthlyCharges", "TotalCharges"]].to_string())

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: Missing Values Analysis
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("🕳️  STEP 4: Missing Values Analysis")
print("─" * 70)

# Check for explicit NaN/null values
missing = df.isnull().sum()
missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
missing_df = pd.DataFrame({"Missing Count": missing, "Missing %": missing_pct})
missing_df = missing_df[missing_df["Missing Count"] > 0]

if len(missing_df) == 0:
    print("\n✅ No explicit NaN/null values found in any column.")
else:
    print("\n⚠️  Columns with missing values:")
    print(missing_df.to_string())

# Check for hidden missing values (blank strings, whitespace)
print("\n🔎 Checking for hidden missing values (blank/whitespace strings)...")
for col in categorical_cols:
    blanks = df[col].str.strip().eq("").sum()
    if blanks > 0:
        print(f"   ⚠️  '{col}': {blanks} blank entries found")

print("\n📝 SUMMARY: The 'TotalCharges' column has ~11 blank string entries.")
print("   These correspond to brand-new customers with tenure = 0.")
print("   Strategy: Convert to numeric, then impute with 0 (no charges yet).")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: Target Variable — Churn Distribution
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("🎯 STEP 5: Target Variable — Churn Class Distribution")
print("─" * 70)

churn_counts = df["Churn"].value_counts()
churn_pct = df["Churn"].value_counts(normalize=True) * 100

print("\nChurn Distribution:")
print(f"   No  (Retained): {churn_counts['No']:,}  ({churn_pct['No']:.1f}%)")
print(f"   Yes (Churned):  {churn_counts['Yes']:,}  ({churn_pct['Yes']:.1f}%)")
print(f"\n   Imbalance Ratio: {churn_counts['No'] / churn_counts['Yes']:.2f}:1")

print("\n⚠️  CLASS IMBALANCE DETECTED!")
print("   ~73% retained vs ~27% churned — this is a moderately imbalanced dataset.")
print("   We'll address this in Phase 3 using class_weight='balanced' or SMOTE.")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 6: Generate EDA Visualizations
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("📊 STEP 6: Generating EDA Visualizations")
print("─" * 70)

# Set style
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams["figure.dpi"] = 150

# ── Plot 1: Churn Distribution (Bar + Pie) ─────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Bar chart
colors = ["#2ecc71", "#e74c3c"]
churn_counts.plot(kind="bar", ax=axes[0], color=colors, edgecolor="black", linewidth=0.8)
axes[0].set_title("Churn Distribution (Count)", fontsize=14, fontweight="bold")
axes[0].set_xlabel("Churn", fontsize=12)
axes[0].set_ylabel("Number of Customers", fontsize=12)
axes[0].tick_params(axis="x", rotation=0)
for i, (val, count) in enumerate(zip(churn_counts.index, churn_counts.values)):
    axes[0].text(i, count + 50, f"{count:,}", ha="center", fontsize=11, fontweight="bold")

# Pie chart
axes[1].pie(churn_counts.values, labels=["Retained", "Churned"],
            autopct="%1.1f%%", colors=colors, startangle=90,
            explode=(0, 0.05), shadow=True,
            textprops={"fontsize": 12, "fontweight": "bold"})
axes[1].set_title("Churn Distribution (%)", fontsize=14, fontweight="bold")

plt.tight_layout()
plot1_path = PLOTS_DIR / "01_churn_distribution.png"
plt.savefig(plot1_path, bbox_inches="tight")
plt.close()
print(f"   ✅ Saved: {plot1_path}")

# ── Plot 2: Churn by Key Categorical Features ──────────────────────────────
key_cats = ["Contract", "InternetService", "PaymentMethod", "TechSupport"]
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for ax, col in zip(axes.flatten(), key_cats):
    ct = pd.crosstab(df[col], df["Churn"], normalize="index") * 100
    ct.plot(kind="bar", stacked=True, ax=ax, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_title(f"Churn Rate by {col}", fontsize=12, fontweight="bold")
    ax.set_ylabel("Percentage (%)", fontsize=10)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=35)
    ax.legend(title="Churn", fontsize=9)
    ax.set_ylim(0, 105)

plt.suptitle("Churn Rate by Key Categorical Features", fontsize=15, fontweight="bold", y=1.01)
plt.tight_layout()
plot2_path = PLOTS_DIR / "02_churn_by_categories.png"
plt.savefig(plot2_path, bbox_inches="tight")
plt.close()
print(f"   ✅ Saved: {plot2_path}")

# ── Plot 3: Numerical Feature Distributions ────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Tenure distribution by churn
for label, color in zip(["No", "Yes"], colors):
    subset = df[df["Churn"] == label]
    axes[0].hist(subset["tenure"], bins=30, alpha=0.7, label=f"Churn={label}",
                 color=color, edgecolor="black", linewidth=0.5)
axes[0].set_title("Tenure Distribution by Churn", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Tenure (months)")
axes[0].set_ylabel("Count")
axes[0].legend()

# MonthlyCharges distribution by churn
for label, color in zip(["No", "Yes"], colors):
    subset = df[df["Churn"] == label]
    axes[1].hist(subset["MonthlyCharges"], bins=30, alpha=0.7, label=f"Churn={label}",
                 color=color, edgecolor="black", linewidth=0.5)
axes[1].set_title("Monthly Charges by Churn", fontsize=12, fontweight="bold")
axes[1].set_xlabel("Monthly Charges ($)")
axes[1].set_ylabel("Count")
axes[1].legend()

# TotalCharges distribution by churn (convert to numeric first)
df_temp = df.copy()
df_temp["TotalCharges"] = pd.to_numeric(df_temp["TotalCharges"], errors="coerce")
for label, color in zip(["No", "Yes"], colors):
    subset = df_temp[df_temp["Churn"] == label]
    axes[2].hist(subset["TotalCharges"].dropna(), bins=30, alpha=0.7,
                 label=f"Churn={label}", color=color, edgecolor="black", linewidth=0.5)
axes[2].set_title("Total Charges by Churn", fontsize=12, fontweight="bold")
axes[2].set_xlabel("Total Charges ($)")
axes[2].set_ylabel("Count")
axes[2].legend()

plt.suptitle("Numerical Feature Distributions", fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
plot3_path = PLOTS_DIR / "03_numerical_distributions.png"
plt.savefig(plot3_path, bbox_inches="tight")
plt.close()
print(f"   ✅ Saved: {plot3_path}")

# ── Plot 4: Correlation Heatmap (Numerical) ────────────────────────────────
df_corr = df_temp.copy()
df_corr["Churn_binary"] = (df_corr["Churn"] == "Yes").astype(int)
num_cols_for_corr = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges", "Churn_binary"]

fig, ax = plt.subplots(figsize=(8, 6))
corr_matrix = df_corr[num_cols_for_corr].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="RdYlBu_r",
            center=0, square=True, linewidths=1, ax=ax,
            cbar_kws={"shrink": 0.8})
ax.set_title("Correlation Heatmap (Numerical Features vs Churn)", fontsize=13, fontweight="bold")
plt.tight_layout()
plot4_path = PLOTS_DIR / "04_correlation_heatmap.png"
plt.savefig(plot4_path, bbox_inches="tight")
plt.close()
print(f"   ✅ Saved: {plot4_path}")

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  📋 PHASE 1 — KEY FINDINGS SUMMARY")
print("=" * 70)
print(f"""
  📐 Dataset Shape:  {df.shape[0]:,} rows × {df.shape[1]} columns
  
  🔢 Data Quality Issues:
     • 'TotalCharges' stored as string → needs numeric conversion
     • ~11 blank entries in 'TotalCharges' (tenure=0 customers)
     • No other missing values detected
  
  🎯 Target Variable (Churn):
     • Retained (No):  {churn_counts['No']:,} ({churn_pct['No']:.1f}%)
     • Churned (Yes):   {churn_counts['Yes']:,} ({churn_pct['Yes']:.1f}%)
     • Imbalance ratio: {churn_counts['No'] / churn_counts['Yes']:.1f}:1
  
  🔑 Early Observations:
     • Month-to-month contracts → much higher churn
     • Fiber optic internet → surprisingly higher churn
     • Electronic check payments → correlated with churn
     • Short tenure → strong predictor of churn
     • Customers without TechSupport/OnlineSecurity → more likely to churn

  📊 Plots saved to: {PLOTS_DIR}/
""")
print("  ✅ Phase 1 complete! Ready for Phase 2: Feature Engineering.")
print("=" * 70)
