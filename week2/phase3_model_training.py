"""
╔══════════════════════════════════════════════════════════════════════╗
║  PHASE 3: Model Training — Logistic Regression & Random Forest     ║
║  Customer Churn Prediction — Telecom Industry                      ║
╚══════════════════════════════════════════════════════════════════════╝

WHY RECALL MATTERS MOST FOR CHURN PREDICTION
─────────────────────────────────────────────
In churn prediction, the cost of a False Negative (predicting "No Churn"
when the customer DOES churn) is far greater than a False Positive:

  • False Negative (FN): We MISS an at-risk customer → they leave → we
    lose their lifetime value (potentially $1000s) AND pay 5-25× more
    to acquire a replacement.

  • False Positive (FP): We flag a loyal customer as at-risk → we offer
    them a retention deal they didn't need → minor cost ($20-50 discount),
    and they might even feel valued, increasing loyalty.

The asymmetry is clear: MISSING a churner costs far more than incorrectly
flagging a loyal customer. Therefore:

  ┌──────────────────────────────────────────────────────────────────┐
  │  RECALL = TP / (TP + FN)                                       │
  │                                                                  │
  │  "Of all customers who ACTUALLY churned,                        │
  │   what % did we correctly identify?"                            │
  │                                                                  │
  │  High Recall → fewer missed churners → more retention saves     │
  └──────────────────────────────────────────────────────────────────┘

We also track F1-Score (harmonic mean of Precision and Recall) to ensure
we don't sacrifice too much precision in pursuit of recall.
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

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
    RocCurveDisplay
)
from sklearn.pipeline import Pipeline

# ── Paths ────────────────────────────────────────────────────────────────────
WEEK2_DIR   = Path(__file__).resolve().parent
DATASET_DIR = WEEK2_DIR / "dataset"
PLOTS_DIR   = WEEK2_DIR / "plots"
MODELS_DIR  = WEEK2_DIR / "models"

print("=" * 70)
print("  PHASE 3: Model Training (Logistic Regression & Random Forest)")
print("=" * 70)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: Load Preprocessed Data
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("📂 STEP 1: Load Preprocessed Data from Phase 2")
print("─" * 70)

# Load raw X and y (we'll apply the preprocessor via pipeline)
X_raw = joblib.load(MODELS_DIR / "X_raw.joblib")
y = joblib.load(MODELS_DIR / "y.joblib")
preprocessor = joblib.load(MODELS_DIR / "preprocessor.joblib")
feature_config = joblib.load(MODELS_DIR / "feature_config.joblib")

print(f"   X shape: {X_raw.shape}")
print(f"   y shape: {y.shape}")
print(f"   Churn distribution: {dict(y.value_counts())}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: Train/Test Split (80/20)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("✂️  STEP 2: Train/Test Split (80/20)")
print("─" * 70)

"""
WHY stratify=y?
  With class imbalance (73.5% vs 26.5%), a random split might give uneven
  churn ratios in train vs test. Stratified splitting ensures both sets
  maintain the same ~26.5% churn rate for fair evaluation.
"""

X_train, X_test, y_train, y_test = train_test_split(
    X_raw, y,
    test_size=0.20,
    random_state=42,
    stratify=y  # Preserve class distribution
)

print(f"\n   Training set: {X_train.shape[0]:,} samples")
print(f"   Test set:     {X_test.shape[0]:,} samples")
print(f"\n   Train churn rate: {y_train.mean():.1%}")
print(f"   Test  churn rate: {y_test.mean():.1%}")
print("   ✅ Stratified split preserved class distribution")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: Build Model Pipelines
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("🏗️  STEP 3: Build Model Pipelines")
print("─" * 70)

"""
WHY class_weight='balanced'?
  Instead of using SMOTE (which synthetically creates new minority samples),
  we use class_weight='balanced' which:
  - Automatically adjusts weights inversely proportional to class frequencies
  - Penalizes misclassifying the minority class (Churn=1) more heavily
  - No data augmentation needed → cleaner pipeline, less risk of data leakage
  - Works natively in both Logistic Regression and Random Forest

  For our case: weight_churn ≈ 7043 / (2 × 1869) ≈ 1.88
                weight_nochurn ≈ 7043 / (2 × 5174) ≈ 0.68
"""

# Pipeline 1: Logistic Regression
lr_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        class_weight="balanced",  # Handle class imbalance
        max_iter=1000,            # Ensure convergence
        random_state=42,
        solver="lbfgs"            # Efficient for medium datasets
    ))
])

# Pipeline 2: Random Forest
rf_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,         # 200 trees (good balance of speed/accuracy)
        class_weight="balanced",  # Handle class imbalance
        random_state=42,
        n_jobs=-1,                # Use all CPU cores
        max_depth=15,             # Prevent overfitting
        min_samples_split=10,     # Minimum samples to split a node
        min_samples_leaf=5        # Minimum samples in a leaf
    ))
])

print("   📊 Logistic Regression Pipeline:")
print("      • Preprocessor → LogisticRegression(class_weight='balanced')")
print("   🌲 Random Forest Pipeline:")
print("      • Preprocessor → RandomForestClassifier(200 trees, balanced)")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: Train Both Models
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("🏋️  STEP 4: Train Both Models")
print("─" * 70)

import time

# Train Logistic Regression
print("\n   Training Logistic Regression...", end=" ")
t0 = time.time()
lr_pipeline.fit(X_train, y_train)
lr_time = time.time() - t0
print(f"Done! ({lr_time:.2f}s)")

# Train Random Forest
print("   Training Random Forest...", end=" ")
t0 = time.time()
rf_pipeline.fit(X_train, y_train)
rf_time = time.time() - t0
print(f"Done! ({rf_time:.2f}s)")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: Evaluate Both Models
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("📊 STEP 5: Model Evaluation — Full Metrics Comparison")
print("─" * 70)


def evaluate_model(pipeline, X_test, y_test, model_name):
    """Evaluate a model and return metrics dictionary."""
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "Model": model_name,
        "Accuracy":  accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall":    recall_score(y_test, y_pred),
        "F1-Score":  f1_score(y_test, y_pred),
        "ROC-AUC":   roc_auc_score(y_test, y_prob),
    }

    print(f"\n   ┌── {model_name} ──{'─' * (50 - len(model_name))}┐")
    for metric, value in metrics.items():
        if metric != "Model":
            # Highlight Recall
            marker = " ◄── KEY METRIC" if metric == "Recall" else ""
            print(f"   │  {metric:<12}: {value:.4f}  ({value:.1%}){marker}")
    print(f"   └{'─' * 56}┘")

    print(f"\n   Classification Report ({model_name}):")
    print("   " + classification_report(y_test, y_pred, 
          target_names=["Retained (0)", "Churned (1)"],
          digits=3).replace("\n", "\n   "))

    return metrics, y_pred, y_prob


# Evaluate both models
lr_metrics, lr_pred, lr_prob = evaluate_model(lr_pipeline, X_test, y_test, "Logistic Regression")
rf_metrics, rf_pred, rf_prob = evaluate_model(rf_pipeline, X_test, y_test, "Random Forest")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 6: Side-by-Side Comparison
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("⚖️  STEP 6: Side-by-Side Model Comparison")
print("─" * 70)

comparison = pd.DataFrame([lr_metrics, rf_metrics]).set_index("Model")
print("\n" + comparison.to_string())

# Determine winner for each metric
print("\n   🏆 Winner per metric:")
for col in comparison.columns:
    winner = comparison[col].idxmax()
    val = comparison[col].max()
    emoji = "🎯" if col == "Recall" else "📊"
    print(f"      {emoji} {col:<12}: {winner} ({val:.4f})")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 7: Generate Evaluation Plots
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("📈 STEP 7: Generate Evaluation Plots")
print("─" * 70)

sns.set_theme(style="whitegrid", font_scale=1.05)

# ── Plot 1: Confusion Matrices ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, pred, name in [(axes[0], lr_pred, "Logistic Regression"),
                        (axes[1], rf_pred, "Random Forest")]:
    cm = confusion_matrix(y_test, pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Retained", "Churned"],
                yticklabels=["Retained", "Churned"],
                linewidths=1, linecolor="white",
                annot_kws={"fontsize": 16, "fontweight": "bold"})
    ax.set_title(f"{name}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual", fontsize=11)

    # Annotate TN, FP, FN, TP
    labels = [["TN", "FP"], ["FN", "TP"]]
    for i in range(2):
        for j in range(2):
            ax.text(j + 0.5, i + 0.75, labels[i][j],
                    ha="center", va="center", fontsize=9, color="gray", alpha=0.7)

plt.suptitle("Confusion Matrices", fontsize=15, fontweight="bold")
plt.tight_layout()
plot_path = PLOTS_DIR / "06_confusion_matrices.png"
plt.savefig(plot_path, bbox_inches="tight")
plt.close()
print(f"   ✅ Saved: {plot_path}")

# ── Plot 2: ROC Curves ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))

for prob, name, color in [(lr_prob, "Logistic Regression", "#3498db"),
                           (rf_prob, "Random Forest", "#e74c3c")]:
    RocCurveDisplay.from_predictions(y_test, prob, name=name, ax=ax,
                                      color=color, linewidth=2)

ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random (AUC = 0.50)")
ax.set_title("ROC Curves — Model Comparison", fontsize=14, fontweight="bold")
ax.legend(loc="lower right", fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plot_path = PLOTS_DIR / "07_roc_curves.png"
plt.savefig(plot_path, bbox_inches="tight")
plt.close()
print(f"   ✅ Saved: {plot_path}")

# ── Plot 3: Metrics Comparison Bar Chart ────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))

metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
x = np.arange(len(metrics_to_plot))
width = 0.35

lr_vals = [lr_metrics[m] for m in metrics_to_plot]
rf_vals = [rf_metrics[m] for m in metrics_to_plot]

bars1 = ax.bar(x - width / 2, lr_vals, width, label="Logistic Regression",
               color="#3498db", edgecolor="black", linewidth=0.5)
bars2 = ax.bar(x + width / 2, rf_vals, width, label="Random Forest",
               color="#e74c3c", edgecolor="black", linewidth=0.5)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.005,
                f"{height:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

ax.set_xlabel("Metric", fontsize=12)
ax.set_ylabel("Score", fontsize=12)
ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(metrics_to_plot, fontsize=11)
ax.set_ylim(0, 1.1)
ax.legend(fontsize=11)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plot_path = PLOTS_DIR / "08_metrics_comparison.png"
plt.savefig(plot_path, bbox_inches="tight")
plt.close()
print(f"   ✅ Saved: {plot_path}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 8: Save the Best Model
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("💾 STEP 8: Save Best Model Pipeline")
print("─" * 70)

# Compare on Recall (our priority metric) and F1 as tiebreaker
print(f"\n   Recall comparison:")
print(f"     Logistic Regression: {lr_metrics['Recall']:.4f}")
print(f"     Random Forest:       {rf_metrics['Recall']:.4f}")

if rf_metrics["Recall"] >= lr_metrics["Recall"]:
    best_model = rf_pipeline
    best_name = "Random Forest"
    best_metrics = rf_metrics
else:
    best_model = lr_pipeline
    best_name = "Logistic Regression"
    best_metrics = lr_metrics

print(f"\n   🏆 Best model (by Recall): {best_name}")

# Save the best pipeline (includes preprocessor + classifier)
model_path = MODELS_DIR / "best_model_pipeline.joblib"
joblib.dump(best_model, model_path)
print(f"   ✅ Saved: {model_path}")

# Also save both pipelines for comparison in Phase 4
joblib.dump(lr_pipeline, MODELS_DIR / "lr_pipeline.joblib")
joblib.dump(rf_pipeline, MODELS_DIR / "rf_pipeline.joblib")
print(f"   ✅ Saved: {MODELS_DIR / 'lr_pipeline.joblib'}")
print(f"   ✅ Saved: {MODELS_DIR / 'rf_pipeline.joblib'}")

# Save train/test split for consistent evaluation in Phase 4
split_data = {
    "X_train": X_train, "X_test": X_test,
    "y_train": y_train, "y_test": y_test,
}
joblib.dump(split_data, MODELS_DIR / "train_test_split.joblib")
print(f"   ✅ Saved: {MODELS_DIR / 'train_test_split.joblib'}")

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3 SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  📋 PHASE 3 — SUMMARY")
print("=" * 70)
print(f"""
  ✂️  Train/Test Split:
     • 80/20 stratified split → {X_train.shape[0]:,} train / {X_test.shape[0]:,} test
     • Churn rate preserved in both sets (~{y_test.mean():.1%})

  ⚖️  Class Imbalance Strategy:
     • Used class_weight='balanced' (auto-adjusts penalty weights)
     • Churn (minority) gets ~1.88× weight; No-churn gets ~0.68× weight

  📊 Results:
     ┌──────────────────┬──────────┬──────────┬──────────┬──────────┬──────────┐
     │ Model            │ Accuracy │Precision │  Recall  │ F1-Score │ ROC-AUC  │
     ├──────────────────┼──────────┼──────────┼──────────┼──────────┼──────────┤
     │ Logistic Reg.    │  {lr_metrics['Accuracy']:.4f}  │  {lr_metrics['Precision']:.4f}  │  {lr_metrics['Recall']:.4f}  │  {lr_metrics['F1-Score']:.4f}  │  {lr_metrics['ROC-AUC']:.4f}  │
     │ Random Forest    │  {rf_metrics['Accuracy']:.4f}  │  {rf_metrics['Precision']:.4f}  │  {rf_metrics['Recall']:.4f}  │  {rf_metrics['F1-Score']:.4f}  │  {rf_metrics['ROC-AUC']:.4f}  │
     └──────────────────┴──────────┴──────────┴──────────┴──────────┴──────────┘

  🎯 Why Recall is the priority metric:
     • FN (missed churner) costs $1000s in lost revenue + replacement
     • FP (false alarm) costs only $20-50 in unnecessary retention offers
     • High Recall = fewer missed churners = more revenue saved

  🏆 Best model: {best_name} (Recall: {best_metrics['Recall']:.4f})
     Saved to: {model_path}

  ✅ Phase 3 complete! Ready for Phase 4: Hyperparameter Tuning & SHAP.
""")
print("=" * 70)
