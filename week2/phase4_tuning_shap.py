"""
╔══════════════════════════════════════════════════════════════════════╗
║  PHASE 4: Hyperparameter Tuning & SHAP Explainability              ║
║  Customer Churn Prediction — Telecom Industry                      ║
╚══════════════════════════════════════════════════════════════════════╝

This phase:
  1. Tunes the Random Forest via RandomizedSearchCV (optimizing Recall)
  2. Compares the tuned model against Phase 3 baselines
  3. Uses SHAP to explain model predictions — global & local
  4. Saves the final production-ready model for the Streamlit dashboard

SHAP (SHapley Additive exPlanations) Primer:
────────────────────────────────────────────
SHAP values come from game theory (Nobel Prize-winning work by Lloyd Shapley).
They answer: "How much did each feature contribute to THIS specific prediction?"

  • Positive SHAP value → pushes prediction toward CHURN
  • Negative SHAP value → pushes prediction toward RETAINED
  • Magnitude = strength of contribution
"""

# ── Imports ──────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
import warnings
import time
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
    make_scorer
)
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Paths ────────────────────────────────────────────────────────────────────
WEEK2_DIR   = Path(__file__).resolve().parent
PLOTS_DIR   = WEEK2_DIR / "plots"
MODELS_DIR  = WEEK2_DIR / "models"

print("=" * 70)
print("  PHASE 4: Hyperparameter Tuning & SHAP Explainability")
print("=" * 70)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: Load Phase 3 Artifacts
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("📂 STEP 1: Load Models & Data from Phase 3")
print("─" * 70)

split_data = joblib.load(MODELS_DIR / "train_test_split.joblib")
X_train = split_data["X_train"]
X_test  = split_data["X_test"]
y_train = split_data["y_train"]
y_test  = split_data["y_test"]

preprocessor   = joblib.load(MODELS_DIR / "preprocessor.joblib")
feature_config = joblib.load(MODELS_DIR / "feature_config.joblib")
rf_pipeline    = joblib.load(MODELS_DIR / "rf_pipeline.joblib")

print(f"   Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}")
print(f"   Features: {len(feature_config['all_feature_names'])}")

# Get Phase 3 baseline metrics for comparison
y_pred_baseline = rf_pipeline.predict(X_test)
y_prob_baseline = rf_pipeline.predict_proba(X_test)[:, 1]
baseline_recall = recall_score(y_test, y_pred_baseline)
baseline_f1     = f1_score(y_test, y_pred_baseline)
print(f"   Phase 3 RF baseline → Recall: {baseline_recall:.4f}, F1: {baseline_f1:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: Define Hyperparameter Search Space
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("🔍 STEP 2: Define Hyperparameter Search Space")
print("─" * 70)

"""
WHY RandomizedSearchCV over GridSearchCV?
  • Grid search tests ALL combinations → O(n^k) evaluations → very slow
  • Randomized search samples N combinations → much faster
  • Research shows random search finds good hyperparameters with fewer
    iterations (Bergstra & Bengio, 2012)

WHY these hyperparameters?
  • n_estimators: More trees → better generalization, but slower
  • max_depth: Controls tree depth → prevents overfitting
  • min_samples_split: Minimum samples to split → regularization
  • min_samples_leaf: Minimum samples in leaf → smoother predictions
  • max_features: Features per split → decorrelates trees
  • class_weight: Rebalancing strategy for imbalanced classes
"""

param_distributions = {
    "classifier__n_estimators":     [100, 200, 300, 400, 500],
    "classifier__max_depth":        [5, 10, 15, 20, 25, None],
    "classifier__min_samples_split": [2, 5, 10, 15, 20],
    "classifier__min_samples_leaf":  [1, 2, 4, 6, 8],
    "classifier__max_features":     ["sqrt", "log2", 0.3, 0.5],
    "classifier__class_weight":     ["balanced", "balanced_subsample"],
}

total_combos = 1
for v in param_distributions.values():
    total_combos *= len(v)
print(f"\n   Total possible combinations: {total_combos:,}")
print(f"   We'll sample 50 random combinations (with 3-fold CV each)")

print("\n   Search space:")
for param, values in param_distributions.items():
    name = param.replace("classifier__", "")
    print(f"     • {name}: {values}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: Run RandomizedSearchCV
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("⚡ STEP 3: Running RandomizedSearchCV (optimizing Recall)")
print("─" * 70)

# Create a fresh pipeline for search
rf_search_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(random_state=42, n_jobs=-1))
])

# Custom scorer: optimize for Recall
recall_scorer = make_scorer(recall_score)

random_search = RandomizedSearchCV(
    estimator=rf_search_pipeline,
    param_distributions=param_distributions,
    n_iter=50,                 # Test 50 random combinations
    cv=3,                      # 3-fold cross-validation
    scoring=recall_scorer,     # Optimize for Recall
    random_state=42,
    n_jobs=-1,                 # Parallel execution
    verbose=1,
    return_train_score=True,   # Track overfitting
)

print("\n   🏋️ Training in progress (50 combinations × 3 folds = 150 fits)...")
t0 = time.time()
random_search.fit(X_train, y_train)
search_time = time.time() - t0
print(f"\n   ✅ Search complete in {search_time:.1f}s")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: Analyze Search Results
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("📊 STEP 4: Analyze Tuning Results")
print("─" * 70)

print(f"\n   Best cross-validation Recall: {random_search.best_score_:.4f}")
print(f"\n   Best hyperparameters:")
for param, value in random_search.best_params_.items():
    name = param.replace("classifier__", "")
    print(f"     • {name}: {value}")

# Show top 5 combinations
results_df = pd.DataFrame(random_search.cv_results_)
results_df = results_df.sort_values("rank_test_score")
print(f"\n   Top 5 parameter combinations:")
print(f"   {'Rank':<6} {'Mean Recall':<14} {'Std':<10} {'Train Recall':<14}")
for _, row in results_df.head(5).iterrows():
    print(f"   {int(row['rank_test_score']):<6} {row['mean_test_score']:<14.4f} "
          f"{row['std_test_score']:<10.4f} {row['mean_train_score']:<14.4f}")

# Check for overfitting (train >> test)
best_idx = random_search.best_index_
train_recall = results_df.iloc[0]["mean_train_score"]
test_recall = results_df.iloc[0]["mean_test_score"]
overfit_gap = train_recall - test_recall
print(f"\n   Overfit check: Train Recall ({train_recall:.4f}) - CV Recall ({test_recall:.4f}) = {overfit_gap:.4f}")
if overfit_gap < 0.10:
    print("   ✅ Gap < 0.10 → No significant overfitting")
else:
    print("   ⚠️ Gap ≥ 0.10 → Some overfitting detected")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: Evaluate Tuned Model on Test Set
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("📈 STEP 5: Evaluate Tuned Random Forest on Test Set")
print("─" * 70)

tuned_rf = random_search.best_estimator_

y_pred_tuned = tuned_rf.predict(X_test)
y_prob_tuned = tuned_rf.predict_proba(X_test)[:, 1]

tuned_metrics = {
    "Accuracy":  accuracy_score(y_test, y_pred_tuned),
    "Precision": precision_score(y_test, y_pred_tuned),
    "Recall":    recall_score(y_test, y_pred_tuned),
    "F1-Score":  f1_score(y_test, y_pred_tuned),
    "ROC-AUC":   roc_auc_score(y_test, y_prob_tuned),
}

baseline_metrics = {
    "Accuracy":  accuracy_score(y_test, y_pred_baseline),
    "Precision": precision_score(y_test, y_pred_baseline),
    "Recall":    recall_score(y_test, y_pred_baseline),
    "F1-Score":  f1_score(y_test, y_pred_baseline),
    "ROC-AUC":   roc_auc_score(y_test, y_prob_baseline),
}

print(f"\n   {'Metric':<12} {'Baseline RF':<14} {'Tuned RF':<14} {'Change':<10}")
print(f"   {'─' * 50}")
for metric in tuned_metrics:
    old = baseline_metrics[metric]
    new = tuned_metrics[metric]
    diff = new - old
    arrow = "▲" if diff > 0 else ("▼" if diff < 0 else "─")
    marker = " ◄── KEY" if metric == "Recall" else ""
    print(f"   {metric:<12} {old:<14.4f} {new:<14.4f} {arrow} {abs(diff):.4f}{marker}")

print(f"\n   Classification Report (Tuned Random Forest):")
print("   " + classification_report(y_test, y_pred_tuned,
      target_names=["Retained (0)", "Churned (1)"],
      digits=3).replace("\n", "\n   "))

# ══════════════════════════════════════════════════════════════════════════════
# STEP 6: SHAP Explainability — Global Feature Importance
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("🔬 STEP 6: SHAP Explainability — Understanding Model Decisions")
print("─" * 70)

"""
HOW TO INTERPRET SHAP PLOTS:
─────────────────────────────

1. SHAP SUMMARY PLOT (Beeswarm):
   • Each dot = one customer for one feature
   • X-axis = SHAP value (impact on churn prediction)
     - Right (positive) → pushes toward CHURN
     - Left (negative) → pushes toward RETAINED
   • Color = feature value (Red = high, Blue = low)
   • Features sorted by importance (top = most impactful)

   Example interpretation:
     "For 'Contract_Two year', blue dots (value=0, meaning NO two-year 
      contract) cluster on the RIGHT → no contract lock-in pushes
      toward churn. Red dots (value=1, meaning HAS two-year contract)
      cluster on LEFT → contract lock-in protects against churn."

2. SHAP DEPENDENCE PLOT:
   • Shows how ONE feature's value affects predictions
   • X-axis = feature value
   • Y-axis = SHAP value (contribution to churn prediction)
   • Color = interaction feature (auto-selected by SHAP)
   • Reveals non-linear relationships and interactions
"""

print("\n   Preparing data for SHAP analysis...")
# Transform test data through the preprocessor
X_test_processed = tuned_rf.named_steps["preprocessor"].transform(X_test)
feature_names = feature_config["all_feature_names"]

# Create a DataFrame with proper feature names for SHAP
X_test_df = pd.DataFrame(X_test_processed, columns=feature_names)

# Extract the classifier from the pipeline
rf_classifier = tuned_rf.named_steps["classifier"]

print("   Computing SHAP values (TreeExplainer — exact, fast for trees)...")
t0 = time.time()
explainer = shap.TreeExplainer(rf_classifier)
shap_values = explainer.shap_values(X_test_df)
shap_time = time.time() - t0
print(f"   ✅ SHAP values computed in {shap_time:.1f}s")

# For binary classification, newer SHAP (0.51+) returns 3D array: (samples, features, classes)
# Older versions return a list of 2D arrays: [class_0_values, class_1_values]
# We want class 1 (Churn) SHAP values
if isinstance(shap_values, list):
    shap_values_churn = shap_values[1]  # Class 1 = Churn
elif shap_values.ndim == 3:
    shap_values_churn = shap_values[:, :, 1]  # 3D: last axis is class
else:
    shap_values_churn = shap_values  # Already 2D

print(f"   SHAP values shape: {shap_values_churn.shape}")

# Convert feature_names to numpy array for safe indexing
feature_names_arr = np.array(feature_names)

# ── Plot 1: SHAP Summary Plot (Beeswarm) ────────────────────────────────────
print("\n   Generating SHAP Summary Plot (Beeswarm)...")
fig, ax = plt.subplots(figsize=(12, 10))
shap.summary_plot(shap_values_churn, X_test_df, 
                  feature_names=feature_names,
                  show=False, max_display=20)
plt.title("SHAP Summary Plot — Top 20 Churn Drivers", fontsize=14, fontweight="bold", pad=20)
plt.tight_layout()
plot_path = PLOTS_DIR / "09_shap_summary.png"
plt.savefig(plot_path, bbox_inches="tight", dpi=150)
plt.close()
print(f"   ✅ Saved: {plot_path}")

# ── Plot 2: SHAP Bar Plot (Mean Absolute SHAP) ─────────────────────────────
print("   Generating SHAP Feature Importance Bar Plot...")
fig, ax = plt.subplots(figsize=(10, 8))
shap.summary_plot(shap_values_churn, X_test_df,
                  feature_names=feature_names,
                  plot_type="bar", show=False, max_display=15)
plt.title("Mean |SHAP| — Feature Importance Ranking", fontsize=14, fontweight="bold", pad=20)
plt.tight_layout()
plot_path = PLOTS_DIR / "10_shap_importance.png"
plt.savefig(plot_path, bbox_inches="tight", dpi=150)
plt.close()
print(f"   ✅ Saved: {plot_path}")

# ── Plot 3: SHAP Dependence Plots (Top 3 features) ─────────────────────────
print("   Generating SHAP Dependence Plots for top features...")

# Find top 3 features by mean absolute SHAP value
mean_abs_shap = np.abs(shap_values_churn).mean(axis=0)
top_features_idx = np.argsort(mean_abs_shap)[::-1][:3]
top_feature_names = feature_names_arr[top_features_idx].tolist()

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax_idx, feat_idx in enumerate(top_features_idx):
    plt.sca(axes[ax_idx])
    shap.dependence_plot(
        int(feat_idx), shap_values_churn, X_test_df,
        feature_names=feature_names,
        show=False, ax=axes[ax_idx]
    )
    axes[ax_idx].set_title(f"SHAP Dependence: {feature_names_arr[feat_idx]}", 
                           fontsize=11, fontweight="bold")

plt.suptitle("SHAP Dependence Plots — Top 3 Churn Drivers", 
             fontsize=14, fontweight="bold", y=1.03)
plt.tight_layout()
plot_path = PLOTS_DIR / "11_shap_dependence.png"
plt.savefig(plot_path, bbox_inches="tight", dpi=150)
plt.close()
print(f"   ✅ Saved: {plot_path}")

# ── Print Top Feature Importances ───────────────────────────────────────────
print("\n   📊 Top 10 Churn Drivers (by mean |SHAP| value):")
print(f"   {'Rank':<6} {'Feature':<40} {'Mean |SHAP|':<14} {'Direction'}")
print(f"   {'─' * 75}")

sorted_idx = np.argsort(mean_abs_shap)[::-1]
for rank, idx in enumerate(sorted_idx[:10], 1):
    name = feature_names_arr[idx]
    importance = mean_abs_shap[idx]
    # Determine dominant direction
    mean_shap = shap_values_churn[:, int(idx)].mean()
    direction = "→ Churn" if mean_shap > 0 else "→ Retain"
    bar = "█" * int(importance * 100)
    print(f"   {rank:<6} {name:<40} {importance:<14.4f} {direction}  {bar}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 7: Save Final Tuned Model
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("💾 STEP 7: Save Final Tuned Model for Dashboard")
print("─" * 70)

# Save the tuned pipeline (this is what the Streamlit app will load)
final_model_path = MODELS_DIR / "tuned_rf_pipeline.joblib"
joblib.dump(tuned_rf, final_model_path)
print(f"   ✅ Saved tuned pipeline:  {final_model_path}")

# Also update the best_model_pipeline to the tuned version
joblib.dump(tuned_rf, MODELS_DIR / "best_model_pipeline.joblib")
print(f"   ✅ Updated best model:    {MODELS_DIR / 'best_model_pipeline.joblib'}")

# Save SHAP explainer and values for potential dashboard use
joblib.dump(explainer, MODELS_DIR / "shap_explainer.joblib")
print(f"   ✅ Saved SHAP explainer:  {MODELS_DIR / 'shap_explainer.joblib'}")

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 4 SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  📋 PHASE 4 — SUMMARY")
print("=" * 70)
print(f"""
  ⚡ Hyperparameter Tuning:
     • Method: RandomizedSearchCV (50 combos × 3-fold CV)
     • Optimized for: Recall (capturing churners)
     • Search time: {search_time:.1f}s

  📊 Performance Improvement:
     • Baseline RF Recall: {baseline_recall:.4f} → Tuned RF Recall: {tuned_metrics['Recall']:.4f}
     • Baseline RF F1:     {baseline_f1:.4f} → Tuned RF F1:     {tuned_metrics['F1-Score']:.4f}

  🔬 Top 5 Churn Drivers (SHAP):
     1. {feature_names_arr[sorted_idx[0]]}
     2. {feature_names_arr[sorted_idx[1]]}
     3. {feature_names_arr[sorted_idx[2]]}
     4. {feature_names_arr[sorted_idx[3]]}
     5. {feature_names_arr[sorted_idx[4]]}

  📈 How to interpret SHAP:
     • Summary plot: features sorted by importance, color = feature value
       (red = high, blue = low), x-axis = impact on churn prediction
     • Dependence plot: shows how one feature's value drives churn probability
     • Positive SHAP = pushes toward CHURN
     • Negative SHAP = pushes toward RETAINED

  💾 Saved:
     • {final_model_path}
     • {MODELS_DIR / 'best_model_pipeline.joblib'}
     • {MODELS_DIR / 'shap_explainer.joblib'}

  ✅ Phase 4 complete! Ready for Phase 5: Streamlit Dashboard.
""")
print("=" * 70)
