# Diabetes Risk Classification System

## Overview
This repository contains a machine learning pipeline for predicting diabetes risk using the Pima Indians Diabetes Database. The project emphasizes algorithmic explainability, robust preprocessing, and ethical bias auditing.

## Architecture

The project is structured into two main components:

1. **Jupyter Notebook (`diabetes_risk_classification.ipynb`)**: An end-to-end analytical pipeline split into five phases:
   - **Phase 1**: Statistical Exploration (EDA)
   - **Phase 2**: Normalization and Feature Engineering (Median imputation, Standard Scaling)
   - **Phase 3**: Modeling (SVM with RBF Kernel vs. 3-layer PyTorch MLP)
   - **Phase 4**: Explainability (SHAP summary and force plots)
   - **Phase 5**: Ethical Audit (Slicing analysis evaluating False Negative Rates across demographic cohorts)
   
2. **Streamlit Dashboard (`dashboard.py`)**: A web interface that dynamically trains the PyTorch MLP model and visualizes the Phase 5 Slicing Analysis. It allows for interactive adjustment of age thresholds to evaluate demographic bias.

## Technical Stack
- Data Manipulation and EDA: pandas, numpy, seaborn, matplotlib
- Machine Learning: scikit-learn (SVM, Preprocessing)
- Deep Learning: PyTorch (MLP)
- Explainable AI: SHAP
- Web Dashboard: Streamlit

## Setup and Execution

### Requirements
Ensure you have the required dependencies installed:
```bash
pip install pandas numpy scikit-learn torch shap matplotlib seaborn streamlit
```

### Running the Dashboard
To launch the interactive Slicing Analysis dashboard, run:
```bash
streamlit run dashboard.py
```
