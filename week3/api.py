import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Path Configuration ---
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "credit_risk_model.joblib"
CONFIG_PATH = BASE_DIR / "models" / "feature_config.joblib"

# --- Model Initialization ---
def load_assets():
    """Loads the model pipeline and feature configuration from disk."""
    if not MODEL_PATH.exists():
        logger.error(f"Model artifact missing at {MODEL_PATH}")
        raise FileNotFoundError("Model file not found. Run training pipeline first.")
    
    try:
        model = joblib.load(MODEL_PATH)
        config = joblib.load(CONFIG_PATH)
        logger.info("ML assets loaded successfully.")
        return model, config
    except Exception as e:
        logger.critical(f"Failed to load assets: {e}")
        raise

model_pipeline, feature_config = load_assets()

# --- Schema Definitions ---

class ApplicantFeatures(BaseModel):
    """
    Input schema for loan applicant credit features.
    
    Validates demographic, financial, and loan-specific data points 
    required by the inference pipeline.
    """
    person_age: int = Field(..., ge=18, le=100, description="Age (18-100)")
    person_income: int = Field(..., ge=0, description="Annual income in USD")
    person_home_ownership: str = Field(..., description="RENT, OWN, MORTGAGE, or OTHER")
    person_emp_length: float | None = Field(None, ge=0, le=60, description="Years of employment")
    loan_intent: str = Field(..., description="Purpose: PERSONAL, EDUCATION, etc.")
    loan_grade: str = Field(..., description="Bank-assigned grade (A-G)")
    loan_amnt: int = Field(..., ge=500, le=35000, description="Requested amount ($500-$35k)")
    loan_int_rate: float | None = Field(None, ge=0, description="Interest rate percentage")
    loan_percent_income: float = Field(..., ge=0, le=1, description="Loan/Income ratio")
    cb_person_default_on_file: str = Field(..., description="Historical default (Y/N)")
    cb_person_cred_hist_length: int = Field(..., ge=0, description="Credit history length")

    model_config = {
        "json_schema_extra": {
            "example": {
                "person_age": 28,
                "person_income": 55000,
                "person_home_ownership": "RENT",
                "person_emp_length": 5.0,
                "loan_intent": "PERSONAL",
                "loan_grade": "B",
                "loan_amnt": 12000,
                "loan_int_rate": 10.5,
                "loan_percent_income": 0.22,
                "cb_person_default_on_file": "N",
                "cb_person_cred_hist_length": 6
            }
        }
    }

class RiskPrediction(BaseModel):
    """Output schema for the default risk prediction and recommendation."""
    applicant_summary: dict
    default_probability: float
    default_probability_pct: str
    risk_tier: str
    recommendation: str

# --- API Instance ---

app = FastAPI(
    title="Credit Risk Assessment API",
    description="Automated credit scoring using Gradient Boosted Decision Trees.",
    version="1.0.0"
)

@app.get("/health", tags=["System"])
def health_check():
    """Verifies service health and model availability."""
    return {"status": "online", "model_loaded": model_pipeline is not None}

@app.post("/predict", response_model=RiskPrediction, tags=["Inference"])
def predict_risk(applicant: ApplicantFeatures):
    """
    Calculates probability of default and returns a business recommendation.
    
    Logic:
        - Low Risk (<30%): Auto-approve.
        - Medium Risk (30-60%): Flags for human underwriter.
        - High Risk (>60%): Auto-decline/Collateral required.
    """
    try:
        # Convert input to DataFrame for Scikit-Learn/XGBoost pipeline compatibility
        input_df = pd.DataFrame([applicant.model_dump()])
        
        # Inference: extract probability of class 1 (default)
        raw_prob = model_pipeline.predict_proba(input_df)[:, 1][0]
        default_prob = float(raw_prob)

        # Business Logic: Tier Classification
        if default_prob < 0.30:
            tier, rec = "Low", "✅ AUTO-APPROVE: Standard terms applied."
        elif default_prob < 0.60:
            tier, rec = "Medium", "⚠️ MANUAL REVIEW: Verify income and documentation."
        else:
            tier, rec = "High", "🚫 DECLINE: High risk. Consider collateralized options."

        return RiskPrediction(
            applicant_summary={
                "age": applicant.person_age,
                "income": f"${applicant.person_income:,}",
                "loan": f"${applicant.loan_amnt:,}"
            },
            default_probability=round(default_prob, 4),
            default_probability_pct=f"{default_prob:.1%}",
            risk_tier=tier,
            recommendation=rec
        )

    except Exception as e:
        logger.error(f"Inference failure: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal processing error during prediction.")

# Deployment Note:
# Start with: uvicorn api:app --host 0.0.0.0 --port 8000