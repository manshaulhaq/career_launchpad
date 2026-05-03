import re
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "sentiment_model"

model = None
tokenizer = None
class_mappings = None

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer, class_mappings
    
    if not MODEL_DIR.exists():
        raise RuntimeError(f"Model directory not found at {MODEL_DIR}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()
    
    mapping_path = MODEL_DIR / "class_mappings.pt"
    if mapping_path.exists():
        class_mappings = torch.load(mapping_path)
    else:
        class_mappings = {0: "Negative", 1: "Neutral", 2: "Positive"}
        
    yield
    
    model = None
    tokenizer = None
    class_mappings = None

app = FastAPI(
    title="Social Media Sentiment API",
    description="Real-time sentiment classification using fine-tuned BERT.",
    version="1.0.0",
    lifespan=lifespan
)

class SentimentRequest(BaseModel):
    review_text: str = Field(..., min_length=2)

class SentimentResponse(BaseModel):
    original_text: str
    cleaned_text: str
    predicted_sentiment: str
    confidence_score: float

@app.get("/health")
def health_check():
    return {"status": "online", "model_active": model is not None}

@app.post("/analyze_sentiment", response_model=SentimentResponse)
async def analyze_sentiment(payload: SentimentRequest):
    try:
        processed_text = clean_text(payload.review_text)
        
        inputs = tokenizer(
            processed_text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=128
        )
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
        
        probabilities = F.softmax(logits, dim=-1).squeeze()
        predicted_class_id = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_class_id].item()
        predicted_label = class_mappings.get(predicted_class_id, "Unknown")
        
        return SentimentResponse(
            original_text=payload.review_text,
            cleaned_text=processed_text,
            predicted_sentiment=predicted_label,
            confidence_score=round(confidence, 4)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))