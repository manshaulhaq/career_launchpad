"""
Social Media Sentiment Analysis — Fine-Tuning BERT
Phases 1-3: Data Loading, Preprocessing, and Model Fine-Tuning
"""

import os
import re
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)

WEEK4_DIR = Path(__file__).resolve().parent
DATASET_PATH = WEEK4_DIR / "dataset" / "sentimentdataset.csv"
MODEL_SAVE_DIR = WEEK4_DIR / "sentiment_model"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

print("="*60)
print("PHASE 1: Data Loading & Exploration")
print("="*60)

if not DATASET_PATH.exists():
    raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")

df = pd.read_csv(DATASET_PATH)
print(f"Dataset shape: {df.shape}")

print("\nMissing values:")
print(df.isnull().sum()[df.isnull().sum() > 0])

if 'Sentiment' not in df.columns:
    raise ValueError("Column 'Sentiment' not found in dataset")

df['Sentiment'] = df['Sentiment'].str.strip()

top_classes = df['Sentiment'].value_counts().nlargest(3).index.tolist()
dff = df[df['Sentiment'].isin(top_classes)].copy()

print(f"\nTop 3 class distribution:")
print(dff['Sentiment'].value_counts())

dff['review_length'] = dff['Text'].apply(lambda x: len(str(x).split()))
print(f"\nReview length distribution (words):")
print(dff['review_length'].describe())

print("\n" + "="*60)
print("PHASE 2: Text Preprocessing & Tokenization")
print("="*60)

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

dff['Cleaned_Text'] = dff['Text'].apply(clean_text)
print("Text cleaning complete. Sample:")
print(dff[['Text', 'Cleaned_Text']].head(2))

label_encoder = LabelEncoder()
dff['label'] = label_encoder.fit_transform(dff['Sentiment'])
num_labels = len(label_encoder.classes_)

MODEL_NAME = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(
            texts.tolist(),
            truncation=True,
            padding=True,
            max_length=max_length
        )
        self.labels = labels.tolist()

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

print("\n" + "="*60)
print("PHASE 3: Model Fine-Tuning")
print("="*60)

X_train, X_val, y_train, y_val = train_test_split(
    dff['Cleaned_Text'], dff['label'],
    test_size=0.2, random_state=42, stratify=dff['label']
)

train_dataset = SentimentDataset(X_train, y_train, tokenizer)
val_dataset = SentimentDataset(X_val, y_val, tokenizer)

print(f"Training samples: {len(train_dataset)} | Validation samples: {len(val_dataset)}")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels
)
model.resize_token_embeddings(len(tokenizer))

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='weighted', zero_division=0
    )
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

training_args = TrainingArguments(
    output_dir=str(WEEK4_DIR / "results"),
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    warmup_steps=10,
    weight_decay=0.01,
    logging_dir=str(WEEK4_DIR / "logs"),
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

print("\nStarting fine-tuning...")
trainer.train()

print("\nFinal evaluation metrics:")
metrics = trainer.evaluate()
for key, value in metrics.items():
    if key.startswith("eval_"):
        print(f"  {key.replace('eval_', '').capitalize()}: {value:.4f}")

print(f"\nSaving model to {MODEL_SAVE_DIR}")
model.save_pretrained(MODEL_SAVE_DIR)
tokenizer.save_pretrained(MODEL_SAVE_DIR)

mappings = {index: label for index, label in enumerate(label_encoder.classes_)}
torch.save(mappings, MODEL_SAVE_DIR / "class_mappings.pt")

print("\nTraining complete. Model ready for production.")