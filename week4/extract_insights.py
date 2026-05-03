import os
import re
import argparse
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def clean_text(text: str) -> str:
    """Basic text cleaning similar to the model's preprocessing."""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_top_keywords(texts, n_keywords=10):
    """Uses TF-IDF to extract the most important keywords from a list of texts."""
    if not texts:
        return []
        
    # We use stop_words='english' to remove common uninformative words
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000, ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    # Sum the TF-IDF scores for each word across all documents
    summed_tfidf = tfidf_matrix.sum(axis=0)
    
    # Create a list of (word, score) tuples
    words_freq = [(word, summed_tfidf[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
    
    # Sort by score descending
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    
    return words_freq[:n_keywords]

def main(input_csv):
    print("="*60)
    print(f"Loading Batch Inference Results: {input_csv}")
    print("="*60)
    
    if not os.path.exists(input_csv):
        print(f"[ERROR] File not found: {input_csv}")
        return

    df = pd.read_csv(input_csv)
    
    # Handle different possible column names for inference results
    text_col = 'Text' if 'Text' in df.columns else 'text'
    sentiment_col = 'Sentiment' if 'Sentiment' in df.columns else 'predicted_sentiment'
    
    if text_col not in df.columns or sentiment_col not in df.columns:
        print(f"[ERROR] Could not find required columns. Found: {df.columns.tolist()}")
        return
        
    print(f"[INFO] Processed {len(df)} records.")
    
    # Clean text
    df['Cleaned_Text'] = df[text_col].apply(clean_text)
    
    # Normalize sentiment labels for grouping (strip spaces and capitalize)
    df['Normalized_Sentiment'] = df[sentiment_col].str.strip().str.capitalize()
    
    # Filter for Positive and Negative subsets
    positive_texts = df[df['Normalized_Sentiment'] == 'Positive']['Cleaned_Text'].tolist()
    negative_texts = df[df['Normalized_Sentiment'] == 'Negative']['Cleaned_Text'].tolist()
    
    print(f"[INFO] Found {len(positive_texts)} Positive reviews and {len(negative_texts)} Negative reviews.\n")
    
    # Extract keywords
    top_positive = extract_top_keywords(positive_texts, n_keywords=10)
    top_negative = extract_top_keywords(negative_texts, n_keywords=10)
    
    print("="*60)
    print("ACTIONABLE BUSINESS INSIGHTS")
    print("="*60)
    
    print("\n🟢 TOP DRIVING KEYWORDS FOR POSITIVE SENTIMENT:")
    for i, (word, score) in enumerate(top_positive, 1):
        print(f"  {i}. {word} (Impact Score: {score:.2f})")
    
    print("\n🔴 TOP DRIVING KEYWORDS FOR NEGATIVE SENTIMENT:")
    for i, (word, score) in enumerate(top_negative, 1):
        print(f"  {i}. {word} (Impact Score: {score:.2f})")

    print("\n[RECOMMENDATIONS]")
    if top_positive:
        pos_words = [w[0] for w in top_positive[:3]]
        print(f"-> Leverage marketing campaigns highlighting '{pos_words[0]}', '{pos_words[1]}', and '{pos_words[2]}'.")
    if top_negative:
        neg_words = [w[0] for w in top_negative[:3]]
        print(f"-> Investigate product/service friction related to '{neg_words[0]}', '{neg_words[1]}', and '{neg_words[2]}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract insights from sentiment analysis batch results.")
    parser.add_argument(
        "--input", 
        type=str, 
        default="dataset/sentimentdataset.csv", 
        help="Path to the CSV file containing the batch inference results."
    )
    args = parser.parse_args()
    
    main(args.input)
