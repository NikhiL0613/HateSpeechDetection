"""
ETL Pipeline - With NLTK NLP Processing
Uses NLTK WordNet Lemmatizer for proper NLP
"""

import re
import os
import pickle
import numpy as np
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack, csr_matrix

RANDOM_STATE = 42

print("Loading NLTK lemmatizer...")
lemmatizer = WordNetLemmatizer()
print("NLTK loaded!")

STOPWORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
    "your", "yours", "yourself", "yourselves", "he", "him", "his",
    "himself", "she", "her", "hers", "herself", "it", "its", "itself",
    "they", "them", "their", "theirs", "themselves", "what", "which",
    "who", "whom", "this", "that", "these", "those", "am", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "having",
    "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if",
    "or", "because", "as", "until", "while", "of", "at", "by", "for",
    "with", "about", "against", "between", "through", "during", "before",
    "after", "above", "below", "to", "from", "up", "down", "in", "out",
    "on", "off", "over", "under", "again", "further", "then", "once",
    "here", "there", "when", "where", "why", "how", "all", "both",
    "each", "few", "more", "most", "other", "some", "such", "no", "nor",
    "not", "only", "own", "same", "so", "than", "too", "very", "s", "t",
    "can", "will", "just", "don", "should", "now", "d", "ll", "m", "o",
    "re", "ve", "y",
}

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")
SPECIAL_RE = re.compile(r"[^a-zA-Z\s]")
MULTI_SPACE = re.compile(r"\s+")


def clean_and_lemmatize(text):
    """Clean text and apply NLTK WordNet lemmatization"""
    text = str(text).lower()
    text = URL_RE.sub("", text)
    text = MENTION_RE.sub("", text)
    text = HASHTAG_RE.sub(r"\1", text)
    text = SPECIAL_RE.sub(" ", text)
    text = MULTI_SPACE.sub(" ", text).strip()
    tokens = [
        lemmatizer.lemmatize(w)
        for w in text.split()
        if w not in STOPWORDS and len(w) > 1
    ]
    return " ".join(tokens)


def extract_meta_features(df):
    feats = pd.DataFrame()
    feats["char_count"] = df["text"].str.len()
    feats["word_count"] = df["text"].str.split().str.len()
    feats["avg_word_len"] = feats["char_count"] / (feats["word_count"] + 1)
    feats["uppercase_ratio"] = df["raw_text"].apply(
        lambda t: sum(c.isupper() for c in str(t)) / (len(str(t)) + 1)
    )
    feats["exclamation_count"] = df["raw_text"].str.count("!")
    feats["question_count"] = df["raw_text"].str.count(r"\?")
    feats["unique_word_ratio"] = df["text"].apply(
        lambda t: len(set(t.split())) / (len(t.split()) + 1)
    )
    scaler = StandardScaler()
    return scaler.fit_transform(feats.values), scaler


def run_pipeline(csv_path="data/dataset.csv", output_dir="data/processed"):
    os.makedirs(output_dir, exist_ok=True)
    print("-" * 60)
    print("ETL PIPELINE (with NLTK NLP)")
    print("-" * 60)

    print("\n[1/6] Loading data...")
    df = pd.read_csv(csv_path)
    print(f"  {len(df):,} samples loaded")
    print(f"  Labels:\n{df['label'].value_counts().to_string()}")

    print("\n[2/6] Cleaning text...")
    df["raw_text"] = df["text"]

    print("\n[3/6] NLTK lemmatization...")
    df["text"] = df["text"].apply(clean_and_lemmatize)
    df = df[df["text"].str.strip() != ""].reset_index(drop=True)
    print(f"  {len(df):,} lemmatized samples")

    print("\n[4/6] Building TF-IDF features...")
    tfidf = TfidfVectorizer(
        max_features=15000,
        ngram_range=(1, 3),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )
    X_tfidf = tfidf.fit_transform(df["text"])
    print(f"  TF-IDF shape: {X_tfidf.shape}")

    print("\n[5/6] Extracting meta features...")
    meta, meta_scaler = extract_meta_features(df)
    X = hstack([X_tfidf, csr_matrix(meta)])
    print(f"  Combined feature matrix: {X.shape}")

    print("\n[6/6] Splitting data...")
    y = df["label"].values
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=RANDOM_STATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.118, stratify=y_tmp, random_state=RANDOM_STATE
    )
    print(f"  Train: {X_train.shape[0]:,}")
    print(f"  Val:   {X_val.shape[0]:,}")
    print(f"  Test:  {X_test.shape[0]:,}")

    for name, obj in {"X_train": X_train, "X_val": X_val, "X_test": X_test,
                      "y_train": y_train, "y_val": y_val, "y_test": y_test,
                      "tfidf_vectorizer": tfidf, "meta_scaler": meta_scaler}.items():
        with open(f"{output_dir}/{name}.pkl", "wb") as f:
            pickle.dump(obj, f)
    df.to_csv(f"{output_dir}/cleaned_dataset.csv", index=False)
    print(f"\nSaved to {output_dir}/")


if __name__ == "__main__":
    run_pipeline()
