"""
src/preprocessing.py
Data Refinery Agent - cleans and validates raw review data.
This script prepares raw e-commerce reviews for machine learning classification.
"""

import re
import string
import pandas as pd
import nltk
import emoji
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# 1. Initialization and Dependency Resolution
# We download necessary NLTK packages for stopwords, lemmatization, and tokenization.
for pkg in ["stopwords", "wordnet", "omw-1.4", "punkt"]:
    nltk.download(pkg, quiet=True)

STOPWORDS   = set(stopwords.words("english"))
lemmatizer  = WordNetLemmatizer()

# Linguistic normalization map: Converts online slang to standard English terms.
SLANG_MAP = {
    "gr8": "great", "luv": "love", "pls": "please", "plz": "please",
    "bcoz": "because", "coz": "because", "u": "you", "r": "are",
    "cn": "can", "wud": "would", "shud": "should", "gud": "good",
    "brt": "bright", "msg": "message", "thx": "thanks", "ty": "thanks",
    "lol": "", "omg": "", "wtf": "",
}

# -----------------------------------------------------------------------------
# Text cleaning pipeline
# -----------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """
    Applies a comprehensive cleaning pipeline to a single string of text.
    Handles HTML tags, URLs, emojis, punctuation, and linguistic normalization.
    """
    if pd.isnull(text):
        return ""

    # Convert to lowercase to ensure consistency across the vocabulary
    text = str(text).lower()

    # URL Removal: Eliminates web links that don't contribute to sentiment logic
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)

    # HTML Tag Removal: Cleans out noise from web-scraped content
    text = re.sub(r"<.*?>", " ", text)

    # Emoji Stripping: Per the 'professional aesthetic' requirement, we remove emojis.
    # While demojize preserves signal, the user requested no emojis across the site/code.
    text = emoji.replace_emoji(text, replace='')

    # Punctuation & Special Character Removal: Simplifies the feature space
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Numerical Removal: Removes digits as they usually lack sentiment polarity
    text = re.sub(r"\d+", " ", text)

    # Slang Normalization and Tokenization
    tokens = text.split()
    tokens = [SLANG_MAP.get(tok, tok) for tok in tokens]

    # Stopword Filtering + Lemmatization
    # We remove generic words (the, is, at) and reduce words to their root (e.g., 'running' -> 'run').
    tokens = [
        lemmatizer.lemmatize(tok)
        for tok in tokens
        if tok not in STOPWORDS and len(tok) > 2
    ]

    return " ".join(tokens)


# -----------------------------------------------------------------------------
# DataFrame-level validation and batch preprocessing
# -----------------------------------------------------------------------------

# Internal mapping to standardize class labels found in different dataset versions
LABEL_MAP = {
    "positive": "positive",
    "pos":      "positive",
    "negative": "negative",
    "neg":      "negative",
    "neutral":  "neutral",
    "neu":      "neutral",
}


def validate_and_preprocess(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Standardizes the dataset structure and cleans the content in bulk.
    """
    df = df.copy()
    # Remove leading/trailing spaces from column names to prevent mapping errors
    df.columns = [c.strip() for c in df.columns]

    # Column Mapping: Automatically detects text and label columns based on common names
    text_candidates  = ["Summary", "Review", "text", "comment", "review_text"]
    label_candidates = ["Sentiment", "label", "Target", "sentiment", "class"]

    if "Text" not in df.columns:
        for c in text_candidates:
            if c in df.columns:
                if verbose: print(f"[Preprocessing] Mapping column '{c}' -> 'Text'")
                df.rename(columns={c: "Text"}, inplace=True)
                break
    
    if "Label" not in df.columns:
        for c in label_candidates:
            if c in df.columns:
                if verbose: print(f"[Preprocessing] Mapping column '{c}' -> 'Label'")
                df.rename(columns={c: "Label"}, inplace=True)
                break

    initial_row_count = len(df)

    # 1. Integrity Check: Drop rows missing the primary Text or Label
    df.dropna(subset=["Text", "Label"], inplace=True)

    # 2. Duplicate Management: Ensure each review is unique to avoid overfitting
    df.drop_duplicates(subset=["Text"], inplace=True)

    # 3. Label Normalization: Map varied input strings to {positive, negative, neutral}
    df["Label"] = df["Label"].str.strip().str.lower().map(LABEL_MAP)

    # 4. Cleanup: Drop rows that couldn't be mapped to our target sentiments
    df.dropna(subset=["Label"], inplace=True)

    # 5. Dimensional Filter: Ignore reviews with < 3 words as they lack sufficient context
    df = df[df["Text"].apply(lambda x: len(str(x).split()) >= 3)]

    # 6. Pipeline Execution: Apply the individual clean_text function in parallel/batch
    df["clean_text"] = df["Text"].apply(clean_text)

    # 7. Final Sanity Check: Remove any rows that ended up empty after cleaning
    df = df[df["clean_text"].str.strip().astype(bool)]
    df.reset_index(drop=True, inplace=True)

    if verbose:
        print(f"[Preprocessing] {initial_row_count} -> {len(df)} rows retained.")
        print(f"[Preprocessing] Final class distribution:\n{df['Label'].value_counts()}\n")

    return df

if __name__ == "__main__":
    import sys, os
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "data/Dataset-SA.csv"

    if not os.path.exists(csv_path):
        print(f"[ERROR] Source file not found: {csv_path}")
        sys.exit(1)

    raw_data = pd.read_csv(csv_path)
    processed_data = validate_and_preprocess(raw_data)
    processed_data.to_csv("data/clean_dataset.csv", index=False)
    print("[Done] Results saved to data/clean_dataset.csv")
