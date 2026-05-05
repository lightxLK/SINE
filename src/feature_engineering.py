"""
src/feature_engineering.py
Feature Engineering Agent — TF-IDF (baseline) + BERT (primary).
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder


# ─────────────────────────────────────────────────────────────────────────────
#  Label encoding
# ─────────────────────────────────────────────────────────────────────────────

def encode_labels(series: pd.Series) -> tuple[np.ndarray, LabelEncoder]:
    le = LabelEncoder()
    encoded = le.fit_transform(series)
    print(f"[Labels] Classes: {list(le.classes_)}")
    return encoded, le


# ─────────────────────────────────────────────────────────────────────────────
#  TF-IDF
# ─────────────────────────────────────────────────────────────────────────────

def build_tfidf(
    train_texts: list[str],
    val_texts:   list[str],
    max_features: int = 15_000,
    ngram_range:  tuple = (1, 2),
    save_path:    str  = "models/tfidf_vectorizer.pkl",
) -> tuple:
    """
    Fit TF-IDF on train set, transform both splits.
    Returns (X_train, X_val, vectorizer).
    """
    vec = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=True,          # log normalisation — helps a lot
        min_df=2,
        strip_accents="unicode",
    )
    X_train = vec.fit_transform(train_texts)
    X_val   = vec.transform(val_texts)

    joblib.dump(vec, save_path)
    print(f"[TF-IDF] Vocabulary size: {len(vec.vocabulary_)}")
    print(f"[TF-IDF] Saved → {save_path}")

    return X_train, X_val, vec


def load_tfidf(path: str = "models/tfidf_vectorizer.pkl") -> TfidfVectorizer:
    return joblib.load(path)


# ─────────────────────────────────────────────────────────────────────────────
#  BERT tokenisation + Dataset
# ─────────────────────────────────────────────────────────────────────────────

def build_bert_datasets(
    train_texts: list[str],
    val_texts:   list[str],
    train_labels: np.ndarray,
    val_labels:   np.ndarray,
    model_name: str = "bert-base-uncased",
    max_length: int = 128,
):
    """
    Tokenise text and return PyTorch Dataset objects.
    Lazy import so TF-IDF path doesn't need torch installed.
    """
    import torch
    from transformers import BertTokenizer

    tokenizer = BertTokenizer.from_pretrained(model_name)

    def _encode(texts):
        return tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors=None,        # list-of-dicts, handled by Dataset
        )

    train_enc = _encode(list(train_texts))
    val_enc   = _encode(list(val_texts))

    class ReviewDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels    = list(labels)

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
            return item

    return ReviewDataset(train_enc, train_labels), ReviewDataset(val_enc, val_labels), tokenizer
