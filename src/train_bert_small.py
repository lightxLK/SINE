"""
src/train_bert_small.py
Strategic fine-tuning of BERT on a balanced small subset for CPU efficacy.
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from src.preprocessing       import validate_and_preprocess
from src.feature_engineering import encode_labels, build_bert_datasets
from src.models              import train_bert

def run_bert_small(csv_path="data/Dataset-SA.csv", sample_size=1000):
    """
    Samples 'sample_size' per class to keep it fast on CPU 
    while maximizing context learning.
    """
    print(f"\n[BERT] Starting context-aware training on CPU...")
    
    # ── Load + preprocess ─────────────────────────────────────────────────
    raw_df = pd.read_csv(csv_path)
    df     = validate_and_preprocess(raw_df)
    
    # ── Balanced Sampling ─────────────────────────────────────────────────
    classes = df['Label'].unique()
    dfs     = []
    actual_sample = min(sample_size, df['Label'].value_counts().min())
    
    print(f"[BERT] Sampling {actual_sample} rows per class ({len(classes)*actual_sample} total)...")
    for c in classes:
        dfs.append(df[df['Label'] == c].sample(actual_sample, random_state=42))
    
    balanced_df = pd.concat(dfs).sample(frac=1, random_state=42).reset_index(drop=True)
    
    # ── Label encoding ────────────────────────────────────────────────────
    balanced_df["label_encoded"], le = encode_labels(balanced_df["Label"])
    
    # ── Train/Val split ───────────────────────────────────────────────────
    train_texts, val_texts, y_train, y_val = train_test_split(
        balanced_df["clean_text"],
        balanced_df["label_encoded"],
        test_size=0.2,
        stratify=balanced_df["label_encoded"],
        random_state=42,
    )
    
    # ── BERT Datasets ─────────────────────────────────────────────────────
    train_ds, val_ds, _ = build_bert_datasets(
         list(train_texts), list(val_texts),
         y_train.values, y_val.values,
    )
    
    # ── Fine-tune ─────────────────────────────────────────────────────────
    model, trainer = train_bert(
        train_ds, val_ds,
        n_labels=len(le.classes_),
        epochs=3,      # 3 epochs is enough for sarcasm
        batch_size=8,  # Good for CPU memory
    )
    
    print(f"\n[BERT] Context-Aware model training complete (CPU optimized).")

if __name__ == "__main__":
    run_bert_small(sample_size=1000)
