"""
src/train_pipeline.py
Master script - runs the full training pipeline end-to-end.
This orchestrates data loading, preprocessing, classical ML, and deep learning phases.
"""

import argparse
import os
import sys

# Ensure the root directory is in the python path for absolute imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from sklearn.model_selection import train_test_split

# Import local pipeline components
from src.preprocessing import validate_and_preprocess
from src.feature_engineering import build_tfidf, build_bert_datasets, encode_labels
from src.models import (
    apply_smote,
    train_logistic_regression,
    train_random_forest,
    train_xgboost,
    train_bert,
)
from src.evaluation import (
    evaluate_model,
    plot_confusion_matrix,
    neutral_class_analysis,
    compare_models,
)

def ensure_dirs():
    """Create necessary directories for saving artifacts if they don't exist."""
    for d in ["models", "reports"]:
        os.makedirs(d, exist_ok=True)

def run_classical(df: pd.DataFrame, le):
    """
    Executes the classical Machine Learning workflow:
    TF-IDF Vectorization -> SMOTE (Oversampling) -> LR / RF / XGBoost Training.
    """
    # 1. Split data into training and validation sets
    X_train_t, X_val_t, y_train, y_val = train_test_split(
        df["clean_text"],
        df["label_encoded"],
        test_size=0.2,
        stratify=df["label_encoded"],
        random_state=42,
    )

    # 2. Build TF-IDF feature matrices
    X_train_tfidf, X_val_tfidf, _ = build_tfidf(
        list(X_train_t), list(X_val_t)
    )

    # 3. Apply SMOTE to balance classes (essential for handling 'Neutral' class deficiency)
    X_sm, y_sm = apply_smote(X_train_tfidf, y_train)

    results = []

    # --- Logistic Regression phase ---
    lr_model  = train_logistic_regression(X_sm, y_sm)
    lr_preds  = lr_model.predict(X_val_tfidf)
    results.append(evaluate_model(y_val, lr_preds, le, "Logistic Regression"))
    plot_confusion_matrix(y_val, lr_preds, le, "LR", "reports/cm_lr.png")
    neutral_class_analysis(y_val, lr_preds, le)

    # --- Random Forest phase ---
    rf_model  = train_random_forest(X_sm, y_sm)
    rf_preds  = rf_model.predict(X_val_tfidf)
    results.append(evaluate_model(y_val, rf_preds, le, "Random Forest"))
    plot_confusion_matrix(y_val, rf_preds, le, "RF", "reports/cm_rf.png")
    neutral_class_analysis(y_val, rf_preds, le)

    # --- XGBoost phase ---
    xgb_model = train_xgboost(X_sm, y_sm, n_classes=len(le.classes_))
    xgb_preds = xgb_model.predict(X_val_tfidf)
    results.append(evaluate_model(y_val, xgb_preds, le, "XGBoost"))
    plot_confusion_matrix(y_val, xgb_preds, le, "XGBoost", "reports/cm_xgb.png")
    neutral_class_analysis(y_val, xgb_preds, le)

    # Compare and return aggregated results
    compare_models(results)
    return results

def run_bert(df: pd.DataFrame, le):
    """
    Executes the Deep Learning workflow:
    Tokenization -> DistilRoBERTa / BERT Fine-tuning.
    """
    # 1. Split data
    X_train_t, X_val_t, y_train, y_val = train_test_split(
        df["clean_text"],
        df["label_encoded"],
        test_size=0.2,
        stratify=df["label_encoded"],
        random_state=42,
    )

    # 2. Prepare Torch-compatible datasets
    train_ds, val_ds, tokenizer = build_bert_datasets(
        list(X_train_t), list(X_val_t),
        y_train.values, y_val.values,
    )

    # 3. Execute Trainer-based fine-tuning
    model, trainer = train_bert(
        train_ds, val_ds,
        n_labels=len(le.classes_),
    )

    # 4. Perform final evaluation of the transformer model
    preds_out = trainer.predict(val_ds)
    preds     = preds_out.predictions.argmax(axis=1) # Get the class index with highest probability
    result    = evaluate_model(y_val.values, preds, le, "BERT")
    plot_confusion_matrix(y_val.values, preds, le, "BERT", "reports/cm_bert.png")
    neutral_class_analysis(y_val.values, preds, le)
    return [result]

def main():
    """Main entry point for the training orchestrator."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",  default="data/Dataset-SA.csv", help="Path to training data")
    parser.add_argument("--mode", choices=["classical", "bert", "all"], default="classical", help="Training mode")
    args = parser.parse_args()

    ensure_dirs()

    print(f"\n{'='*55}")
    print("  Agentic Sentiment Intelligence System - Training")
    print(f"{'='*55}\n")

    # 1. Load and perform initial data validation/preprocessing
    raw_df = pd.read_csv(args.csv)
    df     = validate_and_preprocess(raw_df)

    # 2. Encode categorical labels (Pos/Neg/Neu) into numerical IDs
    df["label_encoded"], le = encode_labels(df["Label"])

    # 3. Run chosen mode sequence
    all_results = []

    if args.mode in ("classical", "all"):
        all_results += run_classical(df, le)

    if args.mode in ("bert", "all"):
        all_results += run_bert(df, le)

    # 4. Final Leaderboard Display
    if len(all_results) > 1:
        print("\n\n Final Leaderboard:")
        compare_models(all_results)

    print("\n[Pipeline] Complete. Models saved to ./models/")

if __name__ == "__main__":
    main()
