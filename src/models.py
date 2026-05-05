"""
src/models.py
Model Orchestrator — Logistic Regression, Random Forest, XGBoost, BERT.
"""

import os
import joblib
import numpy as np
from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from xgboost                 import XGBClassifier
from imblearn.over_sampling  import SMOTE


# ─────────────────────────────────────────────────────────────────────────────
#  SMOTE balancing
# ─────────────────────────────────────────────────────────────────────────────

def apply_smote(X, y, random_state: int = 42):
    """Oversample minority classes to balance dataset."""
    sm = SMOTE(random_state=random_state, k_neighbors=3)
    X_res, y_res = sm.fit_resample(X, y)
    print(f"[SMOTE] {dict(zip(*np.unique(y_res, return_counts=True)))}")
    return X_res, y_res


# ─────────────────────────────────────────────────────────────────────────────
#  Classical ML helpers
# ─────────────────────────────────────────────────────────────────────────────

def _cv_score(model, X, y, cv: int = 5) -> float:
    skf    = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=skf, scoring="f1_weighted", n_jobs=-1)
    return scores.mean()


def train_logistic_regression(X_train, y_train, save_path: str = "models/lr.pkl"):
    model = LogisticRegression(
        max_iter=1000,
        C=1.0,
        class_weight="balanced",
        solver="lbfgs",
    )
    cv = _cv_score(model, X_train, y_train)
    model.fit(X_train, y_train)
    joblib.dump(model, save_path)
    print(f"[LR]  CV F1={cv:.4f} | saved → {save_path}")
    return model


def train_random_forest(X_train, y_train, save_path: str = "models/rf.pkl"):
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    cv = _cv_score(model, X_train, y_train)
    model.fit(X_train, y_train)
    joblib.dump(model, save_path)
    print(f"[RF]  CV F1={cv:.4f} | saved → {save_path}")
    return model


def train_xgboost(X_train, y_train, n_classes: int = 3, save_path: str = "models/xgb.pkl"):
    model = XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        num_class=n_classes,
        objective="multi:softprob",
        n_jobs=-1,
        random_state=42,
    )
    cv = _cv_score(model, X_train, y_train)
    model.fit(X_train, y_train)
    joblib.dump(model, save_path)
    print(f"[XGB] CV F1={cv:.4f} | saved → {save_path}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
#  BERT fine-tuning
# ─────────────────────────────────────────────────────────────────────────────

def train_bert(
    train_dataset,
    val_dataset,
    n_labels: int = 3,
    output_dir: str = "models/bert_model",
    epochs: int = 4,
    batch_size: int = 16,
    lr: float = 2e-5,
):
    """Fine-tune bert-base-uncased for sequence classification."""
    from transformers import (
        BertForSequenceClassification,
        Trainer,
        TrainingArguments,
        EarlyStoppingCallback,
    )
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[BERT] Training on: {device}")

    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=n_labels,
    )

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_dir="logs",
        logging_steps=20,
        fp16=torch.cuda.is_available(),   # auto mixed precision on GPU
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()
    model.save_pretrained(output_dir)
    print(f"[BERT] Saved → {output_dir}")
    return model, trainer


# ─────────────────────────────────────────────────────────────────────────────
#  Load helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_classical_model(path: str):
    return joblib.load(path)


def load_bert_model(path: str = "models/bert_model"):
    from transformers import BertForSequenceClassification, BertTokenizer
    model     = BertForSequenceClassification.from_pretrained(path)
    tokenizer = BertTokenizer.from_pretrained(path)
    return model, tokenizer
