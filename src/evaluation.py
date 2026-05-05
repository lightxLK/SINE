"""
src/evaluation.py
Critic Agent — metrics, confusion matrix, class-wise analysis, SHAP.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Core evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(
    y_true,
    y_pred,
    label_encoder=None,
    model_name: str = "Model",
) -> dict:
    """Print and return all key metrics."""
    target_names = list(label_encoder.classes_) if label_encoder else None

    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average="weighted")
    report = classification_report(y_true, y_pred, target_names=target_names)

    print(f"\n{'─'*50}")
    print(f"  {model_name}")
    print(f"{'─'*50}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1 (weighted): {f1:.4f}")
    print(report)

    return {"model": model_name, "accuracy": acc, "f1_weighted": f1}


# ─────────────────────────────────────────────────────────────────────────────
#  Confusion matrix plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    y_true,
    y_pred,
    label_encoder=None,
    model_name: str = "Model",
    save_path: str = None,
):
    classes = list(label_encoder.classes_) if label_encoder else None
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"[Critic] Saved confusion matrix → {save_path}")
    plt.show()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  Class-wise analysis (important for Neutral)
# ─────────────────────────────────────────────────────────────────────────────

def neutral_class_analysis(y_true, y_pred, label_encoder):
    """Deep-dive into Neutral class misclassification."""
    classes = list(label_encoder.classes_)
    if "neutral" not in classes:
        print("[Critic] No 'neutral' class found — skipping.")
        return

    neutral_idx = classes.index("neutral")
    mask = np.array(y_true) == neutral_idx

    neutral_true  = np.array(y_true)[mask]
    neutral_pred  = np.array(y_pred)[mask]

    correct = (neutral_true == neutral_pred).sum()
    total   = mask.sum()
    recall  = correct / total if total > 0 else 0

    print(f"\n[Critic] Neutral class — {correct}/{total} correct ({recall:.2%} recall)")

    # What does neutral get misclassified as?
    wrong_pred = neutral_pred[neutral_true != neutral_pred]
    if len(wrong_pred):
        unique, counts = np.unique(wrong_pred, return_counts=True)
        for u, c in zip(unique, counts):
            print(f"  → classified as '{classes[u]}': {c} times")


# ─────────────────────────────────────────────────────────────────────────────
#  Model comparison table
# ─────────────────────────────────────────────────────────────────────────────

def compare_models(results: list[dict]) -> pd.DataFrame:
    """Takes list of dicts from evaluate_model(), returns a ranked DataFrame."""
    df = pd.DataFrame(results).sort_values("f1_weighted", ascending=False)
    df.reset_index(drop=True, inplace=True)
    print("\n[Critic] Model Comparison:")
    print(df.to_string(index=False))
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  SHAP explainability (classical ML only)
# ─────────────────────────────────────────────────────────────────────────────

def explain_with_shap(model, X_val, feature_names=None, max_display: int = 20):
    """Generate SHAP summary plot for tree-based models."""
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        X_dense   = X_val.toarray() if hasattr(X_val, "toarray") else X_val
        shap_vals = explainer.shap_values(X_dense)

        shap.summary_plot(
            shap_vals,
            X_dense,
            feature_names=feature_names,
            max_display=max_display,
            show=True,
        )
    except Exception as e:
        print(f"[SHAP] Skipped: {e}")
