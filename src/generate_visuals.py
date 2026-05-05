"""
src/generate_visuals.py
Generates REAL evaluation plots (Confusion Matrix, ROC Curve) for the UI dashboard.
This script bridges the gap between model training and executive visualization.
"""

import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

def generate_visuals():
    """
    Evaluates the trained classical models to create visuals for the presentation.
    """
    print("[Visualizer] Loading models and data...")
    
    # 1. Path Configuration
    MODEL_PATH      = "models/rf.pkl"         # Random Forest binary
    VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"
    DATA_PATH       = "data/Dataset-SA.csv"
    OUTPUT_DIR      = "reports"               # Saving to reports for central access
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 2. Dependency Check: Ensure the training pipeline has been run
    if not os.path.exists(MODEL_PATH):
        print("[Visualizer] Error: RF model missing. Run training first.")
        return
        
    # Load serialized model and vectorizer
    rf    = joblib.load(MODEL_PATH)
    tfidf = joblib.load(VECTORIZER_PATH)
    
    # 3. Data Sampling for Evaluation
    # We take a random sample of 1000 reviews to generate a representative matrix.
    df = pd.read_csv(DATA_PATH).sample(1000)
    # Combine Summary and Review text to match the feature engineering used in training.
    X = tfidf.transform(df['Summary'].fillna("") + " " + df['Review'].fillna(""))
    y_true = df['Sentiment']
    y_pred = rf.predict(X)
    
    # 4. Confusion Matrix Generation
    # Visualizes where the model is succeeding and where it is confusing classes (e.g., Neu vs Neg).
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred, labels=['negative', 'neutral', 'positive'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Neg', 'Neu', 'Pos'], 
                yticklabels=['Neg', 'Neu', 'Pos'])
    plt.title("Confusion Matrix (Inference Lab)")
    plt.savefig(f"{OUTPUT_DIR}/cm_rf.png", transparent=True)
    print(f"[Visualizer] Saved: {OUTPUT_DIR}/cm_rf.png")
    
    # 5. ROC Curve Generation (Receiver Operating Characteristic)
    # Measures the diagnostic ability of the classifier across different threshold settings.
    # We binarize labels to calculate curves for each sentiment class.
    y_true_bin = label_binarize(y_true, classes=['negative', 'neutral', 'positive'])
    y_score = rf.predict_proba(X)
    
    plt.figure(figsize=(8, 6))
    colors = ['#ff3366', '#facc15', '#00ff88'] # Using dashboard theme colors
    for i, label in enumerate(['Negative', 'Neutral', 'Positive']):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        plt.plot(fpr, tpr, color=colors[i], label=f'{label} (AUC = {auc(fpr, tpr):.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5) # Reference line for random guessing
    plt.title("ROC Curve (Agentic Neural Ensemble)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.savefig(f"{OUTPUT_DIR}/roc_rf.png", transparent=True)
    print(f"[Visualizer] Saved: {OUTPUT_DIR}/roc_rf.png")

if __name__ == "__main__":
    generate_visuals()
