"""
notebooks/eda.py
Exploratory Data Analysis - run this FIRST before training.
This script performs statistical analysis and visualization of the raw dataset.
"""

import argparse
import os
import sys

# Ensure the root directory is in the python path for absolute imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS

# Import local preprocessing logic
from src.preprocessing import validate_and_preprocess

def run_eda(csv_path: str, save_dir: str = "reports"):
    """
    Loads the dataset and generates a series of diagnostic visualizations.
    """
    os.makedirs(save_dir, exist_ok=True)

    # 1. Loading the raw data
    if not os.path.exists(csv_path):
        print(f"[EDA] Error: {csv_path} not found.")
        return

    raw_df = pd.read_csv(csv_path)
    print(f"\n[EDA] Raw shape: {raw_df.shape}")
    print(raw_df.dtypes)
    print("\nFirst 3 rows:")
    print(raw_df.head(3))

    # 2. Check for missing values (Essential for data integrity)
    print("\n[EDA] Null counts:")
    print(raw_df.isnull().sum())

    # 3. Class Distribution Analysis
    # We preprocess the data first to see the final labels (Pos/Neg/Neu)
    df = validate_and_preprocess(raw_df)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    df["Label"].value_counts().plot(kind="bar", ax=axes[0], color=["#22c55e","#facc15","#ef4444"])
    axes[0].set_title("Sentiment Class Distribution")
    axes[0].set_xlabel("Sentiment")
    axes[0].set_ylabel("Count")

    # 4. Platform Distribution Analysis
    # Helps identify which e-commerce platforms contribute most to the training set.
    if "Window" in df.columns:
        df["Window"].value_counts().plot(kind="pie", ax=axes[1], autopct="%1.0f%%")
        axes[1].set_title("Platform Distribution")

    plt.tight_layout()
    fig.savefig(f"{save_dir}/class_platform_distribution.png", dpi=150)
    print(f"[EDA] Saved -> {save_dir}/class_platform_distribution.png")
    # plt.show() # Disabled for headless/script-only environments

    # 5. Text Length Distribution
    # Analyzes word count to see if specific sentiments correlate with review length.
    df["text_len"] = df["clean_text"].apply(lambda x: len(x.split()))

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    for label, grp in df.groupby("Label"):
        grp["text_len"].hist(ax=ax2, bins=30, alpha=0.5, label=label)
    ax2.set_title("Token Length Distribution by Sentiment")
    ax2.set_xlabel("Word count (after cleaning)")
    ax2.legend()
    fig2.savefig(f"{save_dir}/text_length_distribution.png", dpi=150)
    print(f"[EDA] Saved -> {save_dir}/text_length_distribution.png")

    # 6. Global WordCloud Generation
    # Provides a visual summary of high-impact terms across the entire dataset.
    all_text  = " ".join(df["clean_text"].tolist())
    custom_sw = set(STOPWORDS) | {"product", "amazon", "flipkart", "myntra"}

    wc = WordCloud(
        stopwords=custom_sw, background_color="white",
        width=1200, height=500, colormap="viridis"
    ).generate(all_text)

    fig3, ax3 = plt.subplots(figsize=(12, 5))
    ax3.imshow(wc, interpolation="bilinear")
    ax3.axis("off")
    ax3.set_title("WordCloud - All Reviews", fontsize=16, fontweight="bold")
    fig3.savefig(f"{save_dir}/wordcloud_all.png", dpi=150)
    print(f"[EDA] Saved -> {save_dir}/wordcloud_all.png")

    # 7. Per-Sentiment WordCloud Generation
    # Differentiates vocabulary used in Positive vs Negative reviews.
    fig4, axes4 = plt.subplots(1, 3, figsize=(18, 5))
    for ax, label in zip(axes4, ["positive", "neutral", "negative"]):
        subset = df[df["Label"] == label]["clean_text"]
        if len(subset) == 0:
            ax.axis("off")
            continue
        wc_l = WordCloud(
            stopwords=custom_sw, background_color="white",
            width=600, height=300,
            colormap="Greens" if label == "positive" else
                     "Oranges" if label == "neutral" else "Reds"
        ).generate(" ".join(subset))
        ax.imshow(wc_l, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(f"{label.capitalize()} Reviews")
    plt.suptitle("Per-Sentiment WordClouds", fontsize=16, fontweight="bold")
    fig4.savefig(f"{save_dir}/wordcloud_per_label.png", dpi=150)
    print(f"[EDA] Saved -> {save_dir}/wordcloud_per_label.png")

    print(f"\n[EDA] All charts saved to ./{save_dir}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/Dataset-SA.csv")
    args = parser.parse_args()
    run_eda(args.csv)
