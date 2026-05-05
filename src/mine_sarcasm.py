"""
src/mine_sarcasm.py
Intelligent Data Mining - Finding "Ironic Dissonance" for specialized training.
This script isolates sarcastic reviews from a large dataset to train a context-specialist.
"""

import pandas as pd
import os

# Configurable paths for the source dataset and the generated specialized subset
CSV_PATH = "data/Dataset-SA.csv"
OUTPUT_PATH = "data/sarcasm_subset.csv"

def mine_sarcasm():
    """
    Identifies 'Ironic Dissonance' where a low product rating contradicts positive descriptive words.
    """
    print("[Miner] Initializing data mining on 200k reviews...")
    
    # Load headers and raw data from the main dataset
    if not os.path.exists(CSV_PATH):
        print(f"[Miner] Error: {CSV_PATH} not found.")
        return
        
    df = pd.read_csv(CSV_PATH)
    
    # 1. Define "Positive Intensifiers"
    # These are words that often appear in sarcastic reviews (e.g., "Wow, what a great broken product").
    intensifiers = ["impressive", "wow", "great", "excellent", "amazing", 
                   "perfectly", "expected from", "thankful", "so good"]
    
    # 2. Data Cleaning: Convert 'Rate' (star rating) to numeric for logical filtering
    # We drop any rows where the rate is corrupted or missing.
    df['Rate'] = pd.to_numeric(df['Rate'], errors='coerce')
    df = df.dropna(subset=['Rate']) 

    # 3. Filter for 'Hard Cases': Reviews with low star ratings (1 or 2 stars)
    low_rate_df = df[df['Rate'] <= 2].copy()
    
    # 4. Pattern Matching: Find rows where any positive intensifier is used in a negative-rated review
    # This is the 'Ironic Dissonance' detection logic.
    mask = low_rate_df['Review'].str.contains('|'.join(intensifiers), case=False, na=False) | \
           low_rate_df['Summary'].str.contains('|'.join(intensifiers), case=False, na=False)
    
    sarcasm_df = low_rate_df[mask].copy()
    
    # 5. Label Correction
    # CRITICAL: If a review has 1 star but the original label said 'positive', it was likely 
    # misclassified by a naive model. We force these to 'negative' so the model learns the Truth.
    sarcasm_df['Sentiment'] = 'negative'
    
    # 6. Synthetic Oversampling
    # We duplicate the sarcastic samples 10 times to ensure the model gives them priority 
    # during the weight-adjustment phase of training.
    sarcasm_df_heavy = pd.concat([sarcasm_df] * 10, ignore_index=True)
    
    print(f"[Miner] Created {len(sarcasm_df_heavy)} heavy-weight sarcastic samples.")
    
    # 7. Balancing the Dataset
    # We combine the heavy sarcasm data with a balanced mix of regular reviews to prevent bias.
    normal_neg = df[df['Sentiment'] == 'negative'].sample(min(500, len(df[df['Sentiment'] == 'negative'])))
    normal_pos = df[df['Sentiment'] == 'positive'].sample(min(500, len(df[df['Sentiment'] == 'positive'])))
    normal_neu = df[df['Sentiment'] == 'neutral'].sample(min(500, len(df[df['Sentiment'] == 'neutral'])))
    
    # Final Combat Corpus Generation
    final_df = pd.concat([sarcasm_df_heavy, normal_neg, normal_pos, normal_neu]).drop_duplicates()
    final_df = final_df.sample(frac=1).reset_index(drop=True) # Shuffle the rows for better training convergence
    
    # Save the specialized subset to disk
    os.makedirs("data", exist_ok=True)
    final_df.to_csv(OUTPUT_PATH, index=False)
    print(f"[Miner] Specialized Sarcasm Training Set created: {OUTPUT_PATH} ({len(final_df)} rows)")

if __name__ == "__main__":
    mine_sarcasm()
