"""
src/train_sarcasm_specialist.py
Sarcasm Specialist Training — Optimized for Modern Transformers (4.47+)
"""

import os
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding
)
import torch

def train_specialist():
    print("[Specialist] Initializing context-aware training...")
    
    # Load mined dataset
    if not os.path.exists("data/sarcasm_subset.csv"):
        print("[Error] Sarcasm subset not found. Please run src/mine_sarcasm.py first.")
        return

    df = pd.read_csv("data/sarcasm_subset.csv")
    label_map = {"negative": 0, "neutral": 1, "positive": 2}
    df['label'] = df['Sentiment'].map(label_map)
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)
    
    # Contextual Pairing
    df['text'] = df['Summary'].fillna("") + " " + df['Review'].fillna("")
    
    dataset = Dataset.from_pandas(df[['text', 'label']])
    dataset = dataset.train_test_split(test_size=0.1)
    
    model_name = "distilroberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def tokenize_fn(examples):
        return tokenizer(examples['text'], truncation=True, padding=False)
    
    tokenized_ds = dataset.map(tokenize_fn, batched=True)
    
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Modern Training Arguments (4.47+ compatible)
    training_args = TrainingArguments(
        output_dir="./results_sarcasm",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        # 'eval_strategy' is the modern standard
        eval_strategy="epoch", 
        save_strategy="epoch",
        logging_dir="./logs",
        learning_rate=2e-5,
        load_best_model_at_end=True,
        use_cpu=True
    )
    
    # Modern Trainer (Using positional or standard args)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds['train'],
        eval_dataset=tokenized_ds['test'],
        data_collator=data_collator
    )
    
    print("[Specialist] Starting fine-tuning on CPU (Estimated 45m)...")
    trainer.train()
    
    # Save artifacts
    save_path = "models/sarcasm_specialist"
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"[Specialist] SUCCESS! Model saved to {save_path}")

if __name__ == "__main__":
    train_specialist()
