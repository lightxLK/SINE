"""
src/linguistic_engine.py
Linguistic Quality Node (LQN) - Advanced Grammatical & Structural Auditor.
Developed for Capstone Project 2026.
"""

# Core NLP libraries for structure and grammar analysis
import spacy
import re
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch

class LinguisticAuditor:
    """
    The LQN is responsible for denoising text and auditing linguistic quality.
    It uses a combination of rule-based (SpaCy) and neural (T5) approaches.
    """
    
    def __init__(self):
        # 1. Initialize SpaCy for structural linguistic analysis
        # Using the fallback logic to ensure 'en_core_web_sm' is available on the local machine
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            print("[LQN] Spacy model missing. Attempting download...")
            import os
            os.system("python -m spacy download en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
        
        # 2. Initialize the Neural Intelligence Node (Grammar Specialist)
        # We load a Seq2Seq model specialized in grammar synthesis (rewriting).
        print("[LQN] Initializing Offline-Ready Grammar Node (Local Specialist)...")
        
        # Local workspace path to the fine-tuned model weights
        self.model_path = r"c:\Users\Lokesh Patra\Documents\Projects\CapstoneProject26\models\grammar_specialist"
        
        # Loading tokenizer and model onto CPU for democratization across consumer hardware
        self.tokenizer  = AutoTokenizer.from_pretrained(self.model_path)
        self.model      = AutoModelForSeq2SeqLM.from_pretrained(self.model_path)
        self.device     = "cpu"
        self.model.to(self.device)
            
    def normalize_text(self, text):
        """
        Neural Normalization: Transforms messy/informal text into standard English.
        Example: 'i absolutli lik dis' -> 'I absolutely like this'
        """
        try:
            # Prepare the prompt for the T5 grammar synthesis task
            clean_input = text.strip('"\'').strip()
            prompt = f"grammar: {clean_input}"
            
            # Tokenize input and perform neural sequence generation
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=64,    # Limits length to prevent runaway generation
                    num_beams=5,      # High beam count for quality over speed
                    early_stopping=True
                )
            
            # Decode the generated sequence
            correction = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Final cleanup of the generated string
            clean_result = correction.split("grammar:")[-1].strip()
            return clean_result.capitalize() if clean_result else text.capitalize()
            
        except Exception as e:
            # If the neural engine fails, return the original text as a safety fallback
            print(f"[LQN] Intelligent Correction Error: {e}")
            return text.capitalize()

    def audit_text(self, text):
        """
        Performs a linguistic audit of the text to calculate a 'quality score'.
        Uses SpaCy to analyze parts of speech, tenses, and grammatical consistency.
        """
        doc = self.nlp(text)
        report = {
            "score": 100,
            "errors": [],
            "pos_counts": {"Nouns": 0, "Verbs": 0, "Adj": 0},
            "tense_diversity": set(),
            "subject_verb_check": "Stable"
        }
        
        # 1. Punctuation Analysis: Deduct points if the user didn't finish their thought
        if not text.endswith(('.', '!', '?')):
            report["errors"].append("Missing Terminal Punctuation")
            report["score"] -= 10
            
        # 2. Deep Structural Analysis: Iterate through tokens to count POS and track tenses
        for token in doc:
            if token.pos_ == "NOUN": report["pos_counts"]["Nouns"] += 1
            if token.pos_ == "VERB": 
                report["pos_counts"]["Verbs"] += 1
                tense = token.morph.get("Tense")
                if tense: report["tense_diversity"].update(tense)
            if token.pos_ == "ADJ":  report["pos_counts"]["Adj"] += 1

        # 3. Subject-Verb Agreement check: A basic heuristic to detect grammar errors
        # Example: 'He go' vs 'He goes'
        for token in doc:
            if token.dep_ == "nsubj" and token.head.pos_ == "VERB":
                subj_num = token.morph.get("Number")
                verb_num = token.head.morph.get("Number")
                if subj_num and verb_num and subj_num != verb_num:
                    report["errors"].append(f"Possible S-V Disagreement: '{token.text}' -> '{token.head.text}'")
                    report["score"] -= 15
                    report["subject_verb_check"] = "Anomalous"

        # 4. Sentiment Intensity Detector: Look for character inflation (e.g., 'soooo goood')
        if re.search(r'(.)\1{3,}', text): 
            report["errors"].append("Redundant Character Inflation (Hyper-Sentiment)")
            report["score"] -= 5

        # Final Formatting of the report dictionary
        report["tense_diversity"] = list(report["tense_diversity"])
        report["score"] = max(0, report["score"])
        return report
