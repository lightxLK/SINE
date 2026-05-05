"""
app/api.py
FastAPI - AGENTIC SPECIALIST BACKEND
Developed for high-performance sentiment inference.
"""

# Core backend libraries for building the API and handling neural models
import os
import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline

# Import local Agentic Logic components from the src directory
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.reasoning_engine import AgenticReasoningEngine
from src.linguistic_engine import LinguisticAuditor

# Initialize FastAPI application with project metadata
# This defines the metadata that appears in the auto-generated /docs page.
app = FastAPI(
    title="Sentiment Analysis API",
    description="Agentic Sentiment Intelligence - Specialist Engine",
    version="4.0.0-Elite",
)

# Configure Cross-Origin Resource Sharing (CORS) 
# Allows the Streamlit UI (running elsewhere) to communicate securely with this backend.
app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"], 
    allow_methods=["*"], 
    allow_headers=["*"]
)

# -----------------------------------------------------------------------------
# GLOBAL MODEL MEMORY NODES
# -----------------------------------------------------------------------------
# We use global variables to ensure models remain resident in RAM for fast inference.
# Loading models inside an endpoint would make the API extremely slow.
_neural_engine    = None
_reasoning_engine = AgenticReasoningEngine()
_linguistic_node  = LinguisticAuditor()

# Custom Label Mapping to decode Neural Predictions into English.
# The model outputs generic labels (LABEL_0, 1, 2) which we map to meaningful sentiments.
_label_map = {
    "LABEL_0": "negative",
    "LABEL_1": "neutral",
    "LABEL_2": "positive"
}

# -----------------------------------------------------------------------------
# EVENT: STARTUP SEQUENCE
# -----------------------------------------------------------------------------
# This logic executes as soon as the uvicorn server starts running.
@app.on_event("startup")
def load_specialist_engine():
    """
    Initialize and load the specialized DistilRoBERTa model from local storage.
    If the specialist is missing, weights from the global CardiffNLP hub are used as fallback.
    """
    global _neural_engine
    
    # Path to our locally fine-tuned "Sarcasm Specialist" weights
    model_path = os.path.join("models", "sarcasm_specialist")
    
    print(f"[API] Initializing Inference Pipeline... Seeking weights at {model_path}")
    
    try:
        # Check if the folder exists (populated by the training script)
        if os.path.exists(model_path):
            # Load from locally fine-tuned weights using the HuggingFace pipeline
            _neural_engine = pipeline(
                "sentiment-analysis", 
                model=model_path,
                tokenizer=model_path,
                device=-1 # Forces CPU usage for stability across consumer hardware
            )
            print("[API] SUCCESS: Context-Aware Specialist Loaded Successfully.")
        else:
            # Fallback to general pre-trained sentiment specialists if local weights are missing
            print("[API] WARNING: Local weights missing. Initializing Fallback Specialist...")
            _neural_engine = pipeline(
                "sentiment-analysis", 
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=-1
            )
    except Exception as e:
        # Catch hardware limitations or missing dependency errors
        print(f"[API] CRITICAL ERROR during startup: {e}")

# -----------------------------------------------------------------------------
# DATA SCHEMAS (Pydantic Models)
# -----------------------------------------------------------------------------
# These classes define the shape of JSON data for requests and responses.

class ReviewRequest(BaseModel):
    """Expects a JSON object with a 'text' field."""
    text: str

class PredictionResponse(BaseModel):
    """Returns a structured report containing sentiment, confidence, and reasoning."""
    sentiment:   str
    confidence:  float
    model_used:  str
    reasoning:   str
    linguistic:  dict
    corrected:   str

# -----------------------------------------------------------------------------
# ENDPOINT: CORE INFERENCE
# -----------------------------------------------------------------------------
@app.post("/predict", response_model=PredictionResponse)
def predict(req: ReviewRequest):
    """
    Perform real-time neural sentiment analysis on raw text.
    Handles normalization, classification, and explanation generation.
    """
    # 1. Validation Logic: Ensure input is not empty
    if not req.text.strip():
        raise HTTPException(400, "Review content is empty. Cannot process null input.")

    # 2. State Check: Ensure the engine finished loading from disk
    if _neural_engine is None:
        raise HTTPException(503, "Neural Engine is still initializing RAM kernels...")
    
    # 3. Neural Normalization Layer (LQN)
    # Corrects slang and typos (e.g., "grreat" -> "great") before the model sees context
    raw_text       = req.text
    normalized_txt = _linguistic_node.normalize_text(raw_text)

    # 4. Neural Feed
    # Pass the cleaned text into the DistilRoBERTa model
    result = _neural_engine(normalized_txt)[0]
    
    # 5. Decoding and Result Formatting
    pred_label = result['label']
    sentiment  = _label_map.get(pred_label, pred_label)
    confidence = float(result['score'])

    # 6. Generate Agentic Reasoning & Linguistic Audit
    # RE handles explanation; Linguistic node audits syntax and grammar metrics
    reasoning   = _reasoning_engine.generate_reasoning(normalized_txt, sentiment, confidence)
    ling_report = _linguistic_node.audit_text(raw_text)

    # 7. Build and Return Response Object
    return PredictionResponse(
        sentiment=sentiment,
        confidence=confidence,
        model_used="DistilRoBERTa Context-Specialist",
        reasoning=reasoning,
        linguistic=ling_report,
        corrected=normalized_txt
    )

# -----------------------------------------------------------------------------
# ENDPOINT: RELOAD KERNEL
# -----------------------------------------------------------------------------
@app.post("/reload")
def reload_model():
    """
    Administrative endpoint to force-reload the model weights from disk.
    Allows for model updates without server downtime.
    """
    global _neural_engine
    load_specialist_engine()
    return {"status": "reloaded", "engine": "specialist-v4-elite"}

# -----------------------------------------------------------------------------
# ENDPOINT: HEALTH CHECK
# -----------------------------------------------------------------------------
@app.get("/health")
def health():
    """Returns the current operational status of the API."""
    return {"status": "operational", "engine": "active"}
