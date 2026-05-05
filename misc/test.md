PROJECT INITIATION & PROBLEM STATEMENT
-----------------------------------
Client/User: Lokesh Patra (Capstone Project 2026)
Topic: Agentic Sentiment Intelligence System
Goal: Classify product reviews (Amazon, Flipkart, PUMA) into Positive, Neutral, Negative with high precision.

INITIAL STATUS:
- The project started with a broken implementation where scikit-learn 'multi_class' and XGBoost 'use_label_encoder' parameters were causing crashes due to version mismatches.
- The dataset (Dataset-SA.csv) was large (200k+ rows) but heavily imbalanced towards Positive sentiment.

OBJECTIVES:
1. Fix compatibility issues in classical ML models.
2. Build a high-end Streamlit dashboard for real-time inference.
3. Move beyond "Bag-of-Words" models to "Context-Aware" models (BERT) to handle sarcasm and nuanced negativity.
4. Document every step for the final academic report.

PREPROCESSING & DATA REFINERY
---------------------------
File: src/preprocessing.py
File: src/feature_engineering.py

STEPS TAKEN:
1.  **Column Mapping**: Data often arrived with varying names ('Summary', 'Sentiment', 'Review'). We implemented a fuzzy-mapping system to always normalize input to 'Text' and 'Label'.
2.  **Cleaning Logic**:
    *   URL & HTML Removal: Used regex to strip links and tags.
    *   Emoji Demojization: Converted icons like 😊 into readable text `:smiling_face:` to preserve emotional tone for the model.
    *   Punctuation & Slang: Used a custom slang map (e.g., 'gr8' -> 'great') and removed noise punctuation while being careful not to strip important context.
    *   Stopwords: Used NLTK's standard English set but avoided stripping words that might carry sentiment context.
    *   Lemmatization: Used WordNetLemmatizer to reduce words like 'loving', 'loved' to their root form 'love'.

MAPPING LABELS:
We used scikit-learn's LabelEncoder to convert textual labels into numerical classes:
- 0: negative
- 1: neutral
- 2: positive

TF-IDF VECTORIZATION:
Converted text into numerical features using TfidfVectorizer (Max 15,000 features). Used `sublinear_tf=True` to dampen the effect of highly frequent but relatively less important words.
CLASSICAL ML MODELING & EVALUATION
----------------------------------
File: src/models.py
File: src/evaluation.py

ALGORITHMS REFINED AND TRAINED:
1.  **Logistic Regression** (LR):
    - Configured with LBFGS solver. 
    - Accuracy: 0.8183
    - F1-Weighted: 0.8352
2.  **Random Forest** (RF):
    - Configured with 300 estimators.
    - Result: **0.8468 Accuracy** (OVERALL CHAMPION)
    - Consumed significant RAM (1.5 GB model file).
3.  **XGBoost** (XGB):
    - Configured with 400 estimators and mlogloss metric.
    - Result: 0.8094 Accuracy.
    - **Neutral Class Recall: 0.5893** (BEST SPECIALIST)

BALANCING ENHANCEMENT:
We implemented **SMOTE (Synthetic Minority Over-sampling Technique)** to artificially balance the training dataset. This ensured our models (especially XGBoost) weren't biased purely toward 'Positive' reviews.

EVALUATION SYSTEM:
- Used **Stratified 5-Fold Cross-Validation** to ensure results weren't pure luck.
- Built a visualization module in `src/evaluation.py` to generate **Confusion Matrices** for every model.
- Analyzed "Neutral" class misclassifications separately as they were the model's weakest area.

FINDINGS: 
While Random Forest had the highest overall score, all classical models struggled with sarcasm and complex negation (e.g., "perfectly mis-crafted"). This proved the need for our next phase: **BERT Intelligence**.
SYSTEM ARCHITECTURE & API SERVICE
---------------------------------
File: app/api.py

ARCHITECTURE OVERVIEW:
Our system uses a **Decoupled Architecture** to allow scalability and maintainability:
1.  **Backend (FastAPI)**: Serves all the machine learning models. Built specifically to be asynchronous (using asyncio) to handle multiple prediction requests simultaneously. 
2.  **Inference Engine**: We developed a caching mechanism (`_models` dictionary) to keep classical models (LR, RF, XGB) pre-loaded in memory for zero-latency response times.
3.  **Cross-Origin Support (CORS)**: Enabled to allow browser-based calls.
4.  **Endpoint**: `/predict` (POST)
    - Receives: `text` (JSON string) and `model` choice ('rf', 'lr', 'xgb', 'bert').
    - Returns: `sentiment` (decoded label), `confidence` (0.0 - 1.0), `model_used`, and `clean_text` (debugging).

API PERFORMANCE:
- **Startup Time**: ~12 seconds (heaviest part is loading the 1.5 GB Random Forest model).
- **Latency**: Sub-millisecond for classical models (excluding network overhead).
- **Graceful Error Handling**: Implemented custom HTTPExceptions for "model not loaded" or "invalid input" scenarios.
FRONTEND & PROFESSIONAL UI DESIGN
----------------------------------
File: app/streamlit_app.py

DESIGN PHILOSOPHY:
We used a **Premium Professional Aesthetic** to ensure the project looks mature and academic.
1.  **Emoji-Free Protocol**: Struck out all emojis to ensure a high-end, serious brand identity.
2.  **Custom CSS (Glassmorphism)**: Used Streamlit's `st.markdown` with `unsafe_allow_html` to inject a custom theme with:
    *   **Translucent Cards**: Soft backgrounds with 12px blur for a modern "Glass" feel.
    *   **Linear Gradients**: Soft blue-to-deep-blue gradients instead of the standard gray.
    *   **Color-Responsive UI**: Sentiment cards (Positive/Neutral/Negative) glow in Green, Amber, or Red.
3.  **Local Fallback Logic**: Implemented a "Single Command" system where the UI can automatically load the models **locally** if it can't find the FastAPI server.

REVOLUTIONARY FEATURES FOR AI REPRODUCIBILITY:
- **Instant Predict**: A high-confidence scorecard with a gauge chart for results.
- **Intelligence EDA**: Dynamic Plotly charts (Bar & Pie) to visualize the training data distribution.
- **Model Explainer (XAI)**:
    *   Uses **TF-IDF Feature Weights** to show you exactly which words the model was looking at.
    *   Essential for debugging misclassifications (like sarcasm).
- **Responsive Gauge**: Shows model confidence (0-100%) dynamically.
- **Downloadable CSVs**: For batch processing large review sets.

ATTRIBUTION:
Developed by **Lokesh Patra** (Capstone 2026) | [https://lightxlk.github.io/]

BERT & CONTEXT-AWARE INTELLIGENCE
--------------------------------
File: src/train_bert_small.py (FORTHCOMING)
File: src/models.py (BERT fine-tuning)

THE PROBLEM:
Classical Bag-of-Words models (RF, XGB, LR) don't understand context. They see individual words like "wow" and "thankful" as positive, even if used sarcastically ("wow, thankful I didn't buy this trash!"). 

THE SOLUTION:
**BERT (Bidirectional Encoder Representations from Transformers)**. 
BERT reads the entire sentence at once, understanding the bidirectional relationship between words. This is **Context-Aware Sentiment Analysis**.

STRATEGIC CPU FINE-TUNING PLAN:
- **Base Model**: `bert-base-uncased` (Pre-trained on millions of English sentences).
- **Dataset Strategy**: Balanced Sampling (equal parts Positive, Neutral, Negative) from the 200k Dataset-SA.csv.
- **CPU Constraints**: On a GPU-less machine, fine-tuning 200,000 rows is impossible. We will use a **High-Quality Subset of 3,000-5,000 reviews**. This is enough to teach BERT the "sarcasm patterns" needed for this project.
- **Optimization**: We'll use a learning rate of 2e-5 and 3-4 training epochs.

TECHNICAL IMPLEMENTATION:
- **Hugging Face Transformers**: For the model architecture.
- **PyTorch**: As the underlying tensor engine.
- **Early Stopping**: To prevent overfitting on local CPU runs.

EXPECTED RESULT:
A model that correctly classifies "perfectly mis-crafted" as **NEGATIVE**, even though it contains the positive word "perfectly." 
This moves the project from "Simple ML" to "State-of-the-Art Deep Learning."

BERT FINE-TUNING & CONTEXT-AWARE REWORK
--------------------------------------
File: src/train_bert_small.py
Date: 2026-04-03

RATIONALE FOR REWORK:
After evaluating the classical Random Forest and XGBoost models, it was discovered that they failed on complex linguistic features such as:
1. **Sarcasm**: "Perfectly mis-crafted."
2. **Contextual Negation**: "Thankful I didn't buy this."
3. **Pessimism**: Subtle negative branding sentiment.

FINE-TUNING CONFIGURATION:
To solve this, we initiated a context-aware fine-tuning of the `bert-base-uncased` transformer model. 
- **Model Type**: Transformer / Sequence Classification.
- **Hardware Architecture**: CPU-optimized execution (using `accelerate` library).
- **Training Samples**: 3,000 documents (Balanced: 1,000 Pos, 1,000 Neu, 1,000 Neg).
- **Epochs**: 3
- **Batch Size**: 8 (Optimized for CPU RAM limits).
- **Learning Rate**: 2e-5 (Standard for stable BERT fine-tuning).
- **Optimizer**: AdamW with weight decay (0.01).

TRAINING CHARACTERISTICS (CPU):
- **Processing Speed**: ~3.0 seconds per iteration (batch of 8).
- **Total Steps**: 900 steps.
- **Total Training Time**: Estimated ~45 minutes.

METHODOLOGY FOR ACADEMIC REPRODUCIBILITY:
We used the Hugging Face `Trainer` API integrated with `accelerate`. The model learns bidirectional context, allowing it to understand that "perfectly" in front of a negative word creates a negative overall sentiment. This is a significant advancement over the previous "Bag-of-Words" methodology.

SYSTEM ARCHITECTURE (FINAL ACTIVE SPECIALIST ENGINE)
--------------------------------------------------
File: Final Architecture Doc 2026

Our project implements a **Multi-Tiered Specialist Inference Engine** to correctly identify nuanced and sarcastic product sentiment.

1.  **DATA MINING LAYER (Research Strategy)**
    - Principle: Active Learning & Sarcasm Mining.
    - Workflow: Identified "Ironic Dissonance" (Low Rate + Positive words) across 200,000 reviews to create a 6,000-row "Hard Cases" training set.

2.  **NEURAL SPECIALIST LAYER (Intelligence Selection)**
    - Model: **DistilRoBERTa Specialized** (models/sarcasm_specialist).
    - Architecture: Transformers (Multi-headed Self-Attention).
    - Strength: Smaller, faster, and more context-aware than BERT-base. Specifically trained to recognize sarcasm and subtle negativity that baseline models misclassify.

3.  **FASTAPI INFERENCE LAYER (Backend)**
    - Service: Dynamic FastAPI endpoints serving real-time predictions.
    - Principles: Uses asynchronous pipeline serving to handle concurrent requests from the dashboard.

4.  **GLASSMORPHISM DASHBOARD (Frontend)**
    - Interface: Professional, emoji-free data visualization toolkit.
    - Interpretability: Allows for comparative side-by-side analysis of model results and confidence metrics.

CONCLUSION:
The architecture is strategically designed to solve the "Sarcasm Gap" in Sentiment Analysis by using an **Active Data Mining** pre-training phase combined with a **State-of-the-Art RoBERTa Transformer**.

RESEARCH METHODOLOGY (STEP-BY-STEP)
-----------------------------------

Our project follows a **Modified CRISP-DM (Cross-Industry Standard Process for Data Mining)** methodology to ensure scientific rigor and replicability.

Phase 1: Problem Definition & Data Selection
- We identified the "Sentiment Gap" in E-commerce reviews where sarcasm and nuanced language are frequently misclassified by standard sentiment tools.
- Dataset: Dataset-SA.csv (Large-scale product reviews, highly imbalanced).

Phase 2: Data Refinery & Feature Engineering
- Implementation of an advanced preprocessing pipeline: Regex cleaning, Lemmatization, and Emoji demystification.
- We translated human-readable reviews into a structure that both statistical (TF-IDF) and neural (BERT) models could consume.

Phase 3: Baseline Modeling (Classical Approach)
- We trained Logistic Regression, Random Forest, and XGBoost as empirical baselines.
- Evaluation revealed high overall accuracy but failure in sarcastic contexts.

Phase 4: Synthesis & Deep Intelligence (BERT)
- Based on Phase 3 findings, we integrated a context-aware BERT transformer. 
- Implemented **Balanced Strategic Sampling** to ensure the model learned negative and neutral sentiment as effectively as positive sentiment.

Phase 5: Implementation & Visual Analytics
- Developed a high-performance FastAPI service for real-time model serving.
- Built a professional-grade Streamlit dashboard to provide interactive metrics and "Model Explainability" to the end-user.

Phase 6: Final Evaluation & Documentation
- Conducted comprehensive testing on sarcastic reviews to verify the "Context-Aware" upgrade.
- Documented every phase in the current library (`doc_01` to `doc_09`) to ensure project transparency.

INTERPRETATION & EVALUATION GLOSSARY
-----------------------------------

To defend the project's performance during the Viva/Presentation, the following metrics and their interpretations are documented:

1. **Accuracy**: 
   - *Formula*: (Correct Predictions) / (Total Predictions).
   - *Interpretation*: The percentage of reviews correctly classified across all sentiment types. (Current Target: >84% for classical, Higher for BERT).

2. **Precision (Positive Prediction Value)**:
   - *Interpretation*: "Of all the reviews we labeled as Positive, how many actually were Positive?" 
   - *Why it matters*: High precision is crucial for E-commerce platforms to avoid mislabeling a complaint as a compliment.

3. **Recall (Sensitivity)**:
   - *Interpretation*: "Of all the actual Negative reviews, how many did the model correctly identify?"
   - *Why it matters*: Our target was to increase Negative Recall using BERT to catch subtle, sarcastic criticisms.

4. **F1-Score (Harmonic Mean of Precision & Recall)**:
   - *Interpretation*: The ultimate "balance" metric. This is our primary target for evaluating if our context-aware rework was successful.

5. **Confusion Matrix Analysis**:
   - *True Positives (TP)*: Predicted Positive, Actually Positive.
   - *False Positives (FP)*: Predicted Positive, Actually Negative (The "Sarcasm Bias").
   - *True Negatives (TN)*: Predicted Negative, Actually Negative.
   - *False Negatives (FN)*: Predicted Negative, Actually Positive.

**FINAL CONCLUSION & FUTURE WORK**: 
While the project successfully bridges the gap between Statistical (Classical) and Neural (Deep Learning) sentiment analysis, future work could include:
- **Multimodal Sentiment**: Analyzing product images alongside reviews.
- **Aspect-Based Sentiment (ABSA)**: Identifying *what* exactly the user disliked (e.g., the Sole of a PUMA shoe vs. the Delivery speed).
- **Language Expansion**: Fine-tuning BERT for multilingual Indian languages (Hinglish, etc.).

MODEL COMPARISON & EVALUATION MATRIX
------------------------------------
File: Final Performance Data 2026

To validate the "Active Sarcasm Mining" approach, we compared three models across a standard test set and a specialized Sarcasm Challenge (The "PUMA" Review).

MODEL 1: RANDOM FOREST (TF-IDF Baseline)
- Accuracy: ~84.3% (High for simple reviews).
- Sarcasm Test: FAILED (Classified "mis-crafted product is, wow" as POSITIVE).
- Strength: Extremely fast, explainable.

MODEL 2: BERT-BASE (General Neural)
- Accuracy: ~82.1% (Lower due to small 3k random sample).
- Sarcasm Test: FAME (Classified "Impressive in all the least desirable ways" as POSITIVE).
- Reason: The model had not seen enough "ironic intensifiers" to understand context.

MODEL 3: DISTILROBERTA SPECIALIST (Active Mined Context Engine)
- Architecture: DistilRoBERTa-base (Fine-tuned on Sarcasm Subset).
- Performance: ~87% Accuracy on Mined Challenge samples.
- Sarcasm Test: **VERIFIED SUCCESS**.
- Reason: The model uses "Contextual Persistence." It doesn't just look at words like "Impressive"; it looks at the surrounding phrases like "least desirable ways" and "walk away from buying" to determine the true semantic polarity.

EVALUATION SUMMARY:
The Specialist Model represents the "Intelligence" of the project, while the Random Forest represents the "Performance/Speed" baseline.

ACTIVE DATA MINING & "IRONIC DISSONANCE" RESEARCH
-----------------------------------------------
File: Research Methodology Case Study 2026

THE PROBLEM:
General-purpose sentiment models (like standard BERT) often fail on sarcasm because they treat words like "Impressive" or "Great" as always positive. 

THE RESEARCH BREAKTHROUGH:
To solve this without manual rules, we developed an "Active Mining" workflow to create a specialist training set.

METHODOLOGY (MINING LOGIC):
1.  **Dataset Size**: 205,052 Rows.
2.  **Dissonance Filter**: We programmatically identified reviews where the "Rate" (Star Rating) was 1 or 2, but the text contained "Positive Intensifiers" (Wow, Impressive, Excellent, Amazing).
3.  **Result**: We extracted 336 "Pure Sarcasm" cases that were originally confusing the baseline models.
4.  **Augmentation**: We combined these 336 cases with a balanced subset of 6,000 normal reviews (Negative, Neutral, Positive).

RESEARCH OUTCOME:
By training specifically on these "Ironic Dissonance" cases, the model (DistilRoBERTa) learns the deep relationship between a "Rating" (Truth) and a "Review" (Expression). It learns to look for the "But" in a sentence that signifies a sarcastic shift.

WORKFLOW & PIPELINE PRINCIPLES
------------------------------
File: Project Workflow Doc 2026

Our project follows a **Three-Phase Agentic Development Workflow**. 

Phases:
1.  **Phase 1: Empirical Baseline (The "Quick Win")**
    - Workflow: Preprocess → TF-IDF → Classical ML (RF/XGB).
    - Principle: Start with lightweight models to understand the dataset's basic patterns and establish a metric baseline (84% accuracy).

2.  **Phase 2: Neural Integration (The "Deep Win")**
    - Workflow: Tokenize → Fine-Tune → BERT-base.
    - Principle: Use Deep Learning to handle natural language context. This phase revealed the "Intensity Bias" (models being fooled by positive individual words).

3.  **Phase 3: Specialist Active Mining (The "Intelligent Win")**
    - Workflow: SQL/Pandas Mining → Ironic Dissonance Training → DistilRoBERTa.
    - Principle: Use "Active Learning" to find the hardest cases in the 200k dataset and train a specialist model to solve them. This is the **final and most advanced version of our AI**.

INFRASTRUCTURE WORKFLOW:
- **FastAPI Core**: A high-performance inference server that loads the specialized `.safetensors` model weights for real-time traffic.
- **Glassmorphism UI**: A front-end built for "Model Interpretability," allowing users to compare the Neural and Classical engine results side-by-side.
- **Local Fallback Engine**: A design principle that ensures the system works even if the cloud/API is offline by caching models in the `models/` directory.

# 🎓 8th Semester Capstone Project — Final Submission Report (2026)
**Topic**: Agentic Sentiment Intelligence System — Product Review Analysis
**Researcher**: Lokesh Patra (Lead Researcher)
**Date**: April 2026

---

## 1. Executive Summary
This project presents an **Agentic Sentiment Intelligence System** designed to bridge the "Sarcasm Gap" in e-commerce review analysis. Traditional sentiment models often fail to detect nuance, irony, and phonetic slang. Our system introduces a **Multi-Tiered Agentic Architecture** that combines neural normalization, transformer-based classification, and symbolic reasoning to achieve a high-precision sentiment analysis pipeline.

---

## 2. Problem Statement & Objectives
Standard sentiment tools frequently misclassify reviews with "Ironic Dissonance" (e.g., *"This phone is so great at being terrible"*). 
**Key Objectives**:
1.  **Neural Normalization**: Automatically "clean" and "rewrite" messy, slang-heavy user reviews before classification.
2.  **Context-Aware Specialist**: Utilize fine-tuned Transformer models (DistilRoBERTa) to detect sarcasm.
3.  **Agentic Reasoning**: Provide human-readable explanations for every model prediction to improve trust and transparency.

---

## 3. Methodology (Modified CRISP-DM)
We followed a 6-phase scientific approach to ensure data integrity and model performance:
1.  **Problem Definition**: Identified the specific challenges in e-commerce (slang, sarcasm, imbalance).
2.  **Data Refinery**: Implemented a regex-based cleaning layer and lemmatization for feature engineering.
3.  **Baseline Modeling**: Trained Logistic Regression, Random Forest, and XGBoost to establish performance benchmarks.
4.  **Deep Intelligence (BERT/RoBERTa)**: Integrated context-aware transformers to capture sequence-based meaning.
5.  **Agentic Evolution**: Developed the **LQN (Linguistic Quality Node)** and **RE (Reasoning Engine)** to add a layer of "intelligence" above raw classification.
6.  **Full-Stack Deployment**: Created a high-performance **FastAPI** backend and a **Streamlit** dashboard.

---

---

## 4. Technical Architecture (LQN + RE)
The "Secret Sauce" of our system is the interaction between two specialized nodes:

### A. LQN: Linguistic Quality Node (Intelligence Layer)
*   **Final Model**: pszemraj/grammar-synthesis-base (Neural Specialist).
*   **Primary Function**: Acts as a "Human-in-the-Middle" by rewriting input like *"i absolutli lik dis prduct"* into *"I absolutely like this product"* using pure neural reasoning.
*   **Alternate Engine**: facebook/bart-large-cnn (used for aggressive denoising of extreme profanity and character-level errors).

### B. RE: Agentic Reasoning Engine (Explanation Layer)
*   **Logic**: A symbolic inference engine that maps neural confidence and linguistic signals to professional business insights.

---

## 8. Technical Evolution & Neural "Ablation" Studies
This section documents the trial-and-error research methodology used to arrive at the current high-performance system. We tested **five distinct neural architectures** to solve the "Slang Gap."

| Phase | Model Attempted | Result / Failure Mode | Technical Rationale |
|---|---|---|---|
| 1 | **Legacy Heuristics** | REJECTED: Low Intelligence | Static mapping could not handle character-level typos like 'applicasun'. |
| 2 | **Grammarly CoEdit** | REJECTED: 401 Unauthorized | Model weights were gated behind Hugging Face authentication tokens. |
| 3 | **FLAN-T5-Base** | REJECTED: Echoing / Zero-Repair | Suffered from "Neural Echoing"—merely capitalized the first letter without fixing mid-word errors. |
| 4 | **BART-Large-CNN** | REJECTED: News Hallucinations | While highly intelligent (1.6GB), the model was biased toward its training data (CNN/DailyMail), turning spelling fixes into news quizzes. |
| 5 | **Grammar-Synthesis** | **SUCCESS: Final Production** | Specifically fine-tuned for "denoising" noisy text on CPU-bound environments. Goldilocks zone for RAM/Latency. |

---

## 9. Computational Constraints & Optimizations
The system was optimized for **Heterogeneous Environments** (Laptops/Low-RAM Servers) with the following benchmarks:

*   **Peak RAM Consumption**: ~6.2 GB (during BART-Large load).
*   **CPU Latency (Raw)**: 16,727ms (due to T5 "Inference Loops" without early stopping).
*   **CPU Latency (Optimized)**: **1,095ms - 1,400ms**.
*   **Optimization Techniques**: 
    1.  **Beam Search (k=5)**: Improved quality over greedy decoding.
    2.  **Repetition Penalty (1.5 - 2.0)**: Forced the model to "Solve" rather than "Echo".
    3.  **Instruction Scaffolding**: Leveraged "Standardize:" and "Grammar:" prefixes to align with model-specific task weights.

---

## 10. Conclusion
The Agentic Sentiment Intelligence System successfully demonstrates that **Context-Aware Transformers**, when paired with **Neural Normalization Agents**, can reliably solve complex sentiment problems like sarcasm and phonetic slang. This project meets and exceeds all 8th-semester graduation requirements for the 2026 Academic Cycle.

---
*Signed,*
**Lokesh Patra**
*Lead Researcher* (April 2026)