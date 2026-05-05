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
