# System Design Specification: Agentic Sentiment Intelligence System

## Table of Contents

1. [Project Vision and Strategic Problem Definition](#1-project-vision-and-strategic-problem-definition)
2. [High-Level Multi-Tiered Agentic Architecture](#2-high-level-multi-tiered-agentic-architecture)
3. [Deep Dive: Linguistic Quality Node (LQN)](#3-deep-dive-linguistic-quality-node-lqn)
4. [Deep Dive: Agentic Reasoning Engine (RE)](#4-deep-dive-agentic-reasoning-engine-re)
5. [Backend Infrastructure: FastAPI Service Layer](#5-backend-infrastructure-fastapi-service-layer)
6. [Frontend Architecture: Glassmorphism Streamlit Dashboard](#6-frontend-architecture-glassmorphism-streamlit-dashboard)
7. [Computational Optimizations for CPU-Bound Environments](#7-computational-optimizations-for-cpu-bound-environments)
8. [System Evaluation and Performance Metrics](#8-system-evaluation-and-performance-metrics)

---

## 1. Project Vision and Strategic Problem Definition

The "Sarcasm Gap" represents a critical failure point in modern e-commerce sentiment analysis. While traditional models are adept at identifying polarized language, they frequently falter when encountering "Ironic Dissonance"—a phenomenon where the semantic content of a review contradicts its intended sentiment (e.g., "This phone is so great at being terrible"). 

Standard "Bag-of-Words" models, such as Logistic Regression or Random Forests, are fundamentally limited by their reliance on TF-IDF sublinear scaling. These models treat tokens as independent units, often overweighting positive intensifiers like "perfectly" while ignoring the bidirectional relationship and negative modifiers that follow (e.g., "perfectly miscrafted"). 

This architecture provides a multi-tiered resolution to this gap by moving beyond keyword matching toward a sophisticated understanding of bidirectional sequence context and intent.

### Core Objectives

*   **Neural Normalization:** Automatically perform stochastic denoising and rewrite messy, slang-heavy user reviews using generative models before they reach the classification stage.
*   **Context-Aware Specialization:** Utilize fine-tuned Transformer models, specifically a **DistilRoBERTa Specialist**, to detect subtle irony and semantic polarity shifts.
*   **Agentic Reasoning:** Provide human-readable explanations for every model prediction to improve trust and transparency for professional users through symbolic inference.

When sarcasm is misclassified, business intelligence is compromised, leading to skewed data that masks severe product defects under a veneer of false positivity. This architecture is specifically designed to bridge this gap through specialized neural tiers.

---

## 2. High-Level Multi-Tiered Agentic Architecture

A decoupled, tiered architecture is essential for scaling AI services across heterogeneous environments. By separating the normalization, inference, and reasoning layers, the system ensures that each component can be optimized independently for either maximum semantic accuracy or computational efficiency. This system follows a modified CRISP-DM methodology, evolving from empirical baselines to active specialist mining.

### Three-Phase Agentic Development Workflow

1.  **Empirical Baseline:** Establishing a performance ceiling using lightweight classical models (Logistic Regression, Random Forest, XGBoost) to understand basic dataset patterns and establish a metric baseline.
2.  **Neural Integration:** Transitioning to Deep Learning via BERT-base to capture natural language context and identifying where "Intensity Bias" (being fooled by positive keywords) occurs.
3.  **Specialist Active Mining:** Utilizing "Active Learning" to identify the most difficult cases—specifically ironic dissonance—and training a specialist DistilRoBERTa model to resolve them.

### Architectural Layer Functional Map

| Layer Name | Primary Component | Strategic Value |
| :--- | :--- | :--- |
| **Data Mining Layer** | Active Sarcasm Mining | Identifies "Hard Cases" to create a specialized 6,000-row training set via dissonance filtering. |
| **Linguistic Quality Node** | Neural Normalizer | Performs stochastic denoising of phonetic slang and character-level errors. |
| **Neural Specialist Layer** | DistilRoBERTa Specialist | Provides high-precision, context-aware sentiment detection for sarcastic content. |
| **FastAPI Inference Layer** | Asynchronous API | Enables real-time, scalable asynchronous pipeline serving for concurrent requests. |
| **Glassmorphism Layer** | Streamlit Dashboard | Delivers a professional UI with Explainable AI (XAI) features for end-users. |

The decision to transition from a general BERT-base model to a DistilRoBERTa Specialist was driven by the need for both accuracy and efficiency. While BERT-base provides a strong foundation, it is often fooled by intensifiers. The DistilRoBERTa Specialist is smaller, faster, and specifically trained on mined "ironic dissonance" cases, allowing it to maintain an 87% accuracy rate on complex samples. This shift transitions the system to a high-intelligence Linguistic Quality Node.

---

## 3. Deep Dive: Linguistic Quality Node (LQN)

The Linguistic Quality Node (LQN) serves as the system’s "Neural Normalizer." Its primary role is to mitigate the noise introduced by phonetic slang, character-level errors, and "messy" user input. By auditing and rewriting text before it reaches the sentiment engine, the LQN ensures the downstream classifier receives high-quality, standardized data.

### Model Selection Ablation Studies

| Model Name | Failure Mode / Success | Technical Rationale |
| :--- | :--- | :--- |
| `FLAN-T5-base` | **REJECTED**: Neural Echoing | Suffered from "Neural Echoing"; merely capitalized the first letter without repairing mid-word errors or slang. |
| `BART-Large-CNN` | **REJECTED**: News Hallucinations | Biased toward CNN/DailyMail training data; turned spelling fixes into news-style summaries. High 6.2 GB RAM load. |
| `pszemraj/grammar-synthesis-base` | **SUCCESS**: Final Production | Specifically fine-tuned for denoising noisy text on CPU-only environments. Optimal RAM/Latency ratio. |

The LQN operates as a "Human-in-the-Middle" by applying pure neural reasoning to input text. For example, a raw review stating *"i absolutli lik dis prduct"* is programmatically rewritten into *"I absolutely like this product."* This normalization drastically reduces the vocabulary size and prevents misclassifications caused by typos or shorthand (e.g., "gr8" → "great"). By stripping away linguistic noise before the inference phase, the system achieves higher reliability across varied e-commerce data sources.

---

## 4. Deep Dive: Agentic Reasoning Engine (RE)

The Agentic Reasoning Engine (RE) provides the symbolic reasoning required to bridge the gap between "black-box" neural outputs and human-understandable business insights. For professional audiences, this Explainable AI (XAI) layer is critical for building trust in automated decisions.

The RE utilizes Conflict Logic and Symbolic Inference to map neural confidence levels and linguistic signals into actionable reports. If the system detects a mixture of sentiments—such as "Good price but bad quality"—the RE identifies a "Conflict of Signals." It programmatically explains that the Specialist engine prioritized the "Dealbreaker" attribute (quality) over the "Incentive" attribute (price) to reach its negative conclusion.

### Ironic Dissonance Miner Logic

1.  **Detector:** Scans the 200,000+ row dataset for instances where star ratings are low (1 or 2) but text contains "Positive Intensifiers" (e.g., "Wow," "Amazing").
2.  **Correction:** Programmatically identifies these cases as "Pure Sarcasm" and assigns a 'Negative' label to the training sample.
3.  **Synthetic Augmentation:** Applies a 10x weighting to these samples during training to force the model to learn rare linguistic patterns of sarcasm.
4.  **Result:** The model learns that "Wow" paired with a 1-star rating signifies extreme disappointment.

---

## 5. Backend Infrastructure: FastAPI Service Layer

The backend is powered by a FastAPI service layer, chosen for its asynchronous, decoupled design which is ideal for real-time inference. This architecture allows the system to handle multiple prediction requests simultaneously without blocking the user interface.

### API Communication Structure

*   **Request JSON:** `{"text": "Sample review text", "model_choice": "bert"}`
*   **Response JSON:** `{"sentiment": "negative", "confidence": 0.89, "model_used": "distilroberta_specialist", "clean_text": "Normalized sample review text"}`

To ensure sub-millisecond latency for classical models, the system implements an "Inference Engine" caching mechanism where models (LR, RF, XGB) are pre-loaded into a `_models` dictionary. The system's **12-second startup time** is primarily attributed to the loading of the 1.5 GB Random Forest model weights from the disk. Furthermore, the system includes robust error-handling protocols, utilizing custom HTTPExceptions to manage "model not loaded" scenarios or "invalid input" errors.

---

## 6. Frontend Architecture: Glassmorphism Streamlit Dashboard

The frontend adopts a "Premium Professional Aesthetic" through a Streamlit-based dashboard. To maintain a serious academic brand identity, the system follows a strict **Emoji-Free Protocol**, ensuring the focus remains on data and intelligence.

### Key Features

*   **Instant Predict:** A high-confidence scorecard featuring a gauge chart for immediate sentiment visualization.
*   **Intelligence EDA:** Dynamic Plotly-powered visualizations displaying training data distribution and class imbalances.
*   **Model Explainer (XAI):** Uses TF-IDF feature weights to show exactly which words the model prioritized for its prediction.
*   **Batch Processing Utility:** Support for downloadable CSVs to process large review sets in a single session.

The UI is built using custom CSS to implement **Glassmorphism**, characterized by translucent cards with a 12px blur effect and soft linear gradients. The interface is color-responsive, with sentiment cards glowing in Green, Amber, or Red based on the detected polarity. Additionally, a "Local Fallback Logic" design principle ensures system reliability; if the FastAPI server is unreachable, the UI can automatically load models locally from the cached directory.

---

## 7. Computational Optimizations for CPU-Bound Environments

A core requirement of the system is "Hardware Democratization"—ensuring high-tier AI can run on consumer-grade hardware (8GB RAM / CPU-only). Through rigorous optimization, the system's latency was reduced from a raw 16.7 seconds to a performant ~1.1 seconds.

### Optimization Techniques

*   **Beam Search (k=5):** Improved the quality of normalized text over standard greedy decoding.
*   **Repetition Penalty (1.5 - 2.0):** Prevented the generative model from "Neural Echoing" or repeating input text.
*   **Instruction Scaffolding:** Used prefixes like "Standardize:" and "Grammar:" to align with model-specific task weights.
*   **Inference Pruning:** Restricted tokenization sequence length to 128 tokens, achieving a 4x speedup for short product reviews.

### System Resource Benchmarks

| Metric | Measurement |
| :--- | :--- |
| **Peak RAM Consumption** | ~6.2 GB (during `BART-Large` neural operations) |
| **Raw CPU Latency** | 16,727ms (without early stopping/pruning) |
| **Optimized CPU Latency** | 1,095ms - 1,400ms |

---

## 8. System Evaluation and Performance Metrics

The system is assessed using an "Evaluation Matrix" where the **F1-Score** serves as the primary balance metric. This ensures that the system's ability to handle context-aware reworks is measured by its success in both precision and recall, particularly for difficult sarcastic "Negative" reviews.

### Model Comparison & Evaluation Matrix

| Model | Accuracy | Sarcasm Challenge Result | Strategic Strength |
| :--- | :--- | :--- | :--- |
| **Random Forest** | ~84.6% | **FAILED**: Misidentified irony as positive. | High speed; optimal for simple linear reviews. |
| **BERT-base** | ~82.1% | **FAILED**: Fooled by positive intensifiers. | Context-aware but lacked specific irony training. |
| **DistilRoBERTa** | ~87.0% | **SUCCESS**: Verified context-aware polarity. | Contextual Persistence; deep understanding of irony. |

### Interpretation Glossary

*   **Accuracy:** The overall percentage of correct predictions across all classes.
*   **Precision:** The accuracy of positive labels; crucial for avoiding the mislabeling of a complaint as a compliment.
*   **Recall (Sensitivity):** The ability to catch all negative reviews, especially sarcastic ones that bypass standard filters.
*   **Confusion Matrix Analysis:** A performance table of True Positives (TP), False Positives (FP), True Negatives (TN), and False Negatives (FN).

---

### Conclusion and Future Work

The Agentic Sentiment Intelligence System successfully achieves over **87% accuracy** on sarcasm-heavy samples. Future work will focus on:

*   **Multimodal Sentiment:** Analyzing images alongside text for holistic sentiment detection.
*   **Aspect-Based Sentiment Analysis (ABSA):** Identifying specific product features (e.g., "battery life," "screen quality").
*   **Multilingual Expansion:** Handling regional shorthand and diverse e-commerce markets beyond English.

