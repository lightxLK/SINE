# Final Semester Capstone Project: Final Submission Report
## Topic: Agentic Sentiment Intelligence System for Nuanced Product Reviews

---

## 1. Executive Summary
The **Agentic Sentiment Intelligence System** is a state-of-the-art solution designed to bridge the "Sarcasm Gap" in e-commerce review analysis. Traditional sentiment models—while effective at identifying polarized language—frequently fail when faced with irony, nuanced negativity, and phonetic slang. This project introduces a **Multi-Tiered Agentic Architecture** that combines neural normalization, transformer-based classification, and symbolic reasoning. By integrating an **Active Data Mining** workflow, the system identifies "Ironic Dissonance" (conflicts between star ratings and text) to train a specialist engine capable of high-precision sentiment detection in complex scenarios.

---

## 2. Project Timeline & Evolution
The project followed a rigorous 10-week development cycle, evolving from a standard statistical classifier into a sophisticated neural agent.

*   **Weeks 1–2: Initiation & Diagnostic**
    *   Analyzed the initial dataset (200k+ rows) and identified a heavy imbalance toward positive reviews.
    *   Resolved critical version mismatches in baseline libraries (scikit-learn and XGBoost parameters) that caused system crashes during high-dimensional matrix operations.
    *   Established a metric baseline using Logistic Regression.

*   **Weeks 3–4: Data Refinery & Classical Modeling**
    *   Implemented a robust preprocessing pipeline to handle URL stripping, HTML removal, and **Emoji Demojization**.
    *   Benchmarked Random Forest and XGBoost. While accuracy reached 84%, subjective testing revealed a systematic failure in detecting sarcasm.
    *   Applied **SMOTE (Synthetic Minority Over-sampling Technique)** to address the "Neutral" class deficiency.

*   **Weeks 5–6: The Context Revolution**
    *   Pivoted from "Bag-of-Words" methodology to **Context-Aware Transformers**.
    *   Conducted "Ablation Studies" on various neural architectures (FLAN-T5, BART, BERT) to determine the best performance-to-RAM ratio for local CPU execution.
    *   Discovered the "Neural Echoing" problem where large models would repeat input instead of repairing it, leading to the implementation of repetitive penalties.

*   **Weeks 7–8: Agentic System Development**
    *   Architected the **Linguistic Quality Node (LQN)** to normalize user input before it reaches the classifier.
    *   Developed the **Reasoning Engine (RE)** to provide symbolic explanations for model confidence.
    *   Designed the **Glassmorphism Dashboard**, prioritizing a professional, emoji-free academic aesthetic.

*   **Weeks 9–10: Specialist Optimization & Final Polish**
    *   Engineered the **Ironic Dissonance Miner** to extract "Hard Cases" from the 200,000-row dataset.
    *   Fine-tuned the final **DistilRoBERTa Specialist** engine.
    *   Optimized inference latency from 16.7 seconds down to ~1.1 seconds through beam search pruning and model caching.

---

## 3. Problem Statement & Objectives
Standard sentiment tools frequently misclassify reviews with "Ironic Dissonance" (e.g., *"This phone is so great at being terrible"*). 

### Key Objectives:
1.  **Solve the Sarcasm Gap**: Move beyond keyword matching to understanding bidirectional sequence context.
2.  **Neural Normalization**: Automatically "clean" and "rewrite" messy, slang-heavy user reviews before classification.
3.  **Agentic Reasoning**: Provide human-readable explanations ("Why") for every prediction to improve transparency.
4.  **Hardware Democratization**: Ensure the system runs efficiently on consumer-grade hardware (8GB RAM / CPU-only) without requiring expensive GPUs.

---

## 4. Intricate Data Refinery & Feature Engineering
The system employs a "Deep Clean" strategy to preserve sentimental signals that standard tokenizers often discard.

### Technical Detail: Fuzzy Column Mapping
Data often arrived with inconsistent headers. We implemented a fuzzy-matching logic that programmatically detects the semantic purpose of each column, ensuring the pipeline never breaks regardless of the dataset source.

### Semantic Preservation:
*   **Emoji Demojization**: Instead of stripping symbols, icons like 😊 are converted into readable text tokens (`:smiling_face:`). This ensures the transformer captures the "emotive tone" often lost in basic cleaning.
*   **Phonetic Slang Mapping**: A custom dictionary-based map handles contemporary shorthand (e.g., 'gr8' -> 'great', 'applicasun' -> 'application'), reducing the vocabulary size and improving model focus.
*   **Sublinear TF-IDF Tuning**: Applied `sublinear_tf=True` to dampen the impact of highly frequent but low-value words, allowing unique sentimental qualifiers to carry more weight in classical models.

---

## 5. Classical ML Modeling & The "Sarcasm Gap"
We trained three primary baseline models to establish an empirical ceiling.

1.  **Logistic Regression (LR)**: Reached 81.8% accuracy. Used as a high-speed "Sanity Check" for the pipeline.
2.  **Random Forest (RF)**: The "Overall Champion" for simple reviews (84.6% accuracy). However, the model file swelled to 1.5 GB, leading to hardware bottlenecks during deployment.
3.  **XGBoost (XGB)**: The "Neutral Specialist," achieving a record 58.9% recall on the neutral class, which is notoriously difficult to classify.

**The Failure Mode**: Despite high accuracy, all three models failed a "Specialist Challenge"—classifying a review like *"Perfectly mis-crafted"* as positive because they saw the word "Perfectly" and ignored the negative modifier because they lacked context-awareness.

---

## 6. The Agentic Architecture: LQN + RE
The uniqueness of this project lies in its **Agentic Layers**, which act as "Intelligent Buffers" between the user and the raw ML model.

### A. Linguistic Quality Node (LQN)
The LQN acts as a **Neural Normalizer**. It uses a specialized Transformer (fine-tuned for grammar synthesis) to audit the input. If a user types *"i absolutli lik dis prduct"*, the LQN programmatically rewrites this into *"I absolutely like this product"* before the sentiment engine sees it. 
*   **Optimization**: We implemented **Beam Search (k=5)** and **Early Stopping** to ensure this normalization happens in under 400ms on a standard CPU.

### B. Reasoning Engine (RE)
The RE provides **Explainable AI (XAI)**. It analyzes the confidence scores and the interaction between positive/negative keywords. 
*   **Conflict Logic**: If the model detects a word mix (e.g., "Good price but bad quality"), the RE flags this as a "Conflict of Signals" and explains that the Specialist prioritized the "Dealbreaker" attribute (quality) over the "Incentive" attribute (price).

---

## 7. Specialist Active Mining (Ironic Dissonance)
To solve sarcasm without manual labeling, we developed a breakthrough **Active Mining Workflow**:

1.  **The Detector**: We scanned 205,052 reviews for "Dissonance"—instances where the Star Rating was low (1 or 2) but the text contained "Positive Intensifiers" (e.g., "Wow", "Impressive", "Amazing").
2.  **The Correction**: The system programmatically identified these "Pure Sarcasm" cases and forced them to a 'Negative' label.
3.  **Synthetic Augmentation**: We applied a **10x Weighting** to these sarcastic samples during training to ensure the model pays attention to these rare but critical linguistic patterns.
4.  **The Result**: The final model learned that "Wow" when paired with a low rating signifies extreme disappointment, not satisfaction.

---

## 8. Hardware Optimizations & Code Tweaks
The project was intentionally constrained to **CPU-bound execution** to simulate real-world accessibility.

*   **RAM Management**: Optimized the startup time by implementing a decoupled loading system. Heavy components are only loaded into memory when required.
*   **Inference Pruning**: By reducing the Transformer tokenization sequence length to 128 (standard for short product reviews), we achieved a **4x speedup** in prediction time without sacrificing accuracy.
*   **Local Fallback Engine**: A design principle that ensures the system remains functional even if a remote API is offline by caching pre-computed weights in the local directory.

---

## 9. Conclusion & Uniqueness
Our system is unique because it doesn't treat sentiment as a one-step classification task. Instead, it treats it as an **intelligence pipeline**:
1.  **Normalize** the language (LQN).
2.  **Analyze** the context (Specialist Engine).
3.  **Explain** the rationale (Reasoning Engine).

The project successfully achieved >87% accuracy on complex, sarcasm-heavy datasets, proving that agentic frameworks are superior to "black-box" classifiers for real-world NLP applications.

---

## 10. Future Work
*   **Multimodal Sentiment**: Incorporating image analysis (e.g., a photo of a broken product accompanying a review).
*   **Aspect-Based Analysis**: Identifying specific product features (e.g., "Battery Life" vs "Screen Quality") mentioned in reviews.
*   **Multilingual Expansion**: Adapting the system to handle regional shorthand and mixed-language reviews prevalent in diverse e-commerce markets.

---

**Lokesh Patra**
*Lead Researcher (April 2026)*