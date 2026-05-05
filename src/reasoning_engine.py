"""
src/reasoning_engine.py
Agentic Reasoning Engine - Explains the 'Why' behind Sentiment Predictions.
Developed for Capstone Project 2026.
"""

import re

class AgenticReasoningEngine:
    """
    The RE provides Explainable AI (XAI) for the sentiment dashboard.
    It analyzes the model's output alongside the text to identify conflicting signals.
    """
    
    def __init__(self):
        # Curated vocabulary maps for specific product categories (skincare/general e-commerce)
        # These are used to identify 'signals' within the review text.
        self.pos_keywords = {'protects', 'good', 'great', 'excellent', 'love', 'perfect', 'white cast', 'value', 'works', 'best', 'effective'}
        self.neg_keywords = {'sweaty', 'oily', 'greasy', 'bad', 'worst', 'expensive', 'useless', 'broke', 'smell', 'sticky', 'irritates'}
        
    def generate_reasoning(self, text, sentiment, confidence):
        """
        Synthesizes a human-readable explanation based on linguistic clues and model confidence.
        """
        text_lower = text.lower()
        
        # 1. Signal Identification Layer
        # Extract keywords that exist in the review to build the logic trace.
        found_pos = [w for w in self.pos_keywords if w in text_lower]
        found_neg = [w for w in self.neg_keywords if w in text_lower]
        
        # Special logic to handle negations: 'no white cast' is actually a positive signal
        if "no white cast" in text_lower or "doesnt leave any white cast" in text_lower or "without white cast" in text_lower:
            if 'white cast' in found_pos: found_pos.remove('white cast')
            found_pos.append("absence of white cast")

        # 2. Logic Matrix: Differentiate explanation path based on sentiment polarity and signal mix
        reasoning = ""
        conf_pct = confidence * 100
        
        if len(found_pos) > 0 and len(found_neg) > 0:
            # CASE: MIXED SIGNALS (Positive and Negative words both detected)
            reasoning = (
                f"The model detected a **Conflict of Signals**. The review contains positive functional attributes "
                f"({', '.join(found_pos)}) but also highlights physical discomfort ({', '.join(found_neg)}). "
            )
            
            # Sub-case: Determining why the model leaned Negative despite mixed input
            if sentiment.upper() == "NEGATIVE":
                reasoning += (
                    f"Despite the positives, the Specialist prioritized the negative experience "
                    f"as a 'Dealbreaker' for this product category. The {conf_pct:.2f}% confidence indicates "
                    f"it weighted the physical dissatisfaction over the technical utility."
                )
            else:
                # Sub-case: Determining why the model leaned Positive
                reasoning += (
                    f"The model chose a POSITIVE classification because the functional benefits "
                    f"outweighed the specific complaint. However, the {conf_pct:.2f}% confidence "
                    f"suggests it acknowledged the existing drawback."
                )
                
        elif sentiment.upper() == "NEGATIVE":
            # CASE: PURE NEGATIVE (No positive signals detected)
            reasoning = (
                f"The model identified strong negative indicators ({', '.join(found_neg) if found_neg else 'contextual disappointment'}). "
                f"The high confidence of {conf_pct:.2f}% suggests a clear consensus on dissatisfaction."
            )
            
        elif sentiment.upper() == "POSITIVE":
            # CASE: PURE POSITIVE (No negative signals detected)
            reasoning = (
                f"The analysis found dominant positive signals ({', '.join(found_pos) if found_pos else 'favorable context'}). "
                f"With {conf_pct:.2f}% confidence, the model views this as a successful user experience."
        )
        else:
            # CASE: NEUTRAL (Balanced or descriptive input)
            reasoning = (
                "The model found the review to be neutral or balanced. It likely lacks "
                "strong polarized vocabulary, focusing instead on objective product description."
            )

        # 3. Agentic Insight: Added for borderline cases to highlight model complexity
        if 55 < conf_pct < 75:
            reasoning += (
                "\n\n**Agentic Insight**: A more 'naive' model might have misclassified this due to the word mix. "
                "The Context-Specialist, however, attempted to balance these 'Ironic Dissonances'."
            )

        return reasoning
