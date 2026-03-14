import re
import numpy as np
from typing import Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)


class SimpleAIDetector:
    """
    Simple rule-based AI detector using text statistics and patterns
    Provides better-than-random results without requiring model training
    """
    
    def __init__(self):
        # Common AI-generated text patterns
        self.ai_patterns = [
            r'\b(utilization|methodology|framework|algorithmic|computational)\b',
            r'\b(sophisticated|advanced|comprehensive|systematic)\b',
            r'\b(facilitates|enables|generates|produces)\b',
            r'\b(in order to|with the purpose of|for the purpose of)\b',
            r'\b(leveraging|harnessing|optimizing|maximizing)\b'
        ]
        
        # Human writing patterns
        self.human_patterns = [
            r'\b(I|we|you|me|us|our)\b',
            r'\b(actually|really|very|quite|pretty)\b',
            r'\b(like|kind of|sort of|maybe|perhaps)\b',
            r'\b(\w+\'\w+|\w+\'t\b)',  # Contractions
            r'[!?]{2,}',  # Multiple punctuation
            r'\b(oh|wow|hey|oops|uh|um)\b'  # Interjections
        ]
    
    def predict(self, text: str) -> Tuple[float, Dict[str, float]]:
        """
        Predict if text is AI-generated using simple heuristics
        Returns: (ai_probability, confidence_scores)
        """
        if not text or not isinstance(text, str):
            return 0.0, {"ai": 0.0, "human": 1.0}
        
        text_lower = text.lower()
        
        # Calculate various scores
        ai_score = self._calculate_ai_score(text_lower)
        human_score = self._calculate_human_score(text_lower)
        
        # Text statistics
        length_score = self._calculate_length_score(text)
        complexity_score = self._calculate_complexity_score(text)
        
        # Combine scores (weighted average)
        final_ai_prob = (
            ai_score * 0.4 +
            (1 - human_score) * 0.3 +
            complexity_score * 0.2 +
            length_score * 0.1
        )
        
        # Ensure probability is between 0 and 1
        final_ai_prob = max(0.0, min(1.0, final_ai_prob))
        
        return final_ai_prob, {"ai": final_ai_prob, "human": 1.0 - final_ai_prob}
    
    def _calculate_ai_score(self, text: str) -> float:
        """Calculate score based on AI patterns"""
        score = 0.0
        
        for pattern in self.ai_patterns:
            matches = len(re.findall(pattern, text))
            score += matches * 0.1  # Each match adds 0.1 to AI score
        
        return min(1.0, score)
    
    def _calculate_human_score(self, text: str) -> float:
        """Calculate score based on human patterns"""
        score = 0.0
        
        for pattern in self.human_patterns:
            matches = len(re.findall(pattern, text))
            score += matches * 0.15  # Each match adds 0.15 to human score
        
        return min(1.0, score)
    
    def _calculate_length_score(self, text: str) -> float:
        """Longer texts are more likely to be AI-generated"""
        words = text.split()
        if len(words) < 50:
            return 0.2  # Short texts less likely to be AI
        elif len(words) < 200:
            return 0.5  # Medium length
        else:
            return 0.8  # Long texts more likely to be AI
    
    def _calculate_complexity_score(self, text: str) -> float:
        """Complex vocabulary suggests AI generation"""
        words = text.split()
        if len(words) == 0:
            return 0.0
        
        # Calculate average word length
        avg_word_length = sum(len(word) for word in words) / len(words)
        
        # Count complex words (more than 6 characters)
        complex_words = sum(1 for word in words if len(word) > 6)
        complex_ratio = complex_words / len(words)
        
        # Combine metrics
        score = (avg_word_length / 10 * 0.4) + (complex_ratio * 0.6)
        
        return min(1.0, score)
    
    def predict_batch(self, texts: list) -> list:
        """Predict for multiple texts"""
        if not texts:
            return []
        
        results = []
        for text in texts:
            ai_prob, confidence = self.predict(text)
            results.append({
                "text": text,
                "ai_probability": ai_prob,
                "confidence_scores": confidence,
                "prediction": "AI" if ai_prob > 0.5 else "Human"
            })
        
        return results


def create_simple_detector() -> SimpleAIDetector:
    """Factory function to create simple detector"""
    return SimpleAIDetector()