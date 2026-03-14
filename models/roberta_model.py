import logging
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

logger = logging.getLogger(__name__)


class RoBERTaModel:
    """
    RoBERTa model for AI text detection
    Uses pre-trained RoBERTa for sequence classification
    """

    def __init__(self, model_name: str = "Hello-SimpleAI/chatgpt-detector-roberta"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self._initialize_model()

    def _initialize_model(self):
        """Initialize RoBERTa model and tokenizer"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=2,  # Binary classification: AI vs Human
            )
            self.model = self.model.to(self.device)
            self.model.eval()

            logger.info(f"RoBERTa model initialized on device: {self.device}")

        except Exception as e:
            logger.error(f"Error initializing RoBERTa model: {e}")
            raise

    def predict(
        self, text: str, max_length: int = 512
    ) -> Tuple[float, Dict[str, float]]:
        """
        Predict if text is AI-generated
        Returns: (ai_probability, confidence_scores)
        """
        if not text or not isinstance(text, str):
            return 0.0, {"ai": 0.0, "human": 1.0}

        try:
            # Tokenize input
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=max_length,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Make prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]

            # Return AI probability and confidence scores
            ai_prob = float(probabilities[1])  # Assuming index 1 is AI class
            confidence_scores = {
                "ai": float(probabilities[1]),
                "human": float(probabilities[0]),
            }

            return ai_prob, confidence_scores

        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            return 0.0, {"ai": 0.0, "human": 1.0}

    def predict_batch(self, texts: list, max_length: int = 512) -> list:
        """Predict for multiple texts"""
        if not texts:
            return []

        results = []
        for text in texts:
            ai_prob, confidence = self.predict(text, max_length)
            results.append(
                {
                    "text": text,
                    "ai_probability": ai_prob,
                    "confidence_scores": confidence,
                    "prediction": "AI" if ai_prob > 0.5 else "Human",
                }
            )

        return results

    def load_model(self, model_path: str):
        """Load pre-trained model weights"""
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.model = self.model.to(self.device)
            self.model.eval()
            logger.info(f"Loaded model from {model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def save_model(self, model_path: str):
        """Save model weights"""
        try:
            self.model.save_pretrained(model_path)
            self.tokenizer.save_pretrained(model_path)
            logger.info(f"Saved model to {model_path}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise


def create_roberta_model() -> RoBERTaModel:
    """Factory function to create RoBERTa model"""
    return RoBERTaModel()
