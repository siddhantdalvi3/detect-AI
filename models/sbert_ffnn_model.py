import logging
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)


class SBERTFFNNModel:
    """
    SBERT-FFNN model for AI text detection
    Uses Sentence-BERT for embeddings and a feed-forward neural network for classification
    """

    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        self.model_name = model_name
        self.sbert_model = None
        self.ffnn_model = None
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # FFNN architecture
        self.input_dim = 768  # MPNet embedding size
        self.hidden_dims = [512, 256, 128]
        self.output_dim = 2  # Binary classification: AI vs Human

        self._initialize_models()

    def _initialize_models(self):
        """Initialize SBERT and FFNN models"""
        try:
            # Initialize SBERT model
            self.sbert_model = SentenceTransformer(self.model_name)
            self.sbert_model = self.sbert_model.to(self.device)

            # Initialize FFNN model
            self.ffnn_model = self._build_ffnn()
            self.ffnn_model = self.ffnn_model.to(self.device)

            logger.info(f"SBERT-FFNN model initialized on device: {self.device}")

        except Exception as e:
            logger.error(f"Error initializing SBERT-FFNN model: {e}")
            raise

    def _build_ffnn(self) -> nn.Module:
        """Build the feed-forward neural network"""
        layers = []
        input_dim = self.input_dim

        # Add hidden layers
        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            input_dim = hidden_dim

        # Add output layer
        layers.append(nn.Linear(input_dim, self.output_dim))
        layers.append(nn.Softmax(dim=1))

        return nn.Sequential(*layers)

    def get_embeddings(self, texts: list) -> np.ndarray:
        """Get sentence embeddings using SBERT"""
        if not texts:
            return np.array([])

        try:
            embeddings = self.sbert_model.encode(
                texts,
                convert_to_tensor=True,
                device=self.device,
                show_progress_bar=False,
            )
            return embeddings.cpu().numpy()
        except Exception as e:
            logger.error(f"Error getting embeddings: {e}")
            raise

    def predict(self, text: str) -> Tuple[float, Dict[str, float]]:
        """
        Predict if text is AI-generated
        Returns: (ai_probability, confidence_scores)
        """
        if not text or not isinstance(text, str):
            return 0.0, {"ai": 0.0, "human": 1.0}

        try:
            # Get embedding
            embedding = self.get_embeddings([text])
            if embedding.size == 0:
                return 0.0, {"ai": 0.0, "human": 1.0}

            # Convert to tensor
            embedding_tensor = torch.tensor(embedding, dtype=torch.float32).to(
                self.device
            )

            # Make prediction
            with torch.no_grad():
                outputs = self.ffnn_model(embedding_tensor)
                probabilities = outputs.cpu().numpy()[0]

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

    def predict_batch(self, texts: list) -> list:
        """Predict for multiple texts"""
        if not texts:
            return []

        results = []
        for text in texts:
            ai_prob, confidence = self.predict(text)
            results.append(
                {
                    "text": text,
                    "ai_probability": ai_prob,
                    "confidence_scores": confidence,
                    "prediction": "AI" if ai_prob > 0.5 else "Human",
                }
            )

        return results

    def load_weights(self, model_path: str):
        """Load pre-trained weights for FFNN"""
        try:
            if self.ffnn_model:
                self.ffnn_model.load_state_dict(
                    torch.load(model_path, map_location=self.device)
                )
                self.ffnn_model.eval()
                logger.info(f"Loaded weights from {model_path}")
        except Exception as e:
            logger.error(f"Error loading weights: {e}")
            raise

    def save_weights(self, model_path: str):
        """Save FFNN weights"""
        try:
            if self.ffnn_model:
                torch.save(self.ffnn_model.state_dict(), model_path)
                logger.info(f"Saved weights to {model_path}")
        except Exception as e:
            logger.error(f"Error saving weights: {e}")
            raise


def create_sbert_ffnn_model() -> SBERTFFNNModel:
    """Factory function to create SBERT-FFNN model"""
    return SBERTFFNNModel()
