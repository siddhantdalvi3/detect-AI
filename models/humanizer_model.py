import logging
import re
from typing import Any, Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

logger = logging.getLogger(__name__)


class HumanizerModel:
    """
    Humanizer model for rewriting AI-generated text to sound more human
    Uses a language model to suggest human-like alternatives
    """

    def __init__(self, model_name: str = "gpt2"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.generator = None
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self._initialize_model()

    def _initialize_model(self):
        """Initialize the humanizer model"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            self.model = self.model.to(self.device)

            # Set padding token if not exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Create text generation pipeline
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1,
            )

            logger.info(f"Humanizer model initialized on device: {self.device}")

        except Exception as e:
            logger.error(f"Error initializing humanizer model: {e}")
            raise

    def humanize_text(
        self, text: str, context: str = "", max_length: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Generate human-like alternatives for AI-generated text

        Args:
            text: The AI-generated text to humanize
            context: Additional context for better rewriting
            max_length: Maximum length of generated suggestions

        Returns:
            List of suggested human-like alternatives
        """
        if not text or not isinstance(text, str):
            return []

        try:
            # Create prompt for humanization
            prompt = self._create_prompt(text, context)

            # Generate suggestions
            suggestions = self.generator(
                prompt,
                max_length=max_length,
                num_return_sequences=3,
                temperature=0.8,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id,
            )

            # Process and clean suggestions
            humanized_texts = self._process_suggestions(suggestions, prompt)

            return [
                {
                    "original_text": text,
                    "suggestion": suggestion,
                    "confidence": 0.7,  # Placeholder confidence score
                    "improvement_type": self._classify_improvement(text, suggestion),
                }
                for suggestion in humanized_texts
            ]

        except Exception as e:
            logger.error(f"Error in humanizing text: {e}")
            return []

    def _create_prompt(self, text: str, context: str) -> str:
        """Create prompt for humanization"""
        if context:
            prompt = f"Rewrite the following AI-generated text to sound more human-like and natural. Context: {context}\n\nAI text: {text}\n\nHuman-like version:"
        else:
            prompt = f"Rewrite the following AI-generated text to sound more human-like and natural:\n\nAI text: {text}\n\nHuman-like version:"

        return prompt

    def _process_suggestions(self, suggestions: List[Dict], prompt: str) -> List[str]:
        """Process and clean generated suggestions"""
        cleaned_suggestions = []

        for suggestion in suggestions:
            generated_text = suggestion["generated_text"]

            # Remove the prompt part
            if generated_text.startswith(prompt):
                human_text = generated_text[len(prompt) :].strip()
            else:
                human_text = generated_text.strip()

            # Clean up the text
            human_text = re.sub(
                r"^[^a-zA-Z0-9\"\']+", "", human_text
            )  # Remove leading non-alphanumeric
            human_text = re.sub(
                r"[^a-zA-Z0-9\s\.,!?;:\"\']+$", "", human_text
            )  # Remove trailing junk
            human_text = human_text.strip()

            if human_text and len(human_text) > 10:  # Minimum length check
                cleaned_suggestions.append(human_text)

        return cleaned_suggestions

    def _classify_improvement(self, original: str, suggestion: str) -> str:
        """Classify the type of improvement made"""
        original_words = set(original.lower().split())
        suggestion_words = set(suggestion.lower().split())

        # Calculate word overlap
        overlap = (
            len(original_words.intersection(suggestion_words)) / len(original_words)
            if original_words
            else 0
        )

        if overlap < 0.3:
            return "complete_rewrite"
        elif overlap < 0.7:
            return "significant_restructuring"
        else:
            return "subtle_improvement"

    def batch_humanize(
        self, texts: List[str], contexts: List[str] = None, max_length: int = 100
    ) -> List[List[Dict]]:
        """Humanize multiple texts"""
        if not texts:
            return []

        if contexts is None:
            contexts = [""] * len(texts)

        results = []
        for text, context in zip(texts, contexts):
            suggestions = self.humanize_text(text, context, max_length)
            results.append(suggestions)

        return results

    def load_model(self, model_path: str):
        """Load pre-trained model weights"""
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_path)
            self.model = self.model.to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
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


def create_humanizer_model() -> HumanizerModel:
    """Factory function to create humanizer model"""
    return HumanizerModel()
