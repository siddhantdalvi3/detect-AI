import logging
from typing import Dict, List, Any, Tuple
from .sbert_ffnn_model import SBERTFFNNModel
from .distilbert_model import DistilBERTModel
from .roberta_model import RoBERTaModel
from .humanizer_model import HumanizerModel
from .line_analyzer import LineAnalyzer

logger = logging.getLogger(__name__)


class ModelOrchestrator:
    """
    Orchestrator to manage all AI detection models and humanizer
    """

    def __init__(self):
        self.models = {}
        self.humanizer = None
        self._initialize_models()

    def _initialize_models(self):
        """Initialize all detection models and humanizer"""
        try:
            # Initialize detection models
            self.models["sbert_ffnn"] = SBERTFFNNModel()
            self.models["distilbert"] = DistilBERTModel()
            self.models["roberta"] = RoBERTaModel()

            # Initialize humanizer
            self.humanizer = HumanizerModel()

            logger.info("All models initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            raise

    def detect_ai(self, text: str) -> Dict[str, Any]:
        """
        Detect AI content using all three models
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with results from all models
        """
        if not text or not isinstance(text, str):
            return self._create_empty_response()

        try:
            results = {}

            # Get predictions from all models
            for model_name, model in self.models.items():
                ai_prob, confidence = model.predict(text)
                results[model_name] = {
                    "ai_probability": ai_prob,
                    "confidence_scores": confidence,
                    "prediction": "AI" if ai_prob > 0.5 else "Human",
                    "model_type": model_name
                }

            # Calculate ensemble score (average of all models)
            ensemble_prob = sum(result["ai_probability"]
                                for result in results.values()) / len(results)

            response = {
                "text": text,
                "ensemble_score": ensemble_prob,
                "ensemble_prediction":
                "AI" if ensemble_prob > 0.5 else "Human",
                "model_results": results,
                "humanizer_suggestions": []
            }

            # If AI is detected, get humanizer suggestions
            if ensemble_prob > 0.5:
                suggestions = self.humanizer.humanize_text(text)
                response["humanizer_suggestions"] = suggestions

            return response

        except Exception as e:
            logger.error(f"Error in AI detection: {e}")
            return self._create_error_response(text, str(e))

    def detect_ai_from_file(self, file_path: str) -> Dict[str, Any]:
        """
        Detect AI content from a file
        
        Args:
            file_path: Path to the file to analyze
            
        Returns:
            Dictionary with results from all models
        """
        try:
            from ..file_handlers.file_processor import FileProcessor

            # Extract text from file
            text = FileProcessor.extract_text_from_file(file_path)

            # Validate text
            if not FileProcessor.is_valid_text(text):
                return self._create_error_response(
                    "", "Text is too short or invalid for analysis")

            # Process and analyze text
            processed_text = FileProcessor.process_input(text)
            return self.detect_ai(processed_text)

        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            return self._create_error_response(
                "", f"File processing error: {str(e)}")

    def batch_detect(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Detect AI content for multiple texts
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            List of detection results
        """
        if not texts:
            return []

        results = []
        for text in texts:
            result = self.detect_ai(text)
            results.append(result)

        return results

    def detect_ai_line_by_line(self, text: str) -> Dict[str, Any]:
        """
        Detect AI content line by line with detailed sentence analysis
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with line-by-line analysis results
        """
        if not text or not isinstance(text, str):
            return self._create_empty_response()

        try:
            # Perform line-by-line analysis
            line_analysis = LineAnalyzer.analyze_line_by_line(text, self)

            # Get standard detection for overall results
            standard_result = self.detect_ai(text)

            # Generate highlighted HTML
            highlighted_html = LineAnalyzer.generate_highlighted_html(
                text, line_analysis)

            # Get detailed model breakdown
            model_breakdown = LineAnalyzer.get_detailed_model_breakdown(
                line_analysis)

            response = {
                "text":
                text,
                "highlighted_html":
                highlighted_html,
                "overall_analysis":
                line_analysis["overall_analysis"],
                "sentence_analysis":
                line_analysis["sentence_analysis"],
                "model_breakdown":
                model_breakdown,
                "standard_result":
                standard_result,
                "humanizer_suggestions":
                standard_result.get("humanizer_suggestions", [])
            }

            return response

        except Exception as e:
            logger.error(f"Error in line-by-line AI detection: {e}")
            return self._create_error_response(
                text, f"Line analysis error: {str(e)}")

    def detect_ai_from_file_line_by_line(self,
                                         file_path: str) -> Dict[str, Any]:
        """
        Detect AI content from a file with line-by-line analysis
        
        Args:
            file_path: Path to the file to analyze
            
        Returns:
            Dictionary with line-by-line analysis results
        """
        try:
            from ..file_handlers.file_processor import FileProcessor

            # Extract text from file
            text = FileProcessor.extract_text_from_file(file_path)

            # Validate text
            if not FileProcessor.is_valid_text(text):
                return self._create_error_response(
                    "", "Text is too short or invalid for analysis")

            # Process text
            processed_text = FileProcessor.process_input(text)

            # Perform line-by-line analysis
            return self.detect_ai_line_by_line(processed_text)

        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            return self._create_error_response(
                "", f"File processing error: {str(e)}")

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about all loaded models"""
        model_info = {}

        for model_name, model in self.models.items():
            model_info[model_name] = {
                "model_type": type(model).__name__,
                "device":
                str(model.device) if hasattr(model, 'device') else "cpu",
                "status": "loaded"
            }

        model_info["humanizer"] = {
            "model_type":
            type(self.humanizer).__name__ if self.humanizer else "None",
            "status": "loaded" if self.humanizer else "not_loaded"
        }

        return model_info

    def _create_empty_response(self) -> Dict[str, Any]:
        """Create empty response for invalid input"""
        return {
            "text": "",
            "ensemble_score": 0.0,
            "ensemble_prediction": "Human",
            "model_results": {},
            "humanizer_suggestions": [],
            "error": "Invalid or empty input text"
        }

    def _create_error_response(self, text: str,
                               error_msg: str) -> Dict[str, Any]:
        """Create error response"""
        return {
            "text": text,
            "ensemble_score": 0.0,
            "ensemble_prediction": "Human",
            "model_results": {},
            "humanizer_suggestions": [],
            "error": error_msg
        }


def create_model_orchestrator() -> ModelOrchestrator:
    """Factory function to create model orchestrator"""
    return ModelOrchestrator()
