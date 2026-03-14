import logging
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List

from .distilbert_model import DistilBERTModel
from .humanizer_model import HumanizerModel
from .line_analyzer import LineAnalyzer
from .roberta_model import RoBERTaModel
from .sbert_ffnn_model import SBERTFFNNModel
from .simple_detector import SimpleAIDetector

logger = logging.getLogger(__name__)


class ModelOrchestrator:
    """
    Orchestrator with a cascade strategy:
    1) fast path (heuristic + compact model)
    2) heavy escalation for uncertain cases
    3) optional async heavy analysis for delayed results
    """

    def __init__(self):
        self.models = {}
        self.humanizer = None

        self.enable_cascade = self._parse_bool_env("ENABLE_CASCADE_MODE", True)
        self.fast_ai_threshold = float(os.getenv("FAST_AI_THRESHOLD", "0.72"))
        self.fast_human_threshold = float(
            os.getenv("FAST_HUMAN_THRESHOLD", "0.28"))

        self.enable_async_heavy = self._parse_bool_env(
            "ENABLE_ASYNC_HEAVY_ANALYSIS", True)
        self.async_pending_limit = int(os.getenv("ASYNC_PENDING_LIMIT", "50"))
        self.async_result_ttl_seconds = int(
            os.getenv("ASYNC_RESULT_TTL_SECONDS", "1800"))
        self._jobs = {}
        self._jobs_lock = threading.Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, int(os.getenv("ASYNC_HEAVY_WORKERS", "1"))))

        self._initialize_models()

    @staticmethod
    def _parse_bool_env(name: str, default: bool = False) -> bool:
        value = os.getenv(name, str(default)).strip().lower()
        return value in {"1", "true", "t", "yes", "y", "on"}

    def _initialize_models(self):
        """Initialize fast path models and humanizer; heavy models are lazy."""
        try:
            self.models["heuristic"] = SimpleAIDetector()
            self.models["compact"] = DistilBERTModel()
            self.models["sbert_ffnn"] = None
            self.models["roberta"] = None

            self.humanizer = HumanizerModel()
            logger.info("Cascade models initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            raise

    def _ensure_heavy_models_loaded(self):
        if self.models.get("sbert_ffnn") is None:
            self.models["sbert_ffnn"] = SBERTFFNNModel()
        if self.models.get("roberta") is None:
            self.models["roberta"] = RoBERTaModel()

    def _cleanup_old_jobs(self):
        now = time.time()
        with self._jobs_lock:
            stale = [
                job_id for job_id, job in self._jobs.items() if now -
                job.get("updated_at", now) > self.async_result_ttl_seconds
            ]
            for job_id in stale:
                self._jobs.pop(job_id, None)

    def _run_fast_path(self, text: str) -> Dict[str, Any]:
        heuristic_prob, heuristic_conf = self.models["heuristic"].predict(text)
        compact_prob, compact_conf = self.models["compact"].predict(text)

        fast_prob = (0.65 * compact_prob) + (0.35 * heuristic_prob)
        is_uncertain = self.fast_human_threshold < fast_prob < self.fast_ai_threshold

        return {
            "fast_probability": fast_prob,
            "is_uncertain": is_uncertain,
            "model_results": {
                "heuristic": {
                    "ai_probability": heuristic_prob,
                    "confidence_scores": heuristic_conf,
                    "prediction": "AI" if heuristic_prob > 0.5 else "Human",
                    "model_type": "heuristic",
                },
                "compact": {
                    "ai_probability": compact_prob,
                    "confidence_scores": compact_conf,
                    "prediction": "AI" if compact_prob > 0.5 else "Human",
                    "model_type": "compact",
                },
            },
        }

    def _run_heavy_models(self, text: str) -> Dict[str, Any]:
        self._ensure_heavy_models_loaded()
        heavy_results = {}

        for model_name in ("sbert_ffnn", "roberta"):
            model = self.models.get(model_name)
            if model is None:
                continue
            ai_prob, confidence = model.predict(text)
            heavy_results[model_name] = {
                "ai_probability": ai_prob,
                "confidence_scores": confidence,
                "prediction": "AI" if ai_prob > 0.5 else "Human",
                "model_type": model_name,
            }

        return heavy_results

    def _compose_response(
        self,
        text: str,
        model_results: Dict[str, Any],
        include_humanizer: bool = False,
    ) -> Dict[str, Any]:
        if not model_results:
            return self._create_empty_response()

        ensemble_prob = sum(
            r["ai_probability"]
            for r in model_results.values()) / len(model_results)

        response = {
            "text": text,
            "ensemble_score": ensemble_prob,
            "ensemble_prediction": "AI" if ensemble_prob > 0.5 else "Human",
            "model_results": model_results,
            "humanizer_suggestions": [],
        }

        if include_humanizer and ensemble_prob > 0.5:
            response["humanizer_suggestions"] = self.humanizer.humanize_text(
                text)

        return response

    def _submit_heavy_job(self, text: str, include_humanizer: bool):
        self._cleanup_old_jobs()

        with self._jobs_lock:
            pending_jobs = sum(
                1 for job in self._jobs.values()
                if job.get("status") in {"queued", "processing"})
            if pending_jobs >= self.async_pending_limit:
                return None

            request_id = str(uuid.uuid4())
            self._jobs[request_id] = {
                "status": "queued",
                "result": None,
                "error": None,
                "updated_at": time.time(),
            }

        def _job_runner():
            with self._jobs_lock:
                if request_id in self._jobs:
                    self._jobs[request_id]["status"] = "processing"
                    self._jobs[request_id]["updated_at"] = time.time()

            try:
                fast_path = self._run_fast_path(text)
                heavy_results = self._run_heavy_models(text)
                combined = {**fast_path["model_results"], **heavy_results}
                result = self._compose_response(
                    text, combined, include_humanizer=include_humanizer)
                result["processing_status"] = "completed"

                with self._jobs_lock:
                    if request_id in self._jobs:
                        self._jobs[request_id] = {
                            "status": "completed",
                            "result": result,
                            "error": None,
                            "updated_at": time.time(),
                        }
            except Exception as exc:
                logger.exception("Heavy async analysis failed")
                with self._jobs_lock:
                    if request_id in self._jobs:
                        self._jobs[request_id] = {
                            "status": "failed",
                            "result": None,
                            "error": str(exc),
                            "updated_at": time.time(),
                        }

        self._executor.submit(_job_runner)
        return request_id

    def get_async_result(self, request_id: str) -> Dict[str, Any]:
        self._cleanup_old_jobs()
        with self._jobs_lock:
            job = self._jobs.get(request_id)

        if not job:
            return {
                "status": "not_found",
                "request_id": request_id,
                "detail": "Result expired or request_id is invalid",
            }

        return {
            "status": job.get("status", "queued"),
            "request_id": request_id,
            "result": job.get("result"),
            "error": job.get("error"),
        }

    def detect_ai(
        self,
        text: str,
        include_humanizer: bool = False,
        allow_delayed: bool = False,
    ) -> Dict[str, Any]:
        """
        Detect AI content with cascade strategy.
        """
        if not text or not isinstance(text, str):
            return self._create_empty_response()

        try:
            fast_path = self._run_fast_path(text)

            if self.enable_cascade and fast_path["is_uncertain"]:
                if self.enable_async_heavy and allow_delayed:
                    request_id = self._submit_heavy_job(
                        text, include_humanizer)
                    if request_id:
                        response = self._compose_response(
                            text,
                            fast_path["model_results"],
                            include_humanizer=False)
                        response["processing_status"] = "pending"
                        response["request_id"] = request_id
                        response["heavy_analysis_pending"] = True
                        return response

                heavy_results = self._run_heavy_models(text)
                combined_results = {
                    **fast_path["model_results"],
                    **heavy_results
                }
                response = self._compose_response(
                    text,
                    combined_results,
                    include_humanizer=include_humanizer)
                response["processing_status"] = "completed"
                return response

            response = self._compose_response(
                text,
                fast_path["model_results"],
                include_humanizer=include_humanizer)
            response["processing_status"] = "completed"
            response["cascade_skipped_heavy"] = True
            return response

        except Exception as e:
            logger.error(f"Error in AI detection: {e}")
            return self._create_error_response(text, str(e))

    def detect_ai_from_file(
        self,
        file_path: str,
        include_humanizer: bool = False,
        allow_delayed: bool = False,
    ) -> Dict[str, Any]:
        """
        Detect AI content from a file.
        """
        try:
            from ..file_handlers.file_processor import FileProcessor

            text = FileProcessor.extract_text_from_file(file_path)

            if not FileProcessor.is_valid_text(text):
                return self._create_error_response(
                    "", "Text is too short or invalid for analysis")

            processed_text = FileProcessor.process_input(text)
            return self.detect_ai(
                processed_text,
                include_humanizer=include_humanizer,
                allow_delayed=allow_delayed,
            )

        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            return self._create_error_response(
                "", f"File processing error: {str(e)}")

    def batch_detect(self, texts: List[str]) -> List[Dict[str, Any]]:
        if not texts:
            return []

        results = []
        for text in texts:
            results.append(self.detect_ai(text))

        return results

    def detect_ai_line_by_line(self, text: str) -> Dict[str, Any]:
        if not text or not isinstance(text, str):
            return self._create_empty_response()

        try:
            # Detailed mode is explicitly on-demand; load heavy models to preserve richer view.
            self._ensure_heavy_models_loaded()

            line_analysis = LineAnalyzer.analyze_line_by_line(text, self)
            standard_result = self.detect_ai(text)
            highlighted_html = LineAnalyzer.generate_highlighted_html(
                text, line_analysis)
            model_breakdown = LineAnalyzer.get_detailed_model_breakdown(
                line_analysis)

            return {
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
                standard_result.get("humanizer_suggestions", []),
            }

        except Exception as e:
            logger.error(f"Error in line-by-line AI detection: {e}")
            return self._create_error_response(
                text, f"Line analysis error: {str(e)}")

    def detect_ai_from_file_line_by_line(self,
                                         file_path: str) -> Dict[str, Any]:
        try:
            from ..file_handlers.file_processor import FileProcessor

            text = FileProcessor.extract_text_from_file(file_path)

            if not FileProcessor.is_valid_text(text):
                return self._create_error_response(
                    "", "Text is too short or invalid for analysis")

            processed_text = FileProcessor.process_input(text)
            return self.detect_ai_line_by_line(processed_text)

        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            return self._create_error_response(
                "", f"File processing error: {str(e)}")

    def get_model_info(self) -> Dict[str, Any]:
        model_info = {}

        for model_name, model in self.models.items():
            model_info[model_name] = {
                "model_type":
                type(model).__name__ if model else "NotLoaded",
                "device":
                str(model.device)
                if model is not None and hasattr(model, "device") else "cpu",
                "status":
                "loaded" if model is not None else "lazy_not_loaded",
            }

        model_info["humanizer"] = {
            "model_type":
            type(self.humanizer).__name__ if self.humanizer else "None",
            "status": "loaded" if self.humanizer else "not_loaded",
        }

        with self._jobs_lock:
            pending_jobs = sum(
                1 for job in self._jobs.values()
                if job.get("status") in {"queued", "processing"})

        model_info["async_heavy"] = {
            "enabled": self.enable_async_heavy,
            "pending_jobs": pending_jobs,
            "pending_limit": self.async_pending_limit,
        }

        return model_info

    def _create_empty_response(self) -> Dict[str, Any]:
        return {
            "text": "",
            "ensemble_score": 0.0,
            "ensemble_prediction": "Human",
            "model_results": {},
            "humanizer_suggestions": [],
            "processing_status": "failed",
            "error": "Invalid or empty input text",
        }

    def _create_error_response(self, text: str,
                               error_msg: str) -> Dict[str, Any]:
        return {
            "text": text,
            "ensemble_score": 0.0,
            "ensemble_prediction": "Human",
            "model_results": {},
            "humanizer_suggestions": [],
            "processing_status": "failed",
            "error": error_msg,
        }


def create_model_orchestrator() -> ModelOrchestrator:
    return ModelOrchestrator()
