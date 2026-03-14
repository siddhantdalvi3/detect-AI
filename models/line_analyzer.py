import html
import re
from typing import Any, Dict, List, Tuple

import numpy as np


class LineAnalyzer:
    """
    Analyzes text line by line to detect AI-generated content
    with per-sentence detection and model attribution
    """

    @staticmethod
    def split_into_sentences(text: str) -> List[str]:
        """
        Split text into sentences using basic regex pattern
        """
        sentence_endings = r"(?<=[.!?])\s+"
        sentences = re.split(sentence_endings, text.strip())

        # Keep all non-empty fragments so detailed output never drops pasted content.
        return [s.strip() for s in sentences if s.strip()]

    @staticmethod
    def analyze_line_by_line(text: str, model_orchestrator) -> Dict[str, Any]:
        """
        Analyze text line by line and return detailed results
        with per-sentence AI detection and model attribution
        """
        sentences = LineAnalyzer.split_into_sentences(text)

        if not sentences:
            return {
                "overall_analysis": {
                    "ai_probability": 0.0,
                    "prediction": "Human"
                },
                "sentence_analysis": [],
                "model_breakdown": {},
            }

        sentence_results = []
        model_predictions = {
            model_name: []
            for model_name in model_orchestrator.models.keys()
        }

        for sentence in sentences:
            sentence_result = {
                "sentence": sentence,
                "models": {},
                "overall_ai_prob": 0.0,
            }

            for model_name, model in model_orchestrator.models.items():
                ai_prob, confidence = model.predict(sentence)
                sentence_result["models"][model_name] = {
                    "ai_probability": ai_prob,
                    "confidence": confidence,
                    "prediction": "AI" if ai_prob >= 0.6 else "Human",
                }
                model_predictions[model_name].append(ai_prob)

            sentence_ai_probs = [
                result["ai_probability"]
                for result in sentence_result["models"].values()
            ]
            avg_prob = sum(sentence_ai_probs) / len(sentence_ai_probs)
            ai_votes = sum(1 for p in sentence_ai_probs if p >= 0.6)

            sentence_result["overall_ai_prob"] = avg_prob
            sentence_result["overall_prediction"] = (
                "AI" if (avg_prob >= 0.72 or
                         (ai_votes >= 2 and avg_prob >= 0.6)) else "Human")
            sentence_results.append(sentence_result)

        overall_ai_prob = sum(
            result["overall_ai_prob"]
            for result in sentence_results) / len(sentence_results)

        model_breakdown = {}
        for model_name, predictions in model_predictions.items():
            ai_count = sum(1 for p in predictions if p >= 0.6)
            model_breakdown[model_name] = {
                "average_ai_prob": sum(predictions) / len(predictions),
                "ai_sentences_count": ai_count,
                "total_sentences": len(predictions),
                "ai_percentage": (ai_count / len(predictions)) * 100,
            }

        ai_sentence_count = sum(1 for result in sentence_results
                                if result["overall_prediction"] == "AI")
        ai_sentence_ratio = ai_sentence_count / len(sentence_results)
        doc_prediction = ("AI" if (ai_sentence_ratio >= 0.4
                                   or overall_ai_prob >= 0.62) else "Human")

        return {
            "overall_analysis": {
                "ai_probability": overall_ai_prob,
                "prediction": doc_prediction,
                "total_sentences": len(sentences),
                "ai_sentences": ai_sentence_count,
            },
            "sentence_analysis": sentence_results,
            "model_breakdown": model_breakdown,
        }

    @staticmethod
    def generate_highlighted_html(text: str,
                                  analysis_results: Dict[str, Any]) -> str:
        """
        Generate HTML with highlighted AI-generated sentences
        """
        sentences = LineAnalyzer.split_into_sentences(text)
        highlighted_html = ""

        for i, sentence in enumerate(sentences):
            if i < len(analysis_results["sentence_analysis"]):
                sentence_data = analysis_results["sentence_analysis"][i]
                ai_prob = sentence_data["overall_ai_prob"]

                if ai_prob >= 0.78:
                    css_class = "sentence-highlight-ai"
                elif ai_prob >= 0.6:
                    css_class = "sentence-highlight-mixed"
                else:
                    css_class = "sentence-highlight-human"

                # Tooltip in percentage format
                model_tips = [f"Overall AI: {ai_prob * 100:.2f}%"]
                for model_name, model_data in sentence_data["models"].items():
                    model_tips.append(
                        f"{model_name}: {model_data['ai_probability'] * 100:.2f}% ({model_data['prediction']})"
                    )
                tooltip = " | ".join(model_tips)

                # Escape for safe HTML attribute/content rendering
                escaped_tooltip = html.escape(tooltip, quote=True)
                escaped_sentence = html.escape(sentence, quote=False)

                highlighted_html += (
                    f'<span class="{css_class}" data-bs-toggle="tooltip" '
                    f'title="{escaped_tooltip}">{escaped_sentence}</span>')
            else:
                highlighted_html += f"<span>{html.escape(sentence, quote=False)}</span>"

            highlighted_html += " "

        return highlighted_html.strip()

    @staticmethod
    def get_detailed_model_breakdown(
        analysis_results: Dict[str, Any], ) -> Dict[str, Any]:
        """
        Generate detailed breakdown of which models detected AI content
        """
        breakdown = {
            "by_model": {},
            "consensus_analysis": {},
            "sentence_level": []
        }

        # Model-specific breakdown
        for model_name, stats in analysis_results["model_breakdown"].items():
            breakdown["by_model"][model_name] = {
                "detection_rate": f"{stats['ai_percentage']:.1f}%",
                "sentences_detected": stats["ai_sentences_count"],
                "average_confidence": stats["average_ai_prob"],
            }

        # Consensus analysis (where all models agree)
        sentences = analysis_results["sentence_analysis"]
        all_ai = sum(1 for s in sentences if all(
            m["prediction"] == "AI" for m in s["models"].values()))
        all_human = sum(1 for s in sentences if all(
            m["prediction"] == "Human" for m in s["models"].values()))
        mixed = len(sentences) - all_ai - all_human

        breakdown["consensus_analysis"] = {
            "all_models_agree_ai":
            all_ai,
            "all_models_agree_human":
            all_human,
            "mixed_opinions":
            mixed,
            "consensus_percentage":
            ((all_ai + all_human) / len(sentences)) * 100 if sentences else 0,
        }

        # Sentence-level model agreement
        for i, sentence_data in enumerate(sentences):
            ai_models = [
                name for name, data in sentence_data["models"].items()
                if data["prediction"] == "AI"
            ]
            human_models = [
                name for name, data in sentence_data["models"].items()
                if data["prediction"] == "Human"
            ]

            breakdown["sentence_level"].append({
                "sentence_index":
                i,
                "sentence_preview":
                sentence_data["sentence"][:50] +
                "..." if len(sentence_data["sentence"]) > 50 else
                sentence_data["sentence"],
                "ai_models":
                ai_models,
                "human_models":
                human_models,
                "agreement":
                "unanimous_ai"
                if not human_models and ai_models else "unanimous_human"
                if not ai_models and human_models else "mixed",
                "ai_probability":
                sentence_data["overall_ai_prob"],
            })

        return breakdown
