#!/usr/bin/env python3
"""
Test script to verify AI detection models are working correctly
and not giving random ~50% scores
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.distilbert_model import DistilBERTModel
from models.roberta_model import RoBERTaModel
from models.simple_detector import SimpleAIDetector

def test_models():
    """Test all AI detection models with sample texts"""
    
    # Sample texts for testing
    ai_generated_text = """
    The utilization of advanced computational methodologies has enabled 
    the development of sophisticated algorithmic frameworks that 
    systematically analyze complex datasets to extract meaningful insights 
    and facilitate data-driven decision-making processes.
    """
    
    human_written_text = """
    I was actually pretty surprised by the results. We tried this approach 
    because it seemed like the most straightforward way to solve the problem, 
    and honestly, it worked better than I expected. The team did a great job 
    figuring this out together.
    """
    
    print("Testing AI detection models...")
    print("=" * 50)
    
    # Test DistilBERT model
    print("\n1. Testing DistilBERT Model:")
    print("-" * 30)
    
    try:
        distilbert_model = DistilBERTModel()
        
        # Test AI-generated text
        ai_prob, confidence = distilbert_model.predict(ai_generated_text)
        print(f"AI-generated text score: {ai_prob:.3f}")
        print(f"Confidence: {confidence}")
        
        # Test human-written text  
        ai_prob, confidence = distilbert_model.predict(human_written_text)
        print(f"Human-written text score: {ai_prob:.3f}")
        print(f"Confidence: {confidence}")
        
    except Exception as e:
        print(f"DistilBERT model error: {e}")
    
    # Test RoBERTa model
    print("\n2. Testing RoBERTa Model:")
    print("-" * 30)
    
    try:
        roberta_model = RoBERTaModel()
        
        # Test AI-generated text
        ai_prob, confidence = roberta_model.predict(ai_generated_text)
        print(f"AI-generated text score: {ai_prob:.3f}")
        print(f"Confidence: {confidence}")
        
        # Test human-written text
        ai_prob, confidence = roberta_model.predict(human_written_text)
        print(f"Human-written text score: {ai_prob:.3f}")
        print(f"Confidence: {confidence}")
        
    except Exception as e:
        print(f"RoBERTa model error: {e}")
    
    # Test Simple Detector
    print("\n3. Testing Simple Detector:")
    print("-" * 30)
    
    try:
        simple_detector = SimpleAIDetector()
        
        # Test AI-generated text
        ai_prob, confidence = simple_detector.predict(ai_generated_text)
        print(f"AI-generated text score: {ai_prob:.3f}")
        print(f"Confidence: {confidence}")
        
        # Test human-written text
        ai_prob, confidence = simple_detector.predict(human_written_text)
        print(f"Human-written text score: {ai_prob:.3f}")
        print(f"Confidence: {confidence}")
        
    except Exception as e:
        print(f"Simple detector error: {e}")

if __name__ == "__main__":
    test_models()