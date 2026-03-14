#!/bin/bash

# AI Text Detector Setup Script
echo "🚀 Setting up AI Text Detector..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ uv is not installed. Please install uv first:"
    echo "curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Check if Python 3.9+ is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.9+"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
if [[ "$PYTHON_VERSION" < "3.9" ]]; then
    echo "❌ Python 3.9+ is required. Current version: $PYTHON_VERSION"
    exit 1
fi

echo "✅ Python $PYTHON_VERSION detected"

# Install dependencies
echo "📦 Installing dependencies with uv..."
uv sync

if [ $? -eq 0 ]; then
    echo "✅ Dependencies installed successfully"
else
    echo "❌ Failed to install dependencies"
    exit 1
fi

# Create test file for validation
echo "📝 Creating test file..."
cat > test_input.txt << EOF
This is a test text file to verify that the AI detector application is working correctly. 
The application should be able to process this text file and provide analysis results 
from multiple machine learning models including SBERT-FFNN, DistilBERT, and RoBERTa.

This text contains enough content to meet the minimum length requirements for analysis.
The file processor should successfully extract and clean this text for model inference.
EOF

echo "✅ Setup completed!"
echo ""
echo "🎯 To run the application:"
echo "   uvicorn main:app --host 0.0.0.0 --port 8000 --reload"
echo ""
echo "🌐 Then open: http://localhost:8000"
echo ""
echo "📁 Test file created: test_input.txt"
echo ""
echo "Note: First run will download model files (~2GB). Ensure stable internet connection."