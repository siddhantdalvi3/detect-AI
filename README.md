# AI Text Detector

A powerful Python application that detects AI-generated content using multiple machine learning models. Supports text input and file uploads with a modern web interface.

## Features

### 🔍 AI Detection Models
- **SBERT-FFNN**: Sentence-BERT embeddings with feed-forward neural network
- **DistilBERT**: Lightweight BERT model for sequence classification
- **RoBERTa**: Robust BERT model optimized for text classification
- **Ensemble Scoring**: Combined prediction from all models

### 📁 File Support
- **Text files** (.txt)
- **Markdown files** (.md, .markdown)
- **LaTeX files** (.tex)
- **PDF/DOCX** (planned - dependencies pending)

### 🎨 Web Interface
- **Text input** with copy-paste support
- **File upload** with drag & drop functionality
- **Real-time results** with visual progress bars
- **Responsive design** works on desktop and mobile
- **Humanizer suggestions** for AI-detected content

### 🤖 Humanizer Model
- **Text rewriting** to make AI content sound more human
- **Multiple suggestions** with confidence scores
- **Context-aware** improvements using full input text

## Installation

### Prerequisites
- Python 3.9+
- uv package manager
- 2GB+ free disk space for models

### Quick Start

1. **Clone and setup** (if not already done):
```bash
# Navigate to project directory
cd detect-AI

# Install dependencies using uv
uv sync
```

2. **Run the application**:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

3. **Open your browser**:
```
http://localhost:8000
```

## Usage

### Text Input
1. Go to the "Text Input" tab
2. Paste your text (minimum 50 characters)
3. Click "Analyze Text"
4. View results from all models

### File Upload
1. Go to the "File Upload" tab  
2. Drag & drop or select a file (.txt, .md, .tex)
3. Click "Upload & Analyze"
4. See analysis results with file metadata

### Understanding Results
- **Overall Prediction**: Combined score from all models
- **Individual Model Scores**: Confidence percentages for each model
- **Humanizer Suggestions**: Text rewriting options if AI is detected
- **Confidence Levels**: Color-coded indicators (green=human, red=AI)

## API Endpoints

### Detect from Text
```bash
POST /api/detect/text
Content-Type: multipart/form-data

text=<your-text-here>
```

### Detect from File
```bash
POST /api/detect/file
Content-Type: multipart/form-data

file=<your-file-here>
```

### Health Check
```bash
GET /api/health
```

### Model Information
```bash
GET /api/models
```

## Project Structure

```
detect-AI/
├── main.py                 # FastAPI application entry point
├── pyproject.toml          # Project dependencies and configuration
├── uv.lock                 # Dependency lock file
├── models/
│   ├── __init__.py         # Package initialization
│   ├── sbert_ffnn_model.py # SBERT-FFNN model implementation
│   ├── distilbert_model.py # DistilBERT model implementation
│   ├── roberta_model.py    # RoBERTa model implementation
│   ├── humanizer_model.py  # Text humanizer model
│   └── model_orchestrator.py # Model management and orchestration
├── file_handlers/
│   ├── __init__.py         # Package initialization
│   └── file_processor.py   # File processing utilities
├── templates/
│   └── index.html          # Main web interface template
└── static/
    ├── css/
    │   └── style.css       # Custom styling
    └── js/
        └── app.js          # Frontend JavaScript functionality
```

## Model Details

### SBERT-FFNN Model
- **Base Model**: `sentence-transformers/all-mpnet-base-v2`
- **Embedding Size**: 768 dimensions
- **Architecture**: 3-layer feed-forward neural network
- **Output**: Binary classification (AI vs Human)

### DistilBERT Model  
- **Base Model**: `distilbert-base-uncased`
- **Parameters**: 66 million
- **Max Sequence Length**: 512 tokens
- **Specialization**: Text classification

### RoBERTa Model
- **Base Model**: `roberta-base`
- **Parameters**: 125 million  
- **Training**: Optimized for longer texts
- **Strength**: Context understanding

### Humanizer Model
- **Base Model**: `gpt2`
- **Function**: Text rewriting and paraphrasing
- **Output**: Multiple human-like alternatives
- **Features**: Context-aware suggestions

## Configuration

### Environment Variables
```bash
export PYTHONPATH=/path/to/detect-AI
export MODEL_CACHE_DIR=./model_cache
```

### Model Configuration
Models are automatically downloaded on first use and cached locally. The cache directory is typically:
- Linux/Mac: `~/.cache/huggingface/hub`
- Windows: `C:\Users\<user>\.cache\huggingface\hub`

## Performance

### Hardware Requirements
- **Minimum**: 4GB RAM, 2GB disk space
- **Recommended**: 8GB RAM, 4GB disk space, GPU for faster inference
- **Model Loading**: ~2-5 minutes first time (downloads models)
- **Inference Time**: ~1-3 seconds per request

### Memory Usage
- **Base Application**: ~500MB RAM
- **With Models Loaded**: ~2-3GB RAM
- **Peak Usage**: ~4GB during model inference

## Troubleshooting

### Common Issues

1. **Model download fails**:
   ```bash
   # Check internet connection
   # Clear huggingface cache: rm -rf ~/.cache/huggingface
   ```

2. **Memory errors**:
   ```bash
   # Reduce batch size in model configurations
   # Use smaller models or enable GPU
   ```

3. **File upload issues**:
   ```bash
   # Check file permissions
   # Verify supported file formats
   ```

### Logging

Enable debug logging for troubleshooting:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Development

### Adding New Models
1. Create model file in `models/` directory
2. Implement predict() method
3. Add to model orchestrator
4. Update API endpoints

### Adding File Support
1. Extend `FileProcessor` class
2. Add new file extension handling
3. Update allowed extensions in main.py

### Testing
```bash
# Run basic tests
python -m pytest tests/ -v

# Test specific functionality
python -c "from file_handlers.file_processor import FileProcessor; print(FileProcessor.extract_text_from_file('test.txt'))"
```

## License

This project is open source and available under the MIT License.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review model documentation
3. Check HuggingFace model cards for specific model issues

---

**Note**: First run will download ~2GB of model files. Ensure stable internet connection.