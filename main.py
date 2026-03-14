import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
import os
import tempfile
from typing import Optional
import logging

from models.model_orchestrator import ModelOrchestrator
from file_handlers.file_processor import FileProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title="AI Text Detector",
              description="Detect AI-generated text using multiple models",
              version="1.0.0")

# Initialize model orchestrator
model_orchestrator = None

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.on_event("startup")
async def startup_event():
    """Initialize models on application startup"""
    global model_orchestrator
    try:
        model_orchestrator = ModelOrchestrator()
        logger.info("Models initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing models: {e}")
        raise


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serve the main UI page"""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "title": "AI Text Detector"
    })


@app.post("/api/detect/text")
async def detect_ai_from_text(text: str = Form(...)):
    """
    Detect AI content from text input
    
    Args:
        text: Input text to analyze
    
    Returns:
        JSON response with detection results
    """
    try:
        if not text or not isinstance(text, str):
            raise HTTPException(status_code=400, detail="Invalid text input")

        # Process and validate text
        processed_text = FileProcessor.process_input(text)
        if not FileProcessor.is_valid_text(processed_text):
            raise HTTPException(status_code=400,
                                detail="Text is too short for analysis")

        # Detect AI content
        result = model_orchestrator.detect_ai(processed_text)

        return JSONResponse(content=result)

    except Exception as e:
        logger.error(f"Error in text detection: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/detect/file")
async def detect_ai_from_file(file: UploadFile = File(...)):
    """
    Detect AI content from uploaded file
    
    Args:
        file: Uploaded file to analyze
    
    Returns:
        JSON response with detection results
    """
    try:
        # Validate file type
        allowed_extensions = {'.txt', '.md', '.markdown', '.tex'}
        file_extension = os.path.splitext(file.filename)[1].lower()

        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=
                f"File type {file_extension} not supported. Please upload: {', '.join(allowed_extensions)}"
            )

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False,
                                         suffix=file_extension) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name

        try:
            # Detect AI content from file
            result = model_orchestrator.detect_ai_from_file(temp_file_path)

            # Add file info to result
            result["file_info"] = {
                "filename": file.filename,
                "content_type": file.content_type,
                "size": len(content)
            }

            return JSONResponse(content=result)

        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

    except Exception as e:
        logger.error(f"Error in file detection: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    try:
        model_info = model_orchestrator.get_model_info()
        return {
            "status": "healthy",
            "models_loaded": len(model_info) > 0,
            "model_info": model_info
        }
    except Exception as e:
        raise HTTPException(status_code=500,
                            detail=f"Health check failed: {str(e)}")


@app.post("/api/detect/text/detailed")
async def detect_ai_from_text_detailed(text: str = Form(...)):
    """
    Detect AI content from text input with line-by-line analysis
    
    Args:
        text: Input text to analyze
    
    Returns:
        JSON response with detailed line-by-line analysis
    """
    try:
        if not text or not isinstance(text, str):
            raise HTTPException(status_code=400, detail="Invalid text input")

        # Process and validate text
        processed_text = FileProcessor.process_input(text)
        if not FileProcessor.is_valid_text(processed_text):
            raise HTTPException(status_code=400,
                                detail="Text is too short for analysis")

        # Detect AI content with line-by-line analysis
        result = model_orchestrator.detect_ai_line_by_line(processed_text)

        return JSONResponse(content=result)

    except Exception as e:
        logger.error(f"Error in detailed text detection: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/detect/file/detailed")
async def detect_ai_from_file_detailed(file: UploadFile = File(...)):
    """
    Detect AI content from uploaded file with line-by-line analysis
    
    Args:
        file: Uploaded file to analyze
    
    Returns:
        JSON response with detailed line-by-line analysis
    """
    try:
        # Validate file type
        allowed_extensions = {'.txt', '.md', '.markdown', '.tex'}
        file_extension = os.path.splitext(file.filename)[1].lower()

        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=
                f"File type {file_extension} not supported. Please upload: {', '.join(allowed_extensions)}"
            )

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False,
                                         suffix=file_extension) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name

        try:
            # Detect AI content from file with line-by-line analysis
            result = model_orchestrator.detect_ai_from_file_line_by_line(
                temp_file_path)

            # Add file info to result
            result["file_info"] = {
                "filename": file.filename,
                "content_type": file.content_type,
                "size": len(content)
            }

            return JSONResponse(content=result)

        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

    except Exception as e:
        logger.error(f"Error in detailed file detection: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/models")
async def get_models():
    """Get information about loaded models"""
    try:
        model_info = model_orchestrator.get_model_info()
        return JSONResponse(content=model_info)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("main:app",
                host="0.0.0.0",
                port=8000,
                reload=True,
                log_level="info")
