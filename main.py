import uvicorn
import asyncio
import time
from collections import deque
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
import os
import tempfile
import logging

from models.model_orchestrator import ModelOrchestrator
from file_handlers.file_processor import FileProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _parse_bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name, str(default)).strip().lower()
    return value in {"1", "true", "t", "yes", "y", "on"}


HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
RELOAD = _parse_bool_env("RELOAD", False)
LOG_LEVEL = os.getenv("LOG_LEVEL", "info")
MAX_FILE_SIZE_MB = float(os.getenv("MAX_FILE_SIZE_MB", "10"))
MAX_FILE_SIZE_BYTES = int(MAX_FILE_SIZE_MB * 1024 * 1024)
RATE_LIMIT_REQUESTS_PER_MINUTE = int(
    os.getenv("RATE_LIMIT_REQUESTS_PER_MINUTE", "60"))
REQUEST_TIMEOUT_SECONDS = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "30"))
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "2"))
MAX_QUEUED_REQUESTS = int(os.getenv("MAX_QUEUED_REQUESTS", "10"))
REQUEST_QUEUE_TIMEOUT_SECONDS = float(
    os.getenv("REQUEST_QUEUE_TIMEOUT_SECONDS", "5"))
TEXT_RATE_LIMIT_REQUESTS_PER_MINUTE = int(
    os.getenv("TEXT_RATE_LIMIT_REQUESTS_PER_MINUTE",
              str(RATE_LIMIT_REQUESTS_PER_MINUTE)))
FILE_RATE_LIMIT_REQUESTS_PER_MINUTE = int(
    os.getenv("FILE_RATE_LIMIT_REQUESTS_PER_MINUTE",
              str(RATE_LIMIT_REQUESTS_PER_MINUTE)))
MODELS_RATE_LIMIT_REQUESTS_PER_MINUTE = int(
    os.getenv("MODELS_RATE_LIMIT_REQUESTS_PER_MINUTE",
              str(RATE_LIMIT_REQUESTS_PER_MINUTE)))
TEXT_QUEUE_TIMEOUT_SECONDS = float(
    os.getenv("TEXT_QUEUE_TIMEOUT_SECONDS",
              str(REQUEST_QUEUE_TIMEOUT_SECONDS)))
FILE_QUEUE_TIMEOUT_SECONDS = float(
    os.getenv("FILE_QUEUE_TIMEOUT_SECONDS",
              str(REQUEST_QUEUE_TIMEOUT_SECONDS)))
MODELS_QUEUE_TIMEOUT_SECONDS = float(
    os.getenv("MODELS_QUEUE_TIMEOUT_SECONDS",
              str(REQUEST_QUEUE_TIMEOUT_SECONDS)))
TEXT_MAX_QUEUED_REQUESTS = int(
    os.getenv("TEXT_MAX_QUEUED_REQUESTS", str(MAX_QUEUED_REQUESTS)))
FILE_MAX_QUEUED_REQUESTS = int(
    os.getenv("FILE_MAX_QUEUED_REQUESTS", str(MAX_QUEUED_REQUESTS)))
MODELS_MAX_QUEUED_REQUESTS = int(
    os.getenv("MODELS_MAX_QUEUED_REQUESTS", str(MAX_QUEUED_REQUESTS)))
TRUST_PROXY_HEADERS = _parse_bool_env("TRUST_PROXY_HEADERS", False)
LOCAL_DEV_IGNORE_LIMITS = _parse_bool_env("LOCAL_DEV_IGNORE_LIMITS", True)

allowed_origins_raw = os.getenv("ALLOWED_ORIGINS", "*")
ALLOWED_ORIGINS = [
    origin.strip() for origin in allowed_origins_raw.split(",")
    if origin.strip()
]
if not ALLOWED_ORIGINS:
    ALLOWED_ORIGINS = ["*"]

rate_limit_state = {}
rate_limit_lock = asyncio.Lock()
processing_time_lock = asyncio.Lock()
processing_avg_seconds = 2.0
processing_samples = 0


class RequestQueueManager:

    def __init__(self, max_concurrent: int, max_queued: int):
        self.max_concurrent = max(1, max_concurrent)
        self.max_queued = max(0, max_queued)
        self._running = 0
        self._queue = deque()
        self._lock = asyncio.Lock()

    async def acquire(self, wait_timeout_seconds: float):
        async with self._lock:
            if self._running < self.max_concurrent and not self._queue:
                self._running += 1
                return True, "acquired"

            if len(self._queue) >= self.max_queued:
                return False, "full"

            ticket = asyncio.get_running_loop().create_future()
            self._queue.append(ticket)

        try:
            await asyncio.wait_for(ticket, timeout=wait_timeout_seconds)
            return True, "queued"
        except asyncio.TimeoutError:
            async with self._lock:
                try:
                    self._queue.remove(ticket)
                    return False, "timeout"
                except ValueError:
                    # Ticket was already popped and granted while timing out.
                    pass

            await ticket
            return True, "queued"

    async def release(self):
        next_ticket = None

        async with self._lock:
            while self._queue:
                candidate = self._queue.popleft()
                if not candidate.done():
                    next_ticket = candidate
                    break

            if next_ticket is None:
                self._running = max(0, self._running - 1)

        if next_ticket is not None:
            next_ticket.set_result(True)

    async def get_stats(self):
        async with self._lock:
            return {
                "running": self._running,
                "queued": len(self._queue),
                "max_concurrent": self.max_concurrent,
                "max_queued": self.max_queued,
            }


request_queue_manager = RequestQueueManager(MAX_CONCURRENT_REQUESTS,
                                            MAX_QUEUED_REQUESTS)

# Create FastAPI app
app = FastAPI(title="AI Text Detector",
              description="Detect AI-generated text using multiple models",
              version="1.0.0")

app.add_middleware(CORSMiddleware,
                   allow_origins=ALLOWED_ORIGINS,
                   allow_credentials=True,
                   allow_methods=["*"],
                   allow_headers=["*"])
app.add_middleware(GZipMiddleware, minimum_size=1000)


def _is_protected_api_path(path: str) -> bool:
    if path in {"/api/health", "/api/queue/status"}:
        return False
    if path.startswith("/api/detect/result/"):
        return False
    return path.startswith("/api/")


def _get_endpoint_policy(path: str):
    default_policy = {
        "name": "default",
        "rate_limit": RATE_LIMIT_REQUESTS_PER_MINUTE,
        "queue_timeout": REQUEST_QUEUE_TIMEOUT_SECONDS,
        "queue_cap": MAX_QUEUED_REQUESTS,
    }

    if path.startswith("/api/detect/file"):
        return {
            "name": "file",
            "rate_limit": FILE_RATE_LIMIT_REQUESTS_PER_MINUTE,
            "queue_timeout": FILE_QUEUE_TIMEOUT_SECONDS,
            "queue_cap": min(MAX_QUEUED_REQUESTS, FILE_MAX_QUEUED_REQUESTS),
        }

    if path.startswith("/api/detect/text"):
        return {
            "name": "text",
            "rate_limit": TEXT_RATE_LIMIT_REQUESTS_PER_MINUTE,
            "queue_timeout": TEXT_QUEUE_TIMEOUT_SECONDS,
            "queue_cap": min(MAX_QUEUED_REQUESTS, TEXT_MAX_QUEUED_REQUESTS),
        }

    if path == "/api/models":
        return {
            "name": "models",
            "rate_limit": MODELS_RATE_LIMIT_REQUESTS_PER_MINUTE,
            "queue_timeout": MODELS_QUEUE_TIMEOUT_SECONDS,
            "queue_cap": min(MAX_QUEUED_REQUESTS, MODELS_MAX_QUEUED_REQUESTS),
        }

    return default_policy


def _get_client_ip(request: Request) -> str:
    if TRUST_PROXY_HEADERS:
        xff = request.headers.get("x-forwarded-for", "").strip()
        if xff:
            return xff.split(",")[0].strip()

        xri = request.headers.get("x-real-ip", "").strip()
        if xri:
            return xri

    if request.client and request.client.host:
        return request.client.host
    return "unknown"


async def _check_rate_limit(ip: str, policy_name: str, max_requests: int):
    now = time.time()
    key = f"{policy_name}:{ip}"

    async with rate_limit_lock:
        window_start, count = rate_limit_state.get(key, (now, 0))

        if now - window_start >= 60:
            window_start, count = now, 0

        if count >= max_requests:
            retry_after = max(1, int(60 - (now - window_start)))
            return False, retry_after

        rate_limit_state[key] = (window_start, count + 1)

        # Opportunistic cleanup for stale IP buckets.
        if len(rate_limit_state) > 1000:
            stale_keys = [
                key for key, (start, _) in rate_limit_state.items()
                if now - start >= 120
            ]
            for key in stale_keys:
                rate_limit_state.pop(key, None)

    return True, 0


async def _record_processing_time(duration_seconds: float):
    global processing_avg_seconds, processing_samples

    async with processing_time_lock:
        processing_samples += 1
        if processing_samples == 1:
            processing_avg_seconds = duration_seconds
        else:
            # Exponential moving average smooths spikes but stays responsive.
            processing_avg_seconds = (0.8 * processing_avg_seconds) + (
                0.2 * duration_seconds)


@app.middleware("http")
async def protection_middleware(request: Request, call_next):
    if not _is_protected_api_path(request.url.path):
        return await call_next(request)

    # Local development switch to bypass rate, queue, and timeout enforcement.
    if LOCAL_DEV_IGNORE_LIMITS:
        return await call_next(request)

    policy = _get_endpoint_policy(request.url.path)

    client_ip = _get_client_ip(request)
    allowed, retry_after = await _check_rate_limit(client_ip, policy["name"],
                                                   policy["rate_limit"])
    if not allowed:
        return JSONResponse(
            status_code=429,
            headers={"Retry-After": str(retry_after)},
            content={
                "detail": f"Rate limit exceeded for {policy['name']} endpoint"
            })

    acquired = False
    processing_start = 0.0
    try:
        stats = await request_queue_manager.get_stats()
        if stats["queued"] >= policy["queue_cap"]:
            return JSONResponse(
                status_code=503,
                content={
                    "detail": f"Server is busy. {policy['name']} queue is full"
                })

        acquired, reason = await request_queue_manager.acquire(
            policy["queue_timeout"])
        if not acquired and reason == "full":
            return JSONResponse(
                status_code=503,
                content={
                    "detail": f"Server is busy. {policy['name']} queue is full"
                })
        if not acquired and reason == "timeout":
            return JSONResponse(
                status_code=503,
                content={
                    "detail":
                    f"Server is busy. {policy['name']} queue wait timeout"
                })

        try:
            processing_start = time.perf_counter()
            return await asyncio.wait_for(call_next(request),
                                          timeout=REQUEST_TIMEOUT_SECONDS)
        except asyncio.TimeoutError:
            return JSONResponse(status_code=504,
                                content={"detail": "Request timed out"})
    finally:
        if acquired:
            if processing_start > 0:
                duration = time.perf_counter() - processing_start
                await _record_processing_time(duration)
            await request_queue_manager.release()


# Initialize model orchestrator
model_orchestrator = None

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


async def _read_with_size_limit(file: UploadFile) -> bytes:
    content = await file.read()
    if LOCAL_DEV_IGNORE_LIMITS:
        return content

    if len(content) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=
            f"File too large. Maximum allowed size is {MAX_FILE_SIZE_MB:g} MB")
    return content


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
async def detect_ai_from_text(text: str = Form(...),
                              include_humanizer: bool = Form(False),
                              allow_delayed: bool = Form(False)):
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
        result = model_orchestrator.detect_ai(
            processed_text,
            include_humanizer=include_humanizer,
            allow_delayed=allow_delayed)

        return JSONResponse(content=result)

    except HTTPException:
        raise
    except Exception:
        logger.exception("Error in text detection")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/api/detect/file")
async def detect_ai_from_file(file: UploadFile = File(...),
                              include_humanizer: bool = Form(False),
                              allow_delayed: bool = Form(False)):
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

        content = await _read_with_size_limit(file)

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False,
                                         suffix=file_extension) as temp_file:
            temp_file.write(content)
            temp_file_path = temp_file.name

        try:
            # Detect AI content from file
            result = model_orchestrator.detect_ai_from_file(
                temp_file_path,
                include_humanizer=include_humanizer,
                allow_delayed=allow_delayed)

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

    except HTTPException:
        raise
    except Exception:
        logger.exception("Error in file detection")
        raise HTTPException(status_code=500, detail="Internal server error")


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
    except Exception:
        logger.exception("Health check failed")
        raise HTTPException(status_code=500, detail="Health check failed")


@app.get("/api/queue/status")
async def queue_status():
    """Queue and processing status for frontend live updates"""
    try:
        stats = await request_queue_manager.get_stats()

        async with processing_time_lock:
            avg_seconds = processing_avg_seconds

        estimated_wait_seconds = round(
            (stats["queued"] / max(1, stats["max_concurrent"])) * avg_seconds,
            2)

        policy_snapshot = {
            "text": {
                "rate_limit": TEXT_RATE_LIMIT_REQUESTS_PER_MINUTE,
                "queue_timeout": TEXT_QUEUE_TIMEOUT_SECONDS,
                "queue_cap": min(MAX_QUEUED_REQUESTS, TEXT_MAX_QUEUED_REQUESTS)
            },
            "file": {
                "rate_limit": FILE_RATE_LIMIT_REQUESTS_PER_MINUTE,
                "queue_timeout": FILE_QUEUE_TIMEOUT_SECONDS,
                "queue_cap": min(MAX_QUEUED_REQUESTS, FILE_MAX_QUEUED_REQUESTS)
            },
            "models": {
                "rate_limit": MODELS_RATE_LIMIT_REQUESTS_PER_MINUTE,
                "queue_timeout": MODELS_QUEUE_TIMEOUT_SECONDS,
                "queue_cap": min(MAX_QUEUED_REQUESTS,
                                 MODELS_MAX_QUEUED_REQUESTS)
            },
            "default": {
                "rate_limit": RATE_LIMIT_REQUESTS_PER_MINUTE,
                "queue_timeout": REQUEST_QUEUE_TIMEOUT_SECONDS,
                "queue_cap": MAX_QUEUED_REQUESTS
            },
        }

        return {
            "running":
            stats["running"],
            "queued":
            stats["queued"],
            "max_concurrent":
            stats["max_concurrent"],
            "max_queued":
            stats["max_queued"],
            "avg_processing_seconds":
            round(avg_seconds, 2),
            "estimated_wait_seconds":
            estimated_wait_seconds,
            "async_heavy":
            model_orchestrator.get_model_info().get("async_heavy", {}),
            "policies":
            policy_snapshot,
        }
    except Exception:
        logger.exception("Queue status check failed")
        raise HTTPException(status_code=500, detail="Queue status unavailable")


@app.get("/api/detect/result/{request_id}")
async def get_delayed_detection_result(request_id: str):
    """Retrieve async heavy analysis result when allow_delayed=true was used."""
    try:
        result = model_orchestrator.get_async_result(request_id)
        if result.get("status") == "not_found":
            raise HTTPException(status_code=404, detail=result.get("detail"))
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception:
        logger.exception("Error fetching delayed detection result")
        raise HTTPException(status_code=500, detail="Internal server error")


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

    except HTTPException:
        raise
    except Exception:
        logger.exception("Error in detailed text detection")
        raise HTTPException(status_code=500, detail="Internal server error")


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

        content = await _read_with_size_limit(file)

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False,
                                         suffix=file_extension) as temp_file:
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

    except HTTPException:
        raise
    except Exception:
        logger.exception("Error in detailed file detection")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/models")
async def get_models():
    """Get information about loaded models"""
    try:
        model_info = model_orchestrator.get_model_info()
        return JSONResponse(content=model_info)
    except Exception:
        logger.exception("Error getting model info")
        raise HTTPException(status_code=500, detail="Internal server error")


if __name__ == "__main__":
    uvicorn.run("main:app",
                host=HOST,
                port=PORT,
                reload=RELOAD,
                log_level=LOG_LEVEL,
                timeout_keep_alive=30)
