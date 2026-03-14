import unittest
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

from fastapi.testclient import TestClient

import main as main_app


class StubModelOrchestrator:

    def __init__(self):
        self.text_calls = []
        self.file_calls = []

    def detect_ai(self, text, include_humanizer=False, allow_delayed=False):
        self.text_calls.append({
            "text": text,
            "include_humanizer": include_humanizer,
            "allow_delayed": allow_delayed,
        })
        return {
            "text": text,
            "ensemble_score": 0.2,
            "ensemble_prediction": "Human",
            "model_results": {},
            "humanizer_suggestions": [],
        }

    def detect_ai_from_file(self,
                            file_path,
                            include_humanizer=False,
                            allow_delayed=False):
        self.file_calls.append({
            "file_path": file_path,
            "include_humanizer": include_humanizer,
            "allow_delayed": allow_delayed,
        })
        return {
            "text": "file text",
            "ensemble_score": 0.2,
            "ensemble_prediction": "Human",
            "model_results": {},
            "humanizer_suggestions": [],
        }

    def get_model_info(self):
        return {"stub": {"status": "loaded"}}

    def get_async_result(self, request_id):
        return {
            "status": "completed",
            "request_id": request_id,
            "result": {
                "text": "done",
                "ensemble_score": 0.2,
                "ensemble_prediction": "Human",
                "model_results": {},
                "humanizer_suggestions": [],
            },
            "error": None,
        }


class ApiOptimizationTests(unittest.TestCase):

    _slow_route_added = False

    def setUp(self):
        self._original_orchestrator = main_app.model_orchestrator
        self._original_startup = list(main_app.app.router.on_startup)
        self._original_max_file_size = main_app.MAX_FILE_SIZE_BYTES
        self._original_rate_limit = main_app.RATE_LIMIT_REQUESTS_PER_MINUTE
        self._original_timeout = main_app.REQUEST_TIMEOUT_SECONDS
        self._original_max_concurrency = main_app.MAX_CONCURRENT_REQUESTS
        self._original_max_queue = main_app.MAX_QUEUED_REQUESTS
        self._original_queue_timeout = main_app.REQUEST_QUEUE_TIMEOUT_SECONDS
        self._original_queue_manager = main_app.request_queue_manager
        self._original_text_rate_limit = main_app.TEXT_RATE_LIMIT_REQUESTS_PER_MINUTE
        self._original_file_rate_limit = main_app.FILE_RATE_LIMIT_REQUESTS_PER_MINUTE
        self._original_models_rate_limit = main_app.MODELS_RATE_LIMIT_REQUESTS_PER_MINUTE
        self._original_text_queue_timeout = main_app.TEXT_QUEUE_TIMEOUT_SECONDS
        self._original_file_queue_timeout = main_app.FILE_QUEUE_TIMEOUT_SECONDS
        self._original_models_queue_timeout = main_app.MODELS_QUEUE_TIMEOUT_SECONDS
        self._original_text_queue_cap = main_app.TEXT_MAX_QUEUED_REQUESTS
        self._original_file_queue_cap = main_app.FILE_MAX_QUEUED_REQUESTS
        self._original_models_queue_cap = main_app.MODELS_MAX_QUEUED_REQUESTS
        self._original_local_ignore_limits = main_app.LOCAL_DEV_IGNORE_LIMITS

        main_app.app.router.on_startup = []
        self.stub = StubModelOrchestrator()
        main_app.model_orchestrator = self.stub
        main_app.LOCAL_DEV_IGNORE_LIMITS = False
        main_app.rate_limit_state.clear()
        main_app.request_queue_manager = main_app.RequestQueueManager(
            main_app.MAX_CONCURRENT_REQUESTS, main_app.MAX_QUEUED_REQUESTS)

        if not self.__class__._slow_route_added:

            @main_app.app.get("/api/test/slow")
            async def test_slow_endpoint():
                await asyncio.sleep(0.05)
                return {"status": "ok"}

            self.__class__._slow_route_added = True

        self.client = TestClient(main_app.app)

    def tearDown(self):
        main_app.model_orchestrator = self._original_orchestrator
        main_app.app.router.on_startup = self._original_startup
        main_app.MAX_FILE_SIZE_BYTES = self._original_max_file_size
        main_app.RATE_LIMIT_REQUESTS_PER_MINUTE = self._original_rate_limit
        main_app.REQUEST_TIMEOUT_SECONDS = self._original_timeout
        main_app.MAX_CONCURRENT_REQUESTS = self._original_max_concurrency
        main_app.MAX_QUEUED_REQUESTS = self._original_max_queue
        main_app.REQUEST_QUEUE_TIMEOUT_SECONDS = self._original_queue_timeout
        main_app.request_queue_manager = self._original_queue_manager
        main_app.TEXT_RATE_LIMIT_REQUESTS_PER_MINUTE = self._original_text_rate_limit
        main_app.FILE_RATE_LIMIT_REQUESTS_PER_MINUTE = self._original_file_rate_limit
        main_app.MODELS_RATE_LIMIT_REQUESTS_PER_MINUTE = self._original_models_rate_limit
        main_app.TEXT_QUEUE_TIMEOUT_SECONDS = self._original_text_queue_timeout
        main_app.FILE_QUEUE_TIMEOUT_SECONDS = self._original_file_queue_timeout
        main_app.MODELS_QUEUE_TIMEOUT_SECONDS = self._original_models_queue_timeout
        main_app.TEXT_MAX_QUEUED_REQUESTS = self._original_text_queue_cap
        main_app.FILE_MAX_QUEUED_REQUESTS = self._original_file_queue_cap
        main_app.MODELS_MAX_QUEUED_REQUESTS = self._original_models_queue_cap
        main_app.LOCAL_DEV_IGNORE_LIMITS = self._original_local_ignore_limits
        main_app.rate_limit_state.clear()

    def test_text_endpoint_passes_include_humanizer_flag(self):
        response = self.client.post(
            "/api/detect/text",
            data={
                "text":
                "This is a sufficiently long sample text for API validation tests.",
                "include_humanizer": "true",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(self.stub.text_calls), 1)
        self.assertTrue(self.stub.text_calls[0]["include_humanizer"])

    def test_text_endpoint_passes_allow_delayed_flag(self):
        response = self.client.post(
            "/api/detect/text",
            data={
                "text":
                "This is a sufficiently long sample text for delayed flag testing.",
                "allow_delayed": "true",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(self.stub.text_calls), 1)
        self.assertTrue(self.stub.text_calls[0]["allow_delayed"])

    def test_file_endpoint_rejects_oversized_upload(self):
        main_app.MAX_FILE_SIZE_BYTES = 10

        response = self.client.post(
            "/api/detect/file",
            files={"file": ("sample.txt", b"01234567890", "text/plain")},
        )

        self.assertEqual(response.status_code, 413)
        self.assertIn("File too large", response.json()["detail"])
        self.assertEqual(len(self.stub.file_calls), 0)

    def test_rate_limit_per_ip(self):
        main_app.MODELS_RATE_LIMIT_REQUESTS_PER_MINUTE = 1
        main_app.rate_limit_state.clear()

        response_1 = self.client.get("/api/models")
        response_2 = self.client.get("/api/models")

        self.assertEqual(response_1.status_code, 200)
        self.assertEqual(response_2.status_code, 429)
        self.assertIn("Rate limit exceeded", response_2.json()["detail"])

    def test_endpoint_specific_rate_limit_policy(self):
        main_app.MODELS_RATE_LIMIT_REQUESTS_PER_MINUTE = 1
        main_app.TEXT_RATE_LIMIT_REQUESTS_PER_MINUTE = 3
        main_app.rate_limit_state.clear()

        models_1 = self.client.get("/api/models")
        models_2 = self.client.get("/api/models")

        text_1 = self.client.post(
            "/api/detect/text",
            data={
                "text":
                "This is a sufficiently long sample text for endpoint policy testing."
            },
        )
        text_2 = self.client.post(
            "/api/detect/text",
            data={
                "text":
                "This is another sufficiently long text request for endpoint policy testing."
            },
        )

        self.assertEqual(models_1.status_code, 200)
        self.assertEqual(models_2.status_code, 429)
        self.assertEqual(text_1.status_code, 200)
        self.assertEqual(text_2.status_code, 200)

    def test_request_timeout_limit(self):
        main_app.REQUEST_TIMEOUT_SECONDS = 0.01

        response = self.client.get("/api/test/slow")

        self.assertEqual(response.status_code, 504)
        self.assertEqual(response.json()["detail"], "Request timed out")

    def test_request_queue_limit(self):
        main_app.TEXT_MAX_QUEUED_REQUESTS = 0
        main_app.request_queue_manager = main_app.RequestQueueManager(
            max_concurrent=1, max_queued=10)

        worker_client = TestClient(main_app.app)

        with ThreadPoolExecutor(max_workers=1) as executor:
            slow_request = executor.submit(worker_client.get, "/api/test/slow")
            time.sleep(0.01)

            response = self.client.post(
                "/api/detect/text",
                data={
                    "text":
                    "This is a sufficiently long sample text for queue testing."
                },
            )

            slow_response = slow_request.result()

        self.assertEqual(slow_response.status_code, 200)
        self.assertEqual(response.status_code, 503)
        self.assertIn("queue is full", response.json()["detail"])

    def test_fifo_queue_execution(self):
        main_app.REQUEST_TIMEOUT_SECONDS = 1.0
        main_app.REQUEST_QUEUE_TIMEOUT_SECONDS = 1.0
        main_app.request_queue_manager = main_app.RequestQueueManager(
            max_concurrent=1, max_queued=1)

        worker_client = TestClient(main_app.app)

        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(worker_client.get, "/api/test/slow")
            second = executor.submit(worker_client.get, "/api/test/slow")

            first_response = first.result()
            second_response = second.result()
        elapsed = time.perf_counter() - start

        self.assertEqual(first_response.status_code, 200)
        self.assertEqual(second_response.status_code, 200)
        self.assertGreaterEqual(elapsed, 0.09)

    def test_queue_status_endpoint(self):
        response = self.client.get("/api/queue/status")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("running", payload)
        self.assertIn("queued", payload)
        self.assertIn("max_concurrent", payload)
        self.assertIn("max_queued", payload)
        self.assertIn("avg_processing_seconds", payload)
        self.assertIn("estimated_wait_seconds", payload)
        self.assertIn("policies", payload)

    def test_delayed_result_endpoint(self):
        response = self.client.get("/api/detect/result/test-id")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "completed")
        self.assertEqual(payload["request_id"], "test-id")


if __name__ == "__main__":
    unittest.main()
