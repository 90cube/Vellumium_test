"""
Qwen Image Edit - Queue Worker
===============================
원격 큐 서버에서 작업을 폴링하여 처리하는 GPU 워커.
zimageturbo/worker.py 패턴 기반.
"""

import os
import sys
import io
import base64
import time
import requests
from datetime import datetime

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

sys.path.insert(0, os.path.dirname(__file__))

# Configuration
QUEUE_SERVER_URL = os.getenv("QUEUE_SERVER_URL", "http://localhost:8080")
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "5"))
AUTO_OFFLOAD = os.getenv("AUTO_OFFLOAD", "false").lower() == "true"
WORKER_ID = os.getenv("WORKER_ID", "qwen-edit-1")
API_KEY = os.getenv("API_KEY", "")


def log(msg, level="INFO"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] [{level}] {msg}")


class QwenEditWorker:
    """큐 서버에서 작업을 받아 처리하는 워커."""

    def __init__(self):
        self.engine = None
        log(f"Worker initialized: {WORKER_ID}")
        log(f"  Server: {QUEUE_SERVER_URL}")
        log(f"  Poll interval: {POLL_INTERVAL}s")
        log(f"  Auto offload: {AUTO_OFFLOAD}")

    def _get_headers(self):
        headers = {"Content-Type": "application/json"}
        if API_KEY:
            headers["X-API-Key"] = API_KEY
        return headers

    def _init_engine(self):
        if self.engine is None:
            log("Initializing engine...")
            from engine import QwenImageEditEngine
            self.engine = QwenImageEditEngine()
            self.engine.load()
            log("Engine ready")

    def poll_job(self):
        try:
            response = requests.get(
                f"{QUEUE_SERVER_URL}/api/queue/next",
                headers=self._get_headers(),
                params={"worker_id": WORKER_ID},
                timeout=10,
            )
            if response.status_code == 200:
                job = response.json()
                if job and job.get("job_id"):
                    return job
            elif response.status_code != 204:
                log(f"Poll error: {response.status_code}", "WARN")
        except requests.exceptions.ConnectionError:
            log("Cannot connect to server, retrying...", "WARN")
        except Exception as e:
            log(f"Poll error: {e}", "ERROR")
        return None

    def upload_result(self, job_id: str, image_b64: str = None, success: bool = True, error: str = None):
        try:
            payload = {
                "job_id": job_id,
                "worker_id": WORKER_ID,
                "success": success,
                "image_base64": image_b64 if success else None,
                "error": error,
            }
            response = requests.post(
                f"{QUEUE_SERVER_URL}/api/queue/result",
                headers=self._get_headers(),
                json=payload,
                timeout=30,
            )
            if response.status_code == 200:
                log(f"Result uploaded: {job_id}")
            else:
                log(f"Upload failed: {response.status_code}", "ERROR")
        except Exception as e:
            log(f"Upload error: {e}", "ERROR")

    def process_job(self, job: dict):
        job_id = job.get("job_id")
        prompt = job.get("prompt", "")
        input_images_b64 = job.get("images", [])
        num_steps = job.get("num_steps")
        true_cfg_scale = job.get("true_cfg_scale")
        seed = job.get("seed", 0)

        log(f"Processing job: {job_id}")
        log(f"  prompt: {prompt[:60]}...")
        log(f"  images: {len(input_images_b64)}")

        try:
            self._init_engine()

            # base64 → PIL
            from PIL import Image
            pil_images = []
            for b64 in input_images_b64:
                raw = base64.b64decode(b64)
                pil_images.append(Image.open(io.BytesIO(raw)).convert("RGB"))

            # 추론
            result_image = self.engine.generate(
                images=pil_images,
                prompt=prompt,
                num_steps=num_steps,
                true_cfg_scale=true_cfg_scale,
                seed=seed,
            )

            # PIL → base64
            buf = io.BytesIO()
            result_image.save(buf, format="PNG")
            result_b64 = base64.b64encode(buf.getvalue()).decode()

            self.upload_result(job_id, result_b64, success=True)

        except Exception as e:
            log(f"Job failed: {e}", "ERROR")
            self.upload_result(job_id, success=False, error=str(e))

        finally:
            if AUTO_OFFLOAD and self.engine:
                self.engine.unload()
                self.engine = None

    def run(self):
        log("=" * 50)
        log("Starting worker polling loop...")
        log(f"  Polling {QUEUE_SERVER_URL} every {POLL_INTERVAL}s")
        log("  Press Ctrl+C to stop")
        log("=" * 50)

        try:
            while True:
                job = self.poll_job()
                if job:
                    self.process_job(job)
                time.sleep(POLL_INTERVAL)
        except KeyboardInterrupt:
            log("Worker stopped by user")
            if self.engine:
                self.engine.unload()


if __name__ == "__main__":
    worker = QwenEditWorker()
    worker.run()
