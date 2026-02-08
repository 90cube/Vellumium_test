"""
Local GPU Worker for Remote Queue Server
=========================================
Polls remote server for generation jobs, processes them, and uploads results.
"""

import os
import sys
import time
import base64
import requests
import io
from datetime import datetime
from dotenv import load_dotenv

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

# Load environment variables
load_dotenv()

# Configuration
QUEUE_SERVER_URL = os.getenv("QUEUE_SERVER_URL", "http://localhost:8000")
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "5"))
AUTO_OFFLOAD = os.getenv("AUTO_OFFLOAD", "true").lower() == "true"
WORKER_ID = os.getenv("WORKER_ID", "local-gpu-1")
API_KEY = os.getenv("API_KEY", "")

# Logging helper
def log(msg, level="INFO"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] [{level}] {msg}")


class ImageWorker:
    """Worker that polls server for jobs and processes them."""
    
    def __init__(self):
        self.engine = None
        self.current_model = None
        log(f"🚀 Worker initialized: {WORKER_ID}")
        log(f"   Server: {QUEUE_SERVER_URL}")
        log(f"   Poll interval: {POLL_INTERVAL}s")
        log(f"   Auto offload: {AUTO_OFFLOAD}")
    
    def _get_headers(self):
        """Get request headers with API key if set."""
        headers = {"Content-Type": "application/json"}
        if API_KEY:
            headers["X-API-Key"] = API_KEY
        return headers
    
    def _init_engine(self):
        """Lazy init the model engine."""
        if self.engine is None:
            log("⏳ Initializing model engine...")
            from app_engine import ModelManager
            self.engine = ModelManager()
            log("✅ Engine initialized")
    
    def poll_job(self):
        """Poll server for next available job."""
        try:
            response = requests.get(
                f"{QUEUE_SERVER_URL}/api/queue/next",
                headers=self._get_headers(),
                params={"worker_id": WORKER_ID},
                timeout=10
            )
            
            if response.status_code == 200:
                job = response.json()
                if job and job.get("job_id"):
                    return job
            elif response.status_code == 204:
                # No content - queue empty
                return None
            else:
                log(f"Poll error: {response.status_code}", "WARN")
                
        except requests.exceptions.ConnectionError:
            log("⚠️ Cannot connect to server, retrying...", "WARN")
        except Exception as e:
            log(f"Poll error: {e}", "ERROR")
        
        return None
    
    def load_model(self, model_type: str):
        """Load model if not already loaded."""
        self._init_engine()
        
        if self.current_model == model_type:
            log(f"   Model {model_type} already loaded")
            return
        
        log(f"📥 Loading model: {model_type}...")
        result = self.engine.load_model(model_type)
        self.current_model = model_type
        log(f"✅ Model loaded: {result}")
    
    def generate_image(self, prompt: str, image_data: str = None, steps: int = 8):
        """Generate image from prompt (Turbo: CFG=0.0 fixed, no negative prompt)."""
        log(f"🎨 Generating: {prompt[:50]}...")

        # Decode input image if provided
        input_image = None
        if image_data:
            from PIL import Image
            img_bytes = base64.b64decode(image_data)
            input_image = Image.open(io.BytesIO(img_bytes))

        # Generate (Turbo engine handles CFG internally)
        result_image = self.engine.generate(
            prompt=prompt,
            image=input_image,
            steps=steps,
        )
        
        # Convert to base64
        buffer = io.BytesIO()
        result_image.save(buffer, format="PNG")
        result_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        log("✅ Image generated")
        return result_base64
    
    def upload_result(self, job_id: str, image_base64: str, success: bool = True, error: str = None):
        """Upload result back to server."""
        log(f"📤 Uploading result for job {job_id}...")
        
        try:
            payload = {
                "job_id": job_id,
                "worker_id": WORKER_ID,
                "success": success,
                "image_base64": image_base64 if success else None,
                "error": error
            }
            
            response = requests.post(
                f"{QUEUE_SERVER_URL}/api/queue/result",
                headers=self._get_headers(),
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                log("✅ Result uploaded")
                return True
            else:
                log(f"Upload failed: {response.status_code} - {response.text}", "ERROR")
                return False
                
        except Exception as e:
            log(f"Upload error: {e}", "ERROR")
            return False
    
    def offload_model(self):
        """Unload model to free VRAM."""
        if self.engine and AUTO_OFFLOAD:
            log("🧹 Offloading model...")
            self.engine.unload_model()
            self.current_model = None
            log("✅ Model offloaded, VRAM freed")
    
    def process_job(self, job: dict):
        """Process a single job."""
        job_id = job.get("job_id")
        model_type = job.get("model", "z-image-turbo")
        prompt = job.get("prompt", "")
        input_image = job.get("input_image")  # base64 or None
        steps = job.get("steps", 8)

        log(f"📋 Processing job: {job_id} (model: {model_type})")

        try:
            # 1. Load model
            self.load_model(model_type)

            # 2. Generate image
            result_image = self.generate_image(prompt, input_image, steps)
            
            # 3. Upload result
            self.upload_result(job_id, result_image, success=True)
            
        except Exception as e:
            log(f"❌ Job failed: {e}", "ERROR")
            self.upload_result(job_id, None, success=False, error=str(e))
        
        finally:
            # 4. Offload model if configured
            if AUTO_OFFLOAD:
                self.offload_model()
    
    def run(self):
        """Main worker loop."""
        log("=" * 50)
        log("🔄 Starting worker polling loop...")
        log(f"   Polling {QUEUE_SERVER_URL} every {POLL_INTERVAL}s")
        log("   Press Ctrl+C to stop")
        log("=" * 50)
        
        try:
            while True:
                job = self.poll_job()
                
                if job:
                    self.process_job(job)
                else:
                    # Idle - show heartbeat every minute
                    pass
                
                time.sleep(POLL_INTERVAL)
                
        except KeyboardInterrupt:
            log("\n🛑 Worker stopped by user")
            self.offload_model()


def main():
    worker = ImageWorker()
    worker.run()


if __name__ == "__main__":
    main()
