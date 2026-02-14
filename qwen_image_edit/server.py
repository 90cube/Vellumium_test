"""
Qwen Image Edit - FastAPI HTTP Server
======================================
Node-RED, API 클라이언트 등에서 HTTP로 호출 가능한 추론 서버.

엔드포인트:
  POST /generate     - 이미지 편집 (base64 입력, R2 URL 출력)
  GET  /health       - 서버 상태 확인
  POST /load         - 모델 로딩
  POST /unload       - 모델 해제
"""

import os
import sys
import io
import base64
import time
import uuid
from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PIL import Image
import boto3
from botocore.config import Config as BotoConfig

sys.path.insert(0, os.path.dirname(__file__))
from engine import QwenImageEditEngine

# Configuration
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8200"))
AUTO_LOAD = os.getenv("AUTO_LOAD", "true").lower() == "true"

# Cloudflare R2 Storage
R2_ENDPOINT = os.getenv("R2_ENDPOINT", "")
R2_BUCKET = os.getenv("R2_BUCKET", "vellumium-storage")
R2_ACCESS_KEY = os.getenv("R2_ACCESS_KEY_ID", "")
R2_SECRET_KEY = os.getenv("R2_SECRET_ACCESS_KEY", "")
R2_PUBLIC_URL = os.getenv("R2_PUBLIC_URL", "")

_s3_client = None
def _get_s3():
    global _s3_client
    if _s3_client is None:
        _s3_client = boto3.client(
            "s3",
            endpoint_url=R2_ENDPOINT,
            aws_access_key_id=R2_ACCESS_KEY,
            aws_secret_access_key=R2_SECRET_KEY,
            config=BotoConfig(signature_version="s3v4"),
            region_name="auto",
        )
    return _s3_client

def _upload_to_r2(image_bytes: bytes, filename: str) -> str:
    s3 = _get_s3()
    s3.put_object(Bucket=R2_BUCKET, Key=filename, Body=image_bytes, ContentType="image/png")
    if R2_PUBLIC_URL:
        return f"{R2_PUBLIC_URL}/{filename}"
    return f"{R2_ENDPOINT}/{R2_BUCKET}/{filename}"

engine = QwenImageEditEngine()


@asynccontextmanager
async def lifespan(app: FastAPI):
    if AUTO_LOAD:
        print("Auto-loading model on startup...")
        engine.load()
    yield
    if engine.is_loaded:
        engine.unload()


app = FastAPI(
    title="Qwen Image Edit API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response Models ──────────────────────────────

class GenerateRequest(BaseModel):
    images: list[str] = Field(..., description="Base64 인코딩된 입력 이미지 리스트")
    prompt: str = Field(..., description="편집 프롬프트")
    num_steps: int | None = Field(None, description="추론 스텝 수 (기본: LoRA=8, 베이스=28)")
    true_cfg_scale: float | None = Field(None, description="True CFG 스케일 (기본: LoRA=3.0, 베이스=4.0)")
    seed: int = Field(0, description="랜덤 시드")


class GenerateResponse(BaseModel):
    url: str = Field(..., description="R2에 업로드된 결과 이미지 URL")
    elapsed: float = Field(..., description="추론 시간 (초)")
    num_steps: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    has_lora: bool


# ── Endpoints ──────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        model_loaded=engine.is_loaded,
        has_lora=engine.has_lora,
    )


@app.post("/load")
async def load_model():
    if engine.is_loaded:
        return {"message": "Already loaded"}
    engine.load()
    return {"message": "Model loaded"}


@app.post("/unload")
async def unload_model():
    if not engine.is_loaded:
        return {"message": "Not loaded"}
    engine.unload()
    return {"message": "Model unloaded"}


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    if not engine.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded. POST /load first.")

    # base64 → PIL.Image
    pil_images = []
    for i, b64 in enumerate(req.images):
        try:
            raw = base64.b64decode(b64)
            pil_images.append(Image.open(io.BytesIO(raw)).convert("RGB"))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image at index {i}: {e}")

    num_steps = req.num_steps or (8 if engine.has_lora else 28)

    t0 = time.time()
    result_image = engine.generate(
        images=pil_images,
        prompt=req.prompt,
        num_steps=req.num_steps,
        true_cfg_scale=req.true_cfg_scale,
        seed=req.seed,
    )
    elapsed = time.time() - t0

    # PIL.Image → R2 upload
    buf = io.BytesIO()
    result_image.save(buf, format="PNG")
    image_bytes = buf.getvalue()

    filename = f"qwen-edit/{uuid.uuid4().hex}.png"
    url = _upload_to_r2(image_bytes, filename)

    return GenerateResponse(
        url=url,
        elapsed=round(elapsed, 1),
        num_steps=num_steps,
    )


if __name__ == "__main__":
    uvicorn.run("server:app", host=HOST, port=PORT, log_level="info")
