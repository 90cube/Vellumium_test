# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Vellumium is a local AI image generation platform built around the **Z-Image** model family (by Tongyi-MAI / Alibaba PAI). It provides a Gradio-based web UI and a queue-based worker system for generating images using Z-Image-Turbo with ControlNet, LoRA, and inpainting support.

The project contains Korean (한국어) UI strings and documentation alongside English content.

## Repository Structure

- **`zimageturbo/`** — Main application (the active codebase)
  - `app_engine.py` — Core application: FastAPI server + Gradio UI + `ModelManager` class (model loading, ControlNet, LoRA, generation)
  - `worker.py` — Queue-based GPU worker that polls a remote server for generation jobs
  - `cache_dit/` — Vendored [Cache-DiT](https://github.com/ali-vilab/Cache-DiT) library for inference acceleration (DiT block caching)
  - `models/` — Local model weights (gitignored, ~40GB+)
    - `z_image_turbo/` — Base Z-Image-Turbo transformer + scheduler
    - `Personalized_Model/` — ControlNet `.safetensors` weights (Union, Tile, Lite variants)
    - `LoRA/` — LoRA adapter weights (`.safetensors`)
    - `text_encoders/` — Qwen3 text encoder
    - `vae/` — VAE weights
  - `z_image_control_2.1.yaml` — ControlNet layer configuration (15 control layers + 2 refiner layers)
  - `asset/` — Sample control images (pose, canny, depth, hed, mask, etc.)
  - `.env` — Worker configuration (QUEUE_SERVER_URL, WORKER_ID, API_KEY)
  - `venv/` — Python virtual environment (gitignored)
- **`zimagebase/`** — Documentation for the non-turbo Z-Image base model (reference only)
- **`Logo/`** — Brand assets at various resolutions

## Running the Application

### Start the Gradio UI server
```bash
cd zimageturbo
venv\Scripts\python.exe app_engine.py
# Or use the batch file:
start_server.bat
```
Server starts at `http://localhost:7860`.

### Start the queue worker
```bash
cd zimageturbo
venv\Scripts\python.exe worker.py
# Or use the batch file:
start_worker.bat
```
Worker polls the queue server defined in `.env`.

### Install dependencies
```bash
cd zimageturbo
python -m venv venv
venv\Scripts\pip.exe install -r requirements.txt
```
`requirements_backup.txt` contains a full pinned freeze for reproducibility (torch 2.10.0+cu130, diffusers 0.36.0, transformers 5.1.0, gradio 6.5.1).

## Architecture

### ModelManager (app_engine.py)
Central singleton (`engine`) managing the full lifecycle:
1. **Model Loading** — Two paths: `videox_fun` library (preferred, enables ControlNet via `ZImageControlPipeline`) or `diffusers` fallback (`ZImagePipeline`)
2. **ControlNet** — Weights loaded via `load_state_dict(strict=False)` into the transformer; dynamically scans `models/Personalized_Model/` for available `.safetensors` files
3. **LoRA** — Applied via diffusers' `load_lora_weights()` + `fuse_lora()`
4. **Cache-DiT** — Optional inference acceleration using `DBCacheConfig` with residual diff thresholding
5. **Generation** — Handles text-to-image, ControlNet-guided generation, and inpainting in a single `generate()` method with automatic fallback for simpler pipelines

### Worker (worker.py)
Polling loop that: fetches jobs from a remote queue server → lazy-initializes `ModelManager` → generates images → uploads base64 results. Supports auto-offload to free VRAM between jobs.

### Key Model Details
- **Weight dtype**: `torch.bfloat16` (use `torch.float16` for V100/RTX 2080 Ti)
- **Z-Image-Turbo**: 8-step distilled model, `guidance_scale=0.0`, no negative prompting
- **Z-Image Base**: Full model, 28-50 steps, `guidance_scale=3.0-5.0`, supports negative prompts and CFG
- **ControlNet control_context_scale**: Optimal range 0.65-1.00 (default 0.75)
- **Resolution presets**: 1MP (up to 1536x640) and 2MP (up to 1920x1088)
- **VRAM**: ~20-24GB with `model_cpu_offload`, ~42-48GB full load; minimum 16GB VRAM + 32GB RAM

### Hardcoded Paths
`app_engine.py` uses hardcoded `BASE_DIR = r"E:\Vellumium\zimageturbo"`. Update this if the project moves.

## ControlNet Model Variants

| File | Use Case | Steps |
|------|----------|-------|
| `*-Union-2.1-2601-8steps.safetensors` | Multi-condition control (Canny, HED, Depth, Pose, MLSD) | 8 |
| `*-Tile-2.1-2601-8steps.safetensors` | Super resolution (up to 2048x2048) | 8 |
| `*-lite-2601-8steps.safetensors` | Lighter control, fewer layers, lower VRAM | 8 |

The 2601 versions fix artifacts and mask leakage from earlier iterations.

## Conventions

- Windows-first development (batch files, backslash paths, MSYS/Git Bash compatible)
- No test suite currently exists
- The `cache_dit/` directory is vendored third-party code — avoid modifying it
- Model weights are large binary files excluded via `.gitignore`
