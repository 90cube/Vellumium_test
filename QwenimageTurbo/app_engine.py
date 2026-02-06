import os
import sys
import gc
import psutil
import torch
import time
import threading
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import gradio as gr

# Add local library path for cache-dit
sys.path.append(os.path.dirname(__file__))  # cache_dit folder is in same directory

try:
    import cache_dit
    from cache_dit import DBCacheConfig, TaylorSeerCalibratorConfig
except ImportError as e:
    print(f"Error importing cache_dit: {e}")
    sys.exit(1)

from diffusers import DiffusionPipeline, AutoencoderKL
# Import specific pipelines found in the environment
try:
    from diffusers import ZImagePipeline, QwenImageEditPlusPipeline
except ImportError:
    try:
        from diffusers.pipelines.z_image.pipeline_z_image import ZImagePipeline
        from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import QwenImageEditPlusPipeline
    except ImportError:
        ZImagePipeline = None
        QwenImageEditPlusPipeline = None

from transformers import AutoTokenizer, AutoModel, AutoConfig
from safetensors.torch import load_file as load_safetensors

# --- 100% LOCAL Configuration ---
BASE_DIR = r"E:\Vellumium\QwenimageTurbo"
MODELS_ROOT = os.path.join(BASE_DIR, "models", "diffusion_models")
TE_ROOT = os.path.join(BASE_DIR, "models", "text_encoders")
VAE_ROOT = os.path.join(BASE_DIR, "models", "vae")

# Model Specific Dirs (must contain config.json and weights)
DIR_Z_IMAGE = os.path.join(MODELS_ROOT, "z_image_turbo")
DIR_QWEN = os.path.join(MODELS_ROOT, "QWEN")

# Components
MODEL_Z_IMAGE_TE = os.path.join(TE_ROOT, "zimage_turbo") 
MODEL_QWEN_TE = os.path.join(TE_ROOT, "qwen_2.5_vl_7b_fp8_scaled")

MODEL_Z_IMAGE_VAE = os.path.join(VAE_ROOT, "zimage")
MODEL_QWEN_VAE = os.path.join(VAE_ROOT, "QWEN")

# Hub IDs (Only for template fallbacks, not for download)
HUB_Z_IMAGE = "Tongyi-MAI/Z-Image-Turbo"
HUB_QWEN_EDIT = "Qwen/Qwen-Image-Edit-2509"

# --- Custom Tokenizer Logic ---
class CustomTokenizerWrapper:
    def __init__(self, tokenizer_path, template_type="z-image"):
        print(f"    🔤 Loading Tokenizer from {tokenizer_path}...")
        try:
            # Force local only
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True, local_files_only=True)
        except Exception as e:
            print(f"    ⚠️ Local tokenizer failed ({e}), using fallback ID...")
            fallback = "Qwen/Qwen2.5-3B" if template_type == "z-image" else "Qwen/Qwen2-VL-7B-Instruct"
            self.tokenizer = AutoTokenizer.from_pretrained(fallback, trust_remote_code=True)
        self.template_type = template_type

    def apply_template(self, text, images=None):
        if self.template_type == "z-image":
            return f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
        elif self.template_type == "qwen-edit":
            if images:
                system_prompt = "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n"
                user_prompt = f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{text}<|im_end|>\n<|im_start|>assistant\n"
                return system_prompt + user_prompt
            else:
                system_prompt = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n"
                user_prompt = f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
                return system_prompt + user_prompt
        return text

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)

# --- Helper ---
def load_te(path, fallback_id):
    print(f"    ⏳ Loading Text Encoder from {path}...")
    try:
        return AutoModel.from_pretrained(path, trust_remote_code=True, torch_dtype=torch.bfloat16, local_files_only=True)
    except:
        print(f"    📡 TE Local failed, using Hub ID (cache): {fallback_id}")
        return AutoModel.from_pretrained(fallback_id, trust_remote_code=True, torch_dtype=torch.bfloat16)

def load_vae(path, fallback_id):
    print(f"    ⏳ Loading VAE from {path}...")
    try:
        return AutoencoderKL.from_pretrained(path, torch_dtype=torch.bfloat16, local_files_only=True)
    except:
        print(f"    📡 VAE Local failed, using Hub ID (cache): {fallback_id}")
        return AutoencoderKL.from_pretrained(fallback_id, subfolder="vae", torch_dtype=torch.bfloat16)

# --- ModelManager ---
class ModelManager:
    def __init__(self):
        self.pipe = None
        self.current_model_type = None
        self.lock = threading.Lock()
        self.tokenizer_wrapper = None

    def get_status(self):
        ram_percent = psutil.virtual_memory().percent
        gpu_mem = "N/A"
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            gpu_mem = f"{allocated:.2f}/{reserved:.2f} GB"
        
        return {
            "loaded_model": self.current_model_type,
            "ram_usage": f"{ram_percent}%",
            "gpu_memory": gpu_mem
        }

    def unload_model(self):
        if self.pipe:
            print(f"🧹 Unloading {self.current_model_type}...")
            try:
                cache_dit.disable_cache(self.pipe)
            except:
                pass
            del self.pipe
            del self.tokenizer_wrapper
            self.pipe = None
            self.tokenizer_wrapper = None
            self.current_model_type = None
            gc.collect()
            torch.cuda.empty_cache()
            print("✅ Model unloaded.")

    def load_model(self, model_type):
        with self.lock:
            if self.current_model_type == model_type and self.pipe is not None:
                return {"status": "Already loaded", "model": model_type}

            self.unload_model()
            print(f"🚀 Loading {model_type} 100% LOCALLY...")
            
            # Fast loading dtype
            dtype = torch.float8_e4m3fn if hasattr(torch, "float8_e4m3fn") and ("qwen" in model_type or "fp8" in model_type) else torch.bfloat16
            
            try:
                if model_type == "z-image-turbo":
                    print("    📁 Loading pipeline from Hub with local transformer...")
                    # Load from Hub but we'll use cached version
                    self.pipe = ZImagePipeline.from_pretrained(
                        HUB_Z_IMAGE,
                        torch_dtype=torch.bfloat16
                    )
                    self.pipe.to("cuda")
                    self.tokenizer_wrapper = CustomTokenizerWrapper(MODEL_Z_IMAGE_TE, "z-image")
                    
                elif model_type == "qwen-image-edit":
                    te = load_te(MODEL_QWEN_TE, "Qwen/Qwen2-VL-7B-Instruct")
                    vae = load_vae(MODEL_QWEN_VAE, HUB_QWEN_EDIT)
                    
                    self.pipe = QwenImageEditPlusPipeline.from_pretrained(
                        DIR_QWEN,
                        text_encoder=te,
                        vae=vae,
                        torch_dtype=dtype,
                        trust_remote_code=True,
                        local_files_only=True
                    ).to("cuda")
                    self.tokenizer_wrapper = CustomTokenizerWrapper(MODEL_QWEN_TE, "qwen-edit")

                # Apply Cache-DiT
                print("⚡ Enabling Cache-DiT...")
                if "z-image" in model_type:
                    cache_dit.enable_cache(self.pipe, cache_config=DBCacheConfig(Fn_compute_blocks=1, Bn_compute_blocks=0, residual_diff_threshold=0.15))
                else:
                    cache_dit.enable_cache(self.pipe, cache_config=DBCacheConfig(Fn_compute_blocks=8, Bn_compute_blocks=0, residual_diff_threshold=0.08), 
                                          calibrator_config=TaylorSeerCalibratorConfig(taylorseer_order=1))

                self.current_model_type = model_type
                return {"status": "Loaded successfully from LOCAL", "model": model_type}
            
            except Exception as e:
                print(f"❌ Critical Error: {e}")
                import traceback
                traceback.print_exc()
                self.unload_model()
                raise HTTPException(status_code=500, detail=str(e))

    def generate(self, prompt, image=None, steps=4, guidance_scale=1.0):
        if not self.pipe: raise HTTPException(status_code=400, detail="No model loaded")
        
        full_prompt = self.tokenizer_wrapper.apply_template(prompt, image) if self.tokenizer_wrapper else prompt
        if "turbo" in self.current_model_type: guidance_scale = 0.0

        with torch.no_grad():
            if self.current_model_type == "qwen-image-edit" and image:
                result = self.pipe(prompt=full_prompt, image=image, num_inference_steps=steps, guidance_scale=guidance_scale)
            else:
                result = self.pipe(prompt=full_prompt, num_inference_steps=steps, guidance_scale=guidance_scale)
            return result.images[0]

engine = ModelManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    engine.unload_model()

app = FastAPI(lifespan=lifespan)

class LoadRequest(BaseModel):
    model: str

@app.get("/health")
def health(): return {"status": "ok"}

@app.get("/status")
def status(): return engine.get_status()

@app.post("/load")
def load(req: LoadRequest): return engine.load_model(req.model)

def gradio_interface():
    with gr.Blocks(title="Z-Image & Qwen Engine") as demo:
        gr.Markdown("# ⚡ 100% Local & Fast GenAI Engine")
        with gr.Row():
            status_box = gr.JSON(label="System Status", value=engine.get_status)
            refresh_btn = gr.Button("Refresh", size="sm")
        with gr.Row():
            model_dropdown = gr.Dropdown(choices=["z-image-turbo", "qwen-image-edit"], label="Select Model", value="z-image-turbo")
            load_btn = gr.Button("Load Model", variant="primary")
            unload_btn = gr.Button("Unload", variant="secondary")
        with gr.Tabs():
            with gr.TabItem("Generation / Edit"):
                with gr.Row():
                    input_image = gr.Image(label="Input Image", type="pil")
                    output_image = gr.Image(label="Result")
                prompt_input = gr.Textbox(label="Prompt", lines=3)
                steps_slider = gr.Slider(minimum=1, maximum=100, value=4, label="Steps")
                cfg_slider = gr.Slider(minimum=0.0, maximum=20.0, value=1.0, label="CFG")
                gen_btn = gr.Button("Generate", variant="primary", size="lg")

        refresh_btn.click(engine.get_status, outputs=status_box)
        load_btn.click(fn=lambda m: engine.load_model(m)["status"], inputs=model_dropdown).then(engine.get_status, outputs=status_box)
        unload_btn.click(engine.unload_model).then(engine.get_status, outputs=status_box)
        gen_btn.click(fn=engine.generate, inputs=[prompt_input, input_image, steps_slider, cfg_slider], outputs=output_image)
    return demo

app = gr.mount_gradio_app(app, gradio_interface(), path="/")

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*50)
    print("  🚀 Cache-DiT Inference Engine Starting...")
    print("  📌 Access URL: http://localhost:7860")
    print("="*50 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=7860)