import os
import sys
import gc
import psutil
import torch
import time
import threading
import glob
from contextlib import asynccontextmanager

# Fix Windows cp949 encoding for emoji/unicode output
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, List, Dict, Tuple
import gradio as gr
from PIL import Image
import numpy as np
from omegaconf import OmegaConf

# Add local library path for cache-dit
sys.path.append(os.path.dirname(__file__))  # cache_dit folder is in same directory

try:
    import cache_dit
    from cache_dit import DBCacheConfig, TaylorSeerCalibratorConfig
except ImportError as e:
    print(f"Warning: cache_dit not available: {e}")
    cache_dit = None

from diffusers import FlowMatchEulerDiscreteScheduler, AutoencoderKL, ZImageTransformer2DModel
from diffusers import ZImagePipeline
from diffusers import FlowMatchLCMScheduler
from transformers import AutoTokenizer
from safetensors.torch import load_file as load_safetensors


def _convert_z_image_state_dict(state_dict):
    """Convert original Z-Image checkpoint keys to diffusers format.
    Handles fused QKV → separate Q/K/V and key renaming.
    (Mirrors convert_z_image_transformer_checkpoint_to_diffusers in diffusers.)"""
    RENAME_DICT = {
        "final_layer.": "all_final_layer.2-1.",
        "x_embedder.": "all_x_embedder.2-1.",
        ".attention.out.bias": ".attention.to_out.0.bias",
        ".attention.k_norm.weight": ".attention.norm_k.weight",
        ".attention.q_norm.weight": ".attention.norm_q.weight",
        ".attention.out.weight": ".attention.to_out.0.weight",
    }

    # Step 1: Rename keys
    for old_key in list(state_dict.keys()):
        new_key = old_key
        for src, dst in RENAME_DICT.items():
            new_key = new_key.replace(src, dst)
        if new_key != old_key:
            state_dict[new_key] = state_dict.pop(old_key)

    # Step 2: Split fused QKV → separate Q/K/V
    for key in list(state_dict.keys()):
        if ".attention.qkv.weight" in key:
            fused = state_dict.pop(key)
            q, k, v = torch.chunk(fused, 3, dim=0)
            state_dict[key.replace(".attention.qkv.weight", ".attention.to_q.weight")] = q
            state_dict[key.replace(".attention.qkv.weight", ".attention.to_k.weight")] = k
            state_dict[key.replace(".attention.qkv.weight", ".attention.to_v.weight")] = v

    return state_dict

# --- GPU Optimization for RTX 5090 ---
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
torch.set_float32_matmul_precision('high')

# --- 100% LOCAL Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_ROOT = os.path.join(BASE_DIR, "models")
LORA_ROOT = os.path.join(MODELS_ROOT, "LoRA")
CONTROLNET_ROOT = os.path.join(MODELS_ROOT, "Personalized_Model")
Z_IMAGE_ROOT = os.path.join(MODELS_ROOT, "z_image_turbo")
TE_ROOT = os.path.join(MODELS_ROOT, "text_encoders")
VAE_ROOT = os.path.join(MODELS_ROOT, "vae")
TOKENIZER_ROOT = os.path.join(MODELS_ROOT, "tokenizer")
CONFIG_PATH = os.path.join(BASE_DIR, "z_image_control_2.1.yaml")

# --- Transformer Model Options (동적 스캔) ---
def scan_transformer_models() -> Dict[str, str]:
    """Scan available transformer model files.
    Returns dict of display_name -> path_or_subfolder."""
    models = {}
    # Standard diffusers subfolder (bf16)
    std_path = os.path.join(Z_IMAGE_ROOT, "transformer", "diffusion_pytorch_model.safetensors")
    if os.path.exists(std_path):
        size_gb = os.path.getsize(std_path) / (1024**3)
        models[f"BF16 ({size_gb:.1f}GB)"] = "subfolder"
    # Single-file .safetensors in z_image_turbo/
    for f in glob.glob(os.path.join(Z_IMAGE_ROOT, "*.safetensors")):
        fname = os.path.basename(f)
        size_gb = os.path.getsize(f) / (1024**3)
        models[f"{fname} ({size_gb:.1f}GB)"] = f
    return models

# --- ControlNet Model Options (동적 스캔) ---
def get_controlnet_models() -> Dict[str, str]:
    """Dynamically scan ControlNet models from Personalized_Model folder."""
    models = {"None": None}
    if os.path.exists(CONTROLNET_ROOT):
        for f in glob.glob(os.path.join(CONTROLNET_ROOT, "*.safetensors")):
            filename = os.path.basename(f)
            # Create friendly name from filename
            name = filename.replace(".safetensors", "").replace("Z-Image-Turbo-Fun-Controlnet-", "")
            models[name] = filename
    return models

# Lite model config (fewer control layers)
LITE_CONFIG = {
    "control_layers_places": [0, 14, 28],
    "control_refiner_layers_places": [0, 1],
    "add_control_noise_refiner": True,
    "add_control_noise_refiner_correctly": True,
    "control_in_dim": 33
}

# --- Resolution Presets ---
RESOLUTION_1MP = {
    "1:1 (Square)": (1024, 1024),
    "3:2 (Landscape)": (832, 1216),
    "2:3 (Portrait)": (1216, 832),
    "4:3 (Classic)": (896, 1152),
    "16:9 (Widescreen)": (768, 1344),
    "21:9 (Ultrawide)": (640, 1536),
}

RESOLUTION_2MP = {
    "1:1 (Square)": (1408, 1408),
    "3:2 (Landscape)": (1152, 1728),
    "4:3 (Classic)": (1216, 1664),
    "16:9 (Widescreen)": (1088, 1920),
}

# --- ControlNet Types ---
CONTROL_TYPES = ["None", "Canny", "HED", "Depth", "Pose", "MLSD"]

# --- LoRA Scanner ---
def scan_lora_names() -> List[str]:
    """Scan available LoRA files, return sorted name list (no 'None')."""
    loras = []
    if os.path.exists(LORA_ROOT):
        for f in glob.glob(os.path.join(LORA_ROOT, "*.safetensors")):
            name = os.path.splitext(os.path.basename(f))[0]
            loras.append(name)
    return sorted(loras)

def scan_controlnets() -> List[str]:
    """Scan available ControlNet files."""
    models = get_controlnet_models()
    return list(models.keys())

# --- Image Preprocessing ---
def preprocess_image(image: Image.Image, target_size: Tuple[int, int]) -> torch.Tensor:
    """Preprocess PIL image to tensor [B, C, H, W] normalized to [0, 1]."""
    if image is None:
        return None
    image = image.convert("RGB").resize((target_size[1], target_size[0]), Image.LANCZOS)
    img_array = np.array(image).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
    return img_tensor

def preprocess_mask(mask: Image.Image, target_size: Tuple[int, int]) -> torch.Tensor:
    """Preprocess mask image to tensor [B, 1, H, W]."""
    if mask is None:
        return None
    mask = mask.convert("L").resize((target_size[1], target_size[0]), Image.LANCZOS)
    mask_array = np.array(mask).astype(np.float32)
    mask_tensor = torch.from_numpy(mask_array).unsqueeze(0).unsqueeze(0)
    return mask_tensor

# --- ModelManager ---
class ModelManager:
    def __init__(self):
        self.pipe = None
        self.current_model_type = None
        self.lock = threading.Lock()
        self.controlnet_loaded = None
        self.loaded_loras = set()  # Track loaded adapter names
        self.transformer = None
        self.vae = None
        self.text_encoder = None
        self.tokenizer = None
        self.scheduler = None
        self.config = None
        self._transformer_choice = None
        self._control_module = None  # Standalone ControlNet module (hook-based)

    def get_status(self):
        ram_percent = psutil.virtual_memory().percent
        gpu_mem = "N/A"
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            gpu_mem = f"{allocated:.2f}/{total:.1f} GB"
        
        available_cn = scan_controlnets()
        
        return {
            "loaded_model": self.current_model_type,
            "controlnet": self.controlnet_loaded if self.controlnet_loaded else "None",
            "available_controlnets": len(available_cn) - 1,  # Exclude "None"
            "loaded_loras": list(self.loaded_loras) if self.loaded_loras else "None",
            "ram_usage": f"{ram_percent}%",
            "gpu_memory": gpu_mem
        }

    def set_scheduler(self, scheduler_type: str):
        """Switch scheduler at runtime without reloading model."""
        if not self.pipe:
            return {"status": "error", "message": "No model loaded"}

        scheduler_dir = os.path.join(Z_IMAGE_ROOT, "scheduler")
        if scheduler_type == "Euler":
            self.pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(scheduler_dir)
        elif scheduler_type == "Euler + Beta":
            self.pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
                scheduler_dir, use_beta_sigmas=True
            )
        elif scheduler_type == "LCM":
            self.pipe.scheduler = FlowMatchLCMScheduler(
                num_train_timesteps=1000, shift=3.0
            )
        elif scheduler_type == "LCM + Beta":
            self.pipe.scheduler = FlowMatchLCMScheduler(
                num_train_timesteps=1000, shift=3.0, use_beta_sigmas=True
            )
        self.scheduler = self.pipe.scheduler
        print(f"    Scheduler changed to: {scheduler_type}")
        return {"status": "success", "scheduler": scheduler_type}

    def unload_model(self):
        if self.pipe:
            print(f"🧹 Unloading {self.current_model_type}...")
            if cache_dit:
                try:
                    cache_dit.disable_cache(self.pipe)
                except:
                    pass
            del self.pipe
            self.pipe = None
        
        # Clean up ControlNet hooks/module
        if self._control_module is not None:
            from zimage_control import remove_control_hooks
            if self.transformer is not None:
                try:
                    remove_control_hooks(self.transformer)
                except Exception:
                    pass
            del self._control_module
            self._control_module = None

        for attr in ['transformer', 'vae', 'text_encoder', 'tokenizer', 'scheduler']:
            if hasattr(self, attr) and getattr(self, attr) is not None:
                delattr(self, attr)
                setattr(self, attr, None)

        self.current_model_type = None
        self.controlnet_loaded = None
        self.loaded_loras = set()
        gc.collect()
        torch.cuda.empty_cache()
        print("✅ Model unloaded.")

    def load_controlnet(self, controlnet_name: str):
        """Load ControlNet as standalone module with forward hooks.
        Base transformer stays untouched on GPU — no pipeline replacement."""
        if not self.pipe:
            return {"status": "error", "message": "No base model loaded. Load a model first."}

        models = get_controlnet_models()

        # --- Unload ControlNet ---
        if controlnet_name == "None" or controlnet_name not in models:
            if self.controlnet_loaded:
                print("    Removing ControlNet hooks...")
                self._remove_controlnet()
                self.controlnet_loaded = None
                print("    ControlNet disabled")
            return {"status": "success", "message": "ControlNet disabled"}

        filename = models[controlnet_name]
        controlnet_path = os.path.join(CONTROLNET_ROOT, filename)

        if not os.path.exists(controlnet_path):
            return {"status": "error", "message": f"File not found: {filename}\nmodels/Personalized_Model/ folder needed"}

        try:
            from zimage_control import load_controlnet_module, install_control_hooks, remove_control_hooks

            print(f"    Loading ControlNet module: {controlnet_name}...")

            # Remove existing hooks if switching ControlNet
            if self._control_module is not None:
                remove_control_hooks(self.transformer)
                del self._control_module
                self._control_module = None
                gc.collect()
                torch.cuda.empty_cache()

            # Disable Cache-DiT (skipped blocks miss control signals)
            if cache_dit:
                try:
                    cache_dit.disable_cache(self.pipe)
                    print("    Cache-DiT disabled (incompatible with ControlNet hooks)")
                except Exception:
                    pass

            # Load standalone control module onto GPU
            weight_dtype = torch.bfloat16
            self._control_module = load_controlnet_module(controlnet_path, device='cuda', dtype=weight_dtype)

            # Install hooks on base transformer (non-destructive)
            install_control_hooks(self.transformer, self._control_module)

            self.controlnet_loaded = controlnet_name
            print(f"    ControlNet ready: {controlnet_name} (hook-based, transformer untouched)")
            return {"status": "success", "message": f"ControlNet loaded: {controlnet_name}"}
        except Exception as e:
            print(f"    ControlNet load failed: {e}")
            import traceback
            traceback.print_exc()
            return {"status": "error", "message": str(e)}

    def _remove_controlnet(self):
        """Remove ControlNet hooks and module. Base transformer stays intact."""
        from zimage_control import remove_control_hooks

        if self._control_module is not None:
            remove_control_hooks(self.transformer)
            del self._control_module
            self._control_module = None
            gc.collect()
            torch.cuda.empty_cache()

        # Re-enable Cache-DiT
        if cache_dit:
            try:
                cache_dit.enable_cache(
                    self.pipe,
                    cache_config=DBCacheConfig(
                        Fn_compute_blocks=2,
                        Bn_compute_blocks=1,
                        residual_diff_threshold=0.12
                    )
                )
                print("    Cache-DiT re-enabled")
            except Exception as e:
                print(f"    Cache-DiT failed: {e}")

    def _encode_control_latent(self, control_image, height, width,
                               inpaint_image=None, mask_image=None):
        """Encode control/inpaint images to latent space for ControlNet hooks.

        For Union ControlNet (additional_in_dim>0):
            Returns [B, 33, H_lat, W_lat] = [ctrl_latent(16), mask(1), inpaint_latent(16)]
        For simple ControlNet (additional_in_dim=0):
            Returns [B, 16, H_lat, W_lat] = ctrl_latent only
        """
        import torch.nn.functional as F
        weight_dtype = torch.bfloat16
        device = "cuda"

        # Encode control image via VAE
        ctrl_tensor = preprocess_image(control_image, (height, width))  # [B,3,H,W] in [0,1]
        ctrl_tensor = (ctrl_tensor * 2.0 - 1.0).to(dtype=weight_dtype, device=device)  # [-1,1]

        with torch.no_grad():
            ctrl_latent = self.vae.encode(ctrl_tensor).latent_dist.mode()
            ctrl_latent = (ctrl_latent - self.vae.config.shift_factor) * self.vae.config.scaling_factor

        # For models with additional_in_dim > 0 (Union): [ctrl(16), mask(1), inpaint(16)] = 33ch
        if self._control_module and self._control_module.additional_in_dim > 0:
            # Prepare inpaint image
            if inpaint_image is not None:
                inp_tensor = preprocess_image(inpaint_image, (height, width))  # [0,1]
                # Apply mask: white=regenerate -> neutral, black=keep -> original
                if mask_image is not None:
                    mask_t = preprocess_mask(mask_image, (height, width)) / 255.0  # [0,1]
                    mask_inv = 1.0 - mask_t  # 0=regenerate, 1=keep
                    inp_tensor = (inp_tensor - 0.5) * mask_inv + 0.5
            else:
                # Dummy neutral gray inpaint image
                inp_tensor = torch.full([1, 3, height, width], 0.5)

            inp_tensor = (inp_tensor * 2.0 - 1.0).to(dtype=weight_dtype, device=device)
            with torch.no_grad():
                inp_latent = self.vae.encode(inp_tensor).latent_dist.mode()
                inp_latent = (inp_latent - self.vae.config.shift_factor) * self.vae.config.scaling_factor

            # Mask channel (downsampled to latent resolution)
            if mask_image is not None:
                mask_t = preprocess_mask(mask_image, (height, width)) / 255.0
                mask_inv = 1.0 - mask_t
                mask_latent = F.interpolate(
                    mask_inv, size=(ctrl_latent.shape[2], ctrl_latent.shape[3]), mode='nearest'
                )
            else:
                mask_latent = torch.zeros([1, 1, ctrl_latent.shape[2], ctrl_latent.shape[3]])

            mask_latent = mask_latent.to(dtype=weight_dtype, device=device)

            # [16ch ctrl + 1ch mask + 16ch inpaint] = 33ch
            ctrl_latent = torch.cat([ctrl_latent, mask_latent, inp_latent], dim=1)

        return ctrl_latent

    def apply_multi_lora(self, lora_scales: Dict[str, float]):
        """Apply multiple LoRAs with individual scales. Scale 0.0 = disabled."""
        if not self.pipe:
            return

        # LoRA only works with diffusers pipelines that support set_adapters
        if not hasattr(self.pipe, 'set_adapters'):
            return

        # Filter to non-zero scales only
        active = {n: s for n, s in lora_scales.items() if s != 0.0}

        if not active:
            # Disable all LoRA adapters (only if any were loaded)
            if self.loaded_loras:
                try:
                    self.pipe.set_adapters([], adapter_weights=[])
                except Exception:
                    pass
            return

        # Lazy-load any LoRAs not yet loaded as adapters
        for name in active:
            if name not in self.loaded_loras:
                lora_path = os.path.join(LORA_ROOT, f"{name}.safetensors")
                if not os.path.exists(lora_path):
                    print(f"    ⚠️ LoRA file not found: {lora_path}")
                    continue
                try:
                    print(f"    ⏳ Loading LoRA adapter: {name}...")
                    self.pipe.load_lora_weights(lora_path, adapter_name=name)
                    self.loaded_loras.add(name)
                    print(f"    ✅ LoRA adapter loaded: {name}")
                except Exception as e:
                    print(f"    ❌ LoRA load failed ({name}): {e}")

        # Activate adapters with weights (only successfully loaded ones)
        names = [n for n in active if n in self.loaded_loras]
        weights = [active[n] for n in names]
        if names:
            self.pipe.set_adapters(names, adapter_weights=weights)
            print(f"    🎭 Active LoRAs: {dict(zip(names, weights))}")

    def load_model(self, model_type="z-image-turbo", transformer_choice=None, progress=None):
        with self.lock:
            if self.current_model_type == model_type and self.pipe is not None:
                return {"status": "Already loaded", "model": model_type}

            self.unload_model()
            print(f"🚀 Loading {model_type}...")

            try:
                # Load ControlNet config (used later by load_controlnet)
                if os.path.exists(CONFIG_PATH):
                    self.config = OmegaConf.load(CONFIG_PATH)
                    print(f"    📄 Config loaded from {CONFIG_PATH}")

                weight_dtype = torch.bfloat16

                def _progress(frac, desc=""):
                    print(f"    ⏳ {desc}...")
                    if progress:
                        progress(frac, desc=desc)

                # Save transformer choice for ControlNet reloading
                self._transformer_choice = transformer_choice

                # --- Determine which transformer weight file to use ---
                tf_models = scan_transformer_models()
                if transformer_choice and transformer_choice in tf_models:
                    tf_path_val = tf_models[transformer_choice]
                else:
                    tf_path_val = "subfolder"  # default to BF16

                # --- 1. Load Transformer (with QKV key conversion) ---
                _progress(0.1, "Transformer 로딩")
                transformer_config_dir = os.path.join(Z_IMAGE_ROOT, "transformer")

                if tf_path_val == "subfolder":
                    weight_path = os.path.join(transformer_config_dir, "diffusion_pytorch_model.safetensors")
                else:
                    weight_path = tf_path_val  # single-file path (e.g. FP8)

                print(f"    📦 Loading weights: {os.path.basename(weight_path)}")
                raw_state_dict = load_safetensors(weight_path)

                # Check if conversion needed (fused QKV present?)
                needs_conversion = any(".attention.qkv.weight" in k for k in raw_state_dict.keys())
                if needs_conversion:
                    print(f"    🔄 Converting fused QKV → separate Q/K/V...")
                    raw_state_dict = _convert_z_image_state_dict(raw_state_dict)

                tf_config = ZImageTransformer2DModel.load_config(transformer_config_dir)
                self.transformer = ZImageTransformer2DModel.from_config(tf_config)
                missing, unexpected = self.transformer.load_state_dict(raw_state_dict, strict=False)
                del raw_state_dict
                gc.collect()

                if missing:
                    print(f"    ⚠️ Missing keys: {len(missing)} (first 5: {missing[:5]})")
                if unexpected:
                    print(f"    ⚠️ Unexpected keys: {len(unexpected)} (first 5: {unexpected[:5]})")
                if not missing and not unexpected:
                    print(f"    ✅ All transformer weights loaded perfectly")

                self.transformer = self.transformer.to(weight_dtype)

                # --- 2. Load VAE ---
                _progress(0.3, "VAE 로딩")
                self.vae = AutoencoderKL.from_pretrained(VAE_ROOT, torch_dtype=weight_dtype)

                # --- 3. Load Text Encoder (Qwen3ForCausalLM per reference) ---
                _progress(0.5, "Text Encoder 로딩")
                from transformers import Qwen3ForCausalLM
                self.text_encoder = Qwen3ForCausalLM.from_pretrained(
                    TE_ROOT, torch_dtype=weight_dtype, low_cpu_mem_usage=True
                )
                print(f"    ✅ Text encoder loaded as Qwen3ForCausalLM")

                # --- 4. Load Tokenizer ---
                _progress(0.65, "Tokenizer 로딩")
                self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ROOT)

                # --- 5. Load Scheduler ---
                _progress(0.7, "Scheduler 로딩")
                scheduler_dir = os.path.join(Z_IMAGE_ROOT, "scheduler")
                self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(scheduler_dir)

                # --- 6. Assemble Pipeline ---
                _progress(0.75, "Pipeline 조립")
                self.pipe = ZImagePipeline(
                    transformer=self.transformer,
                    vae=self.vae,
                    text_encoder=self.text_encoder,
                    tokenizer=self.tokenizer,
                    scheduler=self.scheduler,
                )

                _progress(0.8, "GPU 전송")
                self.pipe.to("cuda")

                # Apply Cache-DiT if available (aggressive settings for RTX 5090)
                if cache_dit:
                    _progress(0.9, "Cache-DiT 적용")
                    try:
                        cache_dit.enable_cache(
                            self.pipe,
                            cache_config=DBCacheConfig(
                                Fn_compute_blocks=2,
                                Bn_compute_blocks=1,
                                residual_diff_threshold=0.12
                            )
                        )
                    except Exception as e:
                        print(f"    ⚠️ Cache-DiT failed: {e}")

                self.current_model_type = model_type
                _progress(1.0, "로딩 완료")
                print(f"✅ {model_type} loaded (Base pipeline — text-to-image)")
                return {"status": "Loaded successfully", "model": model_type}

            except Exception as e:
                print(f"❌ Critical Error: {e}")
                import traceback
                traceback.print_exc()
                self.unload_model()
                raise HTTPException(status_code=500, detail=str(e))

    def generate(
        self,
        prompt: str,
        image: Image.Image = None,
        steps: int = 8,
        height: int = 1024,
        width: int = 1024,
        control_image: Image.Image = None,
        control_type: str = "None",
        control_scale: float = 0.75,
        mask_image: Image.Image = None,
        seed: int = -1,
        lora_scales: Dict[str, float] = None,
    ):
        if not self.pipe:
            raise HTTPException(status_code=400, detail="No model loaded. Please load a model first.")

        # Apply multi-LoRA
        if lora_scales:
            self.apply_multi_lora(lora_scales)

        # Set seed
        generator = None
        if seed >= 0:
            generator = torch.Generator(device="cuda").manual_seed(seed)
        else:
            seed = torch.randint(0, 2**32, (1,)).item()
            generator = torch.Generator(device="cuda").manual_seed(seed)

        with torch.no_grad():
            # Activate ControlNet hooks if loaded and control image provided
            if self.controlnet_loaded and self._control_module is not None:
                if control_image is not None:
                    ctrl_latent = self._encode_control_latent(
                        control_image, height, width,
                        inpaint_image=image, mask_image=mask_image
                    )
                    self._control_module._control_latent = ctrl_latent
                    self._control_module._image_seq_len = (
                        (ctrl_latent.shape[2] // 2) * (ctrl_latent.shape[3] // 2)
                    )
                    self._control_module._control_scale = control_scale
                    self._control_module._active = True
                    print(f"    ControlNet active: scale={control_scale}, latent={ctrl_latent.shape}")
                else:
                    self._control_module._active = False

            # Base kwargs (Turbo: no CFG, no negative prompt)
            gen_kwargs = {
                "prompt": prompt,
                "height": height,
                "width": width,
                "num_inference_steps": steps,
                "guidance_scale": 0.0,
                "generator": generator,
            }

            # --- Inpainting: noise-level masking via callback ---
            # ZImagePipeline is text-to-image only. For inpainting we must
            # blend the denoised latent with the noised original at each step
            # so non-masked areas converge to the original image.
            if image is not None and mask_image is not None:
                import torch.nn.functional as F
                weight_dtype = torch.bfloat16
                device = "cuda"

                # Encode original image to latent
                img_tensor = preprocess_image(image, (height, width))
                img_tensor = (img_tensor * 2.0 - 1.0).to(dtype=weight_dtype, device=device)
                original_latent = self.vae.encode(img_tensor).latent_dist.mode()
                original_latent = (
                    (original_latent - self.vae.config.shift_factor)
                    * self.vae.config.scaling_factor
                ).float()

                # Mask in latent space (white=1=regenerate, black=0=keep)
                mask_t = preprocess_mask(mask_image, (height, width)) / 255.0
                lat_h, lat_w = original_latent.shape[2], original_latent.shape[3]
                mask_latent = F.interpolate(mask_t, size=(lat_h, lat_w), mode='nearest')
                mask_latent = mask_latent.to(device=device, dtype=torch.float32)

                # Generate initial noise (same shape, same generator for reproducibility)
                initial_noise = torch.randn(
                    original_latent.shape, generator=generator, device=device, dtype=torch.float32
                )
                gen_kwargs["latents"] = initial_noise

                # Callback: blend at each denoising step
                def _inpaint_callback(pipe, step_index, timestep, callback_kwargs):
                    latents = callback_kwargs["latents"]
                    # sigma after this step (0.0 at final step = clean image)
                    next_idx = step_index + 1
                    if next_idx < len(pipe.scheduler.sigmas):
                        sigma = pipe.scheduler.sigmas[next_idx].item()
                    else:
                        sigma = 0.0
                    # Original at current noise level: x_t = sigma*noise + (1-sigma)*x_0
                    noised_orig = (
                        sigma * initial_noise + (1.0 - sigma) * original_latent
                    )
                    # mask=1 → use denoised (regenerate), mask=0 → use original (keep)
                    callback_kwargs["latents"] = (
                        mask_latent * latents + (1.0 - mask_latent) * noised_orig
                    )
                    return callback_kwargs

                gen_kwargs["callback_on_step_end"] = _inpaint_callback
                gen_kwargs["callback_on_step_end_tensor_inputs"] = ["latents"]
                print(f"    Inpainting active: mask→latent {mask_latent.shape}, "
                      f"original_latent {original_latent.shape}")

            try:
                result = self.pipe(**gen_kwargs)
                return result.images[0]
            finally:
                # Deactivate ControlNet after generation
                if self._control_module is not None:
                    self._control_module._active = False
                    self._control_module._control_latent = None
                    self._control_module._embedded_control = None
                    self._control_module._temp_data = None

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

def update_resolution(mp_preset: str, aspect_ratio: str):
    """Update resolution based on MP preset and aspect ratio."""
    presets = RESOLUTION_1MP if mp_preset == "1.0 MP (Standard)" else RESOLUTION_2MP
    if aspect_ratio in presets:
        h, w = presets[aspect_ratio]
        return h, w
    return 1024, 1024

def gradio_interface():
    lora_names = scan_lora_names()
    available_cns = scan_controlnets()
    tf_models = scan_transformer_models()
    tf_choices = list(tf_models.keys())

    with gr.Blocks(title="Z-Image Turbo Engine") as demo:
        gr.Markdown("# ⚡ Z-Image Turbo Engine")
        gr.Markdown("8-step 고속 생성 | ControlNet Union/Tile, LoRA 지원 | CFG 불필요")

        with gr.Row():
            status_box = gr.JSON(label="System Status", value=engine.get_status)
            refresh_btn = gr.Button("🔄 Refresh", size="sm")

        with gr.Row():
            transformer_dropdown = gr.Dropdown(
                choices=tf_choices,
                value=tf_choices[0] if tf_choices else None,
                label="🧠 Transformer Model"
            )
            load_btn = gr.Button("📥 Load Model", variant="primary")
            unload_btn = gr.Button("🗑️ Unload", variant="secondary")

        with gr.Row():
            controlnet_dropdown = gr.Dropdown(
                choices=available_cns,
                value="None",
                label="🎛️ ControlNet Model"
            )
            load_cn_btn = gr.Button("⚡ Load ControlNet", variant="secondary")
        
        with gr.Tabs():
            with gr.TabItem("🎨 Generation"):
                with gr.Row():
                    with gr.Column(scale=1):
                        # Resolution Settings
                        gr.Markdown("### 📐 Resolution")
                        mp_preset = gr.Radio(
                            choices=["1.0 MP (Standard)", "2.0 MP (High)"],
                            value="1.0 MP (Standard)",
                            label="Resolution Preset"
                        )
                        aspect_ratio = gr.Dropdown(
                            choices=list(RESOLUTION_1MP.keys()),
                            value="1:1 (Square)",
                            label="Aspect Ratio"
                        )
                        with gr.Row():
                            height_input = gr.Number(value=1024, label="Height", precision=0)
                            width_input = gr.Number(value=1024, label="Width", precision=0)
                        
                        # Prompt
                        gr.Markdown("### ✏️ Prompt")
                        prompt_input = gr.Textbox(
                            label="Prompt", 
                            lines=4, 
                            placeholder="상세한 프롬프트를 입력하세요. 자세할수록 결과가 안정적입니다."
                        )
                        
                        # Generation Settings
                        gr.Markdown("### ⚙️ Settings")
                        scheduler_dropdown = gr.Dropdown(
                            choices=["Euler", "Euler + Beta", "LCM", "LCM + Beta"],
                            value="Euler",
                            label="Scheduler (LCM+Beta 권장 시도)"
                        )
                        steps_slider = gr.Slider(minimum=4, maximum=12, value=8, step=1, label="Steps (8 권장)")
                        seed_input = gr.Number(value=-1, label="Seed (-1 = random)", precision=0)
                        
                    with gr.Column(scale=1):
                        # Multi-LoRA Sliders
                        gr.Markdown("### 🎭 LoRA (0 = OFF, +/- 방향으로 효과 조절)")
                        lora_sliders = []
                        for lora_name in lora_names:
                            s = gr.Slider(
                                minimum=-2.0, maximum=2.0, value=0.0, step=0.05,
                                label=lora_name
                            )
                            lora_sliders.append(s)

                        # ControlNet Settings
                        gr.Markdown("### 🎛️ ControlNet Input")
                        control_type = gr.Dropdown(
                            choices=CONTROL_TYPES,
                            value="None",
                            label="Control Type (Canny, Depth, Pose 등)"
                        )
                        control_image = gr.Image(label="Control Image", type="pil")
                        control_scale = gr.Slider(
                            minimum=0.0, maximum=1.0, value=0.75, step=0.05, 
                            label="Control Scale (0.65-1.00 권장)"
                        )
                        
                        # Inpainting
                        gr.Markdown("### 🖌️ Inpainting")
                        input_image = gr.Image(label="Input Image", type="pil")
                        mask_image = gr.Image(label="Mask (White = regenerate)", type="pil")
                
                with gr.Row():
                    gen_btn = gr.Button("🚀 Generate", variant="primary", size="lg")
                
                with gr.Row():
                    output_image = gr.Image(label="Result", type="pil")

        # Event handlers
        refresh_btn.click(engine.get_status, outputs=status_box)
        
        def do_load(tf_choice, progress=gr.Progress(track_tqdm=True)):
            engine.load_model("z-image-turbo", transformer_choice=tf_choice, progress=progress)

        load_btn.click(fn=do_load, inputs=transformer_dropdown).then(engine.get_status, outputs=status_box)
        
        def do_load_cn(cn):
            engine.load_controlnet(cn)

        load_cn_btn.click(fn=do_load_cn, inputs=controlnet_dropdown).then(
            engine.get_status, outputs=status_box
        )
        
        unload_btn.click(engine.unload_model).then(engine.get_status, outputs=status_box)
        
        # Resolution update handlers
        def on_mp_change(mp):
            presets = RESOLUTION_1MP if mp == "1.0 MP (Standard)" else RESOLUTION_2MP
            return gr.Dropdown(choices=list(presets.keys()), value=list(presets.keys())[0])
        
        def on_aspect_change(mp, aspect):
            h, w = update_resolution(mp, aspect)
            return h, w
        
        mp_preset.change(on_mp_change, inputs=mp_preset, outputs=aspect_ratio)
        aspect_ratio.change(on_aspect_change, inputs=[mp_preset, aspect_ratio], outputs=[height_input, width_input])
        
        # Generation wrapper: collect LoRA sliders into dict
        def do_generate(prompt, image, steps, height, width,
                        control_img, ctrl_type, ctrl_scale,
                        mask_img, seed, sched_type, *lora_values):
            engine.set_scheduler(sched_type)
            scales = {n: v for n, v in zip(lora_names, lora_values)}
            return engine.generate(
                prompt=prompt, image=image, steps=int(steps),
                height=int(height), width=int(width),
                control_image=control_img, control_type=ctrl_type,
                control_scale=ctrl_scale, mask_image=mask_img,
                seed=int(seed), lora_scales=scales,
            )

        gen_btn.click(
            fn=do_generate,
            inputs=[
                prompt_input, input_image, steps_slider,
                height_input, width_input, control_image, control_type,
                control_scale, mask_image, seed_input,
                scheduler_dropdown,
            ] + lora_sliders,
            outputs=output_image,
        )
    
    return demo

app = gr.mount_gradio_app(app, gradio_interface(), path="/")

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("  🚀 Z-Image Turbo Engine Starting...")
    print("  📌 Access URL: http://localhost:7860")
    print("  ⚡ 8-step distilled | CFG=0.0 | No negative prompt")
    print("  🎛️ ControlNet Union/Tile, LoRA, Resolution Presets")
    print("  📁 ControlNet 위치: models/Personalized_Model/")
    print("="*60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=7860)