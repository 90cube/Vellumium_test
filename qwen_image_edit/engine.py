"""
Qwen-Image-Edit-2511 FP8 + Lightning LoRA Engine
=================================================
Method A 최적화 로딩: safetensors 직접 로드 → FP8 유지 → layerwise_casting.

성능 (RTX 5090, 3장 이미지 736x1011):
  로딩: ~7s | 추론: ~89s (8 steps, true_cfg=3.0) | VRAM 피크: ~22GB
"""

import gc
import time
import json
import torch
from pathlib import Path
from PIL import Image

# 모델 경로 (Vellumium 프로젝트 기준)
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "Qwen-Image-Edit-2511-FP8"
LORA_PATH = MODEL_DIR / "loras" / "Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors"


def _load_transformer_direct(model_dir: Path):
    """Linear만 meta device → safetensors mmap → assign=True로 FP8 직접 로드.

    from_pretrained 대비 ~3x 빠름 (bf16 변환 skip).
    """
    from diffusers import QwenImageTransformer2DModel
    from safetensors.torch import load_file

    transformer_dir = model_dir / "transformer"
    config = QwenImageTransformer2DModel.load_config(str(transformer_dir))

    # Linear만 meta, 나머지(norm, pos_freqs 등)는 CPU bf16
    original_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    _orig_reset = torch.nn.Linear.reset_parameters
    torch.nn.Linear.reset_parameters = lambda self: None
    _orig_init = torch.nn.Linear.__init__

    def _meta_init(self, in_features, out_features, bias=True, device=None, dtype=None):
        _orig_init(self, in_features, out_features, bias=bias, device="meta", dtype=dtype)

    torch.nn.Linear.__init__ = _meta_init
    try:
        transformer = QwenImageTransformer2DModel.from_config(config)
    finally:
        torch.nn.Linear.__init__ = _orig_init
        torch.nn.Linear.reset_parameters = _orig_reset
        torch.set_default_dtype(original_dtype)

    # safetensors 직접 로드 (단일 파일 또는 멀티 샤드)
    single_path = transformer_dir / "diffusion_pytorch_model.safetensors"
    idx_path = transformer_dir / "diffusion_pytorch_model.safetensors.index.json"

    if single_path.exists():
        full_state = load_file(str(single_path), device="cpu")
    elif idx_path.exists():
        with open(idx_path) as f:
            index = json.load(f)
        shard_files = sorted(set(index["weight_map"].values()))
        full_state = {}
        for shard in shard_files:
            tensors = load_file(str(transformer_dir / shard), device="cpu")
            full_state.update(tensors)
    else:
        raise FileNotFoundError(f"No safetensors in {transformer_dir}")

    transformer.load_state_dict(full_state, assign=True)
    del full_state
    gc.collect()
    return transformer


def _fix_unhooked_fp8(transformer):
    """layerwise_casting hook 미적용 FP8 레이어(proj_out 등) → bf16."""
    fixed = []
    for name, module in transformer.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if module.weight.dtype != torch.float8_e4m3fn:
            continue
        has_hook = (
            hasattr(module, "_diffusers_hook")
            and module._diffusers_hook.get_hook("layerwise_casting") is not None
        )
        if not has_hook:
            module.weight = torch.nn.Parameter(
                module.weight.data.to(torch.bfloat16), requires_grad=False
            )
            if module.bias is not None and module.bias.dtype == torch.float8_e4m3fn:
                module.bias = torch.nn.Parameter(
                    module.bias.data.to(torch.bfloat16), requires_grad=False
                )
            fixed.append(name)
    if fixed:
        print(f"    Fixed {len(fixed)} unhooked FP8 layers: {fixed}")
    return fixed


def _fix_lora_fp8(pipe):
    """LoRA 파라미터가 FP8로 캐스트되는 문제 → bf16 복원."""
    fixed = 0
    for name, param in pipe.transformer.named_parameters():
        if "lora_" in name and param.dtype == torch.float8_e4m3fn:
            param.data = param.data.to(torch.bfloat16)
            fixed += 1
    if fixed:
        print(f"    Fixed {fixed} LoRA params → bf16")
    return fixed


class QwenImageEditEngine:
    """Qwen Image Edit 추론 엔진.

    사용법:
        engine = QwenImageEditEngine()
        engine.load()
        result = engine.generate(images, prompt)
        engine.unload()
    """

    def __init__(self, model_dir: Path = MODEL_DIR, lora_path: Path = LORA_PATH):
        self.model_dir = model_dir
        self.lora_path = lora_path if lora_path.exists() else None
        self.pipe = None
        self.has_lora = False

    @property
    def is_loaded(self) -> bool:
        return self.pipe is not None

    def load(self):
        """파이프라인 로딩 (Method A 최적화)."""
        if self.pipe is not None:
            print("Pipeline already loaded")
            return

        from diffusers import QwenImageEditPlusPipeline
        from transformers import Qwen2_5_VLForConditionalGeneration

        t_total = time.time()

        # 1. Transformer (FP8 직접 로드)
        print("  [1/5] Transformer...", end=" ", flush=True)
        t0 = time.time()
        transformer = _load_transformer_direct(self.model_dir)
        print(f"{time.time()-t0:.1f}s")

        # 2. Layerwise casting + unhooked fix
        print("  [2/5] Layerwise casting...", end=" ", flush=True)
        t0 = time.time()
        transformer.enable_layerwise_casting(
            storage_dtype=torch.float8_e4m3fn,
            compute_dtype=torch.bfloat16,
        )
        _fix_unhooked_fp8(transformer)
        print(f"{time.time()-t0:.1f}s")

        # 3. Text encoder
        print("  [3/5] Text encoder...", end=" ", flush=True)
        t0 = time.time()
        text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(self.model_dir / "text_encoder"),
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        print(f"{time.time()-t0:.1f}s")

        # 4. Pipeline 조립
        print("  [4/5] Pipeline...", end=" ", flush=True)
        t0 = time.time()
        self.pipe = QwenImageEditPlusPipeline.from_pretrained(
            str(self.model_dir),
            transformer=transformer,
            text_encoder=text_encoder,
            torch_dtype=torch.bfloat16,
        )
        print(f"{time.time()-t0:.1f}s")

        # 5. LoRA (옵션)
        if self.lora_path:
            print("  [5/5] Lightning LoRA...", end=" ", flush=True)
            t0 = time.time()
            self.pipe.load_lora_weights(str(self.lora_path))
            _fix_lora_fp8(self.pipe)
            self.has_lora = True
            print(f"{time.time()-t0:.1f}s")
        else:
            print("  [5/5] LoRA not found, skipping")

        # CPU offload
        self.pipe.enable_model_cpu_offload()

        total = time.time() - t_total
        alloc = torch.cuda.memory_allocated() / 1024**3
        print(f"  Loaded in {total:.1f}s (GPU: {alloc:.1f}GB)")

    def generate(
        self,
        images: list[str | Image.Image],
        prompt: str,
        num_steps: int | None = None,
        true_cfg_scale: float | None = None,
        seed: int = 0,
    ) -> Image.Image:
        """이미지 편집 추론.

        Args:
            images: 입력 이미지 리스트 (파일 경로 또는 PIL.Image)
            prompt: 편집 프롬프트
            num_steps: 추론 스텝 수 (기본: LoRA=8, 베이스=28)
            true_cfg_scale: True CFG 스케일 (기본: LoRA=3.0, 베이스=4.0)
            seed: 랜덤 시드

        Returns:
            편집된 PIL.Image
        """
        if not self.is_loaded:
            raise RuntimeError("Engine not loaded. Call load() first.")

        if num_steps is None:
            num_steps = 8 if self.has_lora else 28
        if true_cfg_scale is None:
            true_cfg_scale = 3.0 if self.has_lora else 4.0

        # 이미지 로드
        pil_images = []
        for img in images:
            if isinstance(img, str):
                pil_images.append(Image.open(img).convert("RGB"))
            elif isinstance(img, bytes):
                import io
                pil_images.append(Image.open(io.BytesIO(img)).convert("RGB"))
            else:
                pil_images.append(img)

        kwargs = dict(
            image=pil_images,
            prompt=prompt,
            generator=torch.manual_seed(seed),
            num_inference_steps=num_steps,
            guidance_scale=1.0,
            num_images_per_prompt=1,
        )
        if true_cfg_scale > 1.0:
            kwargs["true_cfg_scale"] = true_cfg_scale
            kwargs["negative_prompt"] = " "

        t0 = time.time()
        with torch.inference_mode():
            result = self.pipe(**kwargs)

        elapsed = time.time() - t0
        print(f"  Generated in {elapsed:.1f}s ({num_steps} steps, {elapsed/num_steps:.1f}s/step)")

        return result.images[0]

    def unload(self):
        """모델 해제, VRAM 반환."""
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
            self.has_lora = False
            gc.collect()
            torch.cuda.empty_cache()
            print("  Engine unloaded, VRAM freed")
