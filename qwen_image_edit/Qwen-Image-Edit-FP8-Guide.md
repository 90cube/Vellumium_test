# Qwen-Image-Edit-2511 FP8 + Lightning LoRA

멀티 이미지 편집 모델. FP8 양자화 + Lightning LoRA로 최적화.

## 성능 (RTX 5090 32GB)

| 항목 | 값 |
|------|-----|
| 로딩 | ~7s (Method A) |
| 추론 (8step + cfg3.0) | ~89s (3장 736x1011) |
| VRAM 피크 | ~22GB |
| 원본 대비 속도 | 3x (262s → 89s) |

## 아키텍처

```
QwenImageEditPlusPipeline
├── transformer   QwenImageTransformer2DModel  60 layers  20GB FP8
├── text_encoder  Qwen2_5_VLForConditionalGeneration     8.3GB FP8
├── vae           AutoencoderKLQwenImage                  243MB BF16
├── scheduler     FlowMatchEulerDiscreteScheduler
├── processor     Qwen2VLProcessor
└── tokenizer     Qwen2Tokenizer
```

## 핵심 최적화 기법

### 1. Method A 빠른 로딩 (56s → 7s)

```python
# nn.Linear만 meta device → weight 할당 skip (RAM 0)
# 나머지(norm, pos_freqs 등)는 CPU bf16 정상 초기화
torch.nn.Linear.__init__ = _meta_init  # device="meta"

# safetensors mmap → FP8 dtype 그대로 로드
full_state = load_file("model.safetensors", device="cpu")
transformer.load_state_dict(full_state, assign=True)
```

### 2. Layerwise Casting (FP8 저장 → BF16 연산)

```python
transformer.enable_layerwise_casting(
    storage_dtype=torch.float8_e4m3fn,
    compute_dtype=torch.bfloat16,
)
```

- 레이어별로 forward 시에만 FP8→BF16 업캐스트
- 스텝당 수 밀리초 오버헤드 (이미 측정치에 포함)

### 3. Unhooked FP8 Fix (필수)

diffusers의 `DEFAULT_SKIP_MODULES_PATTERN`:
```
("pos_embed", "patch_embed", "norm", "^proj_in$", "^proj_out$")
```

이 패턴에 해당하는 모듈은 layerwise_casting hook이 안 걸림.
FP8 직접 로드 시 이 레이어 weight가 FP8로 남아 dtype 에러 발생.

```python
# hook 여부 확인 → 없으면 bf16 복원
has_hook = (
    hasattr(module, "_diffusers_hook")
    and module._diffusers_hook.get_hook("layerwise_casting") is not None
)
if not has_hook:
    module.weight = nn.Parameter(module.weight.data.to(torch.bfloat16))
```

### 4. LoRA FP8 Fix (필수)

layerwise_casting 상태에서 LoRA 로드 시, lora_A/lora_B weight가 FP8로 캐스트됨.
이 레이어는 hook이 없어서 연산 시 `addmm_cuda not implemented for Float8_e4m3fn` 에러.

```python
for name, param in pipe.transformer.named_parameters():
    if "lora_" in name and param.dtype == torch.float8_e4m3fn:
        param.data = param.data.to(torch.bfloat16)
```

### 5. Lightning LoRA (28 steps → 8 steps)

| 설정 | 스텝 | true_cfg | 시간 | 비고 |
|------|------|----------|------|------|
| LoRA + 최속 | 8 | 1.0 (OFF) | 58s | 품질 약간 저하 |
| **LoRA + 밸런스** | **8** | **3.0** | **89s** | **추천** |
| 원본 | 28 | 4.0 | 262s | LoRA 없이 |

`true_cfg_scale > 1.0` → 매 스텝 transformer **2회** 실행 (조건부 + 무조건부).

## 파일 구조

```
Vellumium/
├── Qwen-Image-Edit-2511-FP8/     ← .gitignore됨 (29GB)
│   ├── model_index.json
│   ├── transformer/               20GB
│   ├── text_encoder/              8.3GB
│   ├── vae/                       243MB
│   ├── processor/
│   ├── tokenizer/
│   ├── scheduler/
│   └── loras/                     811MB
│       └── Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors
│
└── qwen_image_edit/               ← git 추적됨
    ├── engine.py                  코어 엔진 (로딩/추론)
    ├── server.py                  FastAPI HTTP API
    ├── worker.py                  큐 폴링 워커
    ├── .env                       설정
    ├── start_server.bat
    ├── start_worker.bat
    └── Qwen-Image-Edit-FP8-Guide.md  ← 이 파일
```

## 사용법

### HTTP API 서버

```bash
# 서버 시작 (http://localhost:8200)
start_server.bat
# 또는: python server.py
```

API 문서: http://localhost:8200/docs

```bash
# curl 예시
curl -X POST http://localhost:8200/generate \
  -H "Content-Type: application/json" \
  -d '{
    "images": ["<base64_image_1>", "<base64_image_2>"],
    "prompt": "Change the outfit to a blue dress",
    "num_steps": 8,
    "true_cfg_scale": 3.0,
    "seed": 42
  }'
```

### Node-RED 연결

HTTP Request 노드:
- Method: `POST`
- URL: `http://localhost:8200/generate`
- Body: JSON (images base64 배열 + prompt)
- 응답: `image` (base64 PNG)

### 큐 워커

```bash
start_worker.bat
# 큐 서버 /api/queue/next 폴링 → 작업 처리 → /api/queue/result 업로드
```

### Python에서 직접 사용

```python
from qwen_image_edit.engine import QwenImageEditEngine

engine = QwenImageEditEngine()
engine.load()  # ~7s

result = engine.generate(
    images=["photo1.jpg", "photo2.jpg"],
    prompt="Swap the outfit from image 2 onto image 1",
    num_steps=8,
    true_cfg_scale=3.0,
    seed=42,
)
result.save("output.png")

engine.unload()  # VRAM 해제
```

## 요구사항

```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install diffusers transformers accelerate safetensors peft pillow
pip install fastapi uvicorn python-dotenv requests  # server/worker용
```

- **GPU**: VRAM 22GB+ (RTX 3090/4090/5090)
- **PyTorch**: >= 2.1 (`load_state_dict(assign=True)` 필요)
- **diffusers**: >= 0.36.0 (`QwenImageEditPlusPipeline` 포함)
- **peft**: LoRA 로딩용
