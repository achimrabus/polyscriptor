# Qwen3-VL Hanging Issue - Root Cause Analysis

## Problem
Qwen3-VL inference hangs for 5-20 minutes when loading models on Linux server, but worked fast on Windows.

## Root Causes (Two Issues Found)

### Issue 1: Missing base model files on Linux server
**Status**: ✓ FIXED - Base model downloaded successfully

### What You Have on Linux:
```
models/Qwen3-VL-8B-glagolitic/
└── final_model/              ← ONLY adapter weights (167MB)
    ├── adapter_config.json   ← Points to "Qwen/Qwen3-VL-8B-Instruct"
    ├── adapter_model.safetensors
    └── tokenizer files
```

### What You Should Have (like on Windows):
```
models/Qwen3-VL-8B-glagolitic/
├── config.json               ← Base model config (~4KB)
├── model-*.safetensors       ← Base model weights (~17GB)
├── processor files
└── final_model/
    ├── adapter_config.json
    └── adapter_model.safetensors
```

### Why It Hangs:
1. `PeftModel.from_pretrained()` loads adapter
2. Reads `adapter_config.json` → sees `"base_model_name_or_path": "Qwen/Qwen3-VL-8B-Instruct"`
3. Looks for base model locally → NOT FOUND
4. Tries to download from HuggingFace → **17GB download = hang**

## Solutions

### Option 1: Copy Full Model from Windows (FASTEST)
Copy the entire model directory structure from Windows, not just `final_model/`:
```bash
# On Windows, you should have:
C:\path\to\models\Qwen3-VL-8B-glagolitic\
├── config.json
├── model-00001-of-00004.safetensors
├── model-00002-of-00004.safetensors
├── model-00003-of-00004.safetensors
├── model-00004-of-00004.safetensors
├── preprocessor_config.json
├── tokenizer files
└── final_model/
    └── adapter files

# Copy ALL of this to Linux, not just final_model/
```

### Option 2: Download Base Model Once (ONE-TIME)
Let the download complete once, then it's cached:
```bash
source htr_gui/bin/activate
python -c "
from transformers import Qwen3VLForConditionalGeneration
model = Qwen3VLForConditionalGeneration.from_pretrained(
    'Qwen/Qwen3-VL-8B-Instruct',
    trust_remote_code=True
)
print('Base model downloaded successfully!')
"
```

### Option 3: Use Local Base Model Path
Modify `adapter_config.json` to point to a local base model if you have it elsewhere.

### Issue 2: Multi-GPU device mapping causes generation hanging
**Status**: ✓ FIXED in [inference_qwen3.py:79-86](inference_qwen3.py#L79-L86)

**Root Cause**:
- `device_map="auto"` spreads the model across multiple GPUs (2x L40S on Linux server)
- During `model.generate()`, inter-GPU communication/synchronization hangs indefinitely
- Same code works on Windows (likely single GPU setup, no inter-GPU communication needed)

**Symptoms**:
- Model loads successfully (base + adapter) in ~30-40 seconds
- Processor loads fine
- Generation hangs forever (tested >180 seconds before timeout)
- No error messages, just infinite hang

**Systematic Debugging**:
1. Created [test_qwen3_loading.py](test_qwen3_loading.py) - 7-step test isolating the hang to `model.generate()`
2. Created [test_qwen3_minimal_gen.py](test_qwen3_minimal_gen.py) - tested different device configurations
3. **BREAKTHROUGH**: Single GPU (`device_map="cuda:0"`) works perfectly (<4s inference)
4. **SOLUTION**: Modified [inference_qwen3.py](inference_qwen3.py) to force single GPU when `device="auto"`

**Fix Details**:
```python
# Lines 79-86 in inference_qwen3.py
effective_device_map = device
if device == "auto":
    # Auto-detect: prefer single GPU if available
    if torch.cuda.is_available():
        effective_device_map = "cuda:0"  # Force single GPU
        print(f"  Using single GPU (cuda:0) to avoid multi-GPU hanging issue")
    else:
        effective_device_map = "cpu"
```

**Performance After Fix**:
- Model initialization: ~17-18 seconds (base + adapter loading)
- Inference (50 tokens): ~3-4 seconds
- Total end-to-end: <25 seconds (vs hanging forever before)

## Verification
After both fixes, inference should:
1. Load in <20 seconds (base model + adapter)
2. Generate transcription in <10 seconds
3. Complete end-to-end in <30 seconds

Test with:
```bash
source htr_gui/bin/activate
python test_qwen3_loading.py
```

All 7 steps should pass without hanging.

## Why It Worked on Windows
1. Windows setup had the full base model files alongside the adapter (no download needed)
2. Windows likely used single GPU (no multi-GPU communication issues)
