"""
Polyscriptor Web UI — FastAPI Backend

Thin wrapper around existing HTR engine code. Provides REST API + SSE
for browser-based transcription. All heavy lifting done by the same
modules the PyQt6 GUI uses.

Usage:
    source htr_gui/bin/activate
    python -m uvicorn web.polyscriptor_server:app --host 0.0.0.0 --port 8765

Author: Claude Code
Date: 2026-02-26
"""

import asyncio
import json
import logging
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image, ImageOps
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

log = logging.getLogger("polyscriptor")

# Add project root to path so we can import existing modules
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env from project root (same as the Qt GUI does via CommercialAPIEngine)
try:
    from dotenv import load_dotenv
    _env_path = PROJECT_ROOT / ".env"
    if _env_path.exists():
        load_dotenv(_env_path)
        log.info(f"Loaded environment variables from {_env_path}")
except ImportError:
    pass  # python-dotenv not installed — env vars must be set externally

from htr_engine_base import get_global_registry, HTREngine, TranscriptionResult

# PDF support via PyMuPDF
try:
    import fitz as _fitz  # PyMuPDF
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    log.warning("PyMuPDF not installed — PDF upload disabled. Install with: pip install pymupdf")

# Lazy imports for segmentation (avoid slow startup)
_segmenters_imported = False


def _import_segmenters():
    global _segmenters_imported
    if _segmenters_imported:
        return
    global KrakenLineSegmenter, LineSegmenter, PYLAIA_MODELS
    from kraken_segmenter import KrakenLineSegmenter
    from inference_page import LineSegmenter
    try:
        from inference_pylaia_native import PYLAIA_MODELS
    except ImportError:
        PYLAIA_MODELS = {}
    _segmenters_imported = True


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="Polyscriptor HTR", version="0.1.0")

# Serve static frontend files
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# State — single user, single engine at a time
loaded_engine: Optional[HTREngine] = None
loaded_engine_name: str = ""
loaded_config: dict = {}

# Persistent upload storage (survives server restarts)
UPLOAD_DIR = Path(__file__).parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# Image cache: image_id -> {path, xml_path, pil_image, lines, results}
image_cache: Dict[str, dict] = {}

# Cancel event — set to stop a running transcription stream
cancel_event: asyncio.Event = None  # initialised in startup_event

# Upload TTL: 24 hours
_UPLOAD_TTL_SECONDS = 86400


def _cleanup_old_uploads() -> int:
    """Delete uploads older than TTL and evict their image_cache entries. Returns count deleted."""
    cutoff = time.time() - _UPLOAD_TTL_SECONDS
    deleted = 0
    for f in list(UPLOAD_DIR.iterdir()):
        if f.is_file():
            try:
                if f.stat().st_mtime < cutoff:
                    f.unlink(missing_ok=True)
                    deleted += 1
            except OSError:
                pass
    # Evict stale image_cache entries whose file no longer exists
    for iid in list(image_cache.keys()):
        p = image_cache[iid].get("path")
        if p and not Path(p).exists():
            del image_cache[iid]
    return deleted


async def _periodic_cleanup():
    """Background task: clean up uploads every hour."""
    while True:
        await asyncio.sleep(3600)
        n = _cleanup_old_uploads()
        if n:
            log.info(f"Periodic cleanup: removed {n} old upload file(s).")


# ---------------------------------------------------------------------------
# API key store — web/api_keys.json (gitignored)
# Priority: request value → key store → environment variable
# ---------------------------------------------------------------------------

_KEY_STORE_PATH = Path(__file__).parent / "api_keys.json"

# Mapping from engine/provider name to environment variable
_KEY_ENV_VARS: Dict[str, str] = {
    "openai":   "OPENAI_API_KEY",
    "gemini":   "GOOGLE_API_KEY",
    "claude":   "ANTHROPIC_API_KEY",
    "openwebui": "OPENWEBUI_API_KEY",
}


def _load_key_store() -> Dict[str, str]:
    """Load api_keys.json; return empty dict on any error."""
    try:
        if _KEY_STORE_PATH.exists():
            return json.loads(_KEY_STORE_PATH.read_text()) or {}
    except Exception as e:
        log.warning(f"Could not read api_keys.json: {e}")
    return {}


def _save_key_store(keys: Dict[str, str]) -> None:
    try:
        _KEY_STORE_PATH.write_text(json.dumps(keys, indent=2))
    except Exception as e:
        log.warning(f"Could not write api_keys.json: {e}")


def _resolve_api_key(slot: str, request_value: str) -> str:
    """
    Return the best available API key for a named slot (e.g. 'openai').
    Priority: non-empty request value → api_keys.json → environment variable.
    """
    slot = slot.lower()
    if request_value and request_value.strip():
        return request_value.strip()
    stored = _load_key_store().get(slot, "")
    if stored:
        return stored
    env_var = _KEY_ENV_VARS.get(slot, "")
    return os.environ.get(env_var, "")


def _key_is_saved(slot: str) -> bool:
    return bool(_load_key_store().get(slot.lower()))


def _key_is_in_env(slot: str) -> bool:
    env_var = _KEY_ENV_VARS.get(slot.lower(), "")
    return bool(env_var and os.environ.get(env_var))


# ---------------------------------------------------------------------------
# Startup config (web/server_config.yaml) — optional, auto-load an engine
# ---------------------------------------------------------------------------

def _load_startup_config() -> dict:
    cfg_path = Path(__file__).parent / "server_config.yaml"
    if not cfg_path.exists():
        return {}
    try:
        import yaml
        with open(cfg_path) as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        log.warning(f"Could not read server_config.yaml: {e}")
        return {}


@app.on_event("startup")
async def startup_event():
    """Initialise cancel event, clean old uploads, start periodic cleanup, auto-load engine."""
    global cancel_event
    cancel_event = asyncio.Event()

    # Clean up uploads left over from previous server runs
    n = _cleanup_old_uploads()
    if n:
        log.info(f"Startup cleanup: removed {n} old upload file(s).")

    # Schedule periodic cleanup (every hour)
    asyncio.create_task(_periodic_cleanup())

    # Auto-load default engine from server_config.yaml if present
    cfg = _load_startup_config()
    if not cfg.get("default_engine"):
        return
    engine_name = cfg["default_engine"]
    engine_config = cfg.get("default_config", {})
    log.info(f"Auto-loading engine '{engine_name}' from server_config.yaml ...")
    try:
        registry = get_global_registry()
        engine = registry.get_engine_by_name(engine_name)
        if engine and engine.is_available():
            ok = await asyncio.to_thread(engine.load_model, engine_config)
            if ok:
                global loaded_engine, loaded_engine_name, loaded_config
                loaded_engine = engine
                loaded_engine_name = engine_name
                loaded_config = engine_config
                log.info(f"Auto-loaded '{engine_name}' successfully.")
            else:
                log.warning(f"Auto-load of '{engine_name}' failed (load_model returned False).")
        else:
            log.warning(f"Auto-load: engine '{engine_name}' not found or not available.")
    except Exception as e:
        log.warning(f"Auto-load error: {e}")


# ---------------------------------------------------------------------------
# Config schemas — replaces Qt config widgets for the web UI
# ---------------------------------------------------------------------------

def _get_pylaia_model_options() -> list:
    _import_segmenters()
    return [{"label": k, "value": k} for k in PYLAIA_MODELS.keys()]


def _scan_trocr_models() -> list:
    """Scan models/ directory for TrOCR checkpoints.

    A directory is considered a TrOCR model if it contains
    preprocessor_config.json (TrOCR/ViT-specific) AND config.json
    with model_type == 'vision-encoder-decoder'.
    This avoids picking up PyLaia/CRNN-CTC directories that also
    contain a config.json with training parameters.
    """
    import json as _json
    models_dir = PROJECT_ROOT / "models"
    options = [
        {"label": "kazars24/trocr-base-handwritten-ru (HuggingFace)",
         "value": "kazars24/trocr-base-handwritten-ru",
         "source": "huggingface"},
    ]
    if models_dir.exists():
        for d in sorted(models_dir.iterdir()):
            if not d.is_dir():
                continue
            # Require BOTH preprocessor_config.json AND config.json with
            # model_type == 'vision-encoder-decoder'.
            # preprocessor_config.json is ViT/TrOCR-specific (not in PyLaia).
            # config.json model_type disambiguates from Qwen3 adapters that
            # also ship a preprocessor_config but have no config.json.
            if not (d / "preprocessor_config.json").exists():
                continue
            cfg_path = d / "config.json"
            if not cfg_path.exists():
                continue
            try:
                cfg = _json.load(open(cfg_path))
                if cfg.get("model_type") != "vision-encoder-decoder":
                    continue
            except Exception:
                continue
            options.append({
                "label": d.name,
                "value": str(d),
                "source": "local",
            })
    # Allow custom HuggingFace model ID or local path
    options.append({"label": "Custom HuggingFace ID or local path…", "value": "__custom__"})
    return options


def _scan_vlm_models(engine_type: str = "qwen3") -> list:
    """Scan models/ directory for local VLM checkpoints (LoRA adapters and full models).

    Looks for directories containing adapter_config.json (LoRA fine-tunes) or
    config.json mentioning Qwen/VLM/vision architectures.

    Returns options list ending with a __custom__ sentinel for manual entry.
    """
    models_dir = PROJECT_ROOT / "models"
    options = []

    if models_dir.exists():
        for d in sorted(models_dir.iterdir()):
            if not d.is_dir():
                continue

            # Check for LoRA adapter at top-level
            if (d / "adapter_config.json").exists():
                try:
                    import json as _json
                    with open(d / "adapter_config.json") as f:
                        adapter_cfg = _json.load(f)
                    base = adapter_cfg.get("base_model_name_or_path", "")
                    is_qwen = "qwen" in base.lower() or "qwen" in d.name.lower()
                    is_churro = "churro" in base.lower() or "churro" in d.name.lower()
                    if engine_type == "qwen3" and is_qwen and not is_churro:
                        options.append({
                            "label": f"{d.name} (LoRA → {base})",
                            "value": str(d),
                            "base_model": base,
                            "adapter": str(d),
                        })
                    elif engine_type == "churro" and (is_churro or ("churro" in d.name.lower())):
                        options.append({
                            "label": f"{d.name} (LoRA → {base})",
                            "value": str(d),
                            "base_model": base,
                            "adapter": str(d),
                        })
                except Exception:
                    pass
                continue  # Don't also check final_model subdirs

            # Check for final_model subdirectory with adapter
            final = d / "final_model"
            if final.is_dir() and (final / "adapter_config.json").exists():
                try:
                    import json as _json
                    with open(final / "adapter_config.json") as f:
                        adapter_cfg = _json.load(f)
                    base = adapter_cfg.get("base_model_name_or_path", "")
                    is_qwen = "qwen" in base.lower() or "qwen" in d.name.lower()
                    is_churro = "churro" in base.lower() or "churro" in d.name.lower()
                    if engine_type == "qwen3" and is_qwen and not is_churro:
                        options.append({
                            "label": f"{d.name} (LoRA → {base})",
                            "value": str(final),
                            "base_model": base,
                            "adapter": str(final),
                        })
                    elif engine_type == "churro" and (is_churro or ("churro" in d.name.lower())):
                        options.append({
                            "label": f"{d.name} (LoRA → {base})",
                            "value": str(final),
                            "base_model": base,
                            "adapter": str(final),
                        })
                except Exception:
                    pass

    # Always append a "Custom / HuggingFace" sentinel as the last option
    options.append({
        "label": "Custom / HuggingFace model ID...",
        "value": "__custom__",
    })
    return options


ENGINE_SCHEMAS = {
    "CRNN-CTC (PyLaia-inspired)": lambda: {
        "fields": [
            {"key": "model_path", "type": "select", "label": "Model",
             "options": _get_pylaia_model_options()},
            {"key": "enable_spaces", "type": "checkbox",
             "label": "Convert <space> tokens", "default": True},
        ]
    },
    "TrOCR": lambda: {
        "fields": [
            {"key": "model_path", "type": "select", "label": "Model",
             "options": _scan_trocr_models(),
             "custom_key": "custom_model_path",
             "custom_placeholder": "HuggingFace model ID (e.g. microsoft/trocr-base-handwritten) or absolute local path"},
            {"key": "num_beams", "type": "number", "label": "Beam Search",
             "min": 1, "max": 10, "default": 4},
            {"key": "normalize_background", "type": "checkbox",
             "label": "Normalize Background", "default": False},
        ]
    },
    "Qwen3-VL": lambda: {
        "fields": [
            {"key": "model_preset", "type": "select", "label": "Model",
             "options": _scan_vlm_models("qwen3"),
             "custom_key": "base_model",
             "custom_placeholder": "HuggingFace model ID, e.g. Qwen/Qwen3-VL-8B-Instruct"},
            {"key": "max_image_size", "type": "number", "label": "Max Image Size (px)",
             "min": 512, "max": 4096, "default": 1536},
        ]
    },
    "Churro VLM": lambda: {
        "fields": [
            {"key": "model_preset", "type": "select", "label": "Model",
             "options": _scan_vlm_models("churro"),
             "custom_key": "model_name",
             "custom_placeholder": "HuggingFace model ID, e.g. stanford-oval/churro-3B"},
            {"key": "device", "type": "select", "label": "Device",
             "options": [{"label": "Auto", "value": "auto"},
                         {"label": "GPU 0", "value": "cuda:0"},
                         {"label": "GPU 1", "value": "cuda:1"},
                         {"label": "CPU", "value": "cpu"}]},
            {"key": "max_image_size", "type": "number", "label": "Max Image Size (px)",
             "min": 512, "max": 4096, "default": 2048},
        ]
    },
    "Kraken": lambda: {
        "fields": [
            {"key": "model_path", "type": "text", "label": "Model Path",
             "default": "", "placeholder": "Path to Kraken model file"},
        ]
    },
    "Commercial APIs": lambda: {
        "fields": [
            {"key": "provider", "type": "select", "label": "Provider",
             "options": [
                 {"label": "OpenAI (GPT-4o, o1, …)", "value": "OpenAI"},
                 {"label": "Google Gemini", "value": "Gemini"},
                 {"label": "Anthropic Claude", "value": "Claude"},
             ]},
            {"key": "model", "type": "select", "label": "Model",
             "dynamic": True,
             "dynamic_hint": "Enter API key, then ↻ to load available models",
             # No static lists — always fetch live from the provider API
             "per_provider_options": {},
             "options": [],
             "custom_key": "custom_model_id",
             "custom_placeholder": "e.g. gpt-4.5, gemini-exp-1206, claude-opus-4"},
            {"key": "api_key", "type": "password", "label": "API Key",
             "default": "", "placeholder": "Paste your API key here"},
            {"key": "temperature", "type": "number", "label": "Temperature",
             "min": 0.0, "max": 2.0, "default": 0.0,
             "placeholder": "0.0 = deterministic (recommended for transcription)"},
            {"key": "max_output_tokens", "type": "number", "label": "Max output tokens (optional)",
             "min": 512, "max": 65536, "default": None,
             "placeholder": "Leave blank = model maximum"},
            {"key": "custom_prompt", "type": "text", "label": "Custom Prompt (optional)",
             "default": "",
             "placeholder": "Leave blank for default transcription prompt"},
        ]
    },
    "OpenWebUI": lambda: {
        "fields": [
            {"key": "base_url", "type": "text", "label": "Base URL",
             "default": "https://openwebui.uni-freiburg.de/api",
             "placeholder": "https://your-openwebui-instance/api"},
            {"key": "api_key", "type": "password", "label": "API Key",
             "default": "", "placeholder": "Your OpenWebUI API key"},
            {"key": "model", "type": "select", "label": "Model",
             "dynamic": True,
             "dynamic_hint": "Enter API key & base URL, then ↻ to load available models",
             "options": []},   # populated via /api/engine/OpenWebUI/models
            {"key": "temperature", "type": "number", "label": "Temperature",
             "min": 0.0, "max": 2.0, "default": 0.1},
            {"key": "max_tokens", "type": "number", "label": "Max Tokens",
             "min": 100, "max": 4096, "default": 500},
            {"key": "custom_prompt", "type": "text", "label": "Custom Prompt (optional)",
             "default": "",
             "placeholder": "Leave blank for default transcription prompt"},
        ]
    },
}


# ---------------------------------------------------------------------------
# Request/response models
# ---------------------------------------------------------------------------

class EngineLoadRequest(BaseModel):
    engine_name: str
    config: Dict[str, Any] = {}


class TranscribeRequest(BaseModel):
    image_id: str
    seg_method: str = "kraken"  # kraken, kraken-blla, hpp
    seg_device: str = "cpu"
    max_columns: int = 6          # blla: max sub-columns per region (iterative splitting)
    split_width_fraction: float = 0.40  # blla: min region width (fraction of page) to trigger sub-split
    use_pagexml: bool = True      # use attached PAGE XML for segmentation when available


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/engines")
async def list_engines():
    registry = get_global_registry()
    engines = []
    for engine in registry.get_all_engines():
        available = engine.is_available()
        engines.append({
            "name": engine.get_name(),
            "description": engine.get_description(),
            "available": available,
            "unavailable_reason": engine.get_unavailable_reason() if not available else None,
            "requires_line_segmentation": engine.requires_line_segmentation(),
            "has_config_schema": engine.get_name() in ENGINE_SCHEMAS,
        })
    return engines


@app.get("/api/engine/{name}/config-schema")
async def get_config_schema(name: str):
    if name not in ENGINE_SCHEMAS:
        return {"fields": []}
    schema = ENGINE_SCHEMAS[name]()

    # Inject key_status into password fields so the frontend can show
    # whether a key is already saved or available via environment variable.
    for field in schema.get("fields", []):
        if field.get("type") == "password":
            slot = field["key"]  # e.g. "api_key" — we normalise below
            # Determine the slot name: openwebui, openai, gemini, claude
            if name == "OpenWebUI":
                slot_name = "openwebui"
            elif name == "Commercial APIs":
                slot_name = None  # handled below — report per-provider status
            else:
                slot_name = slot

            if slot_name:
                if _key_is_saved(slot_name):
                    field["key_status"] = "saved"
                elif _key_is_in_env(slot_name):
                    field["key_status"] = "env"
                else:
                    field["key_status"] = "missing"
            elif name == "Commercial APIs":
                # Report status for each provider so the frontend can update
                # the hint dynamically when the user switches provider.
                field["key_status_per_provider"] = {
                    provider: (
                        "saved" if _key_is_saved(provider) else
                        "env"   if _key_is_in_env(provider) else
                        "missing"
                    )
                    for provider in ["openai", "gemini", "claude"]
                }

    return schema


@app.get("/api/engine/status")
async def engine_status():
    return {
        "loaded": loaded_engine is not None and loaded_engine.is_model_loaded(),
        "engine_name": loaded_engine_name,
        "config": loaded_config,
    }


@app.get("/api/engine/{name}/models")
async def get_engine_models(
    name: str,
    api_key: str = "",
    provider: str = "openai",
    base_url: str = "",
):
    """
    Fetch available models for engines whose model list is dynamic.

    - OpenWebUI: queries the OpenWebUI /api/models endpoint
    - Commercial APIs: uses existing fetch_* helpers with fallback lists
    """
    if name == "OpenWebUI":
        resolved = _resolve_api_key("openwebui", api_key)
        if not resolved:
            return {"models": [], "error": "No API key — paste one in the form or set OPENWEBUI_API_KEY"}
        effective_url = base_url.strip() or "https://openwebui.uni-freiburg.de/api"
        try:
            from openai import OpenAI as _OAI  # openai SDK speaks the same protocol
            client = _OAI(
                base_url=effective_url,
                api_key=resolved,
            )
            data = await asyncio.to_thread(lambda: list(client.models.list()))
            models = sorted(m.id for m in data)
            return {"models": models}
        except Exception as e:
            return {"models": [], "error": str(e)}

    elif name == "Commercial APIs":
        prov = provider.lower()
        resolved = _resolve_api_key(prov, api_key)
        try:
            sys.path.insert(0, str(PROJECT_ROOT))
            if prov == "openai":
                from inference_commercial_api import fetch_openai_models
                models = await asyncio.to_thread(fetch_openai_models, resolved or None)
                return {"models": models}
            elif prov == "gemini":
                from inference_commercial_api import fetch_gemini_models
                models = await asyncio.to_thread(fetch_gemini_models, resolved or None)
                return {"models": models}
            elif prov == "claude":
                from inference_commercial_api import fetch_claude_models
                models = await asyncio.to_thread(fetch_claude_models, resolved or None)
                return {"models": models}
            else:
                return {"models": [], "error": f"Unknown provider: {provider}"}
        except Exception as e:
            return {"models": [], "error": str(e)}

    return {"models": [], "error": f"Dynamic model listing not supported for '{name}'"}


@app.post("/api/engine/load")
async def load_engine(req: EngineLoadRequest):
    global loaded_engine, loaded_engine_name, loaded_config

    registry = get_global_registry()
    engine = registry.get_engine_by_name(req.engine_name)
    if not engine:
        raise HTTPException(404, f"Engine '{req.engine_name}' not found")
    if not engine.is_available():
        raise HTTPException(400, f"Engine not available: {engine.get_unavailable_reason()}")

    # Unload previous
    if loaded_engine and loaded_engine.is_model_loaded():
        loaded_engine.unload_model()

    # Expand config based on engine type
    config = dict(req.config)

    if req.engine_name == "TrOCR" and "model_path" in config:
        custom_val = config.pop("custom_model_path", "").strip()
        if config["model_path"] == "__custom__":
            # User entered a custom HuggingFace ID or local path
            if not custom_val:
                raise HTTPException(400, "Please enter a HuggingFace model ID or local path")
            config["model_path"] = custom_val
        # Detect source: local path vs HuggingFace ID
        from pathlib import Path as _P
        if _P(config["model_path"]).exists():
            config["model_source"] = "local"
        else:
            config["model_source"] = "huggingface"

    elif req.engine_name == "Qwen3-VL" and "model_preset" in config:
        # Resolve local VLM preset to base_model + adapter
        preset_val = config.pop("model_preset")
        custom_val = config.pop("base_model", "").strip()
        if preset_val == "__custom__":
            # User typed a custom HuggingFace ID into the extra text field
            config["base_model"] = custom_val or "Qwen/Qwen3-VL-8B-Instruct"
            config["adapter"] = None
        else:
            # Find the matching preset option to get base_model + adapter
            vlm_opts = _scan_vlm_models("qwen3")
            matched = next((o for o in vlm_opts if o["value"] == preset_val), None)
            if matched:
                config["base_model"] = matched.get("base_model", preset_val)
                config["adapter"] = matched.get("adapter")
            else:
                config["base_model"] = preset_val
                config["adapter"] = None

    elif req.engine_name == "Churro VLM" and "model_preset" in config:
        # Resolve local VLM preset to model_name + adapter_path
        preset_val = config.pop("model_preset")
        custom_val = config.pop("model_name", "").strip()
        if preset_val == "__custom__":
            config["model_name"] = custom_val or "stanford-oval/churro-3B"
            config["adapter_path"] = None
        else:
            vlm_opts = _scan_vlm_models("churro")
            matched = next((o for o in vlm_opts if o["value"] == preset_val), None)
            if matched:
                config["model_name"] = matched.get("base_model", preset_val)
                config["adapter_path"] = matched.get("adapter")
            else:
                config["model_name"] = preset_val
                config["adapter_path"] = None

    elif req.engine_name == "Commercial APIs":
        # Resolve __custom__ model sentinel
        if config.get("model") == "__custom__":
            config["model"] = config.pop("model_custom", "").strip() or "gpt-4o"

    # Resolve API keys via priority chain: request → key store → env var
    # Also persist non-empty keys that were explicitly provided
    if req.engine_name == "Commercial APIs":
        provider_slot = config.get("provider", "openai").lower()
        raw_key = config.get("api_key", "")
        resolved = _resolve_api_key(provider_slot, raw_key)
        if not resolved:
            raise HTTPException(400, f"No API key for {config.get('provider')}. "
                                     "Paste a key in the field or set the env variable.")
        config["api_key"] = resolved
        # Save newly-provided key for next time
        if raw_key.strip() and raw_key.strip() != resolved:
            pass  # already resolved from store/env
        if raw_key.strip():
            keys = _load_key_store()
            keys[provider_slot] = raw_key.strip()
            _save_key_store(keys)

    elif req.engine_name == "OpenWebUI":
        raw_key = config.get("api_key", "")
        resolved = _resolve_api_key("openwebui", raw_key)
        if not resolved:
            raise HTTPException(400, "No API key for OpenWebUI. "
                                     "Paste a key in the field or set OPENWEBUI_API_KEY.")
        config["api_key"] = resolved
        if raw_key.strip():
            keys = _load_key_store()
            keys["openwebui"] = raw_key.strip()
            _save_key_store(keys)

    # Strip empty custom_prompt for API engines (use engine default)
    if req.engine_name in ("Commercial APIs", "OpenWebUI"):
        if not config.get("custom_prompt", "").strip():
            config["custom_prompt"] = None

    start = time.time()
    success = await asyncio.to_thread(engine.load_model, config)
    elapsed = time.time() - start

    if not success:
        raise HTTPException(500, "Failed to load model")

    loaded_engine = engine
    loaded_engine_name = req.engine_name
    loaded_config = config

    return {"success": True, "load_time_s": round(elapsed, 2),
            "engine_name": req.engine_name}


@app.get("/api/keys")
async def list_keys():
    """Return saved key slots with masked values (never expose full keys)."""
    store = _load_key_store()
    result = {}
    for slot, key in store.items():
        if key:
            result[slot] = "•" * min(len(key), 8) + key[-4:] if len(key) > 4 else "••••"
    # Also report env-only keys
    for slot, env_var in _KEY_ENV_VARS.items():
        if slot not in result and os.environ.get(env_var):
            result[slot] = f"(from env: {env_var})"
    return result


class SaveKeyRequest(BaseModel):
    slot: str    # e.g. "openai", "gemini", "claude", "openwebui"
    key: str     # empty string = delete


@app.post("/api/keys")
async def save_key(req: SaveKeyRequest):
    """Save or delete an API key in the key store."""
    slot = req.slot.lower()
    if slot not in _KEY_ENV_VARS:
        raise HTTPException(400, f"Unknown key slot '{slot}'. "
                                 f"Valid: {list(_KEY_ENV_VARS.keys())}")
    keys = _load_key_store()
    if req.key.strip():
        keys[slot] = req.key.strip()
        action = "saved"
    else:
        keys.pop(slot, None)
        action = "deleted"
    _save_key_store(keys)
    return {"success": True, "slot": slot, "action": action}


@app.post("/api/engine/unload")
async def unload_engine():
    global loaded_engine, loaded_engine_name, loaded_config
    if loaded_engine:
        loaded_engine.unload_model()
    loaded_engine = None
    loaded_engine_name = ""
    loaded_config = {}
    return {"success": True}


def _register_image(pil_image: Image.Image, filename: str, save_path: Path) -> str:
    """Store a PIL image in the cache and return its image_id."""
    image_id = str(uuid.uuid4())
    image_cache[image_id] = {
        "path": save_path,
        "xml_path": None,
        "pil_image": pil_image,
        "width": pil_image.width,
        "height": pil_image.height,
        "filename": filename,
        "lines": None,
    }
    return image_id


@app.post("/api/image/upload")
async def upload_image(file: UploadFile = File(...)):
    filename = file.filename or "upload"
    is_pdf = (
        filename.lower().endswith(".pdf") or
        (file.content_type or "").startswith("application/pdf")
    )

    content = await file.read()
    if len(content) > 200 * 1024 * 1024:
        raise HTTPException(400, "File too large (max 200MB)")

    # ── PDF: render each page as a separate image ──────────────────────────
    if is_pdf:
        if not PDF_AVAILABLE:
            raise HTTPException(400, "PDF support requires PyMuPDF. Install with: pip install pymupdf")
        try:
            import asyncio
            from concurrent.futures import ThreadPoolExecutor

            def _render_pdf(data: bytes, stem: str) -> list:
                mat = _fitz.Matrix(150 / 72, 150 / 72)
                doc = _fitz.open(stream=data, filetype="pdf")
                results = []
                for i, page in enumerate(doc):
                    pix = page.get_pixmap(matrix=mat, colorspace=_fitz.csRGB)
                    pil_page = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    page_filename = f"{stem}_page{i+1:03d}.png"
                    save_path = UPLOAD_DIR / f"{uuid.uuid4()}.png"
                    pil_page.save(save_path)
                    pid = _register_image(pil_page, page_filename, save_path)
                    results.append({
                        "image_id": pid,
                        "filename": page_filename,
                        "width": pil_page.width,
                        "height": pil_page.height,
                        "page": i + 1,
                    })
                doc.close()
                return results

            stem = Path(filename).stem
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor(max_workers=1) as pool:
                pages_out = await loop.run_in_executor(pool, _render_pdf, content, stem)
            return {
                "is_pdf": True,
                "filename": filename,
                "num_pages": len(pages_out),
                "pages": pages_out,
            }
        except Exception as e:
            raise HTTPException(400, f"Failed to render PDF: {e}")

    # ── Regular image ───────────────────────────────────────────────────────
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image or PDF")

    ext = Path(filename).suffix or ".jpg"
    save_path = UPLOAD_DIR / f"{uuid.uuid4()}{ext}"
    save_path.write_bytes(content)

    try:
        pil_image = Image.open(save_path)
        pil_image = ImageOps.exif_transpose(pil_image)
        pil_image = pil_image.convert("RGB")
    except Exception as e:
        save_path.unlink(missing_ok=True)
        raise HTTPException(400, f"Invalid image: {e}")

    image_id = _register_image(pil_image, filename, save_path)
    return {
        "image_id": image_id,
        "width": pil_image.width,
        "height": pil_image.height,
        "filename": filename,
    }


@app.post("/api/image/{image_id}/xml")
async def upload_xml(image_id: str, file: UploadFile = File(...)):
    """Attach a PAGE XML file to an already-uploaded image."""
    if image_id not in image_cache:
        raise HTTPException(404, "Image not found — upload image first")
    content = await file.read()
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(400, "XML too large (max 10MB)")
    xml_path = UPLOAD_DIR / f"{image_id}.xml"
    xml_path.write_bytes(content)
    image_cache[image_id]["xml_path"] = xml_path
    return {"success": True, "filename": file.filename}


@app.get("/api/image/{image_id}")
async def get_image(image_id: str):
    if image_id not in image_cache:
        raise HTTPException(404, "Image not found")
    return FileResponse(str(image_cache[image_id]["path"]))


@app.get("/api/image/{image_id}/info")
async def image_info(image_id: str):
    if image_id not in image_cache:
        raise HTTPException(404, "Image not found")
    d = image_cache[image_id]
    return {
        "image_id": image_id,
        "filename": d["filename"],
        "width": d["width"],
        "height": d["height"],
        "has_xml": d["xml_path"] is not None,
    }


async def _run_segmentation(img_data: dict, method: str, device: str = "cpu",
                            max_columns: int = 6,
                            split_width_fraction: float = 0.40) -> dict:
    """
    Shared segmentation helper.  Runs the appropriate segmenter, stores
    results in img_data, and returns a serialisable dict ready for SSE or JSON.
    Also populates img_data["line_regions"] with a per-line region index list
    so the transcription loop can tag each line with its column.
    """
    _import_segmenters()
    pil_image = img_data["pil_image"]
    xml_path  = img_data.get("xml_path")

    regions: list = []
    lines: list   = []

    if xml_path is not None:
        from inference_page import PageXMLSegmenter as _PXSeg
        segmenter = _PXSeg(str(xml_path))
        lines = await asyncio.to_thread(segmenter.segment_lines, pil_image)
        source = "pagexml"

    elif method == "kraken-blla":
        segmenter = KrakenLineSegmenter(device=device)
        regions, lines = await asyncio.to_thread(
            segmenter.segment_with_regions, pil_image,
            device=device,
            max_columns=max_columns,
            split_width_fraction=split_width_fraction,
        )
        source = "kraken-blla"

    elif method == "kraken":
        segmenter = KrakenLineSegmenter()
        # Use column-aware segmentation so multi-column pages read correctly
        regions, lines = await asyncio.to_thread(
            segmenter.segment_classical_with_regions, pil_image,
            max_columns=max_columns,
        )
        source = "kraken"

    else:  # hpp
        segmenter = LineSegmenter()
        lines = await asyncio.to_thread(segmenter.segment_lines, pil_image)
        source = "hpp"

    # Build per-line region index (used by transcription loop for column view)
    line_regions: list[int] = []
    if regions:
        offset = 0
        for ri, r in enumerate(regions):
            for _ in r.line_ids:
                line_regions.append(ri)
            offset += len(r.line_ids)
    else:
        line_regions = [0] * len(lines)

    img_data["lines"]        = lines
    img_data["line_regions"] = line_regions
    img_data["seg_source"]   = source
    img_data["seg_regions"]  = [
        {"id": r.id, "bbox": list(r.bbox), "num_lines": len(r.line_ids)}
        for r in regions
    ] if regions else []

    result: dict = {
        "num_lines": len(lines),
        "bboxes":    [list(l.bbox) for l in lines],
        "source":    source,
    }
    if regions:
        result["regions"] = img_data["seg_regions"]
    return result


@app.delete("/api/image/{image_id}/region/{region_index}")
async def delete_region(image_id: str, region_index: int):
    """
    Remove one detected region and its lines from the cached segmentation.
    Returns updated segmentation data in the same format as /segment,
    so the client can redraw the canvas.
    """
    if image_id not in image_cache:
        raise HTTPException(404, "Image not found")
    img_data = image_cache[image_id]

    seg_regions = img_data.get("seg_regions") or []
    if not seg_regions:
        raise HTTPException(400, "No segmentation data — run Segment first")
    if region_index < 0 or region_index >= len(seg_regions):
        raise HTTPException(400, f"Region index out of range (0–{len(seg_regions)-1})")

    lines        = img_data.get("lines") or []
    line_regions = img_data.get("line_regions") or ([0] * len(lines))

    # Keep lines that are NOT in the deleted region; re-index later regions
    new_lines: list = []
    new_line_regions: list = []
    for line, lr in zip(lines, line_regions):
        if lr == region_index:
            continue
        new_lines.append(line)
        new_line_regions.append(lr if lr < region_index else lr - 1)

    new_regions = [r for i, r in enumerate(seg_regions) if i != region_index]

    img_data["lines"]        = new_lines
    img_data["line_regions"] = new_line_regions
    img_data["seg_regions"]  = new_regions

    result: dict = {
        "num_lines": len(new_lines),
        "bboxes":    [list(l.bbox) for l in new_lines],
        "source":    img_data.get("seg_source", "modified"),
    }
    if new_regions:
        result["regions"] = new_regions
    return result


@app.get("/api/image/{image_id}/segment")
async def segment_image(
    image_id: str,
    method: str = "kraken",
    device: str = "cpu",
    max_columns: int = 6,
    split_width_fraction: float = 0.40,
):
    """
    Run segmentation only (no transcription) and return line bboxes as JSON.
    Useful for previewing line layout before transcribing.
    """
    if image_id not in image_cache:
        raise HTTPException(404, "Image not found — upload first")

    try:
        return await _run_segmentation(image_cache[image_id], method, device,
                                       max_columns, split_width_fraction)
    except Exception as e:
        raise HTTPException(500, f"Segmentation failed: {e}")


@app.post("/api/transcribe")
async def transcribe(req: TranscribeRequest):
    if not loaded_engine or not loaded_engine.is_model_loaded():
        raise HTTPException(400, "No engine loaded")
    if req.image_id not in image_cache:
        raise HTTPException(404, "Image not found — upload first")

    img_data = image_cache[req.image_id]
    pil_image = img_data["pil_image"]

    # Reset cancel flag for this transcription run
    cancel_event.clear()

    async def event_stream():
        _import_segmenters()

        try:
            # --- Segmentation ---
            xml_path = img_data.get("xml_path") if req.use_pagexml else None

            if not loaded_engine.requires_line_segmentation():
                # Page-level engine — no segmentation needed
                from inference_page import LineSegment
                lines = [LineSegment(
                    image=pil_image,
                    bbox=(0, 0, pil_image.width, pil_image.height),
                    coords=None,
                )]
                img_data["lines"]        = lines
                img_data["line_regions"] = [0]
                img_data["seg_source"]   = "page"
                img_data["seg_regions"]  = []
                yield _sse("segmentation", {
                    "num_lines": 1,
                    "bboxes": [[0, 0, pil_image.width, pil_image.height]],
                    "source": "page",
                })
            else:
                # Reuse cached segmentation if method matches (e.g. user clicked Segment first)
                cached_lines   = img_data.get("lines")
                cached_source  = img_data.get("seg_source")
                desired_source = "pagexml" if (xml_path and req.use_pagexml) else req.seg_method

                if cached_lines and cached_source == desired_source:
                    lines = cached_lines
                    yield _sse("status", {"message": "Using cached segmentation..."})
                    seg_event: dict = {
                        "num_lines": len(lines),
                        "bboxes":    [list(l.bbox) for l in lines],
                        "source":    cached_source,
                    }
                    if img_data.get("seg_regions"):
                        seg_event["regions"] = img_data["seg_regions"]
                    yield _sse("segmentation", seg_event)
                elif xml_path is not None:
                    yield _sse("status", {"message": "Reading line layout from PAGE XML..."})
                    seg_result = await _run_segmentation(img_data, "pagexml",
                                                         req.seg_device, req.max_columns,
                                                         req.split_width_fraction)
                    lines = img_data["lines"]
                    yield _sse("segmentation", seg_result)
                else:
                    yield _sse("status", {"message": f"Segmenting with {req.seg_method}..."})
                    seg_result = await _run_segmentation(img_data, req.seg_method,
                                                         req.seg_device, req.max_columns,
                                                         req.split_width_fraction)
                    lines = img_data["lines"]
                    yield _sse("segmentation", seg_result)

            # --- Transcription ---
            results = []
            start_time = time.time()
            line_regions = img_data.get("line_regions") or ([0] * len(lines))

            for i, line in enumerate(lines):
                # Check for cancellation before each line
                if cancel_event.is_set():
                    yield _sse("cancelled", {})
                    return

                line_img = line.image if line.image is not None else pil_image.crop(line.bbox)
                img_array = np.array(line_img.convert("RGB"))

                result = await asyncio.to_thread(
                    loaded_engine.transcribe_line, img_array, loaded_config
                )

                text = str(result.text) if hasattr(result, "text") else str(result)
                confidence = None
                if hasattr(result, "confidence") and result.confidence is not None:
                    confidence = float(result.confidence)
                    if confidence > 1:
                        confidence = confidence / 100.0

                line_data = {
                    "index": i,
                    "text": text,
                    "confidence": confidence,
                    "bbox": list(line.bbox),
                    "region": line_regions[i] if i < len(line_regions) else 0,
                }
                results.append(line_data)
                yield _sse("progress", {
                    "current": i + 1,
                    "total": len(lines),
                    "line": line_data,
                })

                # Check for cancellation after each line's progress event
                if cancel_event.is_set():
                    yield _sse("cancelled", {})
                    return

            # Store completed results in image_cache for export
            img_data["results"] = results

            elapsed = time.time() - start_time
            yield _sse("complete", {
                "lines": results,
                "total_time_s": round(elapsed, 2),
                "engine": loaded_engine_name,
            })

        except Exception as e:
            yield _sse("error", {"message": str(e)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Disable nginx buffering if behind proxy
        },
    )


@app.post("/api/transcribe/cancel")
async def cancel_transcription():
    """Signal the running transcription to stop after the current line."""
    if cancel_event is not None:
        cancel_event.set()
    return {"success": True}


@app.post("/api/image/{image_id}/export-xml")
async def export_xml(image_id: str):
    """Export transcription results for image_id as PAGE XML."""
    pretty, stem = _build_xml_bytes(image_id)
    return Response(
        content=pretty,
        media_type="application/xml",
        headers={"Content-Disposition": f'attachment; filename="{stem}.xml"'},
    )


def _build_xml_bytes(image_id: str) -> tuple[bytes, str]:
    """Return (xml_bytes, stem) for a cached image, or raise HTTPException."""
    import xml.etree.ElementTree as ET
    from xml.dom import minidom
    from page_xml_exporter import PageXMLExporter

    if image_id not in image_cache:
        raise HTTPException(404, f"Image {image_id} not found")
    img_data = image_cache[image_id]
    results = img_data.get("results")
    if not results:
        raise HTTPException(400, f"No results for {image_id}")

    filename = img_data.get("filename", img_data["path"].name)
    width = img_data["width"]
    height = img_data["height"]

    class _SegProxy:
        __slots__ = ("bbox", "coords", "text", "confidence")
        def __init__(self, r):
            bbox = r.get("bbox")
            self.bbox = tuple(bbox) if bbox else (0, 0, width, height)
            self.coords = None
            self.text = r.get("text", "")
            self.confidence = r.get("confidence")

    segments = [_SegProxy(r) for r in results]
    exporter = PageXMLExporter(str(filename), width, height)
    root, page = exporter._make_root("Polyscriptor Web UI", None)

    reading_order = ET.SubElement(page, 'ReadingOrder')
    ordered_group = ET.SubElement(reading_order, 'OrderedGroup',
                                  {'id': 'ro_1', 'caption': 'Regions reading order'})
    ET.SubElement(ordered_group, 'RegionRefIndexed', {'index': '0', 'regionRef': 'region_1'})

    text_region = ET.SubElement(page, 'TextRegion',
                                 {'id': 'region_1', 'type': 'paragraph', 'custom': 'readingOrder {index:0;}'})
    if segments:
        x1 = min(s.bbox[0] for s in segments)
        y1 = min(s.bbox[1] for s in segments)
        x2 = max(s.bbox[2] for s in segments)
        y2 = max(s.bbox[3] for s in segments)
        ET.SubElement(text_region, 'Coords').set('points', f'{x1},{y1} {x2},{y1} {x2},{y2} {x1},{y2}')
    for idx, seg in enumerate(segments):
        exporter._add_text_line(text_region, f'line_{idx + 1}', seg, seg.text, idx)

    xml_bytes = ET.tostring(root, encoding='utf-8', method='xml')
    pretty = minidom.parseString(xml_bytes).toprettyxml(indent='  ', encoding='utf-8')
    return pretty, Path(filename).stem


class BatchXMLRequest(BaseModel):
    image_ids: list[str]


@app.post("/api/batch/export-xml")
async def batch_export_xml(req: BatchXMLRequest):
    """Return a ZIP archive containing one PAGE XML file per image."""
    import zipfile, io
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        for image_id in req.image_ids:
            try:
                xml_bytes, stem = _build_xml_bytes(image_id)
                zf.writestr(f"{stem}.xml", xml_bytes)
            except HTTPException:
                pass  # skip images without results
    buf.seek(0)
    return Response(
        content=buf.read(),
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="batch_export.zip"'},
    )


@app.get("/api/kraken/presets")
async def kraken_presets():
    """Return list of available Kraken model presets (local + Zenodo)."""
    try:
        from engines.kraken_engine import KRAKEN_MODELS
    except ImportError:
        return {"presets": []}
    presets = []
    for model_id, info in KRAKEN_MODELS.items():
        presets.append({
            "id": model_id,
            "label": info.get("description", model_id),
            "language": info.get("language", ""),
            "source": info.get("source", ""),
        })
    return {"presets": presets}


@app.get("/api/gpu")
async def gpu_status():
    try:
        import torch
        if not torch.cuda.is_available():
            return {"available": False, "gpus": []}

        # pynvml (nvidia-ml-py) for utilization %; graceful fallback if missing
        nvml_utils: dict[int, dict] = {}
        try:
            import pynvml
            pynvml.nvmlInit()
            for _i in range(pynvml.nvmlDeviceGetCount()):
                h = pynvml.nvmlDeviceGetHandleByIndex(_i)
                u = pynvml.nvmlDeviceGetUtilizationRates(h)
                nvml_utils[_i] = {"gpu_pct": u.gpu, "mem_pct": u.memory}
        except Exception:
            pass  # pynvml unavailable — utilization fields omitted

        gpus = []
        for i in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(i)
            entry: dict = {
                "index": i,
                "name": torch.cuda.get_device_name(i),
                "memory_total_mb": round(total / 1e6),
                "memory_used_mb": round((total - free) / 1e6),
                "memory_free_mb": round(free / 1e6),
            }
            if i in nvml_utils:
                entry["utilization_gpu_pct"] = nvml_utils[i]["gpu_pct"]
                entry["utilization_mem_pct"] = nvml_utils[i]["mem_pct"]
            gpus.append(entry)
        return {"available": True, "gpus": gpus}
    except Exception:
        return {"available": False, "gpus": []}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sse(event: str, data: dict) -> str:
    """Format a Server-Sent Event."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
