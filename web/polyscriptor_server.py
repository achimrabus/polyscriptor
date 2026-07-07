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
import hashlib
import importlib
import json
import logging
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image, ImageOps
from fastapi import Cookie, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
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

from htr_engine_base import get_global_registry, HTREngine, TranscriptionResult, load_runtime_profile
from transcription_metrics import ComparisonMode, TranscriptionMetrics

# Optional deployment profile — lets a deployment customise segmentation
# without baking environment-specific logic into the server. None in normal use.
_RUNTIME_PROFILE = load_runtime_profile()

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

# ---------------------------------------------------------------------------
# Engine pool — Phase 2: shared pool of loaded engine instances
# ---------------------------------------------------------------------------

@dataclass
class EngineSlot:
    """One loaded engine instance in the pool."""
    engine: Any  # HTREngine instance (not the registry singleton)
    engine_name: str
    config: dict
    pool_key: str
    ref_count: int = 0
    last_used: float = field(default_factory=time.time)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

engine_pool: Dict[str, EngineSlot] = {}
pool_lock = asyncio.Lock()

# VRAM budget estimates (GB) for eviction decisions
_ENGINE_VRAM_GB = {
    "CRNN-CTC (PyLaia-inspired)": 2,
    "TrOCR": 3,
    "Qwen3-VL": 18,
    "Churro VLM": 10,
    "Kraken": 2,
    "Party": 4,
    "PaddleOCR": 2,
    "LapaOCR": 18,
    "PaddleOCR-VL": 9,
}
_NO_GPU_ENGINES = {"Commercial APIs", "OpenWebUI", "LightOnOCR", "DeepSeek-OCR"}
_TOTAL_VRAM_GB = 92  # 2x L40S @ 46GB each


# Factory: engine name -> (module, class) for creating fresh instances
_ENGINE_FACTORY = {
    "TrOCR":                       ("engines.trocr_engine",        "TrOCREngine"),
    "CRNN-CTC (PyLaia-inspired)":  ("engines.pylaia_engine",       "PyLaiaEngine"),
    "Qwen3-VL":                    ("engines.qwen3_engine",        "Qwen3Engine"),
    "Churro VLM":                   ("engines.churro_engine",       "ChurroEngine"),
    "Kraken":                       ("engines.kraken_engine",       "KrakenEngine"),
    "Commercial APIs":              ("engines.commercial_api_engine", "CommercialAPIEngine"),
    "Party":                        ("engines.party_engine",        "PartyEngine"),
    "OpenWebUI":                    ("engines.openwebui_engine",    "OpenWebUIEngine"),
    "DeepSeek-OCR":                 ("engines.deepseek_ocr_engine", "DeepSeekOCREngine"),
    "LightOnOCR":                   ("engines.lighton_ocr_engine",  "LightOnOCREngine"),
    "LapaOCR":                      ("engines.lapa_ocr_engine",     "LapaOCREngine"),
    "PaddleOCR":                    ("engines.paddle_engine",       "PaddleOCREngine"),
    "PaddleOCR-VL":                 ("engines.paddle_vl_engine",    "PaddleOCRVLEngine"),
}


def _create_engine_instance(engine_name: str):
    """Create a fresh engine instance (not the registry singleton).

    The registry is used for discovery/availability only.
    Pool slots get their own instances so multiple models can coexist.
    """
    entry = _ENGINE_FACTORY.get(engine_name)
    if not entry:
        return None
    module_name, class_name = entry
    mod = importlib.import_module(module_name)
    cls = getattr(mod, class_name)
    return cls()


def _make_pool_key(engine_name: str, config: dict) -> str:
    """Build a key that uniquely identifies an engine+model combination."""
    if engine_name == "Commercial APIs":
        provider = config.get("provider", "unknown")
        model = config.get("model", "unknown")
        api_key = config.get("api_key", "")
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()[:8] if api_key else "nokey"
        return f"{engine_name}::{provider}::{model}::{key_hash}"

    if engine_name == "OpenWebUI":
        model = config.get("model", "unknown")
        api_key = config.get("api_key", "")
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()[:8] if api_key else "nokey"
        return f"{engine_name}::{model}::{key_hash}"

    if engine_name == "TrOCR":
        return f"{engine_name}::{config.get('model_path', 'default')}"

    if engine_name in ("CRNN-CTC (PyLaia-inspired)", "Kraken"):
        return f"{engine_name}::{config.get('model_path', 'default')}"

    if engine_name == "Qwen3-VL":
        base = config.get("base_model", "default")
        adapter = config.get("adapter", "")
        return f"{engine_name}::{base}::{adapter or 'none'}"

    if engine_name == "Churro VLM":
        return f"{engine_name}::{config.get('model_name', 'default')}"

    if engine_name == "LightOnOCR":
        return f"{engine_name}::{config.get('model_path', 'default')}"

    if engine_name == "LapaOCR":
        base_model = config.get("base_model") or config.get("model_id") or "default"
        adapter = config.get("adapter", "")
        quant = config.get("quantization", "none")
        return f"{engine_name}::{base_model}::{adapter or 'none'}::{quant}"

    # Fallback: hash the config
    config_hash = hashlib.sha256(str(sorted(config.items())).encode()).hexdigest()[:12]
    return f"{engine_name}::{config_hash}"


async def _maybe_evict(new_engine_name: str):
    """Evict LRU slots with ref_count==0 if VRAM is tight. Called UNDER pool_lock."""
    if new_engine_name in _NO_GPU_ENGINES:
        return
    needed = _ENGINE_VRAM_GB.get(new_engine_name, 4)
    used = sum(_ENGINE_VRAM_GB.get(s.engine_name, 4)
               for s in engine_pool.values()
               if s.engine_name not in _NO_GPU_ENGINES)
    if used + needed <= _TOTAL_VRAM_GB:
        return
    # Evict: ref_count==0, oldest first
    candidates = sorted(
        [(k, s) for k, s in engine_pool.items()
         if s.ref_count == 0 and s.engine_name not in _NO_GPU_ENGINES],
        key=lambda x: x[1].last_used
    )
    for key, slot in candidates:
        if used + needed <= _TOTAL_VRAM_GB:
            break
        log.info(f"Evicting engine slot '{key}' (last used {time.time() - slot.last_used:.0f}s ago)")
        try:
            slot.engine.unload_model()
        except Exception as e:
            log.warning(f"Error unloading evicted engine: {e}")
        del engine_pool[key]
        used -= _ENGINE_VRAM_GB.get(slot.engine_name, 4)
    if used + needed > _TOTAL_VRAM_GB:
        log.warning(f"VRAM tight: ~{used}GB used + ~{needed}GB needed > {_TOTAL_VRAM_GB}GB total")


# Compatibility shims — will be removed after full migration
loaded_engine: Optional[HTREngine] = None
loaded_engine_name: str = ""
loaded_config: dict = {}

# Persistent upload storage (survives server restarts)
UPLOAD_DIR = Path(__file__).parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# Upload TTL: 24 hours
_UPLOAD_TTL_SECONDS = 86400

# Session TTL: 2 hours of inactivity
_SESSION_TTL_SECONDS = 7200

# Cookie name for session tracking
# Cookie name is configurable so multiple instances on the same host (e.g. a
# long-running server plus a dev/test instance on another port) don't clobber
# each other's session cookie — browser cookies are scoped per host, not per
# port. Default is unchanged for existing deployments.
_SESSION_COOKIE = os.environ.get("POLYSCRIPTOR_SESSION_COOKIE", "polyscriptor_session")

# API key layer (Phase B1) — opt-in identity for programmatic clients.
# Without a configured key file the registry is disabled and requests are
# handled exactly as before. See web/api_auth.py and API_MULTIUSER_ROADMAP.md.
try:
    from web.api_auth import load_registry_from_env
except ImportError:  # server started with web/ as CWD on sys.path
    from api_auth import load_registry_from_env

api_key_registry = load_registry_from_env()

# Upload pixel cap (Phase B2). Legitimate manuscript facsimiles reach ~180 MP,
# so the cap is generous — it only stops pathological decompression bombs.
_MAX_UPLOAD_PIXELS = int(os.environ.get("POLYSCRIPTOR_MAX_UPLOAD_PIXELS", 600_000_000))


# ---------------------------------------------------------------------------
# Per-user sessions — Phase 1 of multi-user refactoring
# ---------------------------------------------------------------------------

@dataclass
class UserSession:
    session_id: str
    image_cache: Dict[str, dict] = field(default_factory=dict)
    cancel_events: Dict[str, asyncio.Event] = field(default_factory=dict)
    pool_key: Optional[str] = None  # Reference into engine_pool
    comparison_pool_keys: Dict[str, str] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)


sessions: Dict[str, UserSession] = {}


def _get_or_create_session(session_id: Optional[str]) -> tuple[UserSession, bool]:
    """Return (session, created). If session_id is missing/unknown, create a new one."""
    if session_id and session_id in sessions:
        session = sessions[session_id]
        session.last_active = time.time()
        return session, False
    new_id = str(uuid.uuid4())
    session = UserSession(session_id=new_id)
    sessions[new_id] = session
    return session, True


def _cleanup_expired_sessions() -> int:
    """Remove sessions inactive for more than _SESSION_TTL_SECONDS. Returns count removed."""
    cutoff = time.time() - _SESSION_TTL_SECONDS
    expired = [sid for sid, s in sessions.items() if s.last_active < cutoff]
    for sid in expired:
        session = sessions.pop(sid)
        # Release pool references (primary + comparison)
        for pool_key in _iter_session_pool_keys(session):
            _release_pool_reference(pool_key, reason="session expiry")
        # Clean up upload files belonging to this session
        for iid, img_data in session.image_cache.items():
            p = img_data.get("path")
            if p:
                Path(p).unlink(missing_ok=True)
            xp = img_data.get("xml_path")
            if xp:
                Path(xp).unlink(missing_ok=True)
        log.info(f"Expired session {sid[:8]}... ({len(session.image_cache)} images)")
    return len(expired)


_SESSION_PASSTHROUGH_PATHS = {"/api/gpu", "/api/engines", "/api/kraken/presets", "/api/activity"}


@app.middleware("http")
async def session_middleware(request: Request, call_next):
    """Inject session into request.state; set session cookie on new sessions.

    Pure status/discovery routes (GPU poll, engine list) are excluded from
    last_active updates so that background browser polling cannot keep a session
    alive indefinitely and prevent engine-slot eviction.
    """
    # API key check (Phase B1): only rejects when a key is PRESENTED but
    # invalid. No header → anonymous browser session, unchanged behavior.
    api_user = None
    raw_key = request.headers.get("X-API-Key")
    if raw_key and api_key_registry.enabled:
        api_user = api_key_registry.verify(raw_key)
        if api_user is None:
            return JSONResponse(status_code=401, content={"detail": "Invalid API key"})
    request.state.api_user = api_user

    session_id = request.cookies.get(_SESSION_COOKIE)
    session, created = _get_or_create_session(session_id)
    request.state.session = session

    # Don't update last_active for polling-only routes
    if request.url.path in _SESSION_PASSTHROUGH_PATHS:
        session.last_active  # read only — no write
    else:
        session.last_active = time.time()

    response = await call_next(request)

    if api_user is not None and request.url.path.startswith("/api/"):
        api_key_registry.log_usage(api_user, request.method, request.url.path, response.status_code)

    if created or session_id != session.session_id:
        response.set_cookie(
            key=_SESSION_COOKIE,
            value=session.session_id,
            httponly=True,
            samesite="lax",
            max_age=_SESSION_TTL_SECONDS,
        )
    return response


def _get_session(request: Request) -> UserSession:
    """FastAPI dependency: extract session set by middleware."""
    return request.state.session


def _check_admin(request: Request, legacy_localhost_only: bool) -> None:
    """Guard for dangerous endpoints (Phase B2).

    With key auth enabled, only a valid admin key passes — the localhost-IP
    check is retired because it is unreliable behind a reverse proxy
    (request.client.host becomes the proxy IP). Without a key file, each
    endpoint keeps its exact legacy behavior.
    """
    if api_key_registry.enabled:
        user = getattr(request.state, "api_user", None)
        if user is None or not user.is_admin:
            raise HTTPException(status_code=403, detail="Admin API key required (X-API-Key)")
    elif legacy_localhost_only:
        if request.client and request.client.host not in ("127.0.0.1", "::1"):
            raise HTTPException(status_code=403, detail="localhost only")


def _cleanup_old_uploads() -> int:
    """Delete uploads older than TTL and evict image_cache entries across all sessions."""
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
    # Evict stale image_cache entries whose file no longer exists (all sessions)
    for session in sessions.values():
        for iid in list(session.image_cache.keys()):
            p = session.image_cache[iid].get("path")
            if p and not Path(p).exists():
                del session.image_cache[iid]
        _prune_stale_comparison_pool_references(session)
    return deleted


_SLOT_IDLE_TTL_SECONDS = 6 * 3600  # evict loaded engines idle for 6h, regardless of ref_count


def _evict_idle_slots() -> int:
    """Evict engine slots that have not been used for _SLOT_IDLE_TTL_SECONDS.

    Called under no lock — must only be called from _periodic_cleanup (single-threaded).
    The GPU-status poll (/api/gpu) keeps sessions alive indefinitely, so we cannot rely
    on session expiry alone to release VRAM. This independently caps engine residency.
    """
    cutoff = time.time() - _SLOT_IDLE_TTL_SECONDS
    stale = [k for k, s in engine_pool.items() if s.last_used < cutoff
             and s.engine_name not in _NO_GPU_ENGINES]
    for key in stale:
        slot = engine_pool.pop(key)
        log.info(f"Idle eviction: '{slot.engine_name}' (idle {(time.time() - slot.last_used)/3600:.1f}h)")
        try:
            slot.engine.unload_model()
        except Exception as e:
            log.warning(f"unload_model() failed for '{slot.engine_name}': {e}")
        # Invalidate all sessions pointing at this slot
        for session in sessions.values():
            if session.pool_key == key:
                session.pool_key = None
            for slot_label, pool_key in list(session.comparison_pool_keys.items()):
                if pool_key == key:
                    del session.comparison_pool_keys[slot_label]
    return len(stale)


def _iter_session_pool_keys(session: UserSession) -> List[str]:
    """Return all pool keys referenced by this session, including duplicates."""
    refs: List[str] = []
    if session.pool_key:
        refs.append(session.pool_key)
    refs.extend(session.comparison_pool_keys.values())
    return refs


def _release_pool_reference(pool_key: Optional[str], reason: str) -> None:
    """Decrement a slot ref_count and evict immediately when it reaches zero."""
    if not pool_key or pool_key not in engine_pool:
        return
    slot = engine_pool[pool_key]
    slot.ref_count = max(0, slot.ref_count - 1)
    if slot.ref_count == 0:
        log.info(f"Immediate eviction ({reason}): '{slot.engine_name}'")
        try:
            slot.engine.unload_model()
        except Exception as e:
            log.warning(f"unload_model() failed for '{slot.engine_name}': {e}")
        if pool_key in engine_pool:
            del engine_pool[pool_key]


def _attach_comparison_pool_reference(session: UserSession, slot_label: str, pool_key: Optional[str]) -> None:
    """Track a loaded engine slot for comparison use in the current session."""
    if not pool_key:
        return

    previous = session.comparison_pool_keys.get(slot_label)
    if previous == pool_key:
        return

    if previous:
        _release_pool_reference(previous, reason="comparison slot switch")

    if pool_key in engine_pool:
        engine_pool[pool_key].ref_count += 1
        engine_pool[pool_key].last_used = time.time()
    session.comparison_pool_keys[slot_label] = pool_key


def _release_all_comparison_pool_references(session: UserSession, reason: str) -> None:
    """Release every comparison-slot reference held by the session."""
    for slot_label, pool_key in list(session.comparison_pool_keys.items()):
        _release_pool_reference(pool_key, reason=reason)
        session.comparison_pool_keys.pop(slot_label, None)


def _prune_stale_comparison_pool_references(session: UserSession) -> None:
    """Release comparison references whose stored result slots no longer exist."""
    active_slot_ids = set()
    for img_data in session.image_cache.values():
        for slot_id, slot in (img_data.get("result_slots") or {}).items():
            if slot.get("kind") == "comparison" and slot.get("pool_key"):
                active_slot_ids.add(slot_id)

    for slot_label, pool_key in list(session.comparison_pool_keys.items()):
        if slot_label not in active_slot_ids:
            _release_pool_reference(pool_key, reason="stale comparison cleanup")
            session.comparison_pool_keys.pop(slot_label, None)


async def _periodic_cleanup():
    """Background task: clean up uploads + expired sessions + idle engine slots every hour."""
    while True:
        await asyncio.sleep(3600)
        n = _cleanup_old_uploads()
        m = _cleanup_expired_sessions()
        p = _evict_idle_slots()
        q = _cleanup_finished_jobs()
        if n or m or p or q:
            log.info(f"Periodic cleanup: {n} upload(s), {m} session(s), {p} idle engine slot(s), {q} old job(s).")


# ---------------------------------------------------------------------------
# API key resolution — keys never stored or shared server-side (Phase 3)
# Web UI users MUST provide their own keys via browser localStorage.
# Server env vars (.env) are NOT used by the web UI — they exist only for
# the PyQt GUI and CLI tools which run locally on the admin's machine.
# ---------------------------------------------------------------------------

# Known key slots (for validation only — env vars are NOT consulted)
_KEY_SLOTS = {"openai", "gemini", "claude", "openwebui"}


def _resolve_api_key(slot: str, request_value: str) -> str:
    """
    Return the API key from the browser request, or empty string.
    Server env vars are deliberately NOT used as fallback — each web user
    must supply their own key via browser localStorage.
    """
    if request_value and request_value.strip():
        return request_value.strip()
    return ""


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
    """Clean old uploads, start periodic cleanup, auto-load engine."""
    # Clean up uploads left over from previous server runs
    n = _cleanup_old_uploads()
    if n:
        log.info(f"Startup cleanup: removed {n} old upload file(s).")

    # Schedule periodic cleanup (every hour)
    asyncio.create_task(_periodic_cleanup())

    # Start the batch-job worker (Phase B3) — serial, yields to interactive UI
    asyncio.create_task(_job_worker())

    # Auto-load default engine from server_config.yaml if present
    cfg = _load_startup_config()
    if not cfg.get("default_engine"):
        return
    engine_name = cfg["default_engine"]
    engine_config = cfg.get("default_config", {})
    log.info(f"Auto-loading engine '{engine_name}' from server_config.yaml ...")
    try:
        registry = get_global_registry()
        reg_engine = registry.get_engine_by_name(engine_name)
        if reg_engine and reg_engine.is_available():
            engine = _create_engine_instance(engine_name)
            if not engine:
                log.warning(f"Auto-load: cannot create instance for '{engine_name}'.")
                return
            ok = await asyncio.to_thread(engine.load_model, engine_config)
            if ok:
                pool_key = _make_pool_key(engine_name, engine_config)
                slot = EngineSlot(
                    engine=engine, engine_name=engine_name,
                    config=engine_config, pool_key=pool_key,
                    ref_count=0,  # No session owns it yet
                )
                engine_pool[pool_key] = slot
                # Update compat shims
                global loaded_engine, loaded_engine_name, loaded_config
                loaded_engine = engine
                loaded_engine_name = engine_name
                loaded_config = engine_config
                log.info(f"Auto-loaded '{engine_name}' into pool as '{pool_key}'.")
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
    from inference_pylaia_native import _scan_pylaia_models
    _scan_pylaia_models(str(Path(__file__).resolve().parents[1] / "models"))
    options = [{"label": k, "value": k} for k in PYLAIA_MODELS.keys()]
    options.append({"label": "Custom / local path…", "value": "__custom__"})
    return options


def _scan_kraken_models() -> list:
    """Scan models/ directory for local Kraken .mlmodel files and build select options."""
    options = []
    models_root = Path(__file__).resolve().parents[1] / "models"
    if models_root.exists():
        for p in sorted(models_root.rglob("*.mlmodel")):
            rel = str(p.relative_to(models_root.parent))  # e.g. models/kraken_cs/best.mlmodel
            label = f"{p.parent.name}/{p.name}"
            options.append({"label": label, "value": rel, "source": "local"})
    # Zenodo presets from kraken_engine (auto-download on load)
    try:
        from engines.kraken_engine import KRAKEN_MODELS
        for preset_id, info in KRAKEN_MODELS.items():
            if info.get("source") == "zenodo":
                options.append({
                    "label": f"{info.get('label', preset_id)} [Zenodo, auto-download]",
                    "value": f"__zenodo__{preset_id}",
                    "source": "zenodo",
                })
    except Exception:
        pass
    return options


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
        {"label": "Custom HuggingFace ID or local path…", "value": "__custom__"},
        {"label": "kazars24/trocr-base-handwritten-ru (HuggingFace)",
         "value": "kazars24/trocr-base-handwritten-ru",
         "source": "huggingface"},
        {"label": "microsoft/trocr-base-printed — printed text, base",
         "value": "microsoft/trocr-base-printed",
         "source": "huggingface"},
        {"label": "microsoft/trocr-large-printed — printed text, large",
         "value": "microsoft/trocr-large-printed",
         "source": "huggingface"},
        {"label": "dh-unibe/trocr-kurrent — German Kurrent 19th c. (CER 2.66%)",
         "value": "dh-unibe/trocr-kurrent",
         "source": "huggingface"},
        {"label": "dh-unibe/trocr-kurrent-XVI-XVII — German Kurrent 16th–18th c. (CER 5.42%)",
         "value": "dh-unibe/trocr-kurrent-XVI-XVII",
         "source": "huggingface"},
    ]
    if models_dir.exists():
        for d in sorted(models_dir.iterdir()):
            if not d.is_dir():
                continue
            # Require a ViT/TrOCR image-processor config AND config.json with
            # model_type == 'vision-encoder-decoder'.
            # transformers <5 writes preprocessor_config.json; transformers >=5
            # writes a combined processor_config.json instead. Accept either.
            # Both are ViT/TrOCR-specific (not in PyLaia).
            # config.json model_type disambiguates from Qwen3 adapters that
            # also ship a preprocessor_config but have no config.json.
            if not ((d / "preprocessor_config.json").exists()
                    or (d / "processor_config.json").exists()):
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
             "options": _get_pylaia_model_options(),
             "custom_key": "custom_model_path",
             "custom_placeholder": "Absolute path to best_model.pt (e.g. /home/…/models/pylaia_yiddish_20260326/best_model.pt)"},
            {"key": "enable_spaces", "type": "checkbox",
             "label": "Convert <space> tokens", "default": True},
            {"key": "flip_rtl", "type": "checkbox",
             "label": "RTL manuscript (flip line images)", "default": False,
             "hint": "Flip line images horizontally for RTL scripts (Ottoman, Arabic, Hebrew)"},
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
            {"key": "flip_rtl", "type": "checkbox",
             "label": "RTL manuscript (flip line images)", "default": False,
             "hint": "Flip line images horizontally for RTL scripts (Ottoman, Arabic, Hebrew)"},
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
            {"key": "model_path", "type": "select", "label": "Model",
             "options": _scan_kraken_models(),
             "custom_key": "custom_model_path",
             "custom_placeholder": "Absolute path on server, e.g. /home/user/models/my.mlmodel",
             "upload": True},
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
            {"key": "custom_prompt", "type": "textarea", "label": "Custom Prompt (optional)",
             "default": "",
             "rows": 4,
             "placeholder": "Transcribe all handwritten text in this manuscript image. Preserve the original language (Cyrillic, Latin, etc.) and layout. Output only the transcribed text without any additional commentary.",
             "hint": "Leave blank to use the default prompt shown above"},
            {"key": "thinking_mode", "type": "select", "label": "Thinking Mode (Gemini only)",
             "options": [
                 {"label": "Auto (model decides, no cap)", "value": ""},
                 {"label": "Low (budget: 8k tokens)", "value": "low"},
                 {"label": "High (no cap, max reasoning)", "value": "high"},
             ], "default": ""},
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
            {"key": "max_tokens", "type": "number", "label": "Max output tokens (optional)",
             "min": 512, "max": 65536, "default": None,
             "placeholder": "Leave blank = model maximum"},
            {"key": "custom_prompt", "type": "textarea", "label": "Custom Prompt (optional)",
             "default": "",
             "rows": 3,
             "placeholder": "Transcribe all handwritten text in this manuscript image. Preserve the original language (Cyrillic, Latin, etc.) and layout. Output only the transcribed text without any additional commentary.",
             "hint": "Leave blank to use the default prompt shown above"},
        ]
    },
    "LightOnOCR": lambda: {
        "fields": [
            {"key": "model_path", "type": "select", "label": "Model",
             "options": (lambda: [
                 {"label": f"{name} — {info.get('description','')}", "value": info["id"]}
                 for name, info in __import__('lighton_models', fromlist=['LIGHTON_MODELS']).LIGHTON_MODELS.items()
             ] + [{"label": "Custom HuggingFace ID…", "value": "__custom__"}])(),
             "custom_key": "custom_model_path",
             "custom_placeholder": "HuggingFace model ID, e.g. lightonai/LightOnOCR-2-1B-base"},
            {"key": "max_new_tokens", "type": "number", "label": "Max new tokens",
             "min": 32, "max": 512, "default": 128},
        ]
    },
    "LapaOCR": lambda: {
        "fields": [
            {"key": "base_model", "type": "text", "label": "Base model",
             "default": "lapa-llm/lapa-v0.1.2-instruct",
             "placeholder": "HuggingFace base model ID"},
            {"key": "adapter", "type": "text", "label": "LoRA adapter",
             "default": "VmF0x/lapa-ocr-lora",
             "placeholder": "Adapter model ID or local adapter path"},
            {"key": "quantization", "type": "select", "label": "Quantization",
             "default": "none",
             "options": [
                 {"label": "none (best quality, high VRAM)", "value": "none"},
                 {"label": "8bit (lower VRAM)", "value": "8bit"},
                 {"label": "4bit (lowest VRAM)", "value": "4bit"},
             ]},
            {"key": "max_new_tokens", "type": "number", "label": "Max new tokens",
               "min": 64, "max": 1024, "default": 128,
               "hint": "Lower values improve latency and reduce runaway generation."},
              {"key": "max_time_s", "type": "number", "label": "Max generation time per line (s)",
               "min": 5, "max": 600, "default": 90,
               "hint": "Safety cap for one line. Prevents very long hangs during generation."},
            {"key": "prompt", "type": "textarea", "label": "Prompt",
             "default": "Transcribe Ukrainian text literally. Output only the text, no preamble.",
             "rows": 3},
        ]
    },
    "PaddleOCR": lambda: {
        "fields": [
            {"key": "lang", "type": "select", "label": "Language / Script",
             "default": "ch",
             "options": [
                 {"label": "Chinese + English (mixed, recommended default)",  "value": "ch"},
                 {"label": "English",                                          "value": "en"},
                 {"label": "German",                                           "value": "german"},
                 {"label": "French",                                           "value": "french"},
                 {"label": "Japanese",                                         "value": "japan"},
                 {"label": "Korean",                                           "value": "korean"},
                 {"label": "Arabic",                                           "value": "arabic"},
                 {"label": "Cyrillic (Russian/Ukrainian/Bulgarian)",           "value": "cyrillic"},
                 {"label": "Latin script (generic)",                           "value": "latin"},
                 {"label": "Custom (enter code below)",                        "value": "__custom__"},
             ],
             "custom_key": "custom_lang",
             "custom_placeholder": "PaddleOCR lang code, e.g. ru, uk, fr, es, it, pt, …",
             "hint": "One language model per run. 'ch' is bilingual (Chinese+English) and PaddleOCR's strongest model. For mixed-script documents outside this list, run separate passes."},
            {"key": "use_angle_cls", "type": "checkbox",
             "label": "Text-angle classifier (correct 180° rotation)", "default": True},
            {"key": "use_gpu", "type": "checkbox",
             "label": "Use GPU (requires paddlepaddle-gpu)", "default": False},
        ]
    },
    "PaddleOCR-VL": lambda: {
        "fields": [
            {"key": "pipeline_version", "type": "select", "label": "Model version",
             "default": "v1.5",
             "options": [
                 {"label": "v1.5 (benchmarked / published)", "value": "v1.5"},
                 {"label": "v1.6 (newest)",                  "value": "v1.6"},
             ],
             "hint": "0.9B vision-language document parser. Excellent for Chinese and printed/multilingual documents. NOT suitable for Cyrillic/Slavic handwriting (no support — use TrOCR / CRNN-CTC there)."},
            {"key": "use_gpu", "type": "checkbox",
             "label": "Use GPU (strongly recommended)", "default": True},
            {"key": "gpu_index", "type": "number", "label": "GPU index", "default": 0,
             "hint": "Physical GPU pinned via CUDA_VISIBLE_DEVICES. Dense full pages are slow on the native backend; per-line is fast (~1.4s)."},
            {"key": "prompt_label", "type": "text", "label": "Task label (advanced)", "default": "",
             "hint": "Force a recognition task instead of auto-detection. Leave empty for Auto. Advanced: e.g. 'text', 'table', 'formula' — wrong values may error."},
            {"key": "max_new_tokens", "type": "number", "label": "Max new tokens", "default": 0,
             "hint": "0 = pipeline default. Cap generation length to curb runaway repetition on dense pages."},
            {"key": "repetition_penalty", "type": "number", "label": "Repetition penalty", "default": 0,
             "hint": "0 = pipeline default. >1.0 (e.g. 1.05–1.2) discourages repeated tokens."},
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
    text_direction: str = "horizontal-lr"  # reading order for Kraken: horizontal-lr, horizontal-rl, vertical-lr, vertical-rl
    engine_config_overrides: Dict[str, Any] = {}  # live form values merged into stored config at transcription time


class CompareRunRequest(BaseModel):
    image_id: str
    engine_config_overrides: Dict[str, Any] = {}
    label: Optional[str] = None


class GroundTruthCompareRequest(BaseModel):
    image_id: str
    slot_id: Optional[str] = None  # score this slot only; None = score every slot


# ---------------------------------------------------------------------------
# Interactive-priority gate + activity registry + job queue (Phase B3 + C)
# ---------------------------------------------------------------------------
#
# Interactive SSE transcriptions (web UI) always take priority over queued
# batch jobs: the job worker pauses between lines while any interactive
# transcription is running. Jobs run serially through a single worker task.

_interactive_active = 0
_interactive_idle = asyncio.Event()
_interactive_idle.set()


def _interactive_begin() -> None:
    global _interactive_active
    _interactive_active += 1
    _interactive_idle.clear()


def _interactive_end() -> None:
    global _interactive_active
    _interactive_active = max(0, _interactive_active - 1)
    if _interactive_active == 0:
        _interactive_idle.set()


# Activity registry (Phase C): what is computing right now, visible to everyone.
_activity: Dict[str, dict] = {}


def _request_actor(request: Request) -> str:
    """Display name for the activity panel: key user or anonymized session."""
    api_user = getattr(request.state, "api_user", None)
    if api_user is not None:
        return api_user.name
    session = getattr(request.state, "session", None)
    return f"Gast-{session.session_id[:8]}" if session else "Gast"


def _activity_register(entry_id: str, kind: str, who: str, engine: str) -> None:
    _activity[entry_id] = {
        "kind": kind,  # transcribe | compare | job
        "who": who,
        "engine": engine,
        "current": 0,
        "total": 0,
        "started": time.time(),
    }


def _activity_update(entry_id: str, current: int, total: int) -> None:
    entry = _activity.get(entry_id)
    if entry:
        entry["current"] = current
        entry["total"] = total


def _activity_remove(entry_id: str) -> None:
    _activity.pop(entry_id, None)


@dataclass
class TranscriptionJob:
    """One queued batch transcription (Phase B3). Runs against the owner's
    session state: the image must be uploaded and an engine loaded before
    submitting; the job uses whatever engine the session has at run time."""
    job_id: str
    owner: str          # "key:<name>" or "session:<id>"
    display_name: str   # key user name or Gast-<sid8>
    session_id: str
    params: TranscribeRequest
    status: str = "queued"  # queued | running | done | error | cancelled
    created: float = field(default_factory=time.time)
    started: Optional[float] = None
    finished: Optional[float] = None
    current: int = 0
    total: int = 0
    results: Optional[List[dict]] = None
    error: Optional[str] = None
    engine_name: str = ""
    cancel_evt: asyncio.Event = field(default_factory=asyncio.Event)


jobs: Dict[str, TranscriptionJob] = {}
job_queue: asyncio.Queue = asyncio.Queue()
_JOB_RETENTION_SECONDS = 24 * 3600
_ANON_MAX_JOBS = 2  # per-session job cap when no API key is presented

# Daily page quota bookkeeping: (owner, "YYYY-MM-DD") -> pages transcribed
_job_pages_today: Dict[tuple, int] = {}


def _pages_used_today(owner: str) -> int:
    return _job_pages_today.get((owner, time.strftime("%Y-%m-%d")), 0)


def _count_job_pages(owner: str, pages: int = 1) -> None:
    today = time.strftime("%Y-%m-%d")
    # prune other days so the dict cannot grow unbounded
    for key in [k for k in _job_pages_today if k[1] != today]:
        del _job_pages_today[key]
    _job_pages_today[(owner, today)] = _job_pages_today.get((owner, today), 0) + pages


def _serialize_job(job: TranscriptionJob, include_results: bool = True) -> dict:
    out = {
        "job_id": job.job_id,
        "status": job.status,
        "created": job.created,
        "started": job.started,
        "finished": job.finished,
        "progress": {"current": job.current, "total": job.total},
        "engine": job.engine_name,
        "error": job.error,
    }
    if include_results and job.status == "done":
        out["lines"] = job.results
    return out


async def _run_job(job: TranscriptionJob) -> None:
    if job.status == "cancelled":
        return
    job.status = "running"
    job.started = time.time()
    _activity_register(job.job_id, "job", job.display_name, "")
    try:
        session = sessions.get(job.session_id)
        if session is None:
            raise RuntimeError("Session expired before the job ran — re-upload and resubmit")
        img_data = session.image_cache.get(job.params.image_id)
        if img_data is None:
            raise RuntimeError("Image no longer cached — re-upload and resubmit")

        slot, eff_engine, eff_config, eff_engine_name, _pool_key = _resolve_effective_engine(
            session, job.params.engine_config_overrides,
        )
        job.engine_name = eff_engine_name
        _activity[job.job_id]["engine"] = eff_engine_name

        _import_segmenters()
        pil_image = img_data["pil_image"]
        req = job.params
        xml_path = img_data.get("xml_path") if req.use_pagexml else None

        # Segmentation — same decision tree as /api/transcribe
        if not eff_engine.requires_line_segmentation() and not xml_path and not img_data.get("lines"):
            from inference_page import LineSegment
            lines = [LineSegment(image=pil_image,
                                 bbox=(0, 0, pil_image.width, pil_image.height),
                                 coords=None)]
            img_data["lines"] = lines
            img_data["line_regions"] = [0]
            img_data["seg_source"] = "page"
        else:
            cached_lines = img_data.get("lines")
            desired_source = "pagexml" if xml_path else req.seg_method
            if not (cached_lines and img_data.get("seg_source") == desired_source):
                await _run_segmentation(img_data, "pagexml" if xml_path else req.seg_method,
                                        req.seg_device, req.max_columns,
                                        req.split_width_fraction, req.text_direction)
            lines = img_data["lines"]

        line_regions = img_data.get("line_regions") or ([0] * len(lines))
        if eff_engine_name == "LapaOCR" and not eff_config.get("max_time_s"):
            eff_config = dict(eff_config)
            eff_config["max_time_s"] = 90

        results = []
        job.total = len(lines)
        for i, line in enumerate(lines):
            # Interactive web-UI requests take priority: pause between lines
            await _interactive_idle.wait()
            if job.cancel_evt.is_set():
                job.status = "cancelled"
                return

            line_img = line.image if line.image is not None else pil_image.crop(line.bbox)
            img_array = np.array(line_img.convert("RGB"))
            if slot:
                async with slot.lock:
                    slot.last_used = time.time()
                    result = await asyncio.to_thread(eff_engine.transcribe_line, img_array, eff_config)
            else:
                result = await asyncio.to_thread(eff_engine.transcribe_line, img_array, eff_config)

            text = str(result.text) if hasattr(result, "text") else str(result)
            confidence = None
            if hasattr(result, "confidence") and result.confidence is not None:
                confidence = float(result.confidence)
                if confidence > 1:
                    confidence = confidence / 100.0
            results.append({
                "index": i,
                "text": text,
                "confidence": confidence,
                "bbox": list(line.bbox),
                "region": line_regions[i] if i < len(line_regions) else 0,
            })
            job.current = i + 1
            _activity_update(job.job_id, i + 1, len(lines))

        job.results = results
        job.status = "done"
        _count_job_pages(job.owner)
    except HTTPException as e:
        job.status = "error"
        job.error = str(e.detail)
    except Exception as e:
        log.exception(f"Job {job.job_id} failed")
        job.status = "error"
        job.error = str(e)
    finally:
        job.finished = time.time()
        _activity_remove(job.job_id)


async def _job_worker() -> None:
    """Single worker: jobs run strictly serially, yielding to interactive use."""
    while True:
        job = await job_queue.get()
        try:
            await _run_job(job)
        except Exception:
            log.exception("Job worker: unexpected error")
        finally:
            job_queue.task_done()


def _cleanup_finished_jobs() -> int:
    cutoff = time.time() - _JOB_RETENTION_SECONDS
    stale = [jid for jid, j in jobs.items()
             if j.status in ("done", "error", "cancelled") and (j.finished or j.created) < cutoff]
    for jid in stale:
        del jobs[jid]
    return len(stale)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post("/api/v1/jobs", status_code=202)
async def create_job(request: Request, req: TranscribeRequest):
    """Queue a batch transcription (Phase B3). Same session workflow as
    /api/transcribe (upload image + load engine first), but returns 202 with a
    job id for polling instead of holding an SSE stream open. Queued jobs run
    serially and always yield to interactive web-UI transcriptions."""
    session = _get_session(request)
    if req.image_id not in session.image_cache:
        raise HTTPException(404, "Image not found — upload first")

    api_user = getattr(request.state, "api_user", None)
    if api_user is not None:
        owner, display = f"key:{api_user.name}", api_user.name
        max_jobs, quota = api_user.max_jobs, api_user.daily_page_quota
    else:
        owner, display = f"session:{session.session_id}", f"Gast-{session.session_id[:8]}"
        max_jobs, quota = _ANON_MAX_JOBS, None

    active = sum(1 for j in jobs.values() if j.owner == owner and j.status in ("queued", "running"))
    if active >= max_jobs:
        raise HTTPException(429, f"Job limit reached ({max_jobs} queued/running jobs) — wait for jobs to finish")
    if quota is not None and _pages_used_today(owner) >= quota:
        raise HTTPException(429, f"Daily page quota reached ({quota} pages/day)")

    job = TranscriptionJob(job_id=str(uuid.uuid4()), owner=owner, display_name=display,
                           session_id=session.session_id, params=req)
    jobs[job.job_id] = job
    job_queue.put_nowait(job)
    return {"job_id": job.job_id, "status": "queued", "queue_position": job_queue.qsize()}


def _get_owned_job(request: Request, job_id: str) -> TranscriptionJob:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    owner_ids = set()
    api_user = getattr(request.state, "api_user", None)
    if api_user is not None:
        owner_ids.add(f"key:{api_user.name}")
    session = getattr(request.state, "session", None)
    if session is not None:
        owner_ids.add(f"session:{session.session_id}")
    if job.owner not in owner_ids:
        raise HTTPException(403, "Not your job")
    return job


@app.get("/api/v1/jobs")
async def list_jobs(request: Request):
    """List the caller's own jobs (newest first, without result payloads)."""
    api_user = getattr(request.state, "api_user", None)
    session = getattr(request.state, "session", None)
    owner_ids = set()
    if api_user is not None:
        owner_ids.add(f"key:{api_user.name}")
    if session is not None:
        owner_ids.add(f"session:{session.session_id}")
    own = [j for j in jobs.values() if j.owner in owner_ids]
    own.sort(key=lambda j: j.created, reverse=True)
    return {"jobs": [_serialize_job(j, include_results=False) for j in own]}


@app.get("/api/v1/jobs/{job_id}")
async def get_job(request: Request, job_id: str):
    return _serialize_job(_get_owned_job(request, job_id))


@app.delete("/api/v1/jobs/{job_id}")
async def cancel_job(request: Request, job_id: str):
    job = _get_owned_job(request, job_id)
    if job.status in ("done", "error", "cancelled"):
        return {"job_id": job.job_id, "status": job.status}
    job.cancel_evt.set()
    if job.status == "queued":
        job.status = "cancelled"
        job.finished = time.time()
    return {"job_id": job.job_id, "status": "cancelling" if job.status == "running" else job.status}


@app.get("/api/activity")
async def activity_status():
    """Who is computing right now (Phase C) — shown to all users so batch and
    interactive users can see each other and coordinate."""
    now = time.time()
    active = [
        {
            "kind": e["kind"],
            "who": e["who"],
            "engine": e["engine"],
            "current": e["current"],
            "total": e["total"],
            "running_s": round(now - e["started"]),
        }
        for e in sorted(_activity.values(), key=lambda e: e["started"])
    ]
    return {"active": active, "queued_jobs": sum(1 for j in jobs.values() if j.status == "queued")}

@app.get("/")
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/demo")
async def pwa_demo():
    return FileResponse(str(STATIC_DIR / "pwa" / "demo.html"))


@app.get("/manifest.json")
async def pwa_manifest():
    """Serve the PWA manifest from root so scope / start_url are valid."""
    from fastapi.responses import FileResponse as _FR
    return _FR(str(STATIC_DIR / "pwa" / "manifest.json"), media_type="application/manifest+json")


@app.get("/sw.js")
async def pwa_service_worker():
    """Serve the PWA service worker from root scope so it can control /demo."""
    from fastapi.responses import FileResponse as _FR
    resp = _FR(str(STATIC_DIR / "pwa" / "sw.js"), media_type="application/javascript")
    resp.headers["Service-Worker-Allowed"] = "/"
    return resp


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

    # Key status: always "missing" from server perspective — browser localStorage
    # is the only key store. The frontend checks localStorage client-side.
    for field in schema.get("fields", []):
        if field.get("type") == "password":
            field["key_status"] = "missing"

    return schema


@app.get("/api/engine/status")
async def engine_status(request: Request):
    session = _get_session(request)
    if session.pool_key and session.pool_key in engine_pool:
        slot = engine_pool[session.pool_key]
        return {
            "loaded": slot.engine.is_model_loaded(),
            "engine_name": slot.engine_name,
            "config": slot.config,
            "pool_key": session.pool_key,
        }
    # Fallback: compat shim for tests / startup
    return {
        "loaded": loaded_engine is not None and loaded_engine.is_model_loaded(),
        "engine_name": loaded_engine_name,
        "config": loaded_config,
        "pool_key": None,
    }


def _resolve_effective_engine(
    session: UserSession,
    engine_config_overrides: Optional[Dict[str, Any]] = None,
) -> tuple[Optional[EngineSlot], HTREngine, Dict[str, Any], str, Optional[str]]:
    """Resolve the currently loaded engine and apply safe live overrides."""
    if not session.pool_key or session.pool_key not in engine_pool:
        if not loaded_engine or not loaded_engine.is_model_loaded():
            raise HTTPException(400, "No engine loaded")
    slot = engine_pool.get(session.pool_key) if session.pool_key else None
    eff_engine = slot.engine if slot else loaded_engine
    base_config = slot.config if slot else loaded_config
    effective_pool_key = session.pool_key if slot else None

    reload_only_keys = {
        "api_key", "provider", "model", "model_path", "model_source",
        "base_model", "adapter", "model_name", "preset_id", "lang",
        "use_gpu", "venv_path",
    }
    if engine_config_overrides:
        eff_config = dict(base_config)
        for key, value in engine_config_overrides.items():
            if key not in reload_only_keys:
                eff_config[key] = value
    else:
        eff_config = base_config

    eff_engine_name = slot.engine_name if slot else loaded_engine_name
    if not eff_engine or not eff_engine.is_model_loaded():
        raise HTTPException(400, "No engine loaded")

    return slot, eff_engine, eff_config, eff_engine_name, effective_pool_key


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
            return {"models": [], "error": "No API key — paste one in the form"}
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
        if not resolved:
            return {"models": [], "error": "No API key — paste one in the form"}
        try:
            sys.path.insert(0, str(PROJECT_ROOT))
            if prov == "openai":
                from inference_commercial_api import fetch_openai_models
                models = await asyncio.to_thread(fetch_openai_models, resolved)
                return {"models": models}
            elif prov == "gemini":
                from inference_commercial_api import fetch_gemini_models
                models = await asyncio.to_thread(fetch_gemini_models, resolved)
                return {"models": models}
            elif prov == "claude":
                from inference_commercial_api import fetch_claude_models
                models = await asyncio.to_thread(fetch_claude_models, resolved)
                return {"models": models}
            else:
                return {"models": [], "error": f"Unknown provider: {provider}"}
        except Exception as e:
            return {"models": [], "error": str(e)}

    return {"models": [], "error": f"Dynamic model listing not supported for '{name}'"}


@app.post("/api/engine/load")
async def load_engine(request: Request, req: EngineLoadRequest):
    global loaded_engine, loaded_engine_name, loaded_config
    session = _get_session(request)

    registry = get_global_registry()
    reg_engine = registry.get_engine_by_name(req.engine_name)
    if not reg_engine:
        raise HTTPException(404, f"Engine '{req.engine_name}' not found")
    if not reg_engine.is_available():
        raise HTTPException(400, f"Engine not available: {reg_engine.get_unavailable_reason()}")

    # --- Config resolution (unchanged logic) ---
    config = dict(req.config)

    if req.engine_name == "CRNN-CTC (PyLaia-inspired)" and "model_path" in config:
        custom_val = config.pop("custom_model_path", "").strip()
        if config["model_path"] == "__custom__":
            if not custom_val:
                raise HTTPException(400, "Please enter an absolute path to a best_model.pt file")
            config["model_path"] = custom_val
        # else: named preset from PYLAIA_MODELS — engine resolves it

    elif req.engine_name == "Kraken" and "model_path" in config:
        custom_val = config.pop("custom_model_path", "").strip()
        val = config["model_path"]
        if val == "__custom__":
            if not custom_val:
                raise HTTPException(400, "Please enter a path to a local .mlmodel file")
            config["model_path"] = custom_val
        elif val.startswith("__zenodo__"):
            # Zenodo preset: pass preset_id, let engine handle download
            config["preset_id"] = val[len("__zenodo__"):]
            config["model_path"] = None
        # else: relative local path from select (e.g. "models/kraken_cs/best.mlmodel") — use as-is

    elif req.engine_name == "TrOCR" and "model_path" in config:
        custom_val = config.pop("custom_model_path", "").strip()
        if config["model_path"] == "__custom__":
            if not custom_val:
                raise HTTPException(400, "Please enter a HuggingFace model ID or local path")
            config["model_path"] = custom_val
        from pathlib import Path as _P
        if _P(config["model_path"]).exists():
            config["model_source"] = "local"
        else:
            config["model_source"] = "huggingface"

    elif req.engine_name == "Qwen3-VL" and "model_preset" in config:
        preset_val = config.pop("model_preset")
        custom_val = config.pop("base_model", "").strip()
        if preset_val == "__custom__":
            config["base_model"] = custom_val or "Qwen/Qwen3-VL-8B-Instruct"
            config["adapter"] = None
        else:
            vlm_opts = _scan_vlm_models("qwen3")
            matched = next((o for o in vlm_opts if o["value"] == preset_val), None)
            if matched:
                config["base_model"] = matched.get("base_model", preset_val)
                config["adapter"] = matched.get("adapter")
            else:
                config["base_model"] = preset_val
                config["adapter"] = None

    elif req.engine_name == "Churro VLM" and "model_preset" in config:
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

    elif req.engine_name == "LightOnOCR" and "model_path" in config:
        custom_val = config.pop("custom_model_path", "").strip()
        if config["model_path"] == "__custom__":
            if not custom_val:
                raise HTTPException(400, "Please enter a HuggingFace model ID for LightOnOCR")
            config["model_path"] = custom_val

    elif req.engine_name == "PaddleOCR" and "lang" in config:
        if config["lang"] == "__custom__":
            custom_lang = config.pop("custom_lang", "").strip()
            if not custom_lang:
                raise HTTPException(400, "Please enter a PaddleOCR language code")
            config["lang"] = custom_lang
        else:
            config.pop("custom_lang", None)

    elif req.engine_name == "Commercial APIs":
        if config.get("model") == "__custom__":
            config["model"] = config.pop("model_custom", "").strip() or "gpt-4o"

    # Resolve API keys
    if req.engine_name == "Commercial APIs":
        provider_slot = config.get("provider", "openai").lower()
        raw_key = config.get("api_key", "")
        resolved = _resolve_api_key(provider_slot, raw_key)
        if not resolved:
            raise HTTPException(400, f"No API key for {config.get('provider')}. "
                                     "Paste your API key in the field.")
        config["api_key"] = resolved

    elif req.engine_name == "OpenWebUI":
        raw_key = config.get("api_key", "")
        resolved = _resolve_api_key("openwebui", raw_key)
        if not resolved:
            raise HTTPException(400, "No API key for OpenWebUI. "
                                     "Paste your API key in the field.")
        config["api_key"] = resolved

    # Strip empty custom_prompt for API engines (use engine default)
    if req.engine_name in ("Commercial APIs", "OpenWebUI"):
        if not config.get("custom_prompt", "").strip():
            config["custom_prompt"] = None

    # --- Engine pool logic ---
    pool_key = _make_pool_key(req.engine_name, config)

    async with pool_lock:
        # Release previous engine reference for this session
        if session.pool_key:
            _release_pool_reference(session.pool_key, reason="engine switch")
            session.pool_key = None

        # Check if this exact engine+model is already loaded
        if pool_key in engine_pool:
            slot = engine_pool[pool_key]
            slot.ref_count += 1
            slot.last_used = time.time()
            session.pool_key = pool_key
            # Update compat shims
            loaded_engine = slot.engine
            loaded_engine_name = slot.engine_name
            loaded_config = slot.config
            log.info(f"Pool hit: reusing '{pool_key}' (ref_count={slot.ref_count})")
            return {"success": True, "load_time_s": 0.0,
                    "engine_name": req.engine_name, "reused": True, "pool_key": pool_key}

        # Need new slot — evict if VRAM tight
        await _maybe_evict(req.engine_name)

    # Load model OUTSIDE pool_lock (blocking I/O)
    engine = _create_engine_instance(req.engine_name)
    if not engine:
        raise HTTPException(500, f"Cannot create engine instance for '{req.engine_name}'")

    start = time.time()
    success = await asyncio.to_thread(engine.load_model, config)
    elapsed = time.time() - start

    if not success:
        raise HTTPException(500, "Failed to load model")

    slot = EngineSlot(
        engine=engine,
        engine_name=req.engine_name,
        config=config,
        pool_key=pool_key,
        ref_count=1,
        last_used=time.time(),
    )

    async with pool_lock:
        # Double-check: another request may have loaded the same key concurrently
        if pool_key in engine_pool:
            engine.unload_model()
            slot = engine_pool[pool_key]
            slot.ref_count += 1
            slot.last_used = time.time()
        else:
            engine_pool[pool_key] = slot

        session.pool_key = pool_key
        # Update compat shims
        loaded_engine = slot.engine
        loaded_engine_name = slot.engine_name
        loaded_config = slot.config

    log.info(f"Pool miss: loaded '{pool_key}' in {elapsed:.1f}s (pool size={len(engine_pool)})")
    return {"success": True, "load_time_s": round(elapsed, 2),
            "engine_name": req.engine_name, "reused": False, "pool_key": pool_key}


@app.get("/api/keys")
async def list_keys():
    """Keys are stored in browser localStorage only. Server has no key info.

    This endpoint returns an empty dict — it exists for backwards compatibility.
    """
    return {}


@app.post("/api/admin/evict-all")
async def admin_evict_all(request: Request):
    """Force-evict all engine slots from VRAM (admin key; legacy: localhost only)."""
    _check_admin(request, legacy_localhost_only=True)
    async with pool_lock:
        evicted = []
        for key, slot in list(engine_pool.items()):
            try:
                slot.engine.unload_model()
            except Exception as e:
                log.warning(f"admin evict failed for '{key}': {e}")
            del engine_pool[key]
            evicted.append(key)
        for session in sessions.values():
            session.pool_key = None
            session.comparison_pool_keys.clear()
        global loaded_engine, loaded_engine_name, loaded_config
        loaded_engine = None
        loaded_engine_name = ""
        loaded_config = {}
    log.info(f"Admin force-evict: cleared {len(evicted)} slot(s): {evicted}")
    return {"evicted": evicted}


@app.post("/api/engine/unload")
async def unload_engine(request: Request):
    global loaded_engine, loaded_engine_name, loaded_config
    session = _get_session(request)

    async with pool_lock:
        if session.pool_key:
            _release_pool_reference(session.pool_key, reason="explicit unload")
        session.pool_key = None
        _release_all_comparison_pool_references(session, reason="explicit unload")
        # Update compat shims
        loaded_engine = None
        loaded_engine_name = ""
        loaded_config = {}

    return {"success": True}


def _register_image(session: UserSession, pil_image: Image.Image, filename: str, save_path: Path) -> str:
    """Store a PIL image in the session's cache and return its image_id."""
    image_id = str(uuid.uuid4())
    session.image_cache[image_id] = {
        "path": save_path,
        "xml_path": None,
        "pil_image": pil_image,
        "width": pil_image.width,
        "height": pil_image.height,
        "filename": filename,
        "lines": None,
        "results": None,
        "result_slots": {},
        "primary_result_slot": None,
    }
    return image_id


def _describe_model_config(config: Dict[str, Any]) -> str:
    """Return a short model/config label for display in comparison slots."""
    for key in ("model", "model_path", "base_model", "model_name", "preset_id", "lang"):
        value = config.get(key)
        if not value:
            continue
        text = str(value).strip()
        if not text:
            continue
        return Path(text).name or text
    return ""


def _make_result_slot_label(engine_name: str, config: Dict[str, Any], custom_label: Optional[str] = None) -> str:
    """Build a human-readable label for a stored result slot."""
    if custom_label and custom_label.strip():
        return custom_label.strip()
    model_label = _describe_model_config(config)
    if model_label and model_label != engine_name:
        return f"{engine_name} - {model_label}"
    return engine_name


def _make_result_slot_id(prefix: str, engine_name: str, config: Dict[str, Any], pool_key: Optional[str]) -> str:
    """Build a stable result-slot ID for one engine/config combination."""
    basis = pool_key or _make_pool_key(engine_name, config)
    digest = hashlib.sha256(basis.encode("utf-8")).hexdigest()[:10]
    return f"{prefix}-{digest}"


def _serialize_result_slot(slot: Dict[str, Any]) -> Dict[str, Any]:
    """Return the frontend-safe subset of a stored result slot."""
    return {
        "slot_id": slot.get("slot_id"),
        "label": slot.get("label"),
        "engine_name": slot.get("engine_name"),
        "seg_source": slot.get("seg_source"),
        "line_count": slot.get("line_count", 0),
        "pool_key": slot.get("pool_key"),
        "kind": slot.get("kind", "comparison"),
    }


def _store_result_slot(
    img_data: Dict[str, Any],
    *,
    slot_id: str,
    label: str,
    engine_name: str,
    seg_source: str,
    lines: List[Dict[str, Any]],
    pool_key: Optional[str],
    kind: str,
) -> Dict[str, Any]:
    """Persist a named transcription result set for later comparison."""
    slot = {
        "slot_id": slot_id,
        "label": label,
        "engine_name": engine_name,
        "seg_source": seg_source,
        "line_count": len(lines),
        "pool_key": pool_key,
        "kind": kind,
        "created_at": time.time(),
        "lines": lines,
    }
    img_data.setdefault("result_slots", {})[slot_id] = slot
    if kind == "primary":
        img_data["primary_result_slot"] = slot_id
    return slot


def _extract_pagexml_line_texts(content: bytes) -> List[str]:
    """Extract per-line transcription text from a PAGE XML file, in reading order.

    Returns one string per ``TextLine`` (from ``TextEquiv/Unicode``). Lines
    without text are kept as empty strings so index alignment with the cached
    segmentation is preserved. Used for ground-truth CER/WER evaluation.
    """
    import xml.etree.ElementTree as ET

    root = ET.fromstring(content)
    # The PAGE namespace varies by schema year; read it off the root tag,
    # e.g. "{http://schema.primaresearch.org/PAGE/.../2013-07-15}PcGts".
    ns_uri = root.tag[1:root.tag.index("}")] if root.tag.startswith("{") else ""
    texts: List[str] = []
    if ns_uri:
        ns = {"page": ns_uri}
        for text_line in root.findall(".//page:TextLine", ns):
            uni = text_line.find("page:TextEquiv/page:Unicode", ns)
            texts.append((uni.text or "").strip() if uni is not None else "")
    else:
        for text_line in root.findall(".//TextLine"):
            uni = text_line.find("TextEquiv/Unicode")
            texts.append((uni.text or "").strip() if uni is not None else "")
    return texts


def _build_disagreement_payload(
    base_slot: Dict[str, Any],
    comparison_slot: Dict[str, Any],
    mode: ComparisonMode = ComparisonMode.ENGINE_COMPARISON,
) -> Dict[str, Any]:
    """Build frontend-ready comparison metrics for two stored result slots.

    In ``ENGINE_COMPARISON`` mode both slots are engine outputs and metrics are
    symmetric disagreement rates. In ``GROUND_TRUTH`` mode ``base_slot`` is the
    reference (ground truth) and ``comparison_slot`` the hypothesis, so metrics
    are reported as CER/WER.
    """
    base_lines = base_slot.get("lines", [])
    comparison_lines = comparison_slot.get("lines", [])
    line_count = min(len(base_lines), len(comparison_lines))
    labels = TranscriptionMetrics.get_display_labels(mode)

    base_texts = [base_lines[i].get("text", "") for i in range(line_count)]
    comparison_texts = [comparison_lines[i].get("text", "") for i in range(line_count)]
    summary = TranscriptionMetrics.calculate_summary_metrics(
        base_texts,
        comparison_texts,
        mode,
    )

    rows = []
    for i in range(line_count):
        base_line = base_lines[i]
        comparison_line = comparison_lines[i]
        raw_metrics = TranscriptionMetrics.compare_lines(
            base_line.get("text", ""),
            comparison_line.get("text", ""),
        )
        display = TranscriptionMetrics.get_display_metrics(
            raw_metrics,
            mode,
        )
        has_disagreement = display.edit_distance > 0
        # Only ship char-level diff ops for lines that actually differ — keeps the
        # payload small while still letting the client highlight changes.
        diff_ops = (
            [
                {"op": op.operation, "r": op.ref_char, "h": op.hyp_char}
                for op in raw_metrics.diff_ops
            ]
            if has_disagreement
            else []
        )
        rows.append({
            "index": i,
            "region": base_line.get("region", comparison_line.get("region", 0)),
            "bbox": base_line.get("bbox") or comparison_line.get("bbox"),
            "base_text": base_line.get("text", ""),
            "comparison_text": comparison_line.get("text", ""),
            "metrics": {
                "char_rate": round(display.char_rate, 4),
                "word_rate": round(display.word_rate, 4),
                "match_percent": round(display.match_percent, 4),
                "edit_distance": display.edit_distance,
            },
            "has_disagreement": has_disagreement,
            "diff_ops": diff_ops,
        })

    return {
        "base_slot": _serialize_result_slot(base_slot),
        "comparison_slot": _serialize_result_slot(comparison_slot),
        "labels": {
            "char_rate": labels.char_rate,
            "word_rate": labels.word_rate,
            "match_rate": labels.match_rate,
            "macro_char_rate": labels.macro_char_rate,
            "micro_char_rate": labels.micro_char_rate,
            "macro_word_rate": labels.macro_word_rate,
            "color_thresholds": list(labels.color_thresholds),
        },
        "summary": {
            "line_count": summary.line_count,
            "total_edit_distance": summary.total_edit_distance,
            "macro_char_rate": round(summary.macro_char_rate, 4),
            "micro_char_rate": round(summary.micro_char_rate, 4),
            "macro_word_rate": round(summary.macro_word_rate, 4),
            "avg_match_percent": round(summary.avg_match_percent, 4),
            "identical_lines": sum(1 for row in rows if not row["has_disagreement"]),
        },
        "lines": rows,
    }


@app.post("/api/image/upload")
async def upload_image(
    request: Request,
    file: UploadFile = File(...),
    max_dim: Optional[int] = Query(default=None, ge=100, description="Resize long edge to this many pixels (mobile upload only)"),
):
    session = _get_session(request)
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

            def _render_pdf(data: bytes, stem: str, sess: UserSession) -> list:
                mat = _fitz.Matrix(150 / 72, 150 / 72)
                doc = _fitz.open(stream=data, filetype="pdf")
                results = []
                for i, page in enumerate(doc):
                    pix = page.get_pixmap(matrix=mat, colorspace=_fitz.csRGB)
                    pil_page = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    page_filename = f"{stem}_page{i+1:03d}.png"
                    save_path = UPLOAD_DIR / f"{uuid.uuid4()}.png"
                    pil_page.save(save_path)
                    pid = _register_image(sess, pil_page, page_filename, save_path)
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
                pages_out = await loop.run_in_executor(pool, _render_pdf, content, stem, session)
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
        width, height = pil_image.size  # lazy — header only, no full decode yet
    except Exception as e:
        save_path.unlink(missing_ok=True)
        raise HTTPException(400, f"Invalid image: {e}")

    # Decompression-bomb guard: PIL's own MAX_IMAGE_PIXELS is disabled globally
    # (inference_page.py) for legitimate huge facsimiles, so uploads must be
    # capped here BEFORE exif_transpose/convert trigger the full decode.
    if width * height > _MAX_UPLOAD_PIXELS:
        save_path.unlink(missing_ok=True)
        raise HTTPException(
            400,
            f"Image too large: {width}x{height} px "
            f"(max {_MAX_UPLOAD_PIXELS // 1_000_000} megapixels)",
        )

    try:
        pil_image = ImageOps.exif_transpose(pil_image)
        pil_image = pil_image.convert("RGB")
        if max_dim and max(pil_image.width, pil_image.height) > max_dim:
            pil_image.thumbnail((max_dim, max_dim), Image.LANCZOS)
            pil_image.save(save_path)
    except Exception as e:
        save_path.unlink(missing_ok=True)
        raise HTTPException(400, f"Invalid image: {e}")

    image_id = _register_image(session, pil_image, filename, save_path)
    return {
        "image_id": image_id,
        "width": pil_image.width,
        "height": pil_image.height,
        "filename": filename,
    }


@app.post("/api/image/{image_id}/xml")
async def upload_xml(request: Request, image_id: str, file: UploadFile = File(...)):
    """Attach a PAGE XML file to an already-uploaded image."""
    session = _get_session(request)
    if image_id not in session.image_cache:
        raise HTTPException(404, "Image not found — upload image first")
    content = await file.read()
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(400, "XML too large (max 10MB)")
    xml_path = UPLOAD_DIR / f"{image_id}.xml"
    xml_path.write_bytes(content)
    session.image_cache[image_id]["xml_path"] = xml_path
    return {"success": True, "filename": file.filename}


@app.post("/api/image/{image_id}/ground-truth")
async def upload_ground_truth(request: Request, image_id: str, file: UploadFile = File(...)):
    """Attach a ground-truth PAGE XML to an image for CER/WER evaluation.

    The line texts are extracted and stored on the image; comparisons can then
    be scored against this reference via ``/api/compare/ground-truth``.
    """
    session = _get_session(request)
    if image_id not in session.image_cache:
        raise HTTPException(404, "Image not found — upload image first")
    content = await file.read()
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(400, "XML too large (max 10MB)")
    try:
        gt_lines = _extract_pagexml_line_texts(content)
    except Exception as e:
        raise HTTPException(400, f"Could not parse PAGE XML: {e}")
    if not gt_lines:
        raise HTTPException(400, "No TextLine transcriptions found in the PAGE XML")
    session.image_cache[image_id]["ground_truth"] = {
        "lines": gt_lines,
        "filename": file.filename or "ground_truth.xml",
    }
    return {"success": True, "filename": file.filename, "line_count": len(gt_lines)}


@app.delete("/api/image/{image_id}/ground-truth")
async def clear_ground_truth(request: Request, image_id: str):
    """Remove a previously attached ground-truth reference."""
    session = _get_session(request)
    if image_id not in session.image_cache:
        raise HTTPException(404, "Image not found")
    session.image_cache[image_id].pop("ground_truth", None)
    return {"success": True}


@app.get("/api/image/{image_id}")
async def get_image(request: Request, image_id: str):
    session = _get_session(request)
    if image_id not in session.image_cache:
        raise HTTPException(404, "Image not found")
    return FileResponse(str(session.image_cache[image_id]["path"]))


@app.get("/api/image/{image_id}/info")
async def image_info(request: Request, image_id: str):
    session = _get_session(request)
    if image_id not in session.image_cache:
        raise HTTPException(404, "Image not found")
    d = session.image_cache[image_id]
    return {
        "image_id": image_id,
        "filename": d["filename"],
        "width": d["width"],
        "height": d["height"],
        "has_xml": d["xml_path"] is not None,
    }


async def _run_segmentation(img_data: dict, method: str, device: str = "cpu",
                            max_columns: int = 6,
                            split_width_fraction: float = 0.40,
                            text_direction: str = "horizontal-lr") -> dict:
    """
    Shared segmentation helper.  Runs the appropriate segmenter, stores
    results in img_data, and returns a serialisable dict ready for SSE or JSON.
    Also populates img_data["line_regions"] with a per-line region index list
    so the transcription loop can tag each line with its column.
    """
    profile = _RUNTIME_PROFILE
    if profile is not None and hasattr(profile, "segmentation_overrides"):
        method, device = profile.segmentation_overrides(method, device)
    pil_image = img_data["pil_image"]
    xml_path  = img_data.get("xml_path")

    if profile is not None and hasattr(profile, "run_segmentation"):
        result = await asyncio.to_thread(profile.run_segmentation, img_data, method, device)
        if result is not None:
            return result

    _import_segmenters()

    regions: list = []
    lines: list   = []

    xml_region_data: list = []  # TextRegion bboxes from PAGE XML (for visualization)
    if xml_path is not None:
        from inference_page import PageXMLSegmenter as _PXSeg
        segmenter = _PXSeg(str(xml_path))
        lines = await asyncio.to_thread(segmenter.segment_lines, pil_image)
        source = "pagexml"
        xml_region_data = getattr(segmenter, 'region_data', []) or []


    elif method == "kraken-blla":
        segmenter = KrakenLineSegmenter(device=device)
        regions, lines = await asyncio.to_thread(
            segmenter.segment_with_regions, pil_image,
            device=device,
            max_columns=max_columns,
            split_width_fraction=split_width_fraction,
            text_direction=text_direction,
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
    # PAGE XML provides region bboxes directly; Kraken/blla provide SegRegion objects
    if xml_region_data:
        img_data["seg_regions"] = xml_region_data
    elif regions:
        img_data["seg_regions"] = [
            {"id": r.id, "bbox": list(r.bbox), "num_lines": len(r.line_ids)}
            for r in regions
        ]
    else:
        img_data["seg_regions"] = []

    result: dict = {
        "num_lines": len(lines),
        "bboxes":    [list(l.bbox) for l in lines],
        "source":    source,
    }
    if img_data["seg_regions"]:
        result["regions"] = img_data["seg_regions"]
    return result


@app.delete("/api/image/{image_id}/region/{region_index}")
async def delete_region(request: Request, image_id: str, region_index: int):
    """
    Remove one detected region and its lines from the cached segmentation.
    Returns updated segmentation data in the same format as /segment,
    so the client can redraw the canvas.
    """
    session = _get_session(request)
    if image_id not in session.image_cache:
        raise HTTPException(404, "Image not found")
    img_data = session.image_cache[image_id]

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
    request: Request,
    image_id: str,
    method: str = "kraken",
    device: str = "cpu",
    max_columns: int = 6,
    split_width_fraction: float = 0.40,
    text_direction: str = "horizontal-lr",
):
    """
    Run segmentation only (no transcription) and return line bboxes as JSON.
    Useful for previewing line layout before transcribing.
    """
    session = _get_session(request)
    if image_id not in session.image_cache:
        raise HTTPException(404, "Image not found — upload first")

    try:
        return await _run_segmentation(session.image_cache[image_id], method, device,
                                       max_columns, split_width_fraction, text_direction)
    except Exception as e:
        raise HTTPException(500, f"Segmentation failed: {e}")


@app.post("/api/transcribe")
async def transcribe(request: Request, req: TranscribeRequest):
    session = _get_session(request)

    slot, eff_engine, eff_config, eff_engine_name, effective_pool_key = _resolve_effective_engine(
        session,
        req.engine_config_overrides,
    )

    if req.image_id not in session.image_cache:
        raise HTTPException(404, "Image not found — upload first")

    img_data = session.image_cache[req.image_id]
    pil_image = img_data["pil_image"]

    # Per-request cancel event (replaces global cancel_event)
    request_id = str(uuid.uuid4())
    cancel_evt = asyncio.Event()
    session.cancel_events[request_id] = cancel_evt
    actor = _request_actor(request)

    async def event_stream():
        # eff_config wird im LapaOCR-Zweig unten neu zugewiesen; ohne nonlocal
        # würde Python es zur lokalen Variable machen und jeder frühere Lesezugriff
        # (nicht-Lapa-Engines) liefe in einen UnboundLocalError.
        nonlocal eff_config
        _import_segmenters()
        _interactive_begin()  # batch jobs pause while this runs
        _activity_register(request_id, "transcribe", actor, eff_engine_name)

        try:
            # --- Segmentation ---
            xml_path = img_data.get("xml_path") if req.use_pagexml else None

            # A page-level engine (Qwen3-VL, OpenWebUI, …) normally reads the whole
            # page as one unit. But if the user explicitly ran Segment first, honour
            # those cached line segments and transcribe line-by-line: that produces a
            # segmented base the comparison workspace can align against other engines.
            cached_source = img_data.get("seg_source")
            has_explicit_seg = bool(img_data.get("lines")) and cached_source not in (None, "", "page", "unknown")

            if not eff_engine.requires_line_segmentation() and not xml_path and not has_explicit_seg:
                # Page-level engine with no PAGE XML — send whole page as single line
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
                                                         req.split_width_fraction,
                                                         req.text_direction)
                    lines = img_data["lines"]
                    yield _sse("segmentation", seg_result)
                else:
                    yield _sse("status", {"message": f"Segmenting with {req.seg_method}..."})
                    seg_result = await _run_segmentation(img_data, req.seg_method,
                                                         req.seg_device, req.max_columns,
                                                         req.split_width_fraction,
                                                         req.text_direction)
                    lines = img_data["lines"]
                    yield _sse("segmentation", seg_result)

            # --- Transcription ---
            results = []
            token_usage: Dict[str, Any] = {}
            start_time = time.time()
            line_regions = img_data.get("line_regions") or ([0] * len(lines))

            # Lapa can occasionally spend very long on one line; cap per-line generation time by default.
            if eff_engine_name == "LapaOCR" and not eff_config.get("max_time_s"):
                eff_config = dict(eff_config)
                eff_config["max_time_s"] = 90

            for i, line in enumerate(lines):
                # Check for cancellation before each line
                if cancel_evt.is_set():
                    yield _sse("cancelled", {})
                    return

                yield _sse("status", {
                    "message": f"Generating line {i + 1}/{len(lines)}...",
                })

                line_img = line.image if line.image is not None else pil_image.crop(line.bbox)
                img_array = np.array(line_img.convert("RGB"))

                # Use slot lock to serialize access to this engine instance
                if slot:
                    async with slot.lock:
                        slot.last_used = time.time()
                        result = await asyncio.to_thread(
                            eff_engine.transcribe_line, img_array, eff_config
                        )
                else:
                    result = await asyncio.to_thread(
                        eff_engine.transcribe_line, img_array, eff_config
                    )

                text = str(result.text) if hasattr(result, "text") else str(result)
                confidence = None
                if hasattr(result, "confidence") and result.confidence is not None:
                    confidence = float(result.confidence)
                    if confidence > 1:
                        confidence = confidence / 100.0
                # Accumulate token usage and extract thinking text from API engines (e.g. Gemini)
                thinking_text = None
                if hasattr(result, "metadata") and isinstance(result.metadata, dict):
                    tu = result.metadata.get("token_usage")
                    if tu:
                        for k, v in tu.items():
                            if v is not None:
                                token_usage[k] = token_usage.get(k, 0) + v
                    thinking_text = result.metadata.get("thinking_text")

                line_data = {
                    "index": i,
                    "text": text,
                    "confidence": confidence,
                    "bbox": list(line.bbox),
                    "region": line_regions[i] if i < len(line_regions) else 0,
                }
                if thinking_text:
                    line_data["thinking_text"] = thinking_text
                results.append(line_data)
                _activity_update(request_id, i + 1, len(lines))
                progress_data: Dict[str, Any] = {
                    "current": i + 1,
                    "total": len(lines),
                    "line": line_data,
                }
                if token_usage:
                    progress_data["token_usage"] = dict(token_usage)
                yield _sse("progress", progress_data)

                # Check for cancellation after each line's progress event
                if cancel_evt.is_set():
                    yield _sse("cancelled", {})
                    return

            # Store completed results in session image_cache for export
            img_data["results"] = results
            primary_slot = _store_result_slot(
                img_data,
                slot_id="primary",
                label=_make_result_slot_label(eff_engine_name, eff_config),
                engine_name=eff_engine_name,
                seg_source=img_data.get("seg_source", "unknown"),
                lines=results,
                pool_key=effective_pool_key,
                kind="primary",
            )

            elapsed = time.time() - start_time
            complete_data: Dict[str, Any] = {
                "lines": results,
                "total_time_s": round(elapsed, 2),
                "engine": eff_engine_name,
                "seg_source": img_data.get("seg_source", "unknown"),
                "result_slot": _serialize_result_slot(primary_slot),
            }
            if token_usage:
                complete_data["token_usage"] = token_usage
            yield _sse("complete", complete_data)

        except Exception as e:
            log.exception("Transcription error")
            yield _sse("error", {"message": str(e)})
        finally:
            _interactive_end()
            _activity_remove(request_id)
            # Clean up this request's cancel event
            session.cancel_events.pop(request_id, None)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Disable nginx buffering if behind proxy
        },
    )


@app.post("/api/compare/run")
async def compare_run(request: Request, req: CompareRunRequest):
    """Run a second transcription on cached line segments and store it as a comparison slot."""
    session = _get_session(request)
    slot, eff_engine, eff_config, eff_engine_name, effective_pool_key = _resolve_effective_engine(
        session,
        req.engine_config_overrides,
    )

    if req.image_id not in session.image_cache:
        raise HTTPException(404, "Image not found — upload first")

    img_data = session.image_cache[req.image_id]
    base_slot_id = img_data.get("primary_result_slot")
    result_slots = img_data.get("result_slots") or {}
    base_slot = result_slots.get(base_slot_id) if base_slot_id else None
    if not base_slot or not base_slot.get("lines"):
        raise HTTPException(400, "Run a base transcription first before starting a comparison")

    if img_data.get("seg_source") == "page" or not img_data.get("lines"):
        raise HTTPException(400, "Comparison requires cached line segmentation from the base transcription")

    pil_image = img_data["pil_image"]
    lines = img_data["lines"]
    line_regions = img_data.get("line_regions") or ([0] * len(lines))

    request_id = str(uuid.uuid4())
    cancel_evt = asyncio.Event()
    session.cancel_events[request_id] = cancel_evt
    actor = _request_actor(request)

    async def event_stream():
        # Siehe transcribe(): nonlocal nötig wegen LapaOCR-Neuzuweisung von eff_config.
        nonlocal eff_config
        _interactive_begin()  # batch jobs pause while this runs
        _activity_register(request_id, "compare", actor, eff_engine_name)
        try:
            yield _sse("status", {
                "message": f"Running comparison with {eff_engine_name} on cached line segments...",
            })

            results = []
            start_time = time.time()

            if eff_engine_name == "LapaOCR" and not eff_config.get("max_time_s"):
                eff_config = dict(eff_config)
                eff_config["max_time_s"] = 90

            for i, line in enumerate(lines):
                if cancel_evt.is_set():
                    yield _sse("cancelled", {})
                    return

                yield _sse("status", {
                    "message": f"Generating line {i + 1}/{len(lines)}...",
                })

                line_img = line.image if line.image is not None else pil_image.crop(line.bbox)
                img_array = np.array(line_img.convert("RGB"))

                if slot:
                    async with slot.lock:
                        slot.last_used = time.time()
                        result = await asyncio.to_thread(
                            eff_engine.transcribe_line,
                            img_array,
                            eff_config,
                        )
                else:
                    result = await asyncio.to_thread(
                        eff_engine.transcribe_line,
                        img_array,
                        eff_config,
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
                _activity_update(request_id, i + 1, len(lines))
                yield _sse("progress", {
                    "current": i + 1,
                    "total": len(lines),
                    "line": line_data,
                })

            slot_id = _make_result_slot_id("compare", eff_engine_name, eff_config, effective_pool_key)
            comparison_slot = _store_result_slot(
                img_data,
                slot_id=slot_id,
                label=_make_result_slot_label(eff_engine_name, eff_config, req.label),
                engine_name=eff_engine_name,
                seg_source=img_data.get("seg_source", "unknown"),
                lines=results,
                pool_key=effective_pool_key,
                kind="comparison",
            )
            _attach_comparison_pool_reference(session, slot_id, effective_pool_key)

            payload = _build_disagreement_payload(base_slot, comparison_slot)
            payload["total_time_s"] = round(time.time() - start_time, 2)
            yield _sse("complete", payload)
        except Exception as e:
            log.exception("Comparison run error")
            yield _sse("error", {"message": str(e)})
        finally:
            _interactive_end()
            _activity_remove(request_id)
            session.cancel_events.pop(request_id, None)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/compare/ground-truth")
async def compare_ground_truth(request: Request, req: GroundTruthCompareRequest):
    """Score stored transcription slots against an uploaded ground-truth (CER/WER).

    Returns one per-line payload per scored slot (reusing the comparison
    renderer, with the ground truth as the reference column) plus a leaderboard
    sorted by micro CER.
    """
    session = _get_session(request)
    if req.image_id not in session.image_cache:
        raise HTTPException(404, "Image not found")
    img_data = session.image_cache[req.image_id]
    gt = img_data.get("ground_truth")
    if not gt or not gt.get("lines"):
        raise HTTPException(400, "Upload a ground-truth PAGE XML first")
    result_slots = img_data.get("result_slots") or {}
    if not result_slots:
        raise HTTPException(400, "Run at least one transcription before scoring against ground truth")

    gt_slot = {
        "slot_id": "ground_truth",
        "label": f"Ground Truth ({gt.get('filename', 'GT')})",
        "engine_name": "ground_truth",
        "seg_source": "ground_truth",
        "line_count": len(gt["lines"]),
        "kind": "ground_truth",
        "lines": [{"text": t} for t in gt["lines"]],
    }

    if req.slot_id:
        target_ids = [req.slot_id] if req.slot_id in result_slots else []
    else:
        target_ids = list(result_slots.keys())

    runs = []
    leaderboard = []
    for sid in target_ids:
        slot = result_slots.get(sid)
        if not slot or not slot.get("lines"):
            continue
        payload = _build_disagreement_payload(gt_slot, slot, ComparisonMode.GROUND_TRUTH)
        runs.append(payload)
        s = payload["summary"]
        leaderboard.append({
            "slot_id": sid,
            "label": slot.get("label", sid),
            "engine_name": slot.get("engine_name"),
            "kind": slot.get("kind", "comparison"),
            "micro_cer": s["micro_char_rate"],
            "macro_cer": s["macro_char_rate"],
            "macro_wer": s["macro_word_rate"],
            "scored_lines": s["line_count"],
            "gt_line_count": len(gt["lines"]),
        })

    if not runs:
        raise HTTPException(400, "No matching transcription slot to score")

    leaderboard.sort(key=lambda r: r["micro_cer"])
    return {
        "ground_truth": {"filename": gt.get("filename"), "line_count": len(gt["lines"])},
        "runs": runs,
        "leaderboard": leaderboard,
    }


@app.post("/api/transcribe/cancel")
async def cancel_transcription(request: Request):
    """Signal all running transcriptions for this session to stop."""
    session = _get_session(request)
    for evt in session.cancel_events.values():
        evt.set()
    return {"success": True}


@app.post("/api/image/{image_id}/export-xml")
async def export_xml(request: Request, image_id: str):
    """Export transcription results for image_id as PAGE XML."""
    session = _get_session(request)
    pretty, stem = _build_xml_bytes(session, image_id)
    return Response(
        content=pretty,
        media_type="application/xml",
        headers={"Content-Disposition": f'attachment; filename="{stem}.xml"'},
    )


def _build_xml_bytes(session: UserSession, image_id: str) -> tuple[bytes, str]:
    """Return (xml_bytes, stem) for a cached image, or raise HTTPException."""
    import xml.etree.ElementTree as ET
    from xml.dom import minidom
    from page_xml_exporter import PageXMLExporter

    if image_id not in session.image_cache:
        raise HTTPException(404, f"Image {image_id} not found")
    img_data = session.image_cache[image_id]
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


def _build_thinking_bytes(session: UserSession, image_id: str) -> tuple[bytes, str]:
    """Return (thinking_bytes, stem) for a cached image, or raise HTTPException(404) if no thinking."""
    if image_id not in session.image_cache:
        raise HTTPException(404, f"Image {image_id} not found")
    img_data = session.image_cache[image_id]
    results = img_data.get("results")
    if not results:
        raise HTTPException(400, f"No results for {image_id}")
    filename = img_data.get("filename", img_data["path"].name)
    stem = Path(filename).stem
    blocks = []
    for i, r in enumerate(results):
        t = r.get("thinking_text", "")
        if t:
            if len(results) > 1:
                blocks.append(f"=== Line {i + 1} ===\n{t}")
            else:
                blocks.append(t)
    if not blocks:
        raise HTTPException(404, f"No thinking text for {image_id}")
    return "\n\n".join(blocks).encode("utf-8"), stem


def _build_txt_bytes(session: UserSession, image_id: str) -> tuple[bytes, str]:
    """Return (txt_bytes, stem) for a cached image, or raise HTTPException."""
    if image_id not in session.image_cache:
        raise HTTPException(404, f"Image {image_id} not found")
    img_data = session.image_cache[image_id]
    results = img_data.get("results")
    if not results:
        raise HTTPException(400, f"No results for {image_id}")
    filename = img_data.get("filename", img_data["path"].name)
    text = "\n".join(r.get("text", "") for r in results)
    return text.encode("utf-8"), Path(filename).stem


class BatchXMLRequest(BaseModel):
    image_ids: list[str]


@app.post("/api/batch/export-thinking")
async def batch_export_thinking(request: Request, req: BatchXMLRequest):
    """Return a ZIP archive containing one thinking-text file per image (skips pages without thinking)."""
    session = _get_session(request)
    import zipfile, io
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        for image_id in req.image_ids:
            try:
                thinking_bytes, stem = _build_thinking_bytes(session, image_id)
                zf.writestr(f"{stem}_thinking.txt", thinking_bytes)
            except HTTPException:
                pass  # skip pages without thinking
    buf.seek(0)
    return Response(
        content=buf.read(),
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="batch_thinking.zip"'},
    )


@app.post("/api/batch/export-txt")
async def batch_export_txt(request: Request, req: BatchXMLRequest):
    """Return a ZIP archive containing one plain-text file per image."""
    session = _get_session(request)
    import zipfile, io
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        for image_id in req.image_ids:
            try:
                txt_bytes, stem = _build_txt_bytes(session, image_id)
                zf.writestr(f"{stem}.txt", txt_bytes)
            except HTTPException:
                pass  # skip images without results
    buf.seek(0)
    return Response(
        content=buf.read(),
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="batch_export_txt.zip"'},
    )


@app.post("/api/batch/export-xml")
async def batch_export_xml(request: Request, req: BatchXMLRequest):
    """Return a ZIP archive containing one PAGE XML file per image."""
    session = _get_session(request)
    import zipfile, io
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        for image_id in req.image_ids:
            try:
                xml_bytes, stem = _build_xml_bytes(session, image_id)
                zf.writestr(f"{stem}.xml", xml_bytes)
            except HTTPException:
                pass  # skip images without results
    buf.seek(0)
    return Response(
        content=buf.read(),
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="batch_export.zip"'},
    )


@app.get("/api/session")
async def session_info(request: Request):
    """Return info about the current session (useful for debugging)."""
    session = _get_session(request)
    return {
        "session_id": session.session_id[:8] + "...",
        "images": len(session.image_cache),
        "active_transcriptions": len(session.cancel_events),
        "pool_key": session.pool_key,
        "comparison_pool_keys": dict(session.comparison_pool_keys),
        "created_at": session.created_at,
        "last_active": session.last_active,
        "total_sessions": len(sessions),
    }


@app.get("/api/engine/pool")
async def pool_status():
    """Return current engine pool state (admin/debug endpoint)."""
    slots = []
    for key, slot in engine_pool.items():
        slots.append({
            "pool_key": key,
            "engine_name": slot.engine_name,
            "ref_count": slot.ref_count,
            "loaded": slot.engine.is_model_loaded(),
            "last_used": slot.last_used,
            "age_s": round(time.time() - slot.last_used, 0),
        })
    return {
        "pool_size": len(engine_pool),
        "slots": slots,
        "total_sessions": len(sessions),
    }


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


@app.post("/api/models/upload")
async def upload_model(request: Request, file: UploadFile = File(...)):
    """Upload a Kraken .mlmodel file to the server's models/kraken_uploads/ directory.

    Model files are deserialized on load — with key auth enabled this endpoint
    is admin-only. Legacy (no key file): open, protected only by the perimeter.
    """
    _check_admin(request, legacy_localhost_only=False)
    filename = file.filename or "model.mlmodel"
    if not filename.lower().endswith(".mlmodel"):
        raise HTTPException(400, "Only .mlmodel files are accepted")

    content = await file.read()
    if len(content) > 500 * 1024 * 1024:
        raise HTTPException(400, "File too large (max 500 MB)")

    upload_dir = PROJECT_ROOT / "models" / "kraken_uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    # Sanitize filename — keep only safe characters
    safe_name = Path(filename).name
    safe_name = "".join(c for c in safe_name if c.isalnum() or c in "._- ")
    safe_name = safe_name.strip() or "uploaded.mlmodel"

    dest = upload_dir / safe_name
    dest.write_bytes(content)
    log.info(f"Uploaded Kraken model: {dest} ({len(content)} bytes)")

    rel_path = str(dest.relative_to(PROJECT_ROOT))  # e.g. models/kraken_uploads/foo.mlmodel
    return {
        "path": rel_path,
        "filename": safe_name,
        "size": len(content),
        "options": _scan_kraken_models(),  # refreshed list for frontend to repopulate select
    }


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
