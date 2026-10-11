"""
Chandra Engine Plugin

Wraps Chandra 2 (datalab-to/chandra-ocr-2, a 5B vision-language OCR model) as a
whole-page OCR engine for Polyscriptor. Chandra reads the full page and emits
HTML directly — no pre-segmented lines. The worker converts the HTML to plain
text; the raw HTML is kept in the metadata.

Strengths: universal OCR — 90+ languages, printed and handwritten text, tables,
math, forms (SOTA on olmOCR-Bench). Weakness: historical Slavic handwriting —
specialized models (CRNN-CTC, TrOCR fine-tunes) are far more accurate there;
see the benchmark notes in the model registry.

IMPORTANT: Chandra lives in its OWN isolated venv (needs transformers >= 5.x,
which conflicts with the main venv). Default location is ``venv_chandra`` in
the project root; override via ``POLYSCRIPTOR_CHANDRA_VENV`` (or the
``venv_path`` config field).

Setup (one-time):
    python3 -m venv venv_chandra
    source venv_chandra/bin/activate
    pip install "chandra-ocr[hf]"
    # if the bundled torch does not match your CUDA driver:
    pip install --force-reinstall torch torchvision \
        --index-url https://download.pytorch.org/whl/cu128

This engine calls chandra_worker.py as a subprocess inside that venv. The main
venv never imports chandra/transformers 5.x directly. License note: code is
Apache-2.0; model weights are OpenRAIL-M (free for research/personal use).

Batch CLI usage:
    python batch_processing.py --engine Chandra --input-folder pages/
"""

import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

from polyscriptor.core.engine_base import HTREngine, TranscriptionResult

logger = logging.getLogger(__name__)


_DEFAULT_MODEL_ID = "datalab-to/chandra-ocr-2"
_DEFAULT_MAX_NEW_TOKENS = 4096
from polyscriptor.engines._venv_worker import (
    default_project_root, find_venv_python, venv_has_package, run_worker, WorkerError,
)
_PROJECT_ROOT = default_project_root()

# Path to the worker script (same engines/ directory)
_WORKER_SCRIPT = Path(__file__).resolve().parent / "chandra_worker.py"


def _default_venv() -> Path:
    env = os.environ.get("POLYSCRIPTOR_CHANDRA_VENV")
    if env:
        return Path(env).expanduser()
    return _PROJECT_ROOT / "venv_chandra"


_find_venv_python = find_venv_python


_venv_has_probe = lambda p: venv_has_package(p, 'chandra')


class ChandraEngine(HTREngine):
    """
    Chandra 2 whole-page OCR engine (subprocess mode).

    Calls chandra_worker.py via an isolated venv Python interpreter so the
    5B VLM stack (transformers >= 5.x) never conflicts with the main venv.
    """

    def __init__(self):
        self._venv_path: Path = _default_venv()
        self._venv_python: Optional[Path] = None
        self._is_loaded: bool = False


    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    def get_name(self) -> str:
        return "Chandra"

    def get_description(self) -> str:
        return "Chandra 2: 5B vision-language whole-page OCR, 90+ languages (subprocess mode)"

    def get_aliases(self) -> List[str]:
        return ["chandra", "chandra-ocr", "chandra-2"]

    # ------------------------------------------------------------------
    # Availability
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        if _find_venv_python(self._venv_path) is None:
            return False
        return _venv_has_probe(self._venv_path)

    def get_unavailable_reason(self) -> str:
        if _find_venv_python(self._venv_path) is None:
            return (
                f"Chandra venv not found at: {self._venv_path}\n\n"
                "Create it with:\n"
                f"  python3 -m venv {self._venv_path}\n"
                f"  source {self._venv_path}/bin/activate\n"
                "  pip install 'chandra-ocr[hf]'\n"
            )
        if not _venv_has_probe(self._venv_path):
            return (
                f"chandra-ocr not installed in {self._venv_path}\n\n"
                "Install with:\n"
                f"  source {self._venv_path}/bin/activate\n"
                "  pip install 'chandra-ocr[hf]'\n"
            )
        return ""

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------



    # ------------------------------------------------------------------
    # Config get / set
    # ------------------------------------------------------------------



    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_model(self, config: Dict[str, Any]) -> bool:
        """Validate venv + worker script exist. No actual model loading (lazy via subprocess)."""
        venv_path = Path(config.get("venv_path", str(self._venv_path))).expanduser()
        self._venv_path = venv_path

        python = _find_venv_python(venv_path)
        if python is None:
            logger.error("[Chandra] venv Python not found at %s", venv_path)
            return False

        if not _WORKER_SCRIPT.exists():
            logger.error("[Chandra] Worker script not found: %s", _WORKER_SCRIPT)
            return False

        self._venv_python = python
        self._is_loaded = True
        logger.info("[Chandra] Ready — venv: %s, worker: %s", python, _WORKER_SCRIPT)
        return True

    def unload_model(self):
        self._venv_python = None
        self._is_loaded = False

    def is_model_loaded(self) -> bool:
        return self._is_loaded

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def requires_line_segmentation(self) -> bool:
        return False  # Chandra reads the whole page itself

    def transcribe_line(
        self, image: np.ndarray, config: Optional[Dict[str, Any]] = None
    ) -> TranscriptionResult:
        """
        Transcribe a full page image via the Chandra subprocess.

        Despite the method name, page-based engines receive the full page here.
        """
        if not self._is_loaded or self._venv_python is None:
            return TranscriptionResult(text="[Chandra not loaded]", confidence=0.0)

        if config is None:
            config = self.get_config()

        # Write image to a temp file so the subprocess can read it
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            pil_img = Image.fromarray(image) if isinstance(image, np.ndarray) else image
            pil_img.convert("RGB").save(tmp_path)

            config_json = json.dumps({
                "model_id": config.get("model_id") or _DEFAULT_MODEL_ID,
                "use_gpu": config.get("use_gpu", True),
                "prompt_mode": config.get("prompt_mode") or "ocr",
                "custom_prompt": config.get("custom_prompt") or "",
                "max_new_tokens": int(config.get("max_new_tokens")
                                      or _DEFAULT_MAX_NEW_TOKENS),
            })

            env = os.environ.copy()
            # Optional cache override so weights land off the root FS
            cache = os.environ.get("POLYSCRIPTOR_CHANDRA_CACHE")
            if cache:
                env.setdefault("HF_HUB_CACHE", cache)
            # Pin the physical GPU here; the worker addresses it as cuda:0 internally
            # (per feedback_cuda_visible_devices: never combine CUDA_VISIBLE_DEVICES=N + cuda:N).
            if config.get("use_gpu", True):
                env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
                env["CUDA_VISIBLE_DEVICES"] = str(int(config.get("gpu_index", 0)))

            try:
                output = run_worker(
                    self._venv_python, _WORKER_SCRIPT, config_json, tmp_path,
                    timeout=900, logger=logger, tag="Chandra", env=env,
                )
            except WorkerError as e:
                return TranscriptionResult.error_result(f"{e}")

            text = output.get("text", "")
            lines = output.get("lines", [])
            # Chandra returns no per-token confidence. Report None (not 0.0) so
            # the UI shows "no confidence" instead of a misleading 0%.
            return TranscriptionResult(
                text=text,
                confidence=None,
                metadata={
                    "engine": "Chandra",
                    "model": output.get("model_id", _DEFAULT_MODEL_ID),
                    "line_count": len(lines),
                    "use_gpu": output.get("use_gpu"),
                    "prompt_mode": output.get("prompt_mode"),
                    "html": output.get("html", ""),
                },
            )

        except subprocess.TimeoutExpired:
            logger.error("[Chandra] Subprocess timed out after 900s")
            return TranscriptionResult(text="[Chandra timed out]", confidence=0.0)
        except json.JSONDecodeError as e:
            logger.error("[Chandra] Failed to parse worker output: %s", e)
            return TranscriptionResult(text="[Chandra output parse error]", confidence=0.0)
        except Exception as e:
            logger.error("[Chandra] Unexpected error: %s", e)
            return TranscriptionResult.error_result(f"{e}")
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # Capabilities
    # ------------------------------------------------------------------

    def get_capabilities(self) -> Dict[str, bool]:
        return {
            "batch": False,
            "confidence_scores": False,
            "beam_search": False,
            "language_model": True,
            "preprocessing": True,
        }
