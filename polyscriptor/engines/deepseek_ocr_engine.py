"""
DeepSeek-OCR-2 Engine Plugin (subprocess mode)

Wraps DeepSeek-OCR-2 as a page-level OCR engine for Polyscriptor.
DeepSeek-OCR-2 requires transformers==4.46.3, which conflicts with LightOnOCR's
transformers 5.x dependency in the main venv.  The engine therefore runs in an
isolated venv (venv_deepseek) via a subprocess worker — the main venv never
imports DeepSeek or its conflicting transformers version.

Setup (venv_deepseek):
    python -m venv /path/to/dhlab-slavistik/venv_deepseek
    source venv_deepseek/bin/activate
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    pip install "transformers==4.46.3" einops addict easydict pillow
    # Optional (faster inference, requires CUDA toolkit headers):
    pip install flash-attn --no-build-isolation
    deactivate

Key features:
- Two OCR modes: Document OCR (with layout, markdown) and Free OCR (plain text)
- Visual Causal Flow architecture with Flash Attention 2
- ~3B parameters, ~6 GB VRAM
- subprocess isolation: no transformers version conflicts with main venv
"""

import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

from polyscriptor.core.engine_base import HTREngine, TranscriptionResult

logger = logging.getLogger(__name__)


# Default venv path (sibling of the project root, same convention as venv_paddle)
from polyscriptor.engines._venv_worker import (
    default_project_root, find_venv_python, venv_has_package, run_worker, WorkerError,
)
_PROJECT_ROOT = default_project_root()
_DEFAULT_VENV = _PROJECT_ROOT / "venv_deepseek"

# Worker script lives in the same engines/ directory as this file
_WORKER_SCRIPT = Path(__file__).resolve().parent / "deepseek_worker.py"


# ---------------------------------------------------------------------------
# Venv helpers (filesystem-only, no subprocess, no imports)
# ---------------------------------------------------------------------------

_find_venv_python = find_venv_python


_venv_has_probe = lambda p: venv_has_package(p, 'transformers')


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class DeepSeekOCREngine(HTREngine):
    """
    DeepSeek-OCR-2 page-level OCR engine (subprocess mode).

    Calls deepseek_worker.py via the venv_deepseek Python interpreter so
    DeepSeek's transformers==4.46.3 dependency lives in an isolated venv and
    never conflicts with the main venv's transformers 5.x (used by LightOnOCR).
    """

    def __init__(self):
        self._venv_path: Path = _DEFAULT_VENV
        self._venv_python: Optional[Path] = None
        self._is_loaded: bool = False

        self._strip_markdown_checkbox: Optional[QCheckBox] = None
        self._crop_mode_checkbox: Optional[QCheckBox] = None

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    def get_name(self) -> str:
        return "DeepSeek-OCR"

    def get_description(self) -> str:
        return "DeepSeek-OCR-2: Page-level VLM for document OCR with layout support"

    def get_aliases(self) -> List[str]:
        return ["deepseek", "deepseek-ocr"]

    # ------------------------------------------------------------------
    # Availability (filesystem checks only — no subprocess, no imports)
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        if _find_venv_python(self._venv_path) is None:
            return False
        return _venv_has_probe(self._venv_path)

    def get_unavailable_reason(self) -> str:
        if _find_venv_python(self._venv_path) is None:
            return (
                f"venv_deepseek not found at: {self._venv_path}\n\n"
                "Create it with:\n"
                f"  python -m venv {self._venv_path}\n"
                f"  source {self._venv_path}/bin/activate\n"
                '  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121\n'
                '  pip install "transformers==4.46.3" einops addict easydict pillow\n'
                "  # Optional: pip install flash-attn --no-build-isolation\n"
                "  deactivate"
            )
        if not _venv_has_probe(self._venv_path):
            return (
                f"transformers not installed in {self._venv_path}\n\n"
                "Install with:\n"
                f"  source {self._venv_path}/bin/activate\n"
                '  pip install "transformers==4.46.3" einops addict easydict pillow\n'
                "  deactivate"
            )
        return ""

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------





    # ------------------------------------------------------------------
    # Config get / set
    # ------------------------------------------------------------------



    # ------------------------------------------------------------------
    # Model loading (validates venv + worker; actual model loads lazily in subprocess)
    # ------------------------------------------------------------------

    def load_model(self, config: Dict[str, Any]) -> bool:
        """Validate venv + worker script exist. Actual model loading is lazy (in subprocess)."""
        venv_path = Path(config.get("venv_path", str(self._venv_path)))
        self._venv_path = venv_path


        python = _find_venv_python(venv_path)
        if python is None:
            logger.error("[DeepSeek-OCR] venv Python not found at %s", venv_path)
            return False

        if not _WORKER_SCRIPT.exists():
            logger.error("[DeepSeek-OCR] Worker script not found: %s", _WORKER_SCRIPT)
            return False

        self._venv_python = python
        self._is_loaded = True
        logger.info("[DeepSeek-OCR] Ready — venv: %s, worker: %s", python, _WORKER_SCRIPT)
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
        return False  # Page-level model — no pre-segmentation needed

    def transcribe_line(
        self, image: np.ndarray, config: Optional[Dict[str, Any]] = None
    ) -> TranscriptionResult:
        """
        Transcribe a full page image via DeepSeek-OCR-2 subprocess.

        Despite the method name, page-based engines receive the full page here.
        """
        if not self._is_loaded or self._venv_python is None:
            return TranscriptionResult(text="[DeepSeek-OCR not loaded]", confidence=0.0)

        if config is None:
            config = self.get_config()

        # Write image to a temp file so the subprocess can read it
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            pil_img = Image.fromarray(image) if isinstance(image, np.ndarray) else image
            pil_img.convert("RGB").save(tmp_path)

            worker_config = json.dumps({
                "model_id": config.get("model_id", "deepseek-ai/DeepSeek-OCR-2"),
                "device": config.get("device", "cuda:0"),
                "ocr_mode": config.get("ocr_mode", "document"),
                "strip_markdown": config.get("strip_markdown", False),
                "base_size": config.get("base_size", 1024),
                "image_size": config.get("image_size", 640),
                "crop_mode": config.get("crop_mode", True),
                "repetition_penalty": config.get("repetition_penalty", 1.3),
            })

            result = subprocess.run(
                [str(self._venv_python), str(_WORKER_SCRIPT), worker_config, tmp_path],
                capture_output=True, text=True, timeout=300,
                start_new_session=True,  # isolate from terminal SIGINT
            )

            def _parse_worker_output(raw: str) -> dict:
                """Worker stdout may contain model debug lines before the JSON result.
                The JSON is always the last non-empty line."""
                last_line = next(
                    (l for l in reversed(raw.splitlines()) if l.strip()), ""
                )
                return json.loads(last_line)

            if result.returncode != 0:
                try:
                    output = _parse_worker_output(result.stdout)
                    if "error" in output:
                        err_msg = output["error"]
                        tb = output.get("traceback", "")
                        logger.error("[DeepSeek-OCR] Worker error: %s", err_msg)
                        if tb:
                            logger.debug("[DeepSeek-OCR] Traceback:\n%s", tb)
                        return TranscriptionResult.error_result(f"{err_msg}")
                except (json.JSONDecodeError, ValueError):
                    pass
                stderr = result.stderr[-2000:] if result.stderr else "(no stderr)"
                logger.error("[DeepSeek-OCR] Worker exited %d: %s", result.returncode, stderr)
                return TranscriptionResult(text="[DeepSeek-OCR error — see log]", confidence=0.0)

            output = _parse_worker_output(result.stdout)

            if "error" in output:
                logger.error("[DeepSeek-OCR] Worker error: %s", output["error"])
                return TranscriptionResult.error_result(f"{output['error']}")

            text = output.get("text", "")
            return TranscriptionResult(
                text=text,
                confidence=1.0,  # DeepSeek-OCR-2 does not return per-token confidence
                metadata={
                    "engine": "DeepSeek-OCR",
                    "model": output.get("model_id", "deepseek-ai/DeepSeek-OCR-2"),
                    "ocr_mode": output.get("ocr_mode", config.get("ocr_mode", "document")),
                },
            )

        except subprocess.TimeoutExpired:
            logger.error("[DeepSeek-OCR] Subprocess timed out after 300s")
            return TranscriptionResult(text="[DeepSeek-OCR timed out]", confidence=0.0)
        except json.JSONDecodeError as e:
            logger.error("[DeepSeek-OCR] Failed to parse worker output: %s", e)
            return TranscriptionResult(text="[DeepSeek-OCR output parse error]", confidence=0.0)
        except Exception as e:
            logger.error("[DeepSeek-OCR] Unexpected error: %s", e)
            return TranscriptionResult.error_result(f"{e}")
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def transcribe_lines(
        self, images: List[np.ndarray], config: Optional[Dict[str, Any]] = None
    ) -> List[TranscriptionResult]:
        """Transcribe multiple images (each as a full page)."""
        return [self.transcribe_line(img, config) for img in images]

    def supports_batch(self) -> bool:
        return False

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
            "layout_detection": True,
        }
