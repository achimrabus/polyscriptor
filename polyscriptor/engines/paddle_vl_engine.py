"""
PaddleOCR-VL Engine Plugin

Wraps PaddleOCR-VL-1.5 (a 0.9B vision-language document-parsing model) as a
whole-page OCR engine for Polyscriptor. Unlike the classic PaddleOCR engine,
this is a VLM: it reads the full page and emits markdown/text directly — no
pre-segmented lines and no separate detection/recognition models.

IMPORTANT: PaddleOCR-VL lives in its OWN isolated venv so it never disturbs the
classic PaddleOCR engine in venv_paddle. The classic engine does not need
`transformers`; the VL pipeline does. Keeping them apart avoids version
conflicts (paddlex / transformers).

The venv location defaults to ``venv_paddle_vl`` in the project root (same
convention as venv_paddle / venv_deepseek / venv_chandra) and can be pointed
anywhere via the ``POLYSCRIPTOR_PADDLE_VL_VENV`` environment variable (or the
``venv_path`` config field / GUI browse button). Model/HF caches default to
subdirectories inside the venv directory (self-contained); override the cache
root with ``POLYSCRIPTOR_PADDLE_VL_CACHE``.

Setup (one-time; see requirements-paddle-vl.txt):
    python3.12 -m venv venv_paddle_vl
    source venv_paddle_vl/bin/activate
    pip install paddlepaddle-gpu==3.2.1 \
        -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
    pip install -U "paddleocr[doc-parser]>=3.4.0"
    deactivate

This engine calls paddle_vl_worker.py as a subprocess inside that venv. The
main venv never imports PaddleOCR-VL directly. The worker's model/HF cache is
redirected to the cache root so a near-full root filesystem stays clean.

Note on Cyrillic: PaddleOCR-VL targets multilingual *document parsing*; Cyrillic
handwriting is not officially advertised. Treat results as experimental and
compare against TrOCR / CRNN-CTC for historical Slavic manuscripts.

Batch CLI usage:
    python batch_processing.py --engine PaddleOCR-VL --input-folder pages/
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


# Isolated venv location. Override with POLYSCRIPTOR_PADDLE_VL_VENV (or the
# venv_path config field). Defaults to venv_paddle_vl in the project root,
# same convention as the other engine venvs.
def _default_venv() -> Path:
    env = os.environ.get("POLYSCRIPTOR_PADDLE_VL_VENV")
    if env:
        return Path(env).expanduser()
    return Path(__file__).resolve().parent.parent / "venv_paddle_vl"


# Model / HF cache root. Override with POLYSCRIPTOR_PADDLE_VL_CACHE; otherwise
# caches live inside the venv directory so the engine stays self-contained.
def _default_cache_root(venv: Path) -> Path:
    env = os.environ.get("POLYSCRIPTOR_PADDLE_VL_CACHE")
    if env:
        return Path(env).expanduser()
    return venv


_DEFAULT_VENV = _default_venv()
_DEFAULT_CACHE_ROOT = _default_cache_root(_DEFAULT_VENV)

_DEFAULT_PROMPT = "OCR:"
_DEFAULT_PIPELINE_VERSION = "v1.5"

# Path to the worker script (same engines/ directory)
_WORKER_SCRIPT = Path(__file__).resolve().parent / "paddle_vl_worker.py"


from polyscriptor.engines._venv_worker import find_venv_python as _find_venv_python, venv_has_package as _venv_has_probe_fn


_venv_has_probe = lambda p: _venv_has_probe_fn(p, 'paddleocr')


class PaddleOCRVLEngine(HTREngine):
    """
    PaddleOCR-VL whole-page OCR engine (subprocess mode).

    Calls paddle_vl_worker.py via an isolated venv Python interpreter so the
    VLM pipeline (paddleocr[doc-parser] + transformers) never conflicts with
    the classic PaddleOCR engine or the main venv.
    """

    def __init__(self):
        self._venv_path: Path = _DEFAULT_VENV
        self._cache_root: Path = _DEFAULT_CACHE_ROOT
        self._venv_python: Optional[Path] = None
        self._is_loaded: bool = False


    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    def get_name(self) -> str:
        return "PaddleOCR-VL"

    def get_description(self) -> str:
        return "PaddleOCR-VL-1.5: 0.9B vision-language whole-page document parser (subprocess mode)"

    def get_aliases(self) -> List[str]:
        return ["paddle-vl", "paddleocr-vl", "paddleocrvl"]

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
                f"PaddleOCR-VL venv not found at: {self._venv_path}\n\n"
                "Create it with:\n"
                f"  python3.12 -m venv {self._venv_path}\n"
                f"  source {self._venv_path}/bin/activate\n"
                "  pip install paddlepaddle-gpu==3.2.1 "
                "-i https://www.paddlepaddle.org.cn/packages/stable/cu126/\n"
                "  pip install -U 'paddleocr[doc-parser]>=3.4.0'\n"
            )
        if not _venv_has_probe(self._venv_path):
            return (
                f"paddleocr not installed in {self._venv_path}\n\n"
                "Install with:\n"
                f"  source {self._venv_path}/bin/activate\n"
                "  pip install -U 'paddleocr[doc-parser]>=3.4.0'\n"
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
        # Keep the model/HF cache next to the configured venv (unless an explicit
        # POLYSCRIPTOR_PADDLE_VL_CACHE override is set).
        self._cache_root = _default_cache_root(venv_path)

        python = _find_venv_python(venv_path)
        if python is None:
            logger.error("[PaddleOCR-VL] venv Python not found at %s", venv_path)
            return False

        if not _WORKER_SCRIPT.exists():
            logger.error("[PaddleOCR-VL] Worker script not found: %s", _WORKER_SCRIPT)
            return False

        self._venv_python = python
        self._is_loaded = True
        logger.info("[PaddleOCR-VL] Ready — venv: %s, worker: %s", python, _WORKER_SCRIPT)
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
        return False  # PaddleOCR-VL parses the whole page itself

    def transcribe_line(
        self, image: np.ndarray, config: Optional[Dict[str, Any]] = None
    ) -> TranscriptionResult:
        """
        Transcribe a full page image via the PaddleOCR-VL subprocess.

        Despite the method name, page-based engines receive the full page here.
        """
        if not self._is_loaded or self._venv_python is None:
            return TranscriptionResult(text="[PaddleOCR-VL not loaded]", confidence=0.0)

        if config is None:
            config = self.get_config()

        # Write image to a temp file so the subprocess can read it
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            pil_img = Image.fromarray(image) if isinstance(image, np.ndarray) else image
            pil_img.convert("RGB").save(tmp_path)

            config_json = json.dumps({
                "use_gpu": config.get("use_gpu", True),
                "gpu_index": int(config.get("gpu_index", 0)),
                "prompt": config.get("prompt", _DEFAULT_PROMPT),
                "pipeline_version": config.get("pipeline_version", _DEFAULT_PIPELINE_VERSION),
                "prompt_label": config.get("prompt_label") or None,
                "max_new_tokens": config.get("max_new_tokens") or None,
                "repetition_penalty": config.get("repetition_penalty") or None,
                "cache_root": str(self._cache_root),
            })

            # Redirect model/HF caches to /data so the near-full root FS stays clean.
            env = os.environ.copy()
            env.setdefault("HF_HOME", str(self._cache_root / "hf_cache"))
            env.setdefault("PADDLE_PDX_CACHE_HOME", str(self._cache_root / "paddlex_cache"))
            env["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
            # Pin the physical GPU here; the worker addresses it as gpu:0 internally
            # (per feedback_cuda_visible_devices: never combine CUDA_VISIBLE_DEVICES=N + gpu:N).
            if config.get("use_gpu", True):
                env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
                env["CUDA_VISIBLE_DEVICES"] = str(int(config.get("gpu_index", 0)))

            result = subprocess.run(
                [str(self._venv_python), str(_WORKER_SCRIPT), config_json, tmp_path],
                capture_output=True, text=True, timeout=600,
                start_new_session=True,  # isolate from terminal SIGINT
                env=env,
            )

            if result.returncode != 0:
                try:
                    output = json.loads(result.stdout)
                    if "error" in output:
                        err_msg = output["error"]
                        tb = output.get("traceback", "")
                        logger.error("[PaddleOCR-VL] Worker error: %s", err_msg)
                        if tb:
                            logger.debug("[PaddleOCR-VL] Traceback:\n%s", tb)
                        return TranscriptionResult.error_result(f"{err_msg}")
                except (json.JSONDecodeError, ValueError):
                    pass
                stderr = result.stderr[-2000:] if result.stderr else "(no stderr)"
                logger.error("[PaddleOCR-VL] Worker exited %d: %s", result.returncode, stderr)
                return TranscriptionResult(text="[PaddleOCR-VL error — see log]", confidence=0.0)

            output = json.loads(result.stdout)

            if "error" in output:
                logger.error("[PaddleOCR-VL] Worker error: %s", output["error"])
                return TranscriptionResult.error_result(f"{output['error']}")

            text = output.get("text", "")
            lines = output.get("lines", [])
            if not text and lines:
                text = "\n".join(lines)
            # PaddleOCR-VL (markdown pipeline) returns no per-token confidence.
            # Report None (not 0.0) so the UI shows "no confidence" instead of a
            # misleading 0%.
            confidences = output.get("confidences", [])
            mean_conf = float(np.mean(confidences)) if confidences else None

            return TranscriptionResult(
                text=text,
                confidence=mean_conf,
                metadata={
                    "engine": "PaddleOCR-VL",
                    "model": "PaddlePaddle/PaddleOCR-VL-1.5",
                    "line_count": len(lines) if lines else text.count("\n") + 1 if text else 0,
                    "use_gpu": output.get("use_gpu"),
                    "prompt": output.get("prompt"),
                    "has_markdown": bool(output.get("markdown")),
                },
            )

        except subprocess.TimeoutExpired:
            logger.error("[PaddleOCR-VL] Subprocess timed out after 600s")
            return TranscriptionResult(text="[PaddleOCR-VL timed out]", confidence=0.0)
        except json.JSONDecodeError as e:
            logger.error("[PaddleOCR-VL] Failed to parse worker output: %s", e)
            return TranscriptionResult(text="[PaddleOCR-VL output parse error]", confidence=0.0)
        except Exception as e:
            logger.error("[PaddleOCR-VL] Unexpected error: %s", e)
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
