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

from htr_engine_base import HTREngine, TranscriptionResult

logger = logging.getLogger(__name__)

try:
    from PyQt6.QtWidgets import (
        QComboBox, QFileDialog, QGroupBox,
        QHBoxLayout, QLabel, QLineEdit, QPlainTextEdit, QPushButton,
        QSpinBox, QVBoxLayout, QWidget,
    )
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False
    QWidget = object

_DEFAULT_MODEL_ID = "datalab-to/chandra-ocr-2"
_DEFAULT_MAX_NEW_TOKENS = 4096
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Path to the worker script (same engines/ directory)
_WORKER_SCRIPT = Path(__file__).resolve().parent / "chandra_worker.py"


def _default_venv() -> Path:
    env = os.environ.get("POLYSCRIPTOR_CHANDRA_VENV")
    if env:
        return Path(env).expanduser()
    return _PROJECT_ROOT / "venv_chandra"


def _find_venv_python(venv_path: Path) -> Optional[Path]:
    """Return the Python interpreter inside a venv, or None if not found."""
    import sys as _sys
    if _sys.platform == "win32":
        candidates = [venv_path / "Scripts" / "python.exe"]
    else:
        candidates = [venv_path / "bin" / "python", venv_path / "bin" / "python3"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _venv_has_chandra(venv_path: Path) -> bool:
    """Check that chandra-ocr is installed in the venv (filesystem check, no import)."""
    import sys as _sys
    if _sys.platform == "win32":
        candidates = [venv_path / "Lib" / "site-packages" / "chandra"]
    else:
        candidates = list((venv_path / "lib").glob("python*/site-packages/chandra"))
    for sp_dir in candidates:
        if sp_dir.is_dir():
            return True
    return False


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

        # Config widget references
        self._config_widget: Optional[QWidget] = None
        self._venv_edit: Optional[QLineEdit] = None
        self._device_combo: Optional[QComboBox] = None
        self._gpu_index_spin: Optional[QSpinBox] = None
        self._prompt_mode_combo: Optional[QComboBox] = None
        self._custom_prompt_edit: Optional[QPlainTextEdit] = None
        self._max_tokens_spin: Optional[QSpinBox] = None

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
        return _venv_has_chandra(self._venv_path)

    def get_unavailable_reason(self) -> str:
        if _find_venv_python(self._venv_path) is None:
            return (
                f"Chandra venv not found at: {self._venv_path}\n\n"
                "Create it with:\n"
                f"  python3 -m venv {self._venv_path}\n"
                f"  source {self._venv_path}/bin/activate\n"
                "  pip install 'chandra-ocr[hf]'\n"
            )
        if not _venv_has_chandra(self._venv_path):
            return (
                f"chandra-ocr not installed in {self._venv_path}\n\n"
                "Install with:\n"
                f"  source {self._venv_path}/bin/activate\n"
                "  pip install 'chandra-ocr[hf]'\n"
            )
        return ""

    # ------------------------------------------------------------------
    # Configuration widget
    # ------------------------------------------------------------------

    def get_config_widget(self) -> QWidget:
        if self._config_widget is not None:
            return self._config_widget

        widget = QWidget()
        layout = QVBoxLayout()

        # Venv path
        venv_group = QGroupBox("Chandra venv")
        venv_layout = QVBoxLayout()
        venv_layout.addWidget(QLabel("Path to isolated venv:"))
        venv_row = QHBoxLayout()
        self._venv_edit = QLineEdit(str(self._venv_path))
        self._venv_edit.setToolTip(
            "Isolated Python venv with chandra-ocr[hf] installed.\n"
            f"Default: {_default_venv()}\n"
            "Kept separate because Chandra needs transformers >= 5.x."
        )
        venv_row.addWidget(self._venv_edit)
        btn_browse = QPushButton("Browse…")
        btn_browse.clicked.connect(self._browse_venv)
        venv_row.addWidget(btn_browse)
        venv_layout.addLayout(venv_row)
        venv_group.setLayout(venv_layout)
        layout.addWidget(venv_group)

        # Device
        device_group = QGroupBox("Device")
        device_layout = QHBoxLayout()
        device_layout.addWidget(QLabel("Device:"))
        self._device_combo = QComboBox()
        self._device_combo.addItems(["GPU", "CPU"])
        self._device_combo.setToolTip(
            "GPU strongly recommended — 5B model, ~12 GB VRAM in bf16.\n"
            "CPU works but is extremely slow."
        )
        device_layout.addWidget(self._device_combo)
        device_layout.addWidget(QLabel("GPU index:"))
        self._gpu_index_spin = QSpinBox()
        self._gpu_index_spin.setRange(0, 15)
        self._gpu_index_spin.setValue(0)
        self._gpu_index_spin.setToolTip(
            "Physical GPU index (CUDA_VISIBLE_DEVICES). The worker pins this GPU "
            "and addresses it as cuda:0 internally."
        )
        device_layout.addWidget(self._gpu_index_spin)
        device_layout.addStretch()
        device_group.setLayout(device_layout)
        layout.addWidget(device_group)

        # Prompt mode
        prompt_group = QGroupBox("Prompt")
        prompt_layout = QVBoxLayout()
        prompt_row = QHBoxLayout()
        prompt_row.addWidget(QLabel("Mode:"))
        self._prompt_mode_combo = QComboBox()
        self._prompt_mode_combo.addItems(["ocr", "ocr_layout", "custom"])
        self._prompt_mode_combo.setToolTip(
            "ocr        – plain OCR to HTML (default, recommended)\n"
            "ocr_layout – HTML with layout blocks and bounding boxes\n"
            "custom     – use the custom prompt below"
        )
        prompt_row.addWidget(self._prompt_mode_combo)
        prompt_row.addWidget(QLabel("Max new tokens:"))
        self._max_tokens_spin = QSpinBox()
        self._max_tokens_spin.setRange(64, 16384)
        self._max_tokens_spin.setValue(_DEFAULT_MAX_NEW_TOKENS)
        prompt_row.addWidget(self._max_tokens_spin)
        prompt_row.addStretch()
        prompt_layout.addLayout(prompt_row)
        prompt_layout.addWidget(QLabel("Custom prompt (mode = custom):"))
        self._custom_prompt_edit = QPlainTextEdit("")
        self._custom_prompt_edit.setMaximumHeight(60)
        prompt_layout.addWidget(self._custom_prompt_edit)
        prompt_group.setLayout(prompt_layout)
        layout.addWidget(prompt_group)

        # Info
        info = QLabel(
            "Chandra 2 (5B VLM) runs in an isolated venv. First run downloads ~10 GB\n"
            "of weights to the HF cache. Each transcription spawns a subprocess that\n"
            "loads the model (~30 s) and OCRs the page.\n\n"
            "Universal engine: 90+ languages, print + handwriting, tables, math.\n"
            "Good default for modern documents and materials without a specialized\n"
            "model; for historical scripts a fine-tuned CRNN-CTC / TrOCR model is\n"
            "far more accurate."
        )
        info.setStyleSheet("color: gray; font-size: 9pt; padding: 8px;")
        info.setWordWrap(True)
        layout.addWidget(info)

        layout.addStretch()
        widget.setLayout(layout)
        self._config_widget = widget
        return widget

    def _browse_venv(self):
        folder = QFileDialog.getExistingDirectory(
            self._config_widget, "Select Chandra venv directory", str(self._venv_path)
        )
        if folder:
            self._venv_edit.setText(folder)

    # ------------------------------------------------------------------
    # Config get / set
    # ------------------------------------------------------------------

    def get_config(self) -> Dict[str, Any]:
        if self._config_widget is None:
            return {
                "venv_path": str(self._venv_path),
                "model_id": _DEFAULT_MODEL_ID,
                "use_gpu": True,
                "gpu_index": 0,
                "prompt_mode": "ocr",
                "custom_prompt": "",
                "max_new_tokens": _DEFAULT_MAX_NEW_TOKENS,
            }
        return {
            "venv_path": self._venv_edit.text().strip(),
            "model_id": _DEFAULT_MODEL_ID,
            "use_gpu": self._device_combo.currentText() == "GPU",
            "gpu_index": self._gpu_index_spin.value(),
            "prompt_mode": self._prompt_mode_combo.currentText(),
            "custom_prompt": self._custom_prompt_edit.toPlainText().strip(),
            "max_new_tokens": self._max_tokens_spin.value(),
        }

    def set_config(self, config: Dict[str, Any]):
        if self._config_widget is None:
            return
        if "venv_path" in config:
            self._venv_edit.setText(config["venv_path"])
        self._device_combo.setCurrentText("GPU" if config.get("use_gpu", True) else "CPU")
        self._gpu_index_spin.setValue(int(config.get("gpu_index", 0)))
        if config.get("prompt_mode"):
            self._prompt_mode_combo.setCurrentText(config["prompt_mode"])
        self._custom_prompt_edit.setPlainText(config.get("custom_prompt", ""))
        self._max_tokens_spin.setValue(int(config.get("max_new_tokens", _DEFAULT_MAX_NEW_TOKENS)))

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

            result = subprocess.run(
                [str(self._venv_python), str(_WORKER_SCRIPT), config_json, tmp_path],
                capture_output=True, text=True, timeout=900,
                start_new_session=True,  # isolate from terminal SIGINT
                env=env,
            )

            if result.returncode != 0:
                try:
                    output = json.loads(result.stdout)
                    if "error" in output:
                        err_msg = output["error"]
                        tb = output.get("traceback", "")
                        logger.error("[Chandra] Worker error: %s", err_msg)
                        if tb:
                            logger.debug("[Chandra] Traceback:\n%s", tb)
                        return TranscriptionResult(text=f"[Error: {err_msg}]", confidence=0.0)
                except (json.JSONDecodeError, ValueError):
                    pass
                stderr = result.stderr[-2000:] if result.stderr else "(no stderr)"
                logger.error("[Chandra] Worker exited %d: %s", result.returncode, stderr)
                return TranscriptionResult(text="[Chandra error — see log]", confidence=0.0)

            output = json.loads(result.stdout)

            if "error" in output:
                logger.error("[Chandra] Worker error: %s", output["error"])
                return TranscriptionResult(text=f"[Error: {output['error']}]", confidence=0.0)

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
            return TranscriptionResult(text=f"[Error: {e}]", confidence=0.0)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # Capabilities
    # ------------------------------------------------------------------

    def get_capabilities(self) -> Dict[str, bool]:
        return {
            "batch_processing": False,
            "confidence_scores": False,
            "beam_search": False,
            "language_model": True,
            "preprocessing": True,
        }
