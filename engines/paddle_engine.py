"""
PaddleOCR Engine Plugin

Wraps PaddleOCR as a whole-page OCR engine for Polyscriptor.
PaddleOCR performs its own text detection + recognition — no pre-segmented lines needed.

IMPORTANT: PaddleOCR must be installed in a SEPARATE venv (venv_paddle) to avoid
opencv-contrib-python conflicting with the main venv's opencv-python.

Setup:
    python -m venv venv_paddle
    source venv_paddle/bin/activate
    pip install paddlepaddle paddleocr          # CPU
    # OR
    pip install paddlepaddle-gpu paddleocr      # GPU (CUDA 11.2+ / 12.x)
    deactivate

This engine calls paddle_worker.py as a subprocess inside venv_paddle.
The main venv never imports PaddleOCR directly.

Supported language codes (sample):
    'ch'       - Chinese + English (PaddleOCR's strongest model)
    'en'       - English
    'german'   - German
    'fr'       - French
    'japan'    - Japanese
    'korean'   - Korean
    'latin'    - Latin script (generic)
    'arabic'   - Arabic
    'cyrillic' - Cyrillic (Russian, Ukrainian, etc.)
    Full list: https://paddlepaddle.github.io/PaddleOCR/main/en/ppocr/blog/multi_languages.html

Batch CLI usage:
    python batch_processing.py --engine PaddleOCR --input-folder pages/
"""

import json
import logging
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
        QCheckBox, QComboBox, QFileDialog, QGroupBox,
        QHBoxLayout, QLabel, QLineEdit, QPushButton,
        QSpinBox, QVBoxLayout, QWidget,
    )
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False
    QWidget = object

# Default venv path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_VENV = _PROJECT_ROOT / "venv_paddle"

_DEFAULT_LANG = "en"

# Path to the worker script (same engines/ directory)
_WORKER_SCRIPT = Path(__file__).resolve().parent / "paddle_worker.py"


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


def _venv_has_paddleocr(venv_path: Path) -> bool:
    """Check that paddleocr is installed in the venv via filesystem (no subprocess, no import)."""
    import sys as _sys
    if _sys.platform == "win32":
        candidates = [(venv_path / "Lib" / "site-packages" / "paddleocr",)]
    else:
        candidates = list((venv_path / "lib").glob("python*/site-packages/paddleocr"))
    for sp_dir in candidates:
        if sp_dir.is_dir():
            return True
    return False


class PaddleOCREngine(HTREngine):
    """
    PaddleOCR whole-page OCR engine (subprocess mode).

    Calls paddle_worker.py via the venv_paddle Python interpreter so PaddleOCR
    lives in an isolated venv and never conflicts with the main venv's OpenCV.
    """

    def __init__(self):
        self._venv_path: Path = _DEFAULT_VENV
        self._venv_python: Optional[Path] = None
        self._is_loaded: bool = False

        # Config widget references
        self._config_widget: Optional[QWidget] = None
        self._venv_edit: Optional[QLineEdit] = None
        self._lang_edit: Optional[QLineEdit] = None
        self._device_combo: Optional[QComboBox] = None
        self._angle_cls_check: Optional[QCheckBox] = None
        self._det_db_thresh_spin: Optional[QSpinBox] = None

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    def get_name(self) -> str:
        return "PaddleOCR"

    def get_description(self) -> str:
        return "PaddleOCR: multi-language text detection + recognition (subprocess mode)"

    def get_aliases(self) -> List[str]:
        return ["paddle", "paddleocr"]

    # ------------------------------------------------------------------
    # Availability
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        if _find_venv_python(self._venv_path) is None:
            return False
        return _venv_has_paddleocr(self._venv_path)

    def get_unavailable_reason(self) -> str:
        if _find_venv_python(self._venv_path) is None:
            return (
                f"venv_paddle not found at: {self._venv_path}\n\n"
                "Create it with:\n"
                f"  python -m venv {self._venv_path}\n"
                f"  source {self._venv_path}/bin/activate\n"
                "  pip install paddlepaddle paddleocr      # CPU\n"
                "  pip install paddlepaddle-gpu paddleocr  # GPU\n"
            )
        if not _venv_has_paddleocr(self._venv_path):
            return (
                f"paddleocr not installed in {self._venv_path}\n\n"
                "Install with:\n"
                f"  source {self._venv_path}/bin/activate\n"
                "  pip install paddlepaddle paddleocr      # CPU\n"
                "  pip install paddlepaddle-gpu paddleocr  # GPU\n"
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
        venv_group = QGroupBox("PaddleOCR venv")
        venv_layout = QVBoxLayout()
        venv_layout.addWidget(QLabel("Path to venv_paddle:"))
        venv_row = QHBoxLayout()
        self._venv_edit = QLineEdit(str(self._venv_path))
        self._venv_edit.setToolTip(
            "Path to the isolated Python venv with PaddleOCR installed.\n"
            "Default: <project>/venv_paddle\n"
            "Create with: python -m venv venv_paddle && pip install paddlepaddle paddleocr"
        )
        venv_row.addWidget(self._venv_edit)
        btn_browse = QPushButton("Browse…")
        btn_browse.clicked.connect(self._browse_venv)
        venv_row.addWidget(btn_browse)
        venv_layout.addLayout(venv_row)
        venv_group.setLayout(venv_layout)
        layout.addWidget(venv_group)

        # Language
        lang_group = QGroupBox("Language / Script")
        lang_layout = QVBoxLayout()
        lang_layout.addWidget(QLabel("Language code:"))
        self._lang_edit = QLineEdit(_DEFAULT_LANG)
        self._lang_edit.setToolTip(
            "PaddleOCR language code. Examples:\n"
            "  en        – English\n"
            "  ch        – Chinese + English (strongest model)\n"
            "  de/french/german – German / French\n"
            "  ru/uk/bg  – Russian / Ukrainian / Bulgarian (Cyrillic)\n"
            "  la        – Latin\n"
            "  ar        – Arabic\n"
            "  japan/korean – Japanese / Korean\n"
            "Full list: paddlepaddle.github.io/PaddleOCR/main/en/ppocr/blog/multi_languages.html"
        )
        lang_layout.addWidget(self._lang_edit)
        lang_group.setLayout(lang_layout)
        layout.addWidget(lang_group)

        # Device
        device_group = QGroupBox("Device")
        device_layout = QHBoxLayout()
        device_layout.addWidget(QLabel("Device:"))
        self._device_combo = QComboBox()
        self._device_combo.addItems(["CPU", "GPU (cuda:0)"])
        self._device_combo.setToolTip(
            "CPU: always available.\nGPU: requires paddlepaddle-gpu and CUDA."
        )
        device_layout.addWidget(self._device_combo)
        device_layout.addStretch()
        device_group.setLayout(device_layout)
        layout.addWidget(device_group)

        # Recognition options
        opts_group = QGroupBox("Recognition Options")
        opts_layout = QVBoxLayout()
        self._angle_cls_check = QCheckBox("Enable text-angle classifier (use_angle_cls)")
        self._angle_cls_check.setChecked(True)
        self._angle_cls_check.setToolTip(
            "Detect and correct 180° text rotation.\n"
            "Disable to save ~10% inference time on well-oriented pages."
        )
        opts_layout.addWidget(self._angle_cls_check)
        thresh_row = QHBoxLayout()
        thresh_row.addWidget(QLabel("Det. threshold (×100):"))
        self._det_db_thresh_spin = QSpinBox()
        self._det_db_thresh_spin.setRange(1, 99)
        self._det_db_thresh_spin.setValue(30)
        self._det_db_thresh_spin.setToolTip(
            "DB detection threshold × 100 (default 30 = 0.30).\n"
            "Lower = more regions detected; higher = stricter."
        )
        thresh_row.addWidget(self._det_db_thresh_spin)
        thresh_row.addStretch()
        opts_layout.addLayout(thresh_row)
        opts_group.setLayout(opts_layout)
        layout.addWidget(opts_group)

        # Info
        info = QLabel(
            "PaddleOCR runs in an isolated venv (venv_paddle) to avoid\n"
            "OpenCV conflicts. First run downloads model weights (~100–400 MB).\n"
            "Each transcription spawns a subprocess — expect ~5–15s per page."
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
            self._config_widget, "Select venv_paddle directory", str(self._venv_path)
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
                "lang": "en",
                "use_gpu": False,
                "use_angle_cls": True,
                "det_db_thresh": 0.30,
            }
        return {
            "venv_path": self._venv_edit.text().strip(),
            "lang": self._lang_edit.text().strip() or "en",
            "use_gpu": self._device_combo.currentText().startswith("GPU"),
            "use_angle_cls": self._angle_cls_check.isChecked(),
            "det_db_thresh": self._det_db_thresh_spin.value() / 100.0,
        }

    def set_config(self, config: Dict[str, Any]):
        if self._config_widget is None:
            return
        if "venv_path" in config:
            self._venv_edit.setText(config["venv_path"])
        self._lang_edit.setText(config.get("lang", "en"))
        self._device_combo.setCurrentText(
            "GPU (cuda:0)" if config.get("use_gpu", False) else "CPU"
        )
        self._angle_cls_check.setChecked(config.get("use_angle_cls", True))
        thresh_int = round(config.get("det_db_thresh", 0.30) * 100)
        self._det_db_thresh_spin.setValue(max(1, min(99, thresh_int)))

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_model(self, config: Dict[str, Any]) -> bool:
        """Validate venv + worker script exist. No actual model loading (lazy via subprocess)."""
        venv_path = Path(config.get("venv_path", str(self._venv_path)))
        self._venv_path = venv_path

        python = _find_venv_python(venv_path)
        if python is None:
            logger.error("[PaddleOCR] venv Python not found at %s", venv_path)
            return False

        if not _WORKER_SCRIPT.exists():
            logger.error("[PaddleOCR] Worker script not found: %s", _WORKER_SCRIPT)
            return False

        self._venv_python = python
        self._is_loaded = True
        logger.info("[PaddleOCR] Ready — venv: %s, worker: %s", python, _WORKER_SCRIPT)
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
        return False  # PaddleOCR does its own detection

    def transcribe_line(
        self, image: np.ndarray, config: Optional[Dict[str, Any]] = None
    ) -> TranscriptionResult:
        """
        Transcribe a full page image via PaddleOCR subprocess.

        Despite the method name, page-based engines receive the full page here.
        """
        if not self._is_loaded or self._venv_python is None:
            return TranscriptionResult(text="[PaddleOCR not loaded]", confidence=0.0)

        if config is None:
            config = self.get_config()

        # Write image to a temp file so the subprocess can read it
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            pil_img = Image.fromarray(image) if isinstance(image, np.ndarray) else image
            pil_img.convert("RGB").save(tmp_path)

            config_json = json.dumps({
                "lang": config.get("lang", "en"),
                "use_gpu": config.get("use_gpu", False),
                "use_angle_cls": config.get("use_angle_cls", True),
                "det_db_thresh": config.get("det_db_thresh", 0.30),
            })

            result = subprocess.run(
                [str(self._venv_python), str(_WORKER_SCRIPT), config_json, tmp_path],
                capture_output=True, text=True, timeout=180,
                start_new_session=True,  # isolate from terminal SIGINT → prevents core dump on Ctrl+C
            )

            if result.returncode != 0:
                # Worker always writes a JSON error to stdout; parse that first for the best message.
                try:
                    output = json.loads(result.stdout)
                    if "error" in output:
                        err_msg = output["error"]
                        tb = output.get("traceback", "")
                        logger.error("[PaddleOCR] Worker error: %s", err_msg)
                        if tb:
                            logger.debug("[PaddleOCR] Traceback:\n%s", tb)
                        return TranscriptionResult(text=f"[Error: {err_msg}]", confidence=0.0)
                except (json.JSONDecodeError, ValueError):
                    pass
                stderr = result.stderr[-2000:] if result.stderr else "(no stderr)"
                logger.error("[PaddleOCR] Worker exited %d: %s", result.returncode, stderr)
                return TranscriptionResult(text="[PaddleOCR error — see log]", confidence=0.0)

            output = json.loads(result.stdout)

            if "error" in output:
                logger.error("[PaddleOCR] Worker error: %s", output["error"])
                return TranscriptionResult(text=f"[Error: {output['error']}]", confidence=0.0)

            lines = output.get("lines", [])
            confidences = output.get("confidences", [])
            text = "\n".join(lines)
            mean_conf = float(np.mean(confidences)) if confidences else 0.0

            return TranscriptionResult(
                text=text,
                confidence=mean_conf,
                metadata={
                    "engine": "PaddleOCR",
                    "lang": output.get("lang"),
                    "line_count": len(lines),
                    "use_gpu": output.get("use_gpu"),
                },
            )

        except subprocess.TimeoutExpired:
            logger.error("[PaddleOCR] Subprocess timed out after 120s")
            return TranscriptionResult(text="[PaddleOCR timed out]", confidence=0.0)
        except json.JSONDecodeError as e:
            logger.error("[PaddleOCR] Failed to parse worker output: %s", e)
            return TranscriptionResult(text="[PaddleOCR output parse error]", confidence=0.0)
        except Exception as e:
            logger.error("[PaddleOCR] Unexpected error: %s", e)
            return TranscriptionResult(text=f"[Error: {e}]", confidence=0.0)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # Capabilities
    # ------------------------------------------------------------------

    def get_capabilities(self) -> Dict[str, bool]:
        return {
            "batch_processing": False,
            "confidence_scores": True,
            "beam_search": False,
            "language_model": False,
            "preprocessing": True,
        }
