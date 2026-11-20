"""
dots.mocr Engine Plugin (subprocess mode)

Wraps dots.mocr (rednote-hilab/dots.mocr, ~3B, MIT — formerly dots.ocr-1.5) as a
page-level OCR engine for Polyscriptor.  dots.mocr is a Qwen2.5-VL-style
multilingual document-parsing VLM; it needs transformers==4.56.1 + trust_remote_code
+ qwen-vl-utils, which conflicts with the other engine venvs — hence the isolated
venv + subprocess worker pattern.

IMPORTANT: download the weights to a local directory WITHOUT periods in the name
(temporary upstream workaround), e.g. models/DotsMOCR:
    huggingface-cli download rednote-hilab/dots.mocr --local-dir models/DotsMOCR

Setup (venv_dots):
    python -m venv /path/to/dhlab-slavistik/venv_dots
    source venv_dots/bin/activate
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    pip install "transformers==4.56.1" accelerate qwen-vl-utils pillow
    # Optional (faster inference, requires CUDA toolkit headers):
    pip install flash-attn --no-build-isolation
    deactivate

Key features:
- Whole-page OCR via the official "prompt_ocr" prompt (plain text, headers/footers
  excluded by design) or full layout JSON
- Model card: strong on old scans and non-Latin scripts (Tibetan, Kannada, Russian showcases).
  Measured here (Nov 2026, line level): best engine on Latin-script print, Fraktur and
  typewriting, but 31-70 % CER on Cyrillic print and handwriting (see README)
- ~3B parameters, ~9 GB VRAM
- subprocess isolation: no transformers version conflicts
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
        QComboBox, QFileDialog, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
        QPushButton, QSpinBox, QVBoxLayout, QWidget,
    )
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False
    QWidget = object

# Default venv path (project root, same convention as venv_deepseek)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_VENV = _PROJECT_ROOT / "venv_dots"

# Worker script lives in the same engines/ directory as this file
_WORKER_SCRIPT = Path(__file__).resolve().parent / "dots_ocr_worker.py"

_DEFAULT_MODEL_PATH = "models/DotsMOCR"


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


def _venv_has_transformers(venv_path: Path) -> bool:
    """Check that transformers is installed in the venv (filesystem, no subprocess)."""
    import sys as _sys
    if _sys.platform == "win32":
        candidates = [(venv_path / "Lib" / "site-packages" / "transformers",)]
    else:
        candidates = list((venv_path / "lib").glob("python*/site-packages/transformers"))
    for sp_dir in candidates:
        if sp_dir.is_dir():
            return True
    return False


class DotsOCREngine(HTREngine):
    """
    dots.mocr page-level OCR engine (subprocess mode).

    Calls dots_ocr_worker.py via the venv_dots Python interpreter so dots.mocr's
    transformers==4.56.1 dependency lives in an isolated venv and never
    conflicts with the main venv or the other engine venvs.
    """

    def __init__(self):
        self._venv_path: Path = _DEFAULT_VENV
        self._venv_python: Optional[Path] = None
        self._is_loaded: bool = False

        # Config widget references
        self._config_widget: Optional[QWidget] = None
        self._venv_edit: Optional[QLineEdit] = None
        self._model_path_edit: Optional[QLineEdit] = None
        self._prompt_mode_combo: Optional[QComboBox] = None
        self._max_new_tokens_spin: Optional[QSpinBox] = None
        self._device_combo: Optional[QComboBox] = None

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    def get_name(self) -> str:
        return "dots.mocr"

    def get_description(self) -> str:
        return "dots.mocr (rednote, ~3B, MIT): page OCR, best here on Latin-script print and Fraktur; not for Cyrillic"

    def get_aliases(self) -> List[str]:
        return ["dots", "dots-ocr", "dotsocr", "dots.mocr"]

    # ------------------------------------------------------------------
    # Availability (filesystem checks only — no subprocess, no imports)
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        if _find_venv_python(self._venv_path) is None:
            return False
        return _venv_has_transformers(self._venv_path)

    def get_unavailable_reason(self) -> str:
        if _find_venv_python(self._venv_path) is None:
            return (
                f"venv_dots not found at: {self._venv_path}\n\n"
                "Create it with:\n"
                f"  python -m venv {self._venv_path}\n"
                f"  source {self._venv_path}/bin/activate\n"
                '  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121\n'
                '  pip install "transformers==4.56.1" accelerate qwen-vl-utils pillow\n'
                "  deactivate\n\n"
                "Then download the weights to models/DotsMOCR (no periods in the name!):\n"
                "  huggingface-cli download rednote-hilab/dots.mocr --local-dir models/DotsMOCR"
            )
        if not _venv_has_transformers(self._venv_path):
            return (
                f"transformers not installed in {self._venv_path}\n\n"
                "Install with:\n"
                f"  source {self._venv_path}/bin/activate\n"
                '  pip install "transformers==4.56.1" accelerate qwen-vl-utils pillow\n'
                "  deactivate"
            )
        return ""

    # ------------------------------------------------------------------
    # Configuration widget
    # ------------------------------------------------------------------

    def get_config_widget(self) -> QWidget:
        """Create dots.mocr configuration panel."""
        if self._config_widget is not None:
            return self._config_widget

        widget = QWidget()
        layout = QVBoxLayout()

        # Venv path
        venv_group = QGroupBox("dots.mocr venv")
        venv_layout = QVBoxLayout()
        venv_layout.addWidget(QLabel("Path to venv_dots:"))
        venv_row = QHBoxLayout()
        self._venv_edit = QLineEdit(str(self._venv_path))
        self._venv_edit.setToolTip(
            "Path to the isolated Python venv with dots.mocr dependencies.\n"
            "Default: <project>/venv_dots\n"
            'Requires transformers == 4.56.1 and qwen-vl-utils.'
        )
        venv_row.addWidget(self._venv_edit)
        btn_browse = QPushButton("Browse…")
        btn_browse.clicked.connect(self._browse_venv)
        venv_row.addWidget(btn_browse)
        venv_layout.addLayout(venv_row)
        venv_group.setLayout(venv_layout)
        layout.addWidget(venv_group)

        # Model path
        model_group = QGroupBox("Model Weights")
        model_layout = QVBoxLayout()
        model_layout.addWidget(QLabel("Local model directory (NO periods in the name!):"))
        model_row = QHBoxLayout()
        self._model_path_edit = QLineEdit(_DEFAULT_MODEL_PATH)
        self._model_path_edit.setToolTip(
            "Local path to the dots.mocr weights, relative to the project root.\n"
            "The upstream workaround requires a directory name without periods,\n"
            "e.g. models/DotsMOCR — NOT models/dots.mocr.\n\n"
            "Download with:\n"
            "  huggingface-cli download rednote-hilab/dots.mocr --local-dir models/DotsMOCR"
        )
        model_row.addWidget(self._model_path_edit)
        model_btn = QPushButton("Browse…")
        model_btn.clicked.connect(self._browse_model)
        model_row.addWidget(model_btn)
        model_layout.addLayout(model_row)
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)

        # Prompt mode
        mode_group = QGroupBox("Prompt Mode")
        mode_layout = QVBoxLayout()
        self._prompt_mode_combo = QComboBox()
        self._prompt_mode_combo.addItems([
            "OCR (plain text)", "Layout (JSON)", "Scene spotting",
        ])
        mode_layout.addWidget(QLabel("Mode:"))
        mode_layout.addWidget(self._prompt_mode_combo)
        mode_hint = QLabel(
            "OCR: plain text, page headers/footers excluded by design\n"
            "Layout: full layout info (bbox + category + text) as one JSON object\n"
            "Scene spotting: detect + recognise text in photos"
        )
        mode_hint.setStyleSheet("color: gray; font-size: 9pt;")
        mode_hint.setWordWrap(True)
        mode_layout.addWidget(mode_hint)
        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)

        # Output settings
        output_group = QGroupBox("Output Settings")
        output_layout = QVBoxLayout()
        tok_row = QHBoxLayout()
        tok_row.addWidget(QLabel("Max new tokens:"))
        self._max_new_tokens_spin = QSpinBox()
        self._max_new_tokens_spin.setRange(512, 24000)
        self._max_new_tokens_spin.setSingleStep(512)
        self._max_new_tokens_spin.setValue(8192)
        tok_row.addWidget(self._max_new_tokens_spin)
        tok_row.addStretch()
        output_layout.addLayout(tok_row)
        output_group.setLayout(output_layout)
        layout.addWidget(output_group)

        # Device selection
        device_group = QGroupBox("Device")
        device_layout = QHBoxLayout()
        device_layout.addWidget(QLabel("Device:"))
        self._device_combo = QComboBox()
        self._device_combo.addItems(["cuda:0", "cuda:1", "cpu"])
        self._device_combo.setToolTip(
            "GPU device for inference (runs inside venv_dots subprocess).\n"
            "cuda:0 is recommended; use cpu only as fallback (very slow)."
        )
        device_layout.addWidget(self._device_combo)
        device_layout.addStretch()
        device_group.setLayout(device_layout)
        layout.addWidget(device_group)

        # Info
        info = QLabel(
            "dots.mocr runs in an isolated venv (venv_dots) to avoid transformers\n"
            "version conflicts.  First run loads ~6 GB of local weights.\n"
            "Each call spawns a subprocess — expect ~20–90 s per page on GPU."
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
            self._config_widget, "Select venv_dots directory", str(self._venv_path)
        )
        if folder:
            self._venv_edit.setText(folder)

    def _browse_model(self):
        folder = QFileDialog.getExistingDirectory(
            self._config_widget, "Select dots.mocr weights directory",
            str(_PROJECT_ROOT / self._model_path_edit.text().strip())
        )
        if folder:
            try:
                self._model_path_edit.setText(str(Path(folder).relative_to(_PROJECT_ROOT)))
            except ValueError:
                self._model_path_edit.setText(folder)

    # ------------------------------------------------------------------
    # Config get / set
    # ------------------------------------------------------------------

    def get_config(self) -> Dict[str, Any]:
        if self._config_widget is None:
            return {
                "venv_path": str(self._venv_path),
                "model_path": _DEFAULT_MODEL_PATH,
                "prompt_mode": "ocr",
                "max_new_tokens": 8192,
                "device": "cuda:0",
            }

        mode_map = {
            "OCR (plain text)": "ocr",
            "Layout (JSON)": "layout",
            "Scene spotting": "scene",
        }

        return {
            "venv_path": self._venv_edit.text().strip() or str(self._venv_path),
            "model_path": self._model_path_edit.text().strip() or _DEFAULT_MODEL_PATH,
            "prompt_mode": mode_map.get(self._prompt_mode_combo.currentText(), "ocr"),
            "max_new_tokens": self._max_new_tokens_spin.value(),
            "device": self._device_combo.currentText(),
        }

    def set_config(self, config: Dict[str, Any]):
        if self._config_widget is None:
            return

        if "venv_path" in config:
            self._venv_edit.setText(config["venv_path"])
        if "model_path" in config:
            self._model_path_edit.setText(config["model_path"])

        mode_map = {
            "ocr": "OCR (plain text)",
            "layout": "Layout (JSON)",
            "scene": "Scene spotting",
        }
        self._prompt_mode_combo.setCurrentText(
            mode_map.get(config.get("prompt_mode", "ocr"), "OCR (plain text)")
        )

        self._max_new_tokens_spin.setValue(config.get("max_new_tokens", 8192))

        device = config.get("device", "cuda:0")
        idx = self._device_combo.findText(device)
        if idx >= 0:
            self._device_combo.setCurrentIndex(idx)

    # ------------------------------------------------------------------
    # Model loading (validates venv + worker; actual model loads lazily in subprocess)
    # ------------------------------------------------------------------

    def load_model(self, config: Dict[str, Any]) -> bool:
        """Validate venv + worker script exist. Actual model loading is lazy (in subprocess)."""
        venv_path = Path(config.get("venv_path", str(self._venv_path)))
        self._venv_path = venv_path

        # Sync venv_edit if widget is open
        if self._venv_edit is not None:
            self._venv_edit.setText(str(venv_path))

        python = _find_venv_python(venv_path)
        if python is None:
            logger.error("[dots.mocr] venv Python not found at %s", venv_path)
            return False

        if not _WORKER_SCRIPT.exists():
            logger.error("[dots.mocr] Worker script not found: %s", _WORKER_SCRIPT)
            return False

        self._venv_python = python
        self._is_loaded = True
        logger.info("[dots.mocr] Ready — venv: %s, worker: %s", python, _WORKER_SCRIPT)
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
        Transcribe a full page image via the dots.mocr subprocess.

        Despite the method name, page-based engines receive the full page here.
        """
        if not self._is_loaded or self._venv_python is None:
            return TranscriptionResult(text="[dots.mocr not loaded]", confidence=0.0)

        if config is None:
            config = self.get_config()

        # Write image to a temp file so the subprocess can read it
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            pil_img = Image.fromarray(image) if isinstance(image, np.ndarray) else image
            pil_img.convert("RGB").save(tmp_path)

            worker_config = json.dumps({
                "model_path": config.get("model_path", _DEFAULT_MODEL_PATH),
                "device": config.get("device", "cuda:0"),
                "prompt_mode": config.get("prompt_mode", "ocr"),
                "max_new_tokens": config.get("max_new_tokens", 8192),
            })

            result = subprocess.run(
                [str(self._venv_python), str(_WORKER_SCRIPT), worker_config, tmp_path],
                capture_output=True, text=True, timeout=900,
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
                        logger.error("[dots.mocr] Worker error: %s", err_msg)
                        if tb:
                            logger.debug("[dots.mocr] Traceback:\n%s", tb)
                        return TranscriptionResult(text=f"[Error: {err_msg}]", confidence=0.0)
                except (json.JSONDecodeError, ValueError):
                    pass
                stderr = result.stderr[-2000:] if result.stderr else "(no stderr)"
                logger.error("[dots.mocr] Worker exited %d: %s", result.returncode, stderr)
                return TranscriptionResult(text="[dots.mocr error — see log]", confidence=0.0)

            output = _parse_worker_output(result.stdout)

            if "error" in output:
                logger.error("[dots.mocr] Worker error: %s", output["error"])
                return TranscriptionResult(
                    text=f"[Error: {output['error']}]", confidence=0.0
                )

            text = output.get("text", "")
            return TranscriptionResult(
                text=text,
                confidence=1.0,  # dots.mocr does not return per-token confidence
                metadata={
                    "engine": "dots.mocr",
                    "model": output.get("model_path", _DEFAULT_MODEL_PATH),
                    "prompt_mode": output.get("prompt_mode", config.get("prompt_mode", "ocr")),
                },
            )

        except subprocess.TimeoutExpired:
            logger.error("[dots.mocr] Subprocess timed out after 900s")
            return TranscriptionResult(text="[dots.mocr timed out]", confidence=0.0)
        except json.JSONDecodeError as e:
            logger.error("[dots.mocr] Failed to parse worker output: %s", e)
            return TranscriptionResult(text="[dots.mocr output parse error]", confidence=0.0)
        except Exception as e:
            logger.error("[dots.mocr] Unexpected error: %s", e)
            return TranscriptionResult(text=f"[Error: {e}]", confidence=0.0)
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
            "batch_processing": False,
            "confidence_scores": False,
            "beam_search": False,
            "language_model": True,
            "preprocessing": True,
            "layout_detection": True,
        }
