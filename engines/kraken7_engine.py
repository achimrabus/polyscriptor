"""
Kraken 7 Engine Plugin (PP-OCRv6 line recognition, persistent subprocess worker)

kraken 7.1 added a PP-OCRv6 recognition architecture with multilingual base models
(tiny / small / medium, Apache-2.0, Zenodo 21788403 / 21788405 / 21788410); this is
what eScriptorium now offers. kraken 7 cannot live in the main venv (blla, the
Kraken engine and Party need kraken 6), so it runs in venv_kraken7.

Unlike the other subprocess engines, the worker is PERSISTENT: load_model() starts
`kraken7_worker.py --serve`, which loads the model once and then answers one line
per request over stdin/stdout. A page of 60 lines therefore costs one model load
(~1 s) plus ~30 ms per line, instead of one process start per call.

Setup (venv_kraken7):
    python3.12 -m venv venv_kraken7
    venv_kraken7/bin/pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128
    venv_kraken7/bin/pip install kraken==7.1.1

Models (models/kraken7/, see KRAKEN7_MODELS): `kraken get` stores each model under
~/.local/share/htrmopo/<uuid>/; link it to the preset name, e.g. for medium:
    venv_kraken7/bin/kraken get 10.5281/zenodo.21788410
    mkdir -p models/kraken7
    ln -s ~/.local/share/htrmopo/*/medium.safetensors models/kraken7/ppocr_medium.safetensors
small: 10.5281/zenodo.21788405 (small.safetensors), tiny: 10.5281/zenodo.21788403
(tiny.safetensors). Any other kraken 7 recognition model can be loaded as a custom file.
"""

import json
import logging
import os
import selectors
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

from htr_engine_base import HTREngine, TranscriptionResult

logger = logging.getLogger(__name__)

try:
    from PyQt6.QtWidgets import QComboBox, QFormLayout, QLabel, QLineEdit, QWidget
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False
    QWidget = object

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_VENV = _PROJECT_ROOT / "venv_kraken7"
_WORKER_SCRIPT = Path(__file__).resolve().parent / "kraken7_worker.py"
_MODEL_DIR = _PROJECT_ROOT / "models" / "kraken7"

# Preset name -> (file in models/kraken7, label). Base models without fine-tuning.
KRAKEN7_MODELS = {
    "ppocr_medium": ("ppocr_medium.safetensors",
                     "PP-OCRv6 medium (64 MB, multilingual base)"),
    "ppocr_small": ("ppocr_small.safetensors",
                    "PP-OCRv6 small (13 MB, multilingual base)"),
    "ppocr_tiny": ("ppocr_tiny.safetensors",
                   "PP-OCRv6 tiny (3 MB, multilingual base)"),
}
_DEFAULT_MODEL = "ppocr_medium"

_LOAD_TIMEOUT_S = 180
_LINE_TIMEOUT_S = 120


def _venv_python(venv: Path) -> Optional[Path]:
    for c in (venv / "bin" / "python", venv / "Scripts" / "python.exe"):
        if c.exists():
            return c
    return None


def _venv_has_kraken7(venv: Path) -> bool:
    return any(venv.glob("lib/python*/site-packages/kraken-7.*.dist-info"))


def resolve_model(value: str) -> Path:
    """Preset name or a path to a kraken 7 recognition model."""
    if value in KRAKEN7_MODELS:
        return _MODEL_DIR / KRAKEN7_MODELS[value][0]
    return Path(value).expanduser()


def available_presets() -> List[Dict[str, str]]:
    """Presets whose model file exists, for config schemas / combo boxes."""
    return [{"label": label, "value": key}
            for key, (fname, label) in KRAKEN7_MODELS.items()
            if (_MODEL_DIR / fname).exists()]


class _Worker:
    """One persistent kraken7_worker.py --serve process."""

    def __init__(self, python: Path, config: Dict[str, Any]):
        self.proc = subprocess.Popen(
            [str(python), str(_WORKER_SCRIPT), json.dumps(config), "--serve"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, encoding="utf-8", bufsize=1, cwd=str(_PROJECT_ROOT),
            start_new_session=True,  # no SIGINT from the terminal
        )
        self.sel = selectors.DefaultSelector()
        self.sel.register(self.proc.stdout, selectors.EVENT_READ)
        ready = self._read(_LOAD_TIMEOUT_S)
        if not ready.get("ready"):
            self.close()
            raise RuntimeError(ready.get("error", "worker did not start"))

    def alive(self) -> bool:
        return self.proc.poll() is None

    def _read(self, timeout: float) -> dict:
        if not self.sel.select(timeout):
            self.close()
            raise TimeoutError(f"kraken7 worker silent for {timeout:.0f}s")
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError(f"kraken7 worker exited ({self.proc.poll()})")
        return json.loads(line)

    def ask(self, request: dict, timeout: float = _LINE_TIMEOUT_S) -> dict:
        self.proc.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
        self.proc.stdin.flush()
        return self._read(timeout)

    def close(self):
        try:
            if self.alive():
                try:
                    self.proc.stdin.write('{"cmd": "quit"}\n')
                    self.proc.stdin.flush()
                    self.proc.wait(timeout=5)
                except Exception:
                    self.proc.kill()
                    self.proc.wait(timeout=5)
        finally:
            self.sel.close()


class Kraken7Engine(HTREngine):
    """Line-level recognition with kraken 7 models (PP-OCRv6) in venv_kraken7."""

    def __init__(self):
        self._venv = _DEFAULT_VENV
        self._worker: Optional[_Worker] = None
        self._worker_cfg: Optional[Dict[str, Any]] = None
        self._lock = threading.Lock()
        self._tmpdir = Path(tempfile.mkdtemp(prefix="kraken7_"))
        self._config_widget = None
        self._model_combo = None
        self._custom_edit = None
        self._device_combo = None

    # --- identity / availability -------------------------------------------------
    def get_name(self) -> str:
        return "Kraken 7 (PP-OCR)"

    def get_description(self) -> str:
        return ("kraken 7.1 PP-OCRv6 line recognition: multilingual base models "
                "(44 languages, 10 scripts), fast; runs in venv_kraken7")

    def get_aliases(self) -> List[str]:
        return ["kraken7", "ppocr", "pp-ocr", "kraken7-ppocr"]

    def is_available(self) -> bool:
        return _venv_python(self._venv) is not None and _venv_has_kraken7(self._venv)

    def get_unavailable_reason(self) -> str:
        if _venv_python(self._venv) is None:
            return f"venv_kraken7 not found at {self._venv} (setup: see engines/kraken7_engine.py)"
        if not _venv_has_kraken7(self._venv):
            return f"kraken 7 is not installed in {self._venv}"
        return ""

    # --- config ------------------------------------------------------------------
    def get_config_widget(self) -> QWidget:
        if not PYQT_AVAILABLE:
            return None
        if self._config_widget is None:
            w = QWidget()
            form = QFormLayout(w)
            self._model_combo = QComboBox()
            for p in available_presets():
                self._model_combo.addItem(p["label"], p["value"])
            self._model_combo.addItem("Custom model file…", "__custom__")
            form.addRow("Model:", self._model_combo)
            self._custom_edit = QLineEdit()
            self._custom_edit.setPlaceholderText("/path/to/model.safetensors (kraken 7)")
            form.addRow("Custom:", self._custom_edit)
            self._device_combo = QComboBox()
            self._device_combo.addItems(["cuda:0", "cuda:1", "cpu"])
            form.addRow("Device:", self._device_combo)
            form.addRow(QLabel("Expects line images: use a line segmentation."))
            self._config_widget = w
        return self._config_widget

    def get_config(self) -> Dict[str, Any]:
        if self._model_combo is None:
            return {"model_path": _DEFAULT_MODEL, "device": "cuda:0"}
        val = self._model_combo.currentData()
        if val == "__custom__":
            val = self._custom_edit.text().strip()
        return {"model_path": val, "device": self._device_combo.currentText()}

    def set_config(self, config: Dict[str, Any]):
        if self._model_combo is None:
            return
        val = config.get("model_path", _DEFAULT_MODEL)
        idx = self._model_combo.findData(val)
        if idx < 0:
            idx = self._model_combo.findData("__custom__")
            self._custom_edit.setText(str(val))
        self._model_combo.setCurrentIndex(idx)
        di = self._device_combo.findText(config.get("device", "cuda:0"))
        if di >= 0:
            self._device_combo.setCurrentIndex(di)

    # --- model lifecycle ---------------------------------------------------------
    def load_model(self, config: Dict[str, Any]) -> bool:
        python = _venv_python(self._venv)
        if python is None or not _WORKER_SCRIPT.exists():
            logger.error("[Kraken7] venv or worker missing")
            return False
        model = resolve_model(str(config.get("model_path") or _DEFAULT_MODEL))
        if not model.exists():
            logger.error("[Kraken7] model not found: %s", model)
            return False
        cfg = {"model_path": str(model.resolve()), "device": config.get("device", "cuda:0")}
        with self._lock:
            self._stop_worker()
            try:
                self._worker = _Worker(python, cfg)
            except Exception as e:
                logger.error("[Kraken7] worker start failed: %s", e)
                self._worker = None
                return False
            self._worker_cfg = cfg
        logger.info("[Kraken7] ready: %s on %s", model.name, cfg["device"])
        return True

    def _stop_worker(self):
        if self._worker is not None:
            try:
                self._worker.close()
            except Exception as e:
                logger.warning("[Kraken7] worker close: %s", e)
            self._worker = None

    def unload_model(self):
        with self._lock:
            self._stop_worker()
            self._worker_cfg = None

    def is_model_loaded(self) -> bool:
        return self._worker is not None

    def __del__(self):
        try:
            self._stop_worker()
        except Exception:
            pass

    # --- inference ---------------------------------------------------------------
    def requires_line_segmentation(self) -> bool:
        return True

    def transcribe_line(self, image: np.ndarray,
                        config: Optional[Dict[str, Any]] = None) -> TranscriptionResult:
        if self._worker_cfg is None:
            return TranscriptionResult(text="[Kraken 7 not loaded]", confidence=0.0)
        pil = Image.fromarray(image) if isinstance(image, np.ndarray) else image
        fd, tmp = tempfile.mkstemp(suffix=".png", dir=self._tmpdir)
        os.close(fd)
        try:
            pil.convert("RGB").save(tmp)
            with self._lock:
                # restart a worker that died (OOM, crash) with the same config
                if self._worker is None or not self._worker.alive():
                    logger.warning("[Kraken7] worker not running, restarting")
                    self._stop_worker()
                    self._worker = _Worker(_venv_python(self._venv), self._worker_cfg)
                out = self._worker.ask({"image": tmp})
        except Exception as e:
            logger.error("[Kraken7] %s", e)
            with self._lock:
                self._stop_worker()  # next call starts a fresh worker
            return TranscriptionResult(text=f"[Error: {e}]", confidence=0.0)
        finally:
            Path(tmp).unlink(missing_ok=True)
        if "error" in out:
            return TranscriptionResult(text=f"[Error: {out['error']}]", confidence=0.0)
        conf = out.get("confidence")
        return TranscriptionResult(
            text=out.get("text", ""),
            confidence=float(conf) if conf is not None else 1.0,
            metadata={"engine": self.get_name(),
                      "model": Path(self._worker_cfg["model_path"]).name},
        )

    def transcribe_lines(self, images: List[np.ndarray],
                         config: Optional[Dict[str, Any]] = None) -> List[TranscriptionResult]:
        return [self.transcribe_line(img, config) for img in images]

    def supports_batch(self) -> bool:
        return False

    def get_capabilities(self) -> Dict[str, bool]:
        return {"batch_processing": False, "confidence_scores": True, "beam_search": False,
                "language_model": False, "preprocessing": False, "layout_detection": False}
