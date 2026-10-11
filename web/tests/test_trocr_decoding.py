"""TrOCR decoding: no_repeat_ngram_size defaults to 0 and reaches generate().

Checkpoints from optimized_training.py store no_repeat_ngram_size=3 in their
generation_config; unless the call overrides it, generate() inherits it and forbids
legitimate repetitions inside a line.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from engines.trocr_engine import TrOCREngine


class _FakeInference:
    def __init__(self):
        self.calls = []

    def transcribe_line(self, image, **kwargs):
        self.calls.append(kwargs)
        return "text", 0.9, [0.9]


def _engine(**state):
    engine = TrOCREngine()
    engine.model = _FakeInference()
    for k, v in state.items():
        setattr(engine, k, v)
    return engine


def test_default_turns_the_inherited_ngram_block_off():
    engine = _engine()
    engine.transcribe_line(np.zeros((32, 64, 3), dtype=np.uint8), config={})
    assert engine.model.calls[0]["no_repeat_ngram_size"] == 0


def test_value_from_load_config_is_used():
    engine = _engine(_no_repeat_ngram_size=3)
    engine.transcribe_line(np.zeros((32, 64, 3), dtype=np.uint8), config={})
    assert engine.model.calls[0]["no_repeat_ngram_size"] == 3


def test_transcribe_line_signature_defaults_to_zero():
    import inspect
    from inference_page import TrOCRInference
    sig = inspect.signature(TrOCRInference.transcribe_line)
    assert sig.parameters["no_repeat_ngram_size"].default == 0


def test_batch_cli_flag_defaults_to_zero(monkeypatch, tmp_path):
    import batch_processing
    monkeypatch.setattr(sys, "argv", ["batch_processing.py", "--input-folder", str(tmp_path),
                                      "--engine", "TrOCR", "--model-id", "x/y"])
    args = batch_processing.parse_args()
    assert args.no_repeat_ngram_size == 0


# ── Switch in the web form (shared definition, also meant for the PyQt widget) ──

import math

import pytest

from engines.trocr_engine import REPEAT_BLOCK_FIELD, repeat_block_size


@pytest.mark.parametrize("raw, expected", [
    (None, 0), ("", 0), (0, 0), (3, 3), ("3", 3), (3.0, 3), (" 4 ", 4),
    (-2, 0), ("-1", 0), (float("nan"), 0), ("abc", 0), (2.7, 2), (float("inf"), 0), (True, 0),
])
def test_repeat_block_size_coerces_form_values(raw, expected):
    assert repeat_block_size(raw) == expected


def test_field_definition_defaults_to_off():
    assert REPEAT_BLOCK_FIELD["key"] == "no_repeat_ngram_size"
    assert REPEAT_BLOCK_FIELD["type"] == "number"
    assert REPEAT_BLOCK_FIELD["default"] == 0
    assert REPEAT_BLOCK_FIELD["min"] == 0
    assert REPEAT_BLOCK_FIELD["hint"]


def test_per_call_value_from_the_form_reaches_generate():
    engine = _engine()
    engine.transcribe_line(np.zeros((32, 64, 3), dtype=np.uint8),
                           config={"no_repeat_ngram_size": "3"})
    assert engine.model.calls[0]["no_repeat_ngram_size"] == 3


def test_per_call_zero_overrides_a_block_set_at_load():
    engine = _engine(_no_repeat_ngram_size=3)
    engine.transcribe_line(np.zeros((32, 64, 3), dtype=np.uint8),
                           config={"no_repeat_ngram_size": 0})
    assert engine.model.calls[0]["no_repeat_ngram_size"] == 0


def test_invalid_per_call_value_falls_back_to_off():
    engine = _engine(_no_repeat_ngram_size=3)
    engine.transcribe_line(np.zeros((32, 64, 3), dtype=np.uint8),
                           config={"no_repeat_ngram_size": float("nan")})
    assert engine.model.calls[0]["no_repeat_ngram_size"] == 0


def test_metadata_reports_the_block_used():
    engine = _engine()
    res = engine.transcribe_line(np.zeros((32, 64, 3), dtype=np.uint8),
                                 config={"no_repeat_ngram_size": 2})
    assert res.metadata["no_repeat_ngram_size"] == 2


def test_web_schema_carries_the_shared_field():
    from fastapi.testclient import TestClient
    from web.polyscriptor_server import app
    fields = TestClient(app).get("/api/engine/TrOCR/config-schema").json()["fields"]
    field = next(f for f in fields if f["key"] == "no_repeat_ngram_size")
    assert field == REPEAT_BLOCK_FIELD
    keys = [f["key"] for f in fields]
    assert keys.index("no_repeat_ngram_size") == keys.index("num_beams") + 1


def test_web_live_override_is_not_reload_only(monkeypatch):
    """Changing the field in the form must take effect without reloading the model."""
    from types import SimpleNamespace
    import web.polyscriptor_server as srv

    class _Loaded:
        def is_model_loaded(self):
            return True

    monkeypatch.setattr(srv, "loaded_engine", _Loaded())
    monkeypatch.setattr(srv, "loaded_config", {"no_repeat_ngram_size": 0})
    monkeypatch.setattr(srv, "loaded_engine_name", "TrOCR")
    _slot, _eng, cfg, _name, _key = srv._resolve_effective_engine(
        SimpleNamespace(pool_key=None), {"no_repeat_ngram_size": 3})
    assert cfg["no_repeat_ngram_size"] == 3


# ── PyQt widget: same field, built from REPEAT_BLOCK_FIELD ─────────────────────

@pytest.fixture
def qt_engine(monkeypatch):
    pytest.importorskip("PyQt6.QtWidgets")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    engine = TrOCREngine()
    engine.get_config_widget()
    yield engine
    del app


def test_pyqt_spin_box_follows_the_shared_field(qt_engine):
    spin = qt_engine._repeat_spin
    assert spin.minimum() == REPEAT_BLOCK_FIELD["min"]
    assert spin.maximum() == REPEAT_BLOCK_FIELD["max"]
    assert spin.value() == REPEAT_BLOCK_FIELD["default"] == 0
    assert spin.toolTip() == REPEAT_BLOCK_FIELD["hint"]


def test_pyqt_get_config_reports_the_block(qt_engine):
    assert qt_engine.get_config()["no_repeat_ngram_size"] == 0
    qt_engine._repeat_spin.setValue(3)
    assert qt_engine.get_config()["no_repeat_ngram_size"] == 3


def test_pyqt_set_config_restores_and_defaults_to_off(qt_engine):
    qt_engine.set_config({"no_repeat_ngram_size": 2})
    assert qt_engine._repeat_spin.value() == 2
    qt_engine.set_config({})                    # an old saved config without the key
    assert qt_engine._repeat_spin.value() == 0
    qt_engine.set_config({"no_repeat_ngram_size": "junk"})
    assert qt_engine._repeat_spin.value() == 0


def test_pyqt_value_reaches_generate(qt_engine):
    qt_engine.model = _FakeInference()
    qt_engine._repeat_spin.setValue(3)
    # the GUI calls transcribe_line without a config: the widget is read
    qt_engine.transcribe_line(np.zeros((32, 64, 3), dtype=np.uint8))
    assert qt_engine.model.calls[0]["no_repeat_ngram_size"] == 3
