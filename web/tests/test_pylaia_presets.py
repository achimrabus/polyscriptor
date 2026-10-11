"""
CRNN-CTC presets in a fresh clone: published models come from Hugging Face,
presets whose files are missing and that have no Hugging Face repo are hidden.

Issue #15 (GitHub), point 1: the registry listed local folders that only exist on
the developer's machine, so a fresh clone showed presets that could not load.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import inference_pylaia_native as native
import engines.pylaia_engine as pylaia_engine
import web.polyscriptor_server as server_mod


@pytest.fixture
def presets(tmp_path, monkeypatch):
    """A registry with one local, one Hugging Face and one dead preset."""
    ckpt = tmp_path / "best_model.pt"
    ckpt.write_bytes(b"x")
    syms = tmp_path / "symbols.txt"
    syms.write_text("a\n")
    registry = {
        "Local": {"checkpoint": str(ckpt), "syms": str(syms)},
        "Published": {"checkpoint": "models/does_not_exist/best_model.pt",
                      "syms": "models/does_not_exist/symbols.txt",
                      "repo_id": "achimrabus/crnn-ctc-test"},
        "Dead": {"checkpoint": "models/does_not_exist_either/best_model.pt",
                 "syms": "models/does_not_exist_either/symbols.txt"},
    }
    monkeypatch.setattr(native, "PYLAIA_MODELS", registry)
    monkeypatch.setattr(pylaia_engine, "PYLAIA_MODELS", registry)
    monkeypatch.setattr(server_mod, "PYLAIA_MODELS", registry, raising=False)
    monkeypatch.setattr(server_mod, "_import_segmenters", lambda: None)
    monkeypatch.setattr(native, "_scan_pylaia_models", lambda *a, **k: None)
    return registry


def test_preset_source(presets):
    assert native.preset_source(presets["Local"]) == "local"
    assert native.preset_source(presets["Published"]) == "hf"
    assert native.preset_source(presets["Dead"]) is None


def test_local_file_wins_over_repo_id(presets):
    info = dict(presets["Local"], repo_id="achimrabus/crnn-ctc-test")
    assert native.preset_source(info) == "local"


def test_registry_paths_resolve_from_repo_root(monkeypatch, tmp_path):
    # The server is not always started from the project root
    monkeypatch.chdir(tmp_path)
    assert native._local_file("inference_pylaia_native.py") is not None
    assert native._local_file("models/does_not_exist/best_model.pt") is None


def test_web_options_hide_dead_and_mark_hf(presets):
    options = server_mod._get_pylaia_model_options()
    values = [o["value"] for o in options]
    assert values == ["Local", "Published", "__custom__"]
    labels = {o["value"]: o["label"] for o in options}
    assert labels["Local"] == "Local"
    assert "Hugging Face" in labels["Published"]


class _FakeInference:
    def __init__(self, checkpoint_path, syms_path=None, enable_spaces=True, device=None):
        self.checkpoint_path = checkpoint_path
        self.syms_path = syms_path


def test_engine_downloads_published_preset(presets, monkeypatch):
    calls = []

    def fake_download(repo_id, filename):
        calls.append((repo_id, filename))
        return f"/hf/{repo_id}/{filename}"

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    monkeypatch.setattr(pylaia_engine, "PyLaiaInference", _FakeInference)

    engine = pylaia_engine.PyLaiaEngine()
    assert engine.load_model({"model_path": "Published"})
    assert calls == [("achimrabus/crnn-ctc-test", "best_model.pt"),
                     ("achimrabus/crnn-ctc-test", "symbols.txt")]
    assert engine.model.checkpoint_path == "/hf/achimrabus/crnn-ctc-test/best_model.pt"


def test_engine_prefers_local_files(presets, monkeypatch):
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "hf_hub_download",
                        lambda **k: pytest.fail("must not download when local files exist"))
    monkeypatch.setattr(pylaia_engine, "PyLaiaInference", _FakeInference)

    info = dict(presets["Local"], repo_id="achimrabus/crnn-ctc-test")
    presets["Local"] = info
    engine = pylaia_engine.PyLaiaEngine()
    assert engine.load_model({"model_path": "Local"})
    assert engine.model.checkpoint_path == info["checkpoint"]
    assert engine.model.syms_path == info["syms"]


def test_scan_does_not_duplicate_registered_models(tmp_path, monkeypatch):
    # The web server scans models/ with an absolute path, the registry holds
    # relative ones; the same checkpoint must not be listed twice
    model_dir = tmp_path / "models" / "pylaia_example"
    model_dir.mkdir(parents=True)
    (model_dir / "best_model.pt").write_bytes(b"x")
    registry = {"Example": {"checkpoint": "models/pylaia_example/best_model.pt"}}
    monkeypatch.setattr(native, "PYLAIA_MODELS", registry)
    monkeypatch.chdir(tmp_path)
    native._scan_pylaia_models(str(tmp_path / "models"))
    assert list(registry) == ["Example"]


def test_trocr_presets_include_own_published_models():
    values = [o["value"] for o in server_mod._scan_trocr_models()]
    for repo in ("cyrillic-trocr/trocr-church-slavonic-handwritten",
                 "cyrillic-trocr/trocr-ukrainian-handwritten",
                 "cyrillic-trocr/trocr-handwritten-cyrillic",
                 "achimrabus/trocr-glagolitic-handwritten"):
        assert repo in values
    assert values[0] == "__custom__"
