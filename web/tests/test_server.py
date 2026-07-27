"""
Web UI API tests — polyscriptor_server.py

Run with:
    source htr_gui/bin/activate
    pytest web/tests/test_server.py -v

These tests use FastAPI's TestClient (no running server needed).
No GPU or loaded HTR models are required — engine-heavy endpoints
(transcribe, segment) are covered only at the HTTP contract level.
"""

import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image
from fastapi.testclient import TestClient

# Make sure the project root is on the path before importing the server
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import web.polyscriptor_server as server_mod
from web.polyscriptor_server import app

client = TestClient(app)


# ── Fixtures ─────────────────────────────────────────────────────────────────

def _png_bytes(width: int = 200, height: int = 100, color: str = "white") -> bytes:
    """Create a minimal in-memory PNG."""
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buf, format="PNG")
    return buf.getvalue()


def _pdf_bytes(num_pages: int = 1) -> bytes:
    """Create a minimal in-memory PDF using PyMuPDF."""
    import fitz
    doc = fitz.open()
    for i in range(num_pages):
        page = doc.new_page(width=595, height=842)
        page.insert_text((72, 100), f"Test page {i + 1}", fontsize=14)
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    return buf.getvalue()


def _upload_image() -> dict:
    """Upload a test image and return the response JSON."""
    resp = client.post(
        "/api/image/upload",
        files={"file": ("test.png", _png_bytes(), "image/png")},
    )
    assert resp.status_code == 200
    return resp.json()


# ── Static serving ────────────────────────────────────────────────────────────

def test_root_serves_html():
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert b"Polyscriptor" in resp.content or b"<html" in resp.content


# ── Engine API ────────────────────────────────────────────────────────────────

def test_engines_list():
    resp = client.get("/api/engines")
    assert resp.status_code == 200
    engines = resp.json()
    assert isinstance(engines, list)
    assert len(engines) > 0
    # Each engine must have name and available fields
    for eng in engines:
        assert "name" in eng
        assert "available" in eng


def test_engine_status_initially_unloaded():
    resp = client.get("/api/engine/status")
    assert resp.status_code == 200
    data = resp.json()
    assert "loaded" in data
    assert data["loaded"] is False


def test_config_schema_unknown_engine_returns_empty():
    resp = client.get("/api/engine/NonexistentEngine/config-schema")
    assert resp.status_code == 200
    data = resp.json()
    assert data == {"fields": []}


def test_config_schema_crnn_ctc():
    resp = client.get("/api/engine/CRNN-CTC/config-schema")
    assert resp.status_code == 200
    data = resp.json()
    assert "fields" in data
    assert isinstance(data["fields"], list)
    # Fields may be empty in headless test environments without PyQt


# ── Kraken presets ────────────────────────────────────────────────────────────

def test_kraken_presets_returns_list():
    resp = client.get("/api/kraken/presets")
    assert resp.status_code == 200
    data = resp.json()
    assert "presets" in data
    presets = data["presets"]
    assert len(presets) == 2   # 1 local + 1 verified Zenodo recognition model


def test_kraken_presets_local_first():
    resp = client.get("/api/kraken/presets")
    presets = resp.json()["presets"]
    local = [p for p in presets if p["source"] == "local"]
    zenodo = [p for p in presets if p["source"] == "zenodo"]
    assert len(local) == 1
    assert local[0]["id"] == "blla-local"
    assert len(zenodo) == 1   # catmus-print only; arabic-muharaf is a segmentation model, not a recognizer


def test_kraken_presets_schema():
    resp = client.get("/api/kraken/presets")
    for preset in resp.json()["presets"]:
        assert "id" in preset
        assert "label" in preset
        assert "language" in preset
        assert "source" in preset
        assert preset["source"] in ("local", "zenodo")


# ── Image upload ──────────────────────────────────────────────────────────────

def test_upload_png_returns_image_id():
    data = _upload_image()
    assert "image_id" in data
    assert "width" in data and data["width"] == 200
    assert "height" in data and data["height"] == 100
    assert data.get("is_pdf") is None  # not a PDF response


def test_upload_jpeg():
    buf = io.BytesIO()
    Image.new("RGB", (300, 150), "lightblue").save(buf, format="JPEG")
    resp = client.post(
        "/api/image/upload",
        files={"file": ("scan.jpg", buf.getvalue(), "image/jpeg")},
    )
    assert resp.status_code == 200
    assert "image_id" in resp.json()


def test_upload_invalid_file_returns_400():
    resp = client.post(
        "/api/image/upload",
        files={"file": ("notes.txt", b"not an image", "text/plain")},
    )
    assert resp.status_code == 400


def test_upload_pdf_single_page():
    resp = client.post(
        "/api/image/upload",
        files={"file": ("doc.pdf", _pdf_bytes(1), "application/pdf")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["is_pdf"] is True
    assert data["num_pages"] == 1
    assert len(data["pages"]) == 1
    page = data["pages"][0]
    assert "image_id" in page
    assert page["page"] == 1
    assert page["filename"] == "doc_page001.png"
    assert page["width"] > 0 and page["height"] > 0


def test_upload_pdf_multi_page():
    resp = client.post(
        "/api/image/upload",
        files={"file": ("manuscript.pdf", _pdf_bytes(3), "application/pdf")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["num_pages"] == 3
    assert len(data["pages"]) == 3
    for i, page in enumerate(data["pages"], 1):
        assert page["page"] == i
        assert f"_page{i:03d}.png" in page["filename"]


def test_upload_pdf_by_filename_without_content_type():
    """Server should detect PDF by filename even if content-type is octet-stream."""
    resp = client.post(
        "/api/image/upload",
        files={"file": ("scan.pdf", _pdf_bytes(2), "application/octet-stream")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["is_pdf"] is True
    assert data["num_pages"] == 2


# ── Image serving ─────────────────────────────────────────────────────────────

def test_fetch_uploaded_image():
    data = _upload_image()
    image_id = data["image_id"]
    resp = client.get(f"/api/image/{image_id}")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("image/")
    # Verify it decodes as a valid image
    img = Image.open(io.BytesIO(resp.content))
    assert img.width == 200
    assert img.height == 100


def test_fetch_pdf_page_as_image():
    upload = client.post(
        "/api/image/upload",
        files={"file": ("page.pdf", _pdf_bytes(1), "application/pdf")},
    ).json()
    image_id = upload["pages"][0]["image_id"]
    resp = client.get(f"/api/image/{image_id}")
    assert resp.status_code == 200
    img = Image.open(io.BytesIO(resp.content))
    assert img.width > 0


def test_fetch_nonexistent_image_returns_404():
    resp = client.get("/api/image/00000000-0000-0000-0000-000000000000")
    assert resp.status_code == 404


# ── XML attachment ────────────────────────────────────────────────────────────

def test_attach_xml_to_image():
    data = _upload_image()
    image_id = data["image_id"]
    minimal_xml = b"""<?xml version="1.0" encoding="UTF-8"?>
<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15">
  <Page imageFilename="test.png" imageWidth="200" imageHeight="100"/>
</PcGts>"""
    resp = client.post(
        f"/api/image/{image_id}/xml",
        files={"file": ("test.xml", minimal_xml, "text/xml")},
    )
    assert resp.status_code == 200
    assert resp.json()["success"] is True


def test_attach_xml_to_nonexistent_image_returns_404():
    resp = client.post(
        "/api/image/00000000-0000-0000-0000-000000000000/xml",
        files={"file": ("test.xml", b"<xml/>", "text/xml")},
    )
    assert resp.status_code == 404


# ── GPU status ────────────────────────────────────────────────────────────────

def test_gpu_status_returns_valid_response():
    resp = client.get("/api/gpu")
    assert resp.status_code == 200
    data = resp.json()
    # Must have at least a 'available' or 'device' key
    assert "available" in data or "device" in data or "cuda" in str(data).lower()


# ── Transcribe/segment contract ───────────────────────────────────────────────

def test_transcribe_without_loaded_engine_returns_error():
    """Transcription without a loaded engine should fail gracefully (not crash)."""
    data = _upload_image()
    image_id = data["image_id"]
    # Server may return 400 immediately (no engine loaded) or start SSE stream with error event
    resp = client.post(
        "/api/transcribe",
        json={"image_id": image_id, "engine": "CRNN-CTC", "seg_method": "kraken"},
    )
    assert resp.status_code in (200, 400)
    body = resp.content.decode()
    assert "error" in body.lower() or "not loaded" in body.lower() or "detail" in body.lower()


def test_transcribe_nonexistent_image_returns_error():
    resp = client.post(
        "/api/transcribe",
        json={
            "image_id": "00000000-0000-0000-0000-000000000000",
            "engine": "CRNN-CTC",
            "seg_method": "kraken",
        },
    )
    # May return 400 (no engine), 404 (image not found), or 200 SSE with error event
    assert resp.status_code in (200, 400, 404)
    body = resp.content.decode()
    assert "error" in body.lower() or "not found" in body.lower() or "detail" in body.lower()


class _FakeCompareEngine:
    def __init__(self, outputs):
        self._outputs = list(outputs)
        self._idx = 0

    def is_model_loaded(self):
        return True

    def transcribe_line(self, _img_array, _config=None):
        text = self._outputs[self._idx]
        self._idx += 1
        return SimpleNamespace(text=text, confidence=0.91)


class _FakeUnloadableEngine:
    def __init__(self):
        self.unload_calls = 0

    def unload_model(self):
        self.unload_calls += 1

    def is_model_loaded(self):
        return self.unload_calls == 0


def test_compare_without_base_transcription_returns_400(monkeypatch):
    data = _upload_image()
    image_id = data["image_id"]
    monkeypatch.setattr(server_mod, "loaded_engine", _FakeCompareEngine(["alpha"]), raising=False)
    monkeypatch.setattr(server_mod, "loaded_engine_name", "FakeCompare", raising=False)
    monkeypatch.setattr(server_mod, "loaded_config", {"model": "fake-compare"}, raising=False)

    resp = client.post("/api/compare/run", json={"image_id": image_id})
    assert resp.status_code == 400
    assert "base transcription" in resp.json()["detail"].lower()


def test_compare_run_uses_cached_base_results(monkeypatch):
    data = _upload_image()
    image_id = data["image_id"]
    session_id = client.cookies.get("polyscriptor_session")
    session = server_mod.sessions[session_id]
    img_data = session.image_cache[image_id]

    img_data["seg_source"] = "kraken"
    img_data["lines"] = [
        SimpleNamespace(bbox=(0, 0, 50, 20), image=None),
        SimpleNamespace(bbox=(0, 20, 50, 40), image=None),
    ]
    img_data["line_regions"] = [0, 0]

    base_lines = [
        {"index": 0, "text": "alpha", "confidence": 0.95, "bbox": [0, 0, 50, 20], "region": 0},
        {"index": 1, "text": "beta", "confidence": 0.95, "bbox": [0, 20, 50, 40], "region": 0},
    ]
    img_data["results"] = base_lines
    server_mod._store_result_slot(
        img_data,
        slot_id="primary",
        label="Base Engine",
        engine_name="Base Engine",
        seg_source="kraken",
        lines=base_lines,
        pool_key=None,
        kind="primary",
    )

    monkeypatch.setattr(server_mod, "loaded_engine", _FakeCompareEngine(["alpha", "theta"]), raising=False)
    monkeypatch.setattr(server_mod, "loaded_engine_name", "FakeCompare", raising=False)
    monkeypatch.setattr(server_mod, "loaded_config", {"model": "fake-compare"}, raising=False)

    resp = client.post("/api/compare/run", json={"image_id": image_id})
    assert resp.status_code == 200
    body = resp.text
    assert "event: complete" in body
    assert "Char disagreement" in body
    assert "FakeCompare" in body


def test_unload_engine_releases_primary_and_comparison_slots(monkeypatch):
    data = _upload_image()
    _ = data["image_id"]
    session_id = client.cookies.get("polyscriptor_session")
    session = server_mod.sessions[session_id]

    primary_engine = _FakeUnloadableEngine()
    compare_engine = _FakeUnloadableEngine()

    monkeypatch.setattr(server_mod, "engine_pool", {
        "primary-slot": server_mod.EngineSlot(
            engine=primary_engine,
            engine_name="Primary",
            config={},
            pool_key="primary-slot",
            ref_count=1,
        ),
        "compare-slot": server_mod.EngineSlot(
            engine=compare_engine,
            engine_name="Compare",
            config={},
            pool_key="compare-slot",
            ref_count=1,
        ),
    }, raising=False)
    monkeypatch.setattr(server_mod, "loaded_engine", primary_engine, raising=False)
    monkeypatch.setattr(server_mod, "loaded_engine_name", "Primary", raising=False)
    monkeypatch.setattr(server_mod, "loaded_config", {"model": "primary"}, raising=False)

    session.pool_key = "primary-slot"
    session.comparison_pool_keys = {"compare-abc": "compare-slot"}

    resp = client.post("/api/engine/unload")
    assert resp.status_code == 200
    assert resp.json()["success"] is True
    assert primary_engine.unload_calls == 1
    assert compare_engine.unload_calls == 1
    assert session.pool_key is None
    assert session.comparison_pool_keys == {}
    assert server_mod.engine_pool == {}


# ── Region deletion ───────────────────────────────────────────────────────────

def test_delete_region_on_image_without_segmentation_returns_404_or_error():
    """Deleting a region before segmentation should not crash the server."""
    data = _upload_image()
    image_id = data["image_id"]
    resp = client.delete(f"/api/image/{image_id}/region/0")
    # 200 (empty), 400 (no regions), or 404 (not found) — not a 500
    assert resp.status_code in (200, 400, 404)
    assert resp.status_code != 500


# ── Engine pool ──────────────────────────────────────────────────────────────

def test_pool_status_endpoint():
    resp = client.get("/api/engine/pool")
    assert resp.status_code == 200
    data = resp.json()
    assert "pool_size" in data
    assert "slots" in data
    assert isinstance(data["slots"], list)


def test_pool_key_deterministic():
    """Same inputs produce the same pool key."""
    from web.polyscriptor_server import _make_pool_key
    k1 = _make_pool_key("TrOCR", {"model_path": "models/test_model"})
    k2 = _make_pool_key("TrOCR", {"model_path": "models/test_model"})
    assert k1 == k2


def test_pool_key_differentiates_models():
    from web.polyscriptor_server import _make_pool_key
    k1 = _make_pool_key("TrOCR", {"model_path": "models/model_a"})
    k2 = _make_pool_key("TrOCR", {"model_path": "models/model_b"})
    assert k1 != k2


def test_pool_key_api_key_isolation():
    """Different API keys produce different pool keys."""
    from web.polyscriptor_server import _make_pool_key
    k1 = _make_pool_key("Commercial APIs", {"provider": "Gemini", "model": "gemini-pro", "api_key": "key_aaa"})
    k2 = _make_pool_key("Commercial APIs", {"provider": "Gemini", "model": "gemini-pro", "api_key": "key_bbb"})
    assert k1 != k2


def test_engine_factory_creates_independent_instances():
    from web.polyscriptor_server import _create_engine_instance
    e1 = _create_engine_instance("TrOCR")
    e2 = _create_engine_instance("TrOCR")
    assert e1 is not None
    assert e2 is not None
    assert e1 is not e2  # Independent instances


def test_session_shows_pool_key():
    resp = client.get("/api/session")
    assert resp.status_code == 200
    data = resp.json()
    assert "pool_key" in data


def test_engine_status_session_aware():
    """engine_status returns unloaded for a fresh session (no pool_key)."""
    resp = client.get("/api/engine/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["loaded"] is False


def test_model_upload_rejects_non_mlmodel():
    """Upload endpoint rejects files that don't have .mlmodel extension."""
    content = b"not a model"
    resp = client.post(
        "/api/models/upload",
        files={"file": ("model.txt", io.BytesIO(content), "text/plain")},
    )
    assert resp.status_code == 400
    assert "mlmodel" in resp.json()["detail"].lower()


def test_model_upload_accepts_mlmodel(tmp_path):
    """Upload endpoint accepts a .mlmodel file and returns path + refreshed options."""
    # Use a tiny fake .mlmodel (just bytes)
    content = b"\x00fake_mlmodel_data\x00"
    resp = client.post(
        "/api/models/upload",
        files={"file": ("test_model.mlmodel", io.BytesIO(content), "application/octet-stream")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["filename"] == "test_model.mlmodel"
    assert data["path"].endswith("test_model.mlmodel")
    assert isinstance(data["options"], list)
    assert data["size"] == len(content)
    # Verify file actually exists on disk
    from pathlib import Path
    from web.polyscriptor_server import PROJECT_ROOT
    uploaded = PROJECT_ROOT / data["path"]
    assert uploaded.exists()
    uploaded.unlink()  # cleanup


def _slot(slot_id, label, texts):
    lines = [
        {"index": i, "text": t, "confidence": 0.9, "bbox": [0, i * 20, 50, i * 20 + 20], "region": 0}
        for i, t in enumerate(texts)
    ]
    return {
        "slot_id": slot_id,
        "label": label,
        "engine_name": label,
        "seg_source": "kraken",
        "line_count": len(lines),
        "pool_key": None,
        "kind": "primary" if slot_id == "primary" else "comparison",
        "lines": lines,
    }


def test_disagreement_payload_ships_diff_ops_only_for_differing_lines():
    base = _slot("primary", "Base", ["alpha", "beta", "gamma"])
    comp = _slot("compare", "Comp", ["alpha", "betax", "gamma"])

    payload = server_mod._build_disagreement_payload(base, comp)

    assert payload["summary"]["identical_lines"] == 2
    assert payload["summary"]["line_count"] == 3

    rows = payload["lines"]
    # Identical lines: no disagreement, empty diff_ops (payload kept small).
    assert rows[0]["has_disagreement"] is False
    assert rows[0]["diff_ops"] == []
    assert rows[2]["has_disagreement"] is False
    assert rows[2]["diff_ops"] == []

    # Differing line: disagreement flagged, diff_ops present with an insert op.
    assert rows[1]["has_disagreement"] is True
    assert rows[1]["diff_ops"], "differing line must carry diff ops"
    ops = {op["op"] for op in rows[1]["diff_ops"]}
    assert "insert" in ops  # "betax" adds an 'x' over "beta"
    inserted = [op["h"] for op in rows[1]["diff_ops"] if op["op"] == "insert"]
    assert inserted == ["x"]


# ── Ground-truth (CER/WER) evaluation ────────────────────────────────────────

_GT_XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15">
  <Page imageFilename="t.png" imageWidth="200" imageHeight="100">
    <TextRegion id="r1">
      <TextLine id="l1"><Coords points="0,0 10,0 10,10 0,10"/>
        <TextEquiv><Unicode>\xd0\xb8 \xd0\xb8\xd0\xb4\xd1\xa3\xd1\x88\xd0\xb5</Unicode></TextEquiv></TextLine>
      <TextLine id="l2"><Coords points="0,20 10,20 10,30 0,30"/>
        <TextEquiv><Unicode>\xd0\xbf\xd0\xbe\xd1\x83\xd1\x82\xd0\xb5\xd0\xbc\xd1\x8c</Unicode></TextEquiv></TextLine>
    </TextRegion>
  </Page>
</PcGts>"""


def _session_for_image(image_id):
    for s in server_mod.sessions.values():
        if image_id in s.image_cache:
            return s
    return None


def test_extract_pagexml_line_texts():
    texts = server_mod._extract_pagexml_line_texts(_GT_XML)
    assert texts == ["и идѣше", "поутемь"]


def test_extract_pagexml_line_texts_handles_missing_text():
    xml = b"""<?xml version="1.0"?>
<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15">
  <Page><TextRegion id="r1">
    <TextLine id="l1"><TextEquiv><Unicode>only line</Unicode></TextEquiv></TextLine>
    <TextLine id="l2"/>
  </TextRegion></Page></PcGts>"""
    assert server_mod._extract_pagexml_line_texts(xml) == ["only line", ""]


def test_ground_truth_payload_uses_cer_labels_and_is_asymmetric():
    gt_slot = {"slot_id": "ground_truth", "label": "GT",
               "lines": [{"text": "hello"}, {"text": "world"}]}
    hyp_slot = {"slot_id": "e1", "label": "Engine",
                "lines": [{"text": "helo"}, {"text": "world"}]}
    payload = server_mod._build_disagreement_payload(
        gt_slot, hyp_slot, server_mod.ComparisonMode.GROUND_TRUTH,
    )
    assert payload["labels"]["char_rate"] == "CER"
    assert payload["labels"]["word_rate"] == "WER"
    # "hello" vs "helo" → 1 edit / 5 ref chars = 20% CER; "world" identical = 0%.
    assert payload["lines"][0]["metrics"]["char_rate"] == pytest.approx(20.0)
    assert payload["lines"][1]["metrics"]["char_rate"] == pytest.approx(0.0)
    assert payload["summary"]["macro_char_rate"] == pytest.approx(10.0)


def test_upload_ground_truth_returns_line_count():
    image_id = _upload_image()["image_id"]
    resp = client.post(
        f"/api/image/{image_id}/ground-truth",
        files={"file": ("gt.xml", _GT_XML, "text/xml")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["line_count"] == 2


def test_upload_ground_truth_rejects_empty_xml():
    image_id = _upload_image()["image_id"]
    empty = b"""<?xml version="1.0"?>
<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15">
  <Page imageFilename="t.png" imageWidth="200" imageHeight="100"/>
</PcGts>"""
    resp = client.post(
        f"/api/image/{image_id}/ground-truth",
        files={"file": ("gt.xml", empty, "text/xml")},
    )
    assert resp.status_code == 400


def test_compare_ground_truth_requires_gt_first():
    image_id = _upload_image()["image_id"]
    resp = client.post("/api/compare/ground-truth", json={"image_id": image_id})
    assert resp.status_code == 400


def test_compare_ground_truth_scores_and_ranks_slots():
    image_id = _upload_image()["image_id"]
    session = _session_for_image(image_id)
    assert session is not None
    session.image_cache[image_id]["ground_truth"] = {
        "lines": ["hello", "world"], "filename": "gt.xml",
    }
    session.image_cache[image_id]["result_slots"] = {
        "good": {"slot_id": "good", "label": "Good", "engine_name": "E1",
                 "kind": "primary", "lines": [{"text": "hello"}, {"text": "world"}]},
        "bad": {"slot_id": "bad", "label": "Bad", "engine_name": "E2",
                "kind": "comparison", "lines": [{"text": "helo"}, {"text": "wrld"}]},
    }
    resp = client.post("/api/compare/ground-truth", json={"image_id": image_id})
    assert resp.status_code == 200
    data = resp.json()
    assert data["ground_truth"]["line_count"] == 2
    assert len(data["leaderboard"]) == 2
    # Perfect match ranks first with micro CER 0.
    assert data["leaderboard"][0]["slot_id"] == "good"
    assert data["leaderboard"][0]["micro_cer"] == pytest.approx(0.0)
    assert data["leaderboard"][1]["slot_id"] == "bad"
    assert data["leaderboard"][1]["micro_cer"] > 0
    assert data["runs"][0]["labels"]["char_rate"] == "CER"


# ── Segmentation reuse (seg_method omitted) ──────────────────────────────────

def _img(source=None, lines=1, regions=0):
    """Minimal image-cache stub carrying a segmentation state."""
    d = {}
    if source is not None:
        d["seg_source"] = source
        d["lines"] = [SimpleNamespace(bbox=(0, i * 10, 50, i * 10 + 10), image=None)
                      for i in range(lines)]
        d["seg_regions"] = [{"id": f"r_{i}", "bbox": [0, 0, 10, 10], "num_lines": 1}
                            for i in range(regions)]
    return d


def test_cached_segmentation_is_usable_only_for_real_methods():
    assert server_mod._has_usable_segmentation(_img("kraken-blla")) is True
    assert server_mod._has_usable_segmentation(_img("pagexml")) is True
    # Page-level engines park a single full-page pseudo line under "page";
    # that must never be inherited by a line-based run.
    assert server_mod._has_usable_segmentation(_img("page")) is False
    assert server_mod._has_usable_segmentation(_img("unknown")) is False
    assert server_mod._has_usable_segmentation(_img("")) is False
    assert server_mod._has_usable_segmentation({}) is False
    # Source set but no lines cached is not usable either.
    assert server_mod._has_usable_segmentation({"seg_source": "kraken"}) is False


def test_omitted_seg_method_keeps_existing_segmentation():
    """The whole point: /segment with blla + region edits must survive a job
    that does not name a method."""
    img = _img("kraken-blla", lines=18, regions=2)
    assert server_mod._desired_seg_source(img, None, None) == "kraken-blla"


def test_omitted_seg_method_falls_back_to_kraken_without_cache():
    assert server_mod._desired_seg_source({}, None, None) == "kraken"


def test_omitted_seg_method_ignores_page_level_pseudo_segmentation():
    assert server_mod._desired_seg_source(_img("page"), None, None) == "kraken"


def test_explicit_seg_method_still_forces_that_method():
    img = _img("kraken-blla", lines=18)
    assert server_mod._desired_seg_source(img, "hpp", None) == "hpp"
    assert server_mod._desired_seg_source(img, "kraken-blla", None) == "kraken-blla"


def test_attached_pagexml_wins_over_cache_and_method():
    img = _img("kraken-blla", lines=18)
    assert server_mod._desired_seg_source(img, None, "/tmp/page.xml") == "pagexml"
    assert server_mod._desired_seg_source(img, "hpp", "/tmp/page.xml") == "pagexml"


def test_transcribe_request_allows_omitting_seg_method():
    req = server_mod.TranscribeRequest(image_id="x")
    assert req.seg_method is None


def test_job_status_reports_which_segmentation_ran():
    job = server_mod.TranscriptionJob(
        job_id="j1", owner="key:test", display_name="test",
        session_id="s1", params=server_mod.TranscribeRequest(image_id="x"),
    )
    # Nothing ran yet — no claim is made.
    assert "segmentation" not in server_mod._serialize_job(job)

    job.segmentation = {"source": "kraken-blla", "num_lines": 18, "num_regions": 1}
    out = server_mod._serialize_job(job)
    assert out["segmentation"]["source"] == "kraken-blla"
    assert out["segmentation"]["num_lines"] == 18
    assert out["segmentation"]["num_regions"] == 1


def test_cached_pagexml_layout_is_not_reused_once_xml_is_out_of_play():
    """use_pagexml=False leaves xml_path None; silently reusing the XML layout
    would contradict the caller."""
    img = _img("pagexml", lines=12)
    assert server_mod._desired_seg_source(img, None, None) == "kraken"
