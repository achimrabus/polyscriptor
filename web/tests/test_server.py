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

import pytest
from PIL import Image
from fastapi.testclient import TestClient

# Make sure the project root is on the path before importing the server
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

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
    assert len(presets) == 3   # 1 local + 2 verified Zenodo (hallucinated IDs removed in f1ac6a3)


def test_kraken_presets_local_first():
    resp = client.get("/api/kraken/presets")
    presets = resp.json()["presets"]
    local = [p for p in presets if p["source"] == "local"]
    zenodo = [p for p in presets if p["source"] == "zenodo"]
    assert len(local) == 1
    assert local[0]["id"] == "blla-local"
    assert len(zenodo) == 2   # catmus-print + arabic-muharaf (10 hallucinated IDs removed)


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
