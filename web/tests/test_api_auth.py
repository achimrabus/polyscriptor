"""
API key layer tests — web/api_auth.py + middleware integration (Phase B1).

Run with:
    source htr_gui/bin/activate
    pytest web/tests/test_api_auth.py -v

The key layer is opt-in: without a configured key file the server must behave
exactly as before (headers ignored, no 401s).
"""

import hashlib
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import web.polyscriptor_server as server_mod
from web.api_auth import ApiKeyRegistry
from web.polyscriptor_server import app

client = TestClient(app)


# ── Fixtures ─────────────────────────────────────────────────────────────────

def _write_keys_file(path: Path, entries) -> Path:
    lines = ["keys:"]
    for e in entries:
        lines.append(f"  - name: {e['name']}")
        lines.append(f"    key_sha256: {e['key_sha256']}")
        if e.get("admin"):
            lines.append("    admin: true")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _sha256(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@pytest.fixture
def registry_with_keys(tmp_path):
    keys_file = _write_keys_file(tmp_path / "api_keys.yaml", [
        {"name": "alice", "key_sha256": _sha256("alice-key-123")},
        {"name": "bob", "key_sha256": _sha256("bob-key-456"), "admin": True},
    ])
    usage_log = tmp_path / "usage.jsonl"
    return ApiKeyRegistry(keys_file, usage_log=usage_log)


# ── Registry unit tests ──────────────────────────────────────────────────────

def test_registry_disabled_without_file():
    reg = ApiKeyRegistry(None)
    assert reg.enabled is False
    assert reg.verify("anything") is None


def test_registry_disabled_with_missing_file(tmp_path):
    reg = ApiKeyRegistry(tmp_path / "does_not_exist.yaml")
    assert reg.enabled is False


def test_registry_loads_and_verifies(registry_with_keys):
    reg = registry_with_keys
    assert reg.enabled is True
    user = reg.verify("alice-key-123")
    assert user is not None
    assert user.name == "alice"
    assert user.is_admin is False
    admin = reg.verify("bob-key-456")
    assert admin.name == "bob"
    assert admin.is_admin is True
    assert reg.verify("wrong-key") is None
    assert reg.verify("") is None


def test_registry_ignores_malformed_entries(tmp_path):
    keys_file = tmp_path / "api_keys.yaml"
    keys_file.write_text(
        "keys:\n"
        "  - name: broken\n"
        "    key_sha256: nothex\n"          # invalid hash length
        f"  - key_sha256: {_sha256('x')}\n"  # missing name
        f"  - name: ok\n    key_sha256: {_sha256('good-key')}\n",
        encoding="utf-8",
    )
    reg = ApiKeyRegistry(keys_file)
    assert reg.enabled is True
    assert reg.verify("good-key").name == "ok"
    assert reg.verify("x") is None


def test_usage_logging(registry_with_keys):
    reg = registry_with_keys
    user = reg.verify("alice-key-123")
    reg.log_usage(user, "GET", "/api/engines", 200)
    lines = reg.usage_log.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["user"] == "alice"
    assert entry["path"] == "/api/engines"
    assert entry["status"] == 200
    assert "ts" in entry


# ── Middleware integration tests ─────────────────────────────────────────────

@pytest.fixture
def enabled_registry(registry_with_keys, monkeypatch):
    monkeypatch.setattr(server_mod, "api_key_registry", registry_with_keys)
    return registry_with_keys


def test_header_ignored_when_disabled(monkeypatch):
    monkeypatch.setattr(server_mod, "api_key_registry", ApiKeyRegistry(None))
    resp = client.get("/api/engines", headers={"X-API-Key": "whatever"})
    assert resp.status_code == 200


def test_no_header_still_works_when_enabled(enabled_registry):
    resp = client.get("/api/engines")
    assert resp.status_code == 200


def test_invalid_key_rejected_when_enabled(enabled_registry):
    resp = client.get("/api/engines", headers={"X-API-Key": "invalid"})
    assert resp.status_code == 401


def test_valid_key_accepted_and_logged(enabled_registry):
    resp = client.get("/api/engines", headers={"X-API-Key": "alice-key-123"})
    assert resp.status_code == 200
    lines = enabled_registry.usage_log.read_text(encoding="utf-8").strip().splitlines()
    entries = [json.loads(l) for l in lines]
    assert any(e["user"] == "alice" and e["path"] == "/api/engines" for e in entries)


# ── Admin gating for dangerous endpoints (Phase B2) ──────────────────────────

def _upload_mlmodel(headers=None):
    import io
    return client.post(
        "/api/models/upload",
        files={"file": ("b2_test_model.mlmodel", io.BytesIO(b"\x00fake\x00"), "application/octet-stream")},
        headers=headers or {},
    )


def test_model_upload_open_when_disabled(monkeypatch):
    """Legacy behavior: without key file the upload endpoint stays open."""
    monkeypatch.setattr(server_mod, "api_key_registry", ApiKeyRegistry(None))
    assert _upload_mlmodel().status_code == 200


def test_model_upload_requires_admin_key_when_enabled(enabled_registry):
    assert _upload_mlmodel().status_code == 403
    assert _upload_mlmodel({"X-API-Key": "alice-key-123"}).status_code == 403
    assert _upload_mlmodel({"X-API-Key": "bob-key-456"}).status_code == 200


def test_evict_all_requires_admin_key_when_enabled(enabled_registry):
    assert client.post("/api/admin/evict-all").status_code == 403
    assert client.post("/api/admin/evict-all", headers={"X-API-Key": "alice-key-123"}).status_code == 403
    resp = client.post("/api/admin/evict-all", headers={"X-API-Key": "bob-key-456"})
    assert resp.status_code == 200
    assert "evicted" in resp.json()


def test_evict_all_localhost_only_when_disabled(monkeypatch):
    """Legacy behavior: TestClient host is 'testclient' (not localhost) -> 403."""
    monkeypatch.setattr(server_mod, "api_key_registry", ApiKeyRegistry(None))
    assert client.post("/api/admin/evict-all").status_code == 403


# ── Upload pixel cap (Phase B2, decompression-bomb guard) ────────────────────

def test_upload_rejects_oversized_image(monkeypatch):
    import io
    from PIL import Image

    monkeypatch.setattr(server_mod, "_MAX_UPLOAD_PIXELS", 10_000)
    buf = io.BytesIO()
    Image.new("RGB", (200, 100), "white").save(buf, format="PNG")  # 20k px > 10k cap
    resp = client.post(
        "/api/image/upload",
        files={"file": ("big.png", buf.getvalue(), "image/png")},
    )
    assert resp.status_code == 400
    assert "too large" in resp.json()["detail"].lower()


# ── Key generation script ────────────────────────────────────────────────────

def test_generate_api_key_roundtrip(tmp_path):
    from web.generate_api_key import generate_key

    keys_file = tmp_path / "api_keys.yaml"
    raw1 = generate_key("carol", keys_file, admin=False)
    raw2 = generate_key("dave", keys_file, admin=True)
    assert raw1 != raw2

    reg = ApiKeyRegistry(keys_file)
    assert reg.verify(raw1).name == "carol"
    assert reg.verify(raw1).is_admin is False
    assert reg.verify(raw2).name == "dave"
    assert reg.verify(raw2).is_admin is True
