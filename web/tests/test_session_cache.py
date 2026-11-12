"""
Session cookie lifetime and caching of the page files (12.11.2026).

1. The session cookie must be renewed on every real request (sliding expiry). It was only
   set when the session was created, so the browser dropped it two hours after the first
   visit even while the user kept working, and the next request got a new, empty session.
   Polling routes (GPU, engine list, ...) do not renew it, so an idle tab still expires.
2. The page and its static files carry `Cache-Control: no-cache`: the browser keeps them
   but revalidates on each page load (304 when unchanged), so an update of the JS is seen
   on the next reload instead of whenever the browser's heuristic cache runs out.

Run with:
    source htr_gui/bin/activate
    pytest web/tests/test_session_cache.py -v
"""

import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import web.polyscriptor_server as server_mod  # noqa: E402
from web.polyscriptor_server import app  # noqa: E402

COOKIE = server_mod._SESSION_COOKIE


def _set_cookie_headers(resp):
    return [v for k, v in resp.headers.multi_items() if k.lower() == "set-cookie"]


def test_new_session_sets_cookie():
    client = TestClient(app)
    r = client.get("/")
    assert any(h.startswith(f"{COOKIE}=") for h in _set_cookie_headers(r))


def test_cookie_renewed_on_activity():
    client = TestClient(app)
    first = client.get("/")
    sid = first.cookies.get(COOKIE)
    assert sid
    r = client.get("/")   # the client sends the cookie back
    renewed = [h for h in _set_cookie_headers(r) if h.startswith(f"{COOKIE}={sid};")]
    assert renewed, "an active session must get its cookie re-sent with a fresh Max-Age"
    assert f"Max-Age={server_mod._SESSION_TTL_SECONDS}" in renewed[0]


def test_cookie_not_renewed_by_polling():
    client = TestClient(app)
    sid = client.get("/").cookies.get(COOKIE)
    assert sid
    r = client.get("/api/gpu")
    assert not _set_cookie_headers(r), "GPU polling must not keep a session alive"


def test_page_and_static_files_revalidate():
    client = TestClient(app)
    for path in ("/", "/static/app.js", "/static/components/batch-panel.js"):
        r = client.get(path)
        assert r.status_code == 200, path
        assert r.headers.get("cache-control") == "no-cache", path


def test_static_unchanged_gives_304():
    client = TestClient(app)
    r = client.get("/static/app.js")
    etag = r.headers["etag"]
    r2 = client.get("/static/app.js", headers={"If-None-Match": etag})
    assert r2.status_code == 304


def test_api_responses_untouched():
    client = TestClient(app)
    r = client.get("/api/engines")
    assert r.headers.get("cache-control") != "no-cache"
