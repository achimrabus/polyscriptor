"""
API key registry for Polyscriptor — Multi-User light (Phase B1).

Keys identify programmatic clients (identity, audit, later quota); they do NOT
replace the network perimeter (Uni-Netz/VPN). The layer is strictly opt-in:
when no key file is configured, `enabled` stays False and the server behaves
exactly as without this module.

Key file format (YAML, only SHA-256 hashes — never plaintext keys):

    keys:
      - name: alice
        key_sha256: <64 hex chars>
      - name: bob
        key_sha256: <64 hex chars>
        admin: true

Activation: set POLYSCRIPTOR_API_KEYS_FILE, or create web/api_keys.yaml
(the default location, gitignored). Revoke a key by removing its entry and
restarting the server.
"""

import hashlib
import hmac
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

log = logging.getLogger("polyscriptor")

_WEB_DIR = Path(__file__).resolve().parent
DEFAULT_KEYS_FILE = _WEB_DIR / "api_keys.yaml"
DEFAULT_USAGE_LOG = _WEB_DIR / "api_usage_log.jsonl"


@dataclass(frozen=True)
class ApiKeyUser:
    """Identity attached to a request that presented a valid API key."""
    name: str
    is_admin: bool = False


class ApiKeyRegistry:
    """Loads the key file once at construction; verify() is pure lookup."""

    def __init__(self, keys_file: Optional[Path], usage_log: Optional[Path] = None):
        self.keys_file = Path(keys_file) if keys_file else None
        self.usage_log = Path(usage_log) if usage_log else None
        self._by_hash: Dict[str, ApiKeyUser] = {}
        self.enabled = False
        if self.keys_file and self.keys_file.exists():
            self._load()

    def _load(self) -> None:
        import yaml
        try:
            data = yaml.safe_load(self.keys_file.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError) as e:
            log.error(f"API key file {self.keys_file} unreadable — key auth stays OFF: {e}")
            return
        for entry in data.get("keys") or []:
            if not isinstance(entry, dict):
                continue
            name = entry.get("name")
            key_hash = str(entry.get("key_sha256") or "").strip().lower()
            if not name or len(key_hash) != 64:
                log.warning(f"API key entry skipped (need name + 64-hex key_sha256): {entry!r}")
                continue
            self._by_hash[key_hash] = ApiKeyUser(name=str(name), is_admin=bool(entry.get("admin", False)))
        self.enabled = bool(self._by_hash)
        if self.enabled:
            log.info(f"API key auth enabled: {len(self._by_hash)} key(s) from {self.keys_file}")

    def verify(self, raw_key: str) -> Optional[ApiKeyUser]:
        """Return the user for a valid raw key, else None."""
        if not self.enabled or not raw_key:
            return None
        digest = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()
        for key_hash, user in self._by_hash.items():
            if hmac.compare_digest(key_hash, digest):
                return user
        return None

    def log_usage(self, user: ApiKeyUser, method: str, path: str, status: int) -> None:
        """Append one JSONL audit entry. Best-effort — never breaks a request."""
        if not self.usage_log:
            return
        entry = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "user": user.name,
            "method": method,
            "path": path,
            "status": status,
        }
        try:
            with open(self.usage_log, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError as e:
            log.warning(f"API usage log write failed: {e}")


def load_registry_from_env() -> ApiKeyRegistry:
    """Build the registry from POLYSCRIPTOR_API_KEYS_FILE or the default path.

    No env var and no default file → disabled registry (today's behavior).
    """
    env_path = os.environ.get("POLYSCRIPTOR_API_KEYS_FILE", "").strip()
    keys_file = Path(env_path) if env_path else DEFAULT_KEYS_FILE
    usage_log_env = os.environ.get("POLYSCRIPTOR_API_USAGE_LOG", "").strip()
    usage_log = Path(usage_log_env) if usage_log_env else DEFAULT_USAGE_LOG
    return ApiKeyRegistry(keys_file, usage_log=usage_log)
