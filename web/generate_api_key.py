#!/usr/bin/env python3
"""
Generate a Polyscriptor API key and register its hash in the key file.

Usage:
    source htr_gui/bin/activate
    python web/generate_api_key.py --name alice
    python web/generate_api_key.py --name achim --admin

The raw key is printed ONCE and never stored — only its SHA-256 hash goes
into the key file (default: web/api_keys.yaml, gitignored). Revoke a key by
deleting its entry and restarting the server.
"""

import argparse
import hashlib
import secrets
from pathlib import Path

try:
    from web.api_auth import DEFAULT_KEYS_FILE
except ImportError:  # run directly as `python web/generate_api_key.py`
    from api_auth import DEFAULT_KEYS_FILE

_FILE_HEADER = (
    "# Polyscriptor API keys — only SHA-256 hashes, never plaintext keys.\n"
    "# Managed by web/generate_api_key.py; revoke = delete entry + restart server.\n"
)


def generate_key(name: str, keys_file: Path, admin: bool = False) -> str:
    """Create a new key, append its hash entry to keys_file, return the raw key."""
    import yaml

    raw_key = "psk_" + secrets.token_urlsafe(32)
    key_hash = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

    keys_file = Path(keys_file)
    data = {}
    if keys_file.exists():
        data = yaml.safe_load(keys_file.read_text(encoding="utf-8")) or {}
    entries = data.get("keys") or []
    entry = {"name": name, "key_sha256": key_hash}
    if admin:
        entry["admin"] = True
    entries.append(entry)

    body = yaml.safe_dump({"keys": entries}, allow_unicode=True, sort_keys=False)
    keys_file.write_text(_FILE_HEADER + body, encoding="utf-8")
    keys_file.chmod(0o600)
    return raw_key


def main():
    parser = argparse.ArgumentParser(description="Generate a Polyscriptor API key")
    parser.add_argument("--name", required=True, help="User name for this key (audit log identity)")
    parser.add_argument("--admin", action="store_true", help="Mark key as admin (model upload, evict-all)")
    parser.add_argument("--keys-file", type=Path, default=DEFAULT_KEYS_FILE,
                        help=f"Key file to append to (default: {DEFAULT_KEYS_FILE})")
    args = parser.parse_args()

    raw_key = generate_key(args.name, args.keys_file, admin=args.admin)
    print(f"Key file: {args.keys_file}")
    print(f"User:     {args.name}{' (admin)' if args.admin else ''}")
    print()
    print("API key (shown ONCE, store it now — only the hash is saved):")
    print(f"  {raw_key}")
    print()
    print("Use with:  curl -H 'X-API-Key: <key>' http://<server>:8765/api/engines")
    print("Server restart required for the new key to become active.")


if __name__ == "__main__":
    main()
