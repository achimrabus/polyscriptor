#!/usr/bin/env python3
"""
Manage Polyscriptor API keys: create, list, remove.

Usage (on the server, from the repo directory):
    source htr_gui/bin/activate
    python web/generate_api_key.py --name alice            # create key for alice
    python web/generate_api_key.py --name achim --admin    # create admin key
    python web/generate_api_key.py --list                  # show registered keys
    python web/generate_api_key.py --remove alice          # revoke alice's key(s)

The raw key is printed ONCE on creation and never stored — only its SHA-256
hash goes into the key file (default: web/api_keys.yaml, gitignored).
Changes take effect after a server restart (./deploy/restart_server.sh).
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
    "# Managed by web/generate_api_key.py; restart the server after changes.\n"
)


def _load_entries(keys_file: Path) -> list:
    import yaml
    if not keys_file.exists():
        return []
    data = yaml.safe_load(keys_file.read_text(encoding="utf-8")) or {}
    return data.get("keys") or []


def _write_entries(keys_file: Path, entries: list) -> None:
    import yaml
    body = yaml.safe_dump({"keys": entries}, allow_unicode=True, sort_keys=False)
    keys_file.write_text(_FILE_HEADER + body, encoding="utf-8")
    keys_file.chmod(0o600)


def generate_key(name: str, keys_file: Path, admin: bool = False) -> str:
    """Create a new key, append its hash entry to keys_file, return the raw key."""
    raw_key = "psk_" + secrets.token_urlsafe(32)
    key_hash = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

    keys_file = Path(keys_file)
    entries = _load_entries(keys_file)
    entry = {"name": name, "key_sha256": key_hash}
    if admin:
        entry["admin"] = True
    entries.append(entry)
    _write_entries(keys_file, entries)
    return raw_key


def list_keys(keys_file: Path) -> list:
    """Return the entries (names + flags, no raw keys — those are never stored)."""
    return _load_entries(Path(keys_file))


def remove_key(name: str, keys_file: Path) -> int:
    """Remove all entries for `name`. Returns the number of removed entries."""
    keys_file = Path(keys_file)
    entries = _load_entries(keys_file)
    kept = [e for e in entries if e.get("name") != name]
    removed = len(entries) - len(kept)
    if removed:
        _write_entries(keys_file, kept)
    return removed


def main():
    parser = argparse.ArgumentParser(description="Manage Polyscriptor API keys")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--name", help="Create a key for this user name")
    action.add_argument("--list", action="store_true", help="List registered keys")
    action.add_argument("--remove", metavar="NAME", help="Remove all keys of this user")
    parser.add_argument("--admin", action="store_true",
                        help="With --name: mark key as admin (model upload, evict-all)")
    parser.add_argument("--keys-file", type=Path, default=DEFAULT_KEYS_FILE,
                        help=f"Key file (default: {DEFAULT_KEYS_FILE})")
    args = parser.parse_args()

    if args.list:
        entries = list_keys(args.keys_file)
        if not entries:
            print(f"No keys registered ({args.keys_file} missing or empty) — key auth is OFF.")
            return
        print(f"Keys in {args.keys_file}:")
        for e in entries:
            flags = []
            if e.get("admin"):
                flags.append("admin")
            if e.get("max_jobs") is not None:
                flags.append(f"max_jobs={e['max_jobs']}")
            if e.get("daily_page_quota") is not None:
                flags.append(f"daily_page_quota={e['daily_page_quota']}")
            suffix = f"  ({', '.join(flags)})" if flags else ""
            print(f"  - {e.get('name')}{suffix}")
        return

    if args.remove:
        removed = remove_key(args.remove, args.keys_file)
        if removed:
            print(f"Removed {removed} key(s) for '{args.remove}'.")
            print("Restart the server to apply:  ./deploy/restart_server.sh")
        else:
            print(f"No key found for '{args.remove}' in {args.keys_file}.")
        return

    raw_key = generate_key(args.name, args.keys_file, admin=args.admin)
    print(f"Key file: {args.keys_file}")
    print(f"User:     {args.name}{' (admin)' if args.admin else ''}")
    print()
    print("API key (shown ONCE, store it now — only the hash is saved):")
    print(f"  {raw_key}")
    print()
    print("Use with:  curl -H 'X-API-Key: <key>' http://<server>:8765/api/engines")
    print("Restart the server to apply:  ./deploy/restart_server.sh")


if __name__ == "__main__":
    main()
