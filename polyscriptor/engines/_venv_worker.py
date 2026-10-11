"""Shared plumbing for engines whose heavy dependencies live in an isolated venv.

Engines like PaddleOCR, DeepSeek-OCR, PaddleOCR-VL and Chandra cannot import
their model stack into the main environment (conflicting transformers / OpenCV
pins). Instead they run a worker script with the Python interpreter of a
dedicated venv and exchange one JSON object over stdout.

Protocol: ``python <worker.py> <config_json> <image_path>`` prints exactly one
JSON object — either the success payload or ``{"error": str, "traceback": str}``.
"""

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Optional


def default_project_root() -> Path:
    """Repository root when running from a source checkout."""
    return Path(__file__).resolve().parents[2]


def find_venv_python(venv_path: Path) -> Optional[Path]:
    """Return the Python interpreter inside a venv, or None if not found."""
    if sys.platform == "win32":
        candidates = [venv_path / "Scripts" / "python.exe"]
    else:
        candidates = [venv_path / "bin" / "python", venv_path / "bin" / "python3"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def venv_has_package(venv_path: Path, package: str) -> bool:
    """Check a package is installed in the venv via filesystem only (no import, no subprocess)."""
    if sys.platform == "win32":
        candidates = [venv_path / "Lib" / "site-packages" / package]
    else:
        candidates = list((venv_path / "lib").glob(f"python*/site-packages/{package}"))
    return any(sp.is_dir() for sp in candidates)


class WorkerError(Exception):
    """The worker process failed: non-zero exit, unparsable output, or error payload."""


def run_worker(
    venv_python: Path,
    worker_script: Path,
    config_json: str,
    image_path: str,
    *,
    timeout: int = 180,
    logger: Optional[logging.Logger] = None,
    tag: str = "worker",
    env: Optional[dict] = None,
) -> dict:
    """Run a worker subprocess and return its parsed JSON payload.

    Raises ``WorkerError`` when the process fails or reports an error payload.
    Raises ``subprocess.TimeoutExpired`` on timeout (caller decides the message).
    """
    result = subprocess.run(
        [str(venv_python), str(worker_script), config_json, image_path],
        capture_output=True,
        text=True,
        timeout=timeout,
        start_new_session=True,  # isolate from terminal SIGINT → prevents core dump on Ctrl+C
        env=env,
    )

    output = None
    if result.stdout:
        try:
            output = json.loads(result.stdout)
        except (json.JSONDecodeError, ValueError):
            pass

    if result.returncode != 0:
        if isinstance(output, dict) and "error" in output:
            _log_error(logger, tag, output["error"], output.get("traceback", ""))
            raise WorkerError(output["error"])
        stderr = result.stderr[-2000:] if result.stderr else "(no stderr)"
        if logger:
            logger.error("[%s] Worker exited %d: %s", tag, result.returncode, stderr)
        raise WorkerError(f"worker exited {result.returncode}: {stderr}")

    if output is None:
        raise WorkerError(f"worker produced unparsable output: {(result.stdout or '')[:500]}")
    if "error" in output:
        _log_error(logger, tag, output["error"], output.get("traceback", ""))
        raise WorkerError(output["error"])
    return output


def _log_error(logger, tag, error, traceback):
    if logger:
        logger.error("[%s] Worker error: %s", tag, error)
        if traceback:
            logger.debug("[%s] Traceback:\n%s", tag, traceback)
