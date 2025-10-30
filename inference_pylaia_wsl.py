"""
PyLaia Inference via WSL Subprocess

This module provides a simple wrapper for PyLaia inference that calls
the pylaia-htr-decode-ctc command via WSL subprocess.
"""

import subprocess
import tempfile
from pathlib import Path
from typing import Tuple, Optional
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class PyLaiaInferenceWSL:
    """
    PyLaia inference wrapper that uses WSL subprocess to call native PyLaia decode.

    This avoids the complexity of loading PyTorch Lightning checkpoints directly
    and instead uses PyLaia's built-in decode command.
    """

    def __init__(self, checkpoint_path: str, syms_path: str = None):
        """
        Initialize PyLaia inference.

        Args:
            checkpoint_path: Path to .ckpt checkpoint file (Windows path)
            syms_path: Path to symbols file (Windows path). If None, will look in data directory.
        """
        self.checkpoint_path = Path(checkpoint_path)

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Find symbols file
        if syms_path is None:
            # Look in data/pylaia_glagolitic/syms.txt
            syms_path = Path("data/pylaia_glagolitic/syms.txt")

        self.syms_path = Path(syms_path)
        if not self.syms_path.exists():
            raise FileNotFoundError(f"Symbols file not found: {syms_path}")

        logger.info(f"Initialized PyLaia inference with checkpoint: {checkpoint_path}")
        logger.info(f"Using symbols from: {syms_path}")

    def _windows_to_wsl_path(self, windows_path: Path) -> str:
        """Convert Windows path to WSL path."""
        path_str = str(windows_path.absolute()).replace('\\', '/')
        if len(path_str) > 1 and path_str[1] == ':':
            drive = path_str[0].lower()
            path_str = f"/mnt/{drive}{path_str[2:]}"
        return path_str

    def transcribe(self, image: Image.Image) -> Tuple[str, float]:
        """
        Transcribe a single line image.

        Args:
            image: PIL Image of text line

        Returns:
            Tuple of (transcription_text, confidence_score)
        """
        # Save image to temporary file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_img:
            tmp_img_path = Path(tmp_img.name)
            image.save(tmp_img_path)

        # Create temporary list file for PyLaia
        with tempfile.NamedTemporaryFile(mode='w', suffix='.lst', delete=False, encoding='utf-8') as tmp_lst:
            tmp_lst_path = Path(tmp_lst.name)
            # PyLaia list format: image_id image_path
            tmp_lst.write(f"test {tmp_img_path}\n")

        # Create temporary output file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp_out:
            tmp_out_path = Path(tmp_out.name)

        try:
            # Convert paths to WSL format
            wsl_checkpoint = self._windows_to_wsl_path(self.checkpoint_path)
            wsl_syms = self._windows_to_wsl_path(self.syms_path)
            wsl_img_dir = self._windows_to_wsl_path(tmp_img_path.parent)
            wsl_lst = self._windows_to_wsl_path(tmp_lst_path)
            wsl_out = self._windows_to_wsl_path(tmp_out_path)

            # Run PyLaia decode command via WSL
            cmd = f"""cd /mnt/c/Users/Achim/Documents/TrOCR/dhlab-slavistik && \
source venv_pylaia_wsl/bin/activate && \
pylaia-htr-decode-ctc \
--checkpoint {wsl_checkpoint} \
{wsl_syms} \
{wsl_img_dir} \
{wsl_lst} \
--output {wsl_out} \
--print_line_confidence_scores"""

            result = subprocess.run(
                ["wsl", "bash", "-c", cmd],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode != 0:
                logger.error(f"PyLaia decode failed: {result.stderr}")
                return "", 0.0

            # Read output file
            output_text = tmp_out_path.read_text(encoding='utf-8').strip()

            # Parse output (format: "image_id transcription confidence")
            parts = output_text.split(maxsplit=2)
            if len(parts) >= 2:
                text = parts[1]
                confidence = float(parts[2]) if len(parts) >= 3 else 0.0
                return text, confidence
            else:
                return "", 0.0

        except Exception as e:
            logger.error(f"Error during PyLaia inference: {e}")
            return "", 0.0
        finally:
            # Clean up temporary files
            try:
                tmp_img_path.unlink()
                tmp_lst_path.unlink()
                tmp_out_path.unlink()
            except:
                pass


# For compatibility with existing code
PyLaiaInference = PyLaiaInferenceWSL

# Model registry (updated for trained models)
PYLAIA_MODELS = {
    "Glagolitic (trained)": {
        "checkpoint": "models/pylaia_glagolitic/experiment/epoch=35-lowest_va_cer.ckpt",
        "syms": "data/pylaia_glagolitic/syms.txt",
        "description": "PyLaia model trained on Glagolitic manuscripts (CER: 77.0%)"
    }
}
