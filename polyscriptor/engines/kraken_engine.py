"""
Kraken HTR Engine Plugin

Wraps the Kraken OCR system as a plugin for the unified GUI.
Kraken is specialized for historical document OCR with robust segmentation and recognition.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np


def _print(msg: str) -> None:
    """Print with graceful fallback if console can't encode the message (e.g. Windows CP-1252)."""
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", errors="replace").decode("ascii"))

from polyscriptor.core.engine_base import HTREngine, TranscriptionResult


try:
    from kraken import rpred
    from kraken.lib import vgsl, models
    KRAKEN_AVAILABLE = True
except ImportError:
    KRAKEN_AVAILABLE = False


# Local model (included in repo)
LOCAL_BLLA_MODEL = str(Path(__file__).resolve().parent.parent / "data" / "blla.mlmodel")

# Preset Kraken models — local + Zenodo community models (auto-download on first use)
KRAKEN_MODELS = {
    "blla-local": {
        "path": LOCAL_BLLA_MODEL,
        "description": "BLLA Segmentation Model (Local, Default)",
        "language": "multi",
        "source": "local"
    },
    # --- VERIFIED ZENODO MODELS ---
    # CATMuS-Print: printed text, multilingual, verified DOI 10.5281/zenodo.10592716
    "catmus-print": {
        "zenodo_id": "10.5281/zenodo.10592716",
        "description": "CATMuS-Print (Modern Printed Text, multilingual)",
        "language": "multi",
        "source": "zenodo"
    },
    # NOTE: "arabic-muharaf" (DOI 10.5281/zenodo.14295555) was removed from this
    # list — despite the verified DOI, it is a *segmentation* model (Arabic
    # Handwritten Segmentation, Muharaf Corpus), not a text recognizer. This
    # engine always wraps presets in TorchSeqRecognizer for OCR, which cannot
    # load segmentation-type models (raises "Models of type segmentation are
    # not supported by TorchSeqRecognizer"). Kraken segmentation models belong
    # in kraken_segmenter.py's neural segmentation path, not here.
}


class KrakenEngine(HTREngine):
    """Kraken HTR engine plugin."""

    def __init__(self):
        self.model: Optional[Any] = None  # TorchSeqRecognizer


    def get_name(self) -> str:
        return "Kraken"

    def get_description(self) -> str:
        return "Kraken OCR - Specialized for historical documents with .mlmodel support"

    def is_available(self) -> bool:
        return KRAKEN_AVAILABLE

    def get_unavailable_reason(self) -> str:
        if not KRAKEN_AVAILABLE:
            return "Kraken not installed. Install it in a fresh environment together with the rest: pip install -e '.[kraken]' (see README)"
        return ""








    def load_model(self, config: Dict[str, Any]) -> bool:
        """Load Kraken model (local or Zenodo auto-download)."""
        try:
            model_path = config.get("model_path")
            preset_id = config.get("preset_id")

            # Resolve Zenodo preset: download if needed
            if preset_id and preset_id in KRAKEN_MODELS:
                model_info = KRAKEN_MODELS[preset_id]
                if model_info.get("source") == "zenodo":
                    zenodo_id = model_info.get("zenodo_id")
                    model_path = self._download_zenodo_model(zenodo_id, preset_id)
                    if not model_path:
                        print(f"Error: Failed to download Zenodo model '{preset_id}'")
                        return False
                elif model_info.get("source") == "local":
                    model_path = model_info.get("path")

            # Fall back to default local blla model
            if not model_path:
                model_path = LOCAL_BLLA_MODEL
                print(f"No model specified, using default: {model_path}")

            if not Path(model_path).exists():
                print(f"Error: Model file not found: {model_path}")
                print("For Zenodo models, run: kraken get <zenodo_id>")
                return False

            vgsl_model = vgsl.TorchVGSLModel.load_model(model_path)
            from kraken.lib.models import TorchSeqRecognizer
            self.model = TorchSeqRecognizer(vgsl_model, device='cpu')
            print(f"Kraken model loaded from: {model_path}")
            return True

        except Exception as e:
            import traceback
            print(f"Error loading Kraken model: {e}")
            print(traceback.format_exc())
            self.model = None
            return False

    def _download_zenodo_model(self, zenodo_id: str, model_name: str) -> Optional[str]:
        """Download a Kraken model from Zenodo via the htrmopo Python API.

        Saved under the KRAKEN_ZENODO_MODELS_DIR env var if set (e.g. redirected
        to a data volume with more space), otherwise under `kraken_models/` in
        the repo root. Uses htrmopo's `get_model(model_id, path=...)` directly
        instead of shelling out to `kraken get` — that avoids having to guess
        where the CLI's own cache (platformdirs.user_data_dir('htrmopo'), e.g.
        ~/.local/share/htrmopo on Linux) put the file afterwards.
        Returns local path on success, None on failure.
        """
        import os

        default_dir = Path(__file__).parent.parent / "kraken_models"
        models_dir = Path(os.environ.get("KRAKEN_ZENODO_MODELS_DIR", default_dir))
        target_dir = models_dir / model_name
        target_dir.mkdir(parents=True, exist_ok=True)

        existing = sorted(target_dir.glob("*.mlmodel"))
        if existing:
            _print(f"Using cached Zenodo model: {existing[0]}")
            return str(existing[0])

        try:
            from htrmopo import get_model
        except ImportError:
            _print("ERROR: 'htrmopo' not installed. Install with: pip install htrmopo")
            _print(f"Manual download: https://zenodo.org/record/{zenodo_id.split('/')[-1]}")
            return None

        _print(f"Downloading Zenodo model {zenodo_id} …")
        _print(f"Will save to: {target_dir}")
        _print("This may take a few minutes on first use …")

        try:
            result_path = get_model(zenodo_id, path=target_dir)
        except Exception as e:
            _print(f"Download error: {e}")
            _print(f"Manual: kraken get {zenodo_id}  then copy the .mlmodel into {target_dir}/")
            return None

        downloaded = sorted(Path(result_path).glob("*.mlmodel"))
        if not downloaded:
            _print("WARNING: Download succeeded but no .mlmodel file was found in the result")
            return None
        _print(f"Model saved to: {downloaded[0]}")
        return str(downloaded[0])

    def unload_model(self):
        """Unload model from memory."""
        if self.model is not None:
            del self.model
            self.model = None

            # Free GPU memory
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def is_model_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None

    def transcribe_line(self, image: np.ndarray, config: Optional[Dict[str, Any]] = None) -> TranscriptionResult:
        """Transcribe a line image with Kraken."""
        if self.model is None:
            return TranscriptionResult(text="[Model not loaded]", confidence=0.0)

        if config is None:
            config = self.get_config()

        try:
            # Import numpy at the start
            import numpy as np

            # Convert numpy to PIL
            from PIL import Image as PILImage
            if isinstance(image, np.ndarray):
                pil_image = PILImage.fromarray(image)
            else:
                pil_image = image

            # Convert to grayscale first
            if pil_image.mode != 'L':
                pil_image = pil_image.convert('L')

            # IMPORTANT: Do NOT binarize! Kraken models work better with grayscale
            # Modern Kraken models are trained on grayscale images and binarization
            # destroys character details, especially in historical manuscripts
            # The previous median threshold was causing poor recognition quality
            binary_image = pil_image  # Keep original grayscale

            # Create a simple segmentation boundary for the full line image
            # Kraken's rpred needs a Segmentation object with line boundaries
            from kraken.containers import BaselineLine, Segmentation

            height, width = binary_image.height, binary_image.width

            # Create a baseline (horizontal line through the middle)
            # Use 0-indexed coordinates (width-1, height-1 as maximum)
            baseline = [[0, height // 2], [width - 1, height // 2]]

            # Create a boundary polygon (rectangle around the entire image)
            # Use 0-indexed coordinates to avoid "outside of image bounds" error
            boundary = [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]

            # Create a BaselineLine (not BBoxLine - that doesn't support baselines)
            line = BaselineLine(
                id='line_0',
                baseline=baseline,
                boundary=boundary,
                text='',
                tags=None,
                split=None
            )

            # Create Segmentation container
            seg = Segmentation(
                type='baselines',
                imagename='line',
                text_direction='horizontal-lr',
                script_detection=False,
                lines=[line],
                regions={},
                line_orders=[]
            )

            # Run recognition
            bidi = config.get("bidi_reordering", True)

            # Model is already wrapped as TorchSeqRecognizer in load_model()
            # rpred returns a generator
            results = list(rpred.rpred(
                network=self.model,
                im=binary_image,
                bounds=seg,
                bidi_reordering=bidi
            ))

            # Extract text from first result
            if results and len(results) > 0:
                text = results[0].prediction
                confidence = results[0].confidences
                avg_confidence = sum(confidence) / len(confidence) if confidence else 1.0

                return TranscriptionResult(
                    text=text,
                    confidence=avg_confidence,
                    metadata={"model": "kraken"}
                )
            else:
                return TranscriptionResult(text="", confidence=0.0)

        except Exception as e:
            import traceback
            print(f"Error in Kraken transcription: {e}")
            print(traceback.format_exc())
            return TranscriptionResult.error_result(f"{e}")

    def get_capabilities(self) -> Dict[str, bool]:
        """Kraken capabilities."""
        return {
            "batch": False,  # Could be implemented
            "confidence_scores": True,  # Kraken provides per-character confidence
            "beam_search": False,  # Internal to Kraken
            "language_model": False,  # Not explicitly exposed
            "preprocessing": False,  # External binarization recommended
        }


def download_preset_model(preset_name: str) -> Optional[str]:
    """Module-level helper: resolve and (if needed) download a Kraken preset model.

    Used by batch_processing.py and the web server without instantiating KrakenEngine.
    Returns local file path, or None on failure.
    """
    if preset_name not in KRAKEN_MODELS:
        print(f"Unknown Kraken preset: '{preset_name}'. Available: {list(KRAKEN_MODELS)}")
        return None
    info = KRAKEN_MODELS[preset_name]
    if info.get("source") == "local":
        return info.get("path")
    if info.get("source") == "zenodo":
        engine = KrakenEngine.__new__(KrakenEngine)
        return engine._download_zenodo_model(info["zenodo_id"], preset_name)
    return None
