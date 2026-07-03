"""
HTR Engine Plugin System - Base Classes and Registry

This module defines the plugin architecture for HTR (Handwritten Text Recognition) engines.
All HTR engines (TrOCR, Qwen3, CRNN-CTC, Kraken, etc.) implement the HTREngine interface.

Design principles:
- Abstraction: Each engine is self-contained and interchangeable
- Scalability: New engines can be added without modifying existing code
- Consistency: All engines expose the same interface to the GUI
- Flexibility: Each engine can have custom configuration widgets
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import importlib
import os
import numpy as np

try:
    from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False
    QWidget = object


_runtime_profile = None
_runtime_profile_loaded = False


def load_runtime_profile():
    """Return the optional deployment-profile module named by POLYSCRIPTOR_PROFILE.

    A deployment may point this env var at an importable module that customises
    engine discovery, model presets and segmentation (see the optional hooks
    consumed in ``discover_engines``, ``inference_pylaia_native`` and the web
    server). This keeps environment-specific behaviour out of the core code.

    Returns None when the variable is unset or the module cannot be imported.
    The result is cached after the first call.
    """
    global _runtime_profile, _runtime_profile_loaded
    if _runtime_profile_loaded:
        return _runtime_profile
    _runtime_profile_loaded = True
    name = os.environ.get("POLYSCRIPTOR_PROFILE", "").strip()
    if name:
        try:
            _runtime_profile = importlib.import_module(name)
        except ImportError as e:
            print(f"Warning: runtime profile '{name}' could not be imported: {e}")
            _runtime_profile = None
    return _runtime_profile


@dataclass
class TranscriptionResult:
    """Result from HTR engine transcription."""
    text: str
    confidence: float = 1.0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class HTREngine(ABC):
    """Abstract base class for HTR engines.

    All HTR engines must implement this interface to be compatible
    with the GUI and batch processing systems.
    """

    @abstractmethod
    def get_name(self) -> str:
        """Get display name for the engine.

        Returns:
            str: Human-readable engine name (e.g., "TrOCR", "Qwen3 VLM")
        """
        pass

    @abstractmethod
    def get_description(self) -> str:
        """Get brief description of the engine.

        Returns:
            str: One-line description (e.g., "Transformer-based OCR for manuscripts")
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if engine dependencies are installed and functional.

        Returns:
            bool: True if engine can be used, False otherwise
        """
        pass

    @abstractmethod
    def get_unavailable_reason(self) -> str:
        """Get reason why engine is unavailable (if is_available() == False).

        Returns:
            str: Explanation and installation instructions
        """
        pass

    @abstractmethod
    def get_config_widget(self) -> QWidget:
        """Create and return configuration widget for this engine.

        The widget should contain all engine-specific controls (model selection,
        beam search, preprocessing options, etc.). The GUI will embed this widget
        in the configuration panel.

        Returns:
            QWidget: Qt widget with engine configuration controls
        """
        pass

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration from the config widget.

        This method extracts values from the widget controls and returns
        them as a dictionary that can be passed to transcribe_line().

        Returns:
            Dict[str, Any]: Configuration parameters
        """
        pass

    @abstractmethod
    def set_config(self, config: Dict[str, Any]):
        """Set configuration values in the config widget.

        Used to restore saved settings when switching engines.

        Args:
            config: Configuration parameters
        """
        pass

    @abstractmethod
    def load_model(self, config: Dict[str, Any]) -> bool:
        """Load the HTR model with given configuration.

        Args:
            config: Configuration parameters (from get_config())

        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        pass

    @abstractmethod
    def unload_model(self):
        """Unload model from memory to free resources.

        Called when switching to a different engine or closing the application.
        """
        pass

    @abstractmethod
    def is_model_loaded(self) -> bool:
        """Check if model is currently loaded.

        Returns:
            bool: True if model is ready for inference
        """
        pass

    @abstractmethod
    def transcribe_line(self, image: np.ndarray, config: Optional[Dict[str, Any]] = None) -> TranscriptionResult:
        """Transcribe a single line image.

        Args:
            image: Line image as numpy array (RGB, shape: H x W x 3)
            config: Optional configuration overrides

        Returns:
            TranscriptionResult: Transcription text and metadata
        """
        pass

    def requires_line_segmentation(self) -> bool:
        """Check if engine requires pre-segmented lines or can process full pages.

        Returns:
            bool: True if lines must be segmented first (TrOCR, CRNN-CTC),
                  False if engine handles full pages (Qwen3, Commercial APIs)
        """
        return True  # Default: most engines need line segmentation

    def transcribe_lines(self, images: List[np.ndarray], config: Optional[Dict[str, Any]] = None) -> List[TranscriptionResult]:
        """Transcribe multiple line images (batch processing).

        Default implementation calls transcribe_line() for each image.
        Engines can override this for optimized batch processing.

        Args:
            images: List of line images
            config: Optional configuration overrides

        Returns:
            List[TranscriptionResult]: Transcriptions for each image
        """
        return [self.transcribe_line(img, config) for img in images]

    def supports_batch(self) -> bool:
        """Check if engine supports optimized batch processing.

        Returns:
            bool: True if transcribe_lines() is optimized, False if it just loops
        """
        return False

    def get_aliases(self) -> List[str]:
        """Get alternative names for this engine (e.g., short CLI aliases).

        Returns:
            List[str]: Alternative names accepted by the registry (default: none)
        """
        return []

    def get_capabilities(self) -> Dict[str, bool]:
        """Get engine capabilities.

        Returns:
            Dict with capability flags:
            - batch_processing: Supports batch inference
            - confidence_scores: Returns confidence scores
            - beam_search: Supports beam search decoding
            - language_model: Uses language model for post-processing
            - preprocessing: Has built-in preprocessing
        """
        return {
            "batch_processing": self.supports_batch(),
            "confidence_scores": False,
            "beam_search": False,
            "language_model": False,
            "preprocessing": False,
        }


class HTREngineRegistry:
    """Registry of available HTR engines.

    Manages discovery, registration, and instantiation of HTR engines.
    """

    def __init__(self):
        self.engines: List[HTREngine] = []
        self._engine_cache: Dict[str, HTREngine] = {}

    def register(self, engine: HTREngine):
        """Register an HTR engine.

        Args:
            engine: HTREngine instance to register
        """
        self.engines.append(engine)
        self._engine_cache[engine.get_name()] = engine
        for alias in engine.get_aliases():
            self._engine_cache[alias] = engine

    def discover_engines(self):
        """Automatically discover and register all available engines.

        Tries to import each engine module and registers it if available.
        """
        profile = load_runtime_profile()
        if profile is not None and hasattr(profile, "discover_engines"):
            if profile.discover_engines(self):
                return

        # Import and register TrOCR engine
        try:
            from engines.trocr_engine import TrOCREngine
            self.register(TrOCREngine())
        except ImportError as e:
            print(f"Warning: Failed to load TrOCR engine: {e}")

        # Import and register Qwen3 engine
        try:
            from engines.qwen3_engine import Qwen3Engine
            self.register(Qwen3Engine())
        except ImportError as e:
            print(f"Warning: Failed to load Qwen3 engine: {e}")

        # Import and register Churro engine
        try:
            from engines.churro_engine import ChurroEngine
            self.register(ChurroEngine())
        except ImportError as e:
            print(f"Warning: Failed to load Churro engine: {e}")

        # Import and register CRNN-CTC engine
        try:
            from engines.pylaia_engine import PyLaiaEngine
            self.register(PyLaiaEngine())
        except ImportError as e:
            print(f"Warning: Failed to load CRNN-CTC engine: {e}")

        # Import and register Kraken engine
        try:
            from engines.kraken_engine import KrakenEngine
            self.register(KrakenEngine())
        except ImportError as e:
            print(f"Warning: Failed to load Kraken engine: {e}")

        # Import and register Commercial API engine
        try:
            from engines.commercial_api_engine import CommercialAPIEngine
            self.register(CommercialAPIEngine())
        except ImportError as e:
            print(f"Warning: Failed to load Commercial API engine: {e}")

        # Import and register Party engine
        try:
            from engines.party_engine import PartyEngine
            self.register(PartyEngine())
        except ImportError as e:
            print(f"Warning: Failed to load Party engine: {e}")

        # Import and register OpenWebUI engine
        try:
            from engines.openwebui_engine import OpenWebUIEngine
            self.register(OpenWebUIEngine())
        except ImportError as e:
            print(f"Warning: Failed to load OpenWebUI engine: {e}")

        # Import and register DeepSeek-OCR engine
        try:
            from engines.deepseek_ocr_engine import DeepSeekOCREngine
            self.register(DeepSeekOCREngine())
        except ImportError as e:
            print(f"Warning: Failed to load DeepSeek-OCR engine: {e}")

        # Import and register LightOnOCR engine
        try:
            from engines.lighton_ocr_engine import LightOnOCREngine
            self.register(LightOnOCREngine())
        except ImportError as e:
            print(f"Warning: Failed to load LightOnOCR engine: {e}")

        # Import and register LapaOCR engine
        try:
            from engines.lapa_ocr_engine import LapaOCREngine
            self.register(LapaOCREngine())
        except ImportError as e:
            print(f"Warning: Failed to load LapaOCR engine: {e}")

        # Import and register PaddleOCR engine
        try:
            from engines.paddle_engine import PaddleOCREngine
            self.register(PaddleOCREngine())
        except ImportError as e:
            print(f"Warning: Failed to load PaddleOCR engine: {e}")

        # Import and register PaddleOCR-VL engine
        try:
            from engines.paddle_vl_engine import PaddleOCRVLEngine
            self.register(PaddleOCRVLEngine())
        except ImportError as e:
            print(f"Warning: Failed to load PaddleOCR-VL engine: {e}")

    def get_available_engines(self) -> List[HTREngine]:
        """Get list of engines with satisfied dependencies.

        Returns:
            List[HTREngine]: Engines that can be used
        """
        return [e for e in self.engines if e.is_available()]

    def get_all_engines(self) -> List[HTREngine]:
        """Get all registered engines (including unavailable ones).

        Returns:
            List[HTREngine]: All registered engines
        """
        return self.engines

    def get_engine_by_name(self, name: str) -> Optional[HTREngine]:
        """Get engine by display name.

        Args:
            name: Engine display name

        Returns:
            Optional[HTREngine]: Engine instance or None if not found
        """
        return self._engine_cache.get(name)

    def get_engine_names(self) -> List[str]:
        """Get list of available engine names.

        Returns:
            List[str]: Engine display names
        """
        return [e.get_name() for e in self.get_available_engines()]


# Global registry instance (singleton pattern)
_global_registry: Optional[HTREngineRegistry] = None


def get_global_registry() -> HTREngineRegistry:
    """Get global HTR engine registry (singleton).

    Returns:
        HTREngineRegistry: Global registry instance
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = HTREngineRegistry()
        _global_registry.discover_engines()
    return _global_registry


# Convenience function for GUI
def get_available_engine_names() -> List[str]:
    """Get list of available engine names (convenience function).

    Returns:
        List[str]: Engine display names
    """
    return get_global_registry().get_engine_names()
