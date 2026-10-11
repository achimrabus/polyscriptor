"""
HTR Engine Plugin System - Base Classes and Registry

This module defines the plugin architecture for HTR (Handwritten Text Recognition) polyscriptor.engines.
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

_runtime_profile = None
_runtime_profile_loaded = False


def load_runtime_profile():
    """Return the optional deployment-profile module named by POLYSCRIPTOR_PROFILE.

    A deployment may point this env var at an importable module that customises
    engine discovery, model presets and segmentation (see the optional hooks
    consumed in ``discover_engines``, ``polyscriptor.inference.pylaia_native`` and the web
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
    """Result from HTR engine transcription.

    ``error`` is the honest failure channel: engines that could not transcribe
    set it (while keeping the bracketed ``[Error: ...]`` text for older
    consumers). ``is_error`` is the single test consumers should use.
    """
    text: str
    confidence: float = 1.0
    metadata: Dict[str, Any] = None
    error: Optional[str] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    @property
    def is_error(self) -> bool:
        return self.error is not None

    @classmethod
    def error_result(cls, message: str) -> "TranscriptionResult":
        """Standard error result: bracketed text for old consumers, real field for new."""
        return cls(text=f"[Error: {message}]", confidence=0.0, error=message)


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

    def get_config(self) -> Dict[str, Any]:
        """Default configuration for headless use.

        The web UI builds its config from the engine's JSON schema
        (see polyscriptor_server._get_config_schema), so engines receive an
        explicit dict; this default is only the None-config fallback.

        Returns:
            Dict[str, Any]: Configuration parameters
        """
        return {}

    def set_config(self, config: Dict[str, Any]):
        """Store configuration for engines that keep widget-free state."""
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
            - polyscriptor.cli.batch: Supports batch inference
            - confidence_scores: Returns confidence scores
            - beam_search: Supports beam search decoding
            - language_model: Uses language model for post-processing
            - preprocessing: Has built-in preprocessing
        """
        return {
            "batch": self.supports_batch(),
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

        # (module, class) pairs; order defines UI display order.
        engine_specs = [
            ("polyscriptor.engines.trocr_engine", "TrOCREngine"),
            ("polyscriptor.engines.qwen3_engine", "Qwen3Engine"),
            ("polyscriptor.engines.churro_engine", "ChurroEngine"),
            ("polyscriptor.engines.pylaia_engine", "PyLaiaEngine"),
            ("polyscriptor.engines.kraken_engine", "KrakenEngine"),
            ("polyscriptor.engines.commercial_api_engine", "CommercialAPIEngine"),
            ("polyscriptor.engines.party_engine", "PartyEngine"),
            ("polyscriptor.engines.openwebui_engine", "OpenWebUIEngine"),
            ("polyscriptor.engines.deepseek_ocr_engine", "DeepSeekOCREngine"),
            ("polyscriptor.engines.lighton_ocr_engine", "LightOnOCREngine"),
            ("polyscriptor.engines.lapa_ocr_engine", "LapaOCREngine"),
            ("polyscriptor.engines.paddle_engine", "PaddleOCREngine"),
            ("polyscriptor.engines.paddle_vl_engine", "PaddleOCRVLEngine"),
            ("polyscriptor.engines.chandra_engine", "ChandraEngine"),
        ]

        for module_name, class_name in engine_specs:
            try:
                engine_class = importlib.import_module(module_name).__getattribute__(class_name)
                self.register(engine_class())
            except ImportError as e:
                print(f"Warning: Failed to load {class_name}: {e}")
            except Exception as e:  # engine raising at import time must not kill the registry
                print(f"Warning: Failed to load {class_name}: {type(e).__name__}: {e}")

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
