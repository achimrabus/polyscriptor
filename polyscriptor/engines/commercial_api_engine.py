"""
Commercial API Engine Plugin

Wraps commercial HTR APIs (OpenAI, Gemini, Claude) as a unified plugin.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np

from polyscriptor.core.engine_base import HTREngine, TranscriptionResult

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Look for .env in the project root (parent of engines/)
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f"[CommercialAPIEngine] Loaded environment variables from {env_path}")
except ImportError:
    print("[CommercialAPIEngine] Warning: python-dotenv not installed. API keys will not be loaded from .env file.")
    print("Install with: pip install python-dotenv")


try:
    from polyscriptor.inference.commercial_api import (
        OpenAIInference, GeminiInference, ClaudeInference,
        check_api_availability,
        OPENAI_MODELS, GEMINI_MODELS, CLAUDE_MODELS,
        fetch_openai_models, fetch_gemini_models
    )
    COMMERCIAL_API_AVAILABLE = True
    API_AVAILABILITY = check_api_availability()
except ImportError:
    COMMERCIAL_API_AVAILABLE = False
    API_AVAILABILITY = {"openai": False, "gemini": False, "claude": False}
    OPENAI_MODELS = []
    GEMINI_MODELS = []
    CLAUDE_MODELS = []
    fetch_openai_models = lambda api_key=None: []
    fetch_gemini_models = lambda api_key=None: []


class CommercialAPIEngine(HTREngine):
    """Commercial API HTR engine plugin."""

    def __init__(self):
        # Instance attributes (avoid type annotations here for broader runtime compatibility in some environments)
        self.model = None  # Can be OpenAI, Gemini, or Claude
        self._current_provider = None


    def get_name(self) -> str:
        return "Commercial APIs"

    def get_description(self) -> str:
        return "OpenAI GPT-4V, Google Gemini, Anthropic Claude vision APIs"

    def is_available(self) -> bool:
        return COMMERCIAL_API_AVAILABLE and any(API_AVAILABILITY.values())

    def get_unavailable_reason(self) -> str:
        if not COMMERCIAL_API_AVAILABLE:
            return "Commercial API support not available. Install with: pip install openai google-generativeai anthropic"
        if not any(API_AVAILABILITY.values()):
            return "No API libraries installed. Install at least one: openai, google-generativeai, or anthropic"
        return ""


    def _get_api_key_file(self) -> 'Path':
        """Get path to API key storage file."""
        from pathlib import Path
        storage_dir = Path.home() / ".trocr_gui"
        storage_dir.mkdir(exist_ok=True)
        return storage_dir / "api_keys.json"




    def _get_api_key_from_env(self, provider: str) -> Optional[str]:
        """Get API key from environment variables based on provider."""
        env_var_map = {
            "OpenAI": "OPENAI_API_KEY",
            "Gemini": "GOOGLE_API_KEY",
            "Claude": "ANTHROPIC_API_KEY"
        }

        env_var = env_var_map.get(provider)
        if env_var:
            return os.getenv(env_var, "")






    def load_model(self, config: Dict[str, Any]) -> bool:
        """Load (initialize) API client."""
        try:
            provider = config.get("provider", "")
            model_name = config.get("model", "")
            api_key = config.get("api_key", "")

            if not api_key:
                print("Error: No API key provided")
                return False

            # Unload previous model
            self.unload_model()

            # Initialize appropriate client
            if provider == "OpenAI":
                self.model = OpenAIInference(api_key=api_key, model=model_name)
                self._current_provider = "openai"
            elif provider == "Gemini":
                self.model = GeminiInference(api_key=api_key, model=model_name)
                self._current_provider = "gemini"
            elif provider == "Claude":
                self.model = ClaudeInference(api_key=api_key, model=model_name)
                self._current_provider = "claude"
            else:
                return False

            return True

        except Exception as e:
            print(f"Error initializing API client: {e}")
            self.model = None
            self._current_provider = None
            return False

    def unload_model(self):
        """Unload (clear) API client."""
        if self.model is not None:
            del self.model
            self.model = None
            self._current_provider = None

    def is_model_loaded(self) -> bool:
        """Check if API client is initialized."""
        return self.model is not None

    def transcribe_line(self, image: np.ndarray, config: Optional[Dict[str, Any]] = None) -> TranscriptionResult:
        """Transcribe a line image with commercial API."""
        if self.model is None:
            return TranscriptionResult(text="[API client not initialized]", confidence=0.0)

        if config is None:
            config = self.get_config()

        custom_prompt = config.get("custom_prompt")

        try:
            # Convert numpy array to PIL Image
            from PIL import Image
            if isinstance(image, np.ndarray):
                pil_image = Image.fromarray(image)
            else:
                pil_image = image

            # All API clients have transcribe() method
            # It returns a string directly, not a dict
            # Enable retry logic for Gemini to handle content blocking
            if self._current_provider == "gemini":
                thinking_mode = config.get("thinking_mode") or None
                fast_direct = thinking_mode == "low"  # low mode: request immediate output
                temperature = config.get("temperature")
                max_tokens = config.get("max_output_tokens")
                # Treat 0 as "no limit" (HTML number fields send 0 for blank)
                if max_tokens is not None and max_tokens <= 0:
                    max_tokens = None
                # Disable early exit by default for full reasoning quality
                fast_direct_early_exit = bool(config.get("fast_direct_early_exit", False))
                auto_continue = bool(config.get("auto_continue", False))
                max_auto_continuations = int(config.get("max_auto_continuations", 2))
                continuation_min_new_chars = int(config.get("continuation_min_new_chars", 50))
                # Reasoning fallback is off (1.0 = never trigger) unless configured
                reasoning_fallback_threshold = float(config.get("reasoning_fallback_threshold", 1.0))
                fallback_cap = int(config.get("fallback_cap", 8192))
                if thinking_mode == "low":
                    lit_val = int(config.get("low_initial_tokens", 0) or 0)
                    if lit_val > 0:
                        max_tokens = lit_val
                        print(f"🔧 LOW thinking mode: overriding max_output_tokens to {max_tokens}")

                # Debug: show final token budget
                print(f"📊 Final settings: thinking_mode={thinking_mode}, max_output_tokens={max_tokens or 'model default'}, temp={temperature if temperature is not None else 1.0}")

                text = self.model.transcribe(
                    pil_image,
                    prompt=custom_prompt,
                    temperature=temperature if temperature is not None else 0.0,
                    max_output_tokens=max_tokens,  # None = no limit, model uses its own maximum
                    auto_retry_on_block=True,
                    safety_relax=True,
                    verbose_block_logging=True,
                    thinking_mode=thinking_mode,
                    fast_direct=fast_direct,
                    fast_direct_early_exit=fast_direct_early_exit,
                    auto_continue=auto_continue,
                    max_auto_continuations=max_auto_continuations,
                    continuation_min_new_chars=continuation_min_new_chars,
                    reasoning_fallback_threshold=reasoning_fallback_threshold,
                    fallback_max_output_tokens=fallback_cap,
                    record_stats_csv="gemini_runs.csv",
                    apply_restriction_prompt=False  # Let model reason freely — improves transcription quality
                )
            else:
                temperature = config.get("temperature")
                max_tokens = config.get("max_output_tokens")
                # Treat 0 as "no limit" (HTML number fields send 0 for blank)
                if max_tokens is not None and max_tokens <= 0:
                    max_tokens = None
                thinking_mode = config.get("thinking_mode") or None
                text = self.model.transcribe(
                    pil_image,
                    prompt=custom_prompt,
                    temperature=temperature if temperature is not None else 0.0,
                    max_output_tokens=max_tokens,  # None = no limit, model uses its own maximum
                    thinking_mode=thinking_mode,
                )

            meta: Dict[str, Any] = {
                "provider": self._current_provider,
                "model": config.get("model", ""),
            }
            if hasattr(self.model, "last_usage") and self.model.last_usage:
                usage = dict(self.model.last_usage)
                thinking_text = usage.pop("thinking_text", None)
                meta["token_usage"] = usage
                if thinking_text:
                    meta["thinking_text"] = thinking_text
            return TranscriptionResult(
                text=text if text else "",
                confidence=1.0,  # API models don't provide confidence
                metadata=meta,
            )

        except Exception as e:
            print(f"Error in API transcription: {e}")
            import traceback
            traceback.print_exc()
            return TranscriptionResult(text=f"[API Error: {e}]", confidence=0.0)

    def get_capabilities(self) -> Dict[str, bool]:
        """Commercial API capabilities."""
        return {
            "batch": False,  # APIs typically process one at a time
            "confidence_scores": False,  # Most don't provide confidence
            "beam_search": False,  # Internal to API
            "language_model": True,  # All are language models
            "preprocessing": True,  # APIs handle preprocessing
        }

    def requires_line_segmentation(self) -> bool:
        """Commercial APIs can process full pages without segmentation."""
        return False
