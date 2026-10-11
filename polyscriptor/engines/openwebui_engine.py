"""
OpenWebUI Engine Plugin

Wraps the OpenWebUI API (OpenAI-compatible) from uni-freiburg.de as an HTR engine.
Supports multiple models available on the OpenWebUI platform.
"""

from typing import Dict, Any, Optional, List
from pathlib import Path
import numpy as np
from PIL import Image
import io
import base64

from polyscriptor.core.engine_base import HTREngine, TranscriptionResult


try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False


class OpenWebUIEngine(HTREngine):
    """OpenWebUI API HTR engine plugin (OpenAI-compatible)."""

    def __init__(self):
        self.client: Optional[OpenAI] = None
        self._available_models: List[str] = []

        # Store config from load_model for batch processing
        self._loaded_config: Dict[str, Any] = {}


        # Default API configuration
        self.base_url = "https://openwebui.uni-freiburg.de/api"
        
        # Load environment variables from .env file (only once when instantiated)
        self._load_env_variables()

    def _load_env_variables(self):
        """Load environment variables from .env file if available."""
        try:
            from dotenv import load_dotenv
            # Look for .env in the project root (parent of engines/)
            env_path = Path(__file__).parent.parent / ".env"
            if env_path.exists():
                load_dotenv(env_path)
        except ImportError:
            # Silently skip if python-dotenv is not installed
            # Environment variables can still be set via OS
            pass

        # Load environment variables from .env file (if available)
        self._load_env_file()

    def _load_env_file(self):
        """Load environment variables from project root's .env file.
        
        Looks for .env in the project root directory (parent of engines/).
        Silently skips loading if python-dotenv is not installed or if .env doesn't exist.
        
        If .env loading fails or is skipped, the engine will still work if the API key
        is provided through the config dict.
        """
        if not DOTENV_AVAILABLE:
            return
            
        env_path = Path(__file__).parent.parent / ".env"
        if env_path.exists():
            load_dotenv(env_path)

    def get_name(self) -> str:
        return "OpenWebUI"

    def get_description(self) -> str:
        return "OpenWebUI API from openwebui.uni-freiburg.de (OpenAI-compatible, multiple models)"

    def is_available(self) -> bool:
        return OPENAI_AVAILABLE

    def get_unavailable_reason(self) -> str:
        if not OPENAI_AVAILABLE:
            return "OpenAI library not installed. Install with: pip install openai"
        return ""



    def _get_api_key_file(self) -> 'Path':
        """Get path to API key storage file."""
        from pathlib import Path
        storage_dir = Path.home() / ".trocr_gui"
        storage_dir.mkdir(exist_ok=True)
        return storage_dir / "api_keys.json"






    def load_model(self, config: Dict[str, Any]) -> bool:
        """Initialize OpenWebUI client."""
        try:
            api_key = config.get("api_key", "")

            if not api_key:
                print("Error: No API key provided. Paste your key in the field.")
                return False

            # Store config for batch processing (model, temperature, etc.)
            self._loaded_config = config.copy()

            # Save API key for future use

            # Allow base_url override from config (web UI field or CLI)
            if config.get("base_url", "").strip():
                self.base_url = config["base_url"].strip()

            # Initialize client
            self.client = OpenAI(
                base_url=self.base_url,
                api_key=api_key
            )

            model = config.get("model", config.get("model_id", "unknown"))
            print(f"[OpenWebUI] Client initialized with base URL: {self.base_url}, model: {model}")
            return True

        except Exception as e:
            print(f"Error initializing OpenWebUI client: {e}")
            self.client = None
            return False

    def unload_model(self):
        """Unload OpenWebUI client."""
        if self.client is not None:
            self.client = None
        self._loaded_config = {}

    def is_model_loaded(self) -> bool:
        """Check if client is initialized."""
        return self.client is not None

    def transcribe_line(self, image: np.ndarray, config: Optional[Dict[str, Any]] = None) -> TranscriptionResult:
        """Transcribe a line image with OpenWebUI API."""
        if self.client is None:
            return TranscriptionResult(text="[OpenWebUI client not initialized]", confidence=0.0)

        if config is None:
            # First try loaded config (from batch processing), then GUI config
            if self._loaded_config:
                config = self._loaded_config
            else:
                config = self.get_config()

        try:
            # Convert numpy array to PIL Image
            if isinstance(image, np.ndarray):
                pil_image = Image.fromarray(image)
            else:
                pil_image = image

            # Convert to RGB if needed
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')

            # Encode image to base64
            buffered = io.BytesIO()
            pil_image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

            # Prepare prompt
            custom_prompt = config.get("custom_prompt")
            if custom_prompt:
                prompt = custom_prompt
            else:
                prompt = (
                    "Transcribe the text in this historical manuscript line image. "
                    "Return only the transcribed text without any explanation or formatting."
                )

            # Get model and parameters
            model = config.get("model", "gpt-4-vision-preview")
            temperature = config.get("temperature", 0.1)
            max_tokens = config.get("max_tokens")
            # Treat 0 as "no limit" (HTML number fields send 0 for blank)
            if max_tokens is not None and max_tokens <= 0:
                max_tokens = None

            # Call OpenWebUI API (OpenAI-compatible)
            api_kwargs = dict(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img_base64}"
                                }
                            }
                        ]
                    }
                ],
                temperature=temperature,
            )
            if max_tokens is not None:
                api_kwargs["max_tokens"] = max_tokens
            response = self.client.chat.completions.create(**api_kwargs)

            # Extract transcription
            text = response.choices[0].message.content.strip()

            # Extract usage info
            usage = {}
            if hasattr(response, 'usage') and response.usage:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }

            return TranscriptionResult(
                text=text,
                confidence=1.0,  # OpenWebUI doesn't provide confidence
                metadata={
                    "provider": "openwebui",
                    "model": model,
                    "usage": usage
                }
            )

        except Exception as e:
            print(f"Error in OpenWebUI transcription: {e}")
            import traceback
            traceback.print_exc()
            return TranscriptionResult(text=f"[OpenWebUI Error: {e}]", confidence=0.0)

    def get_capabilities(self) -> Dict[str, bool]:
        """OpenWebUI capabilities."""
        return {
            "batch": False,
            "confidence_scores": False,
            "beam_search": False,
            "language_model": True,
            "preprocessing": True,
        }

    def requires_line_segmentation(self) -> bool:
        """OpenWebUI VLMs can process full pages directly without segmentation."""
        return False  # VLMs process full page images
