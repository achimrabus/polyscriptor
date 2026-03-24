"""
Commercial VLM/LLM API inference for manuscript transcription.

Supports:
- OpenAI GPT-4 Vision / GPT-4o
- Google Gemini Pro Vision / Gemini Flash
- Anthropic Claude 3 (Opus, Sonnet, Haiku)

Usage:
    # OpenAI
    api = OpenAIInference(api_key="sk-...")
    text = api.transcribe(image)

    # Gemini
    api = GeminiInference(api_key="...")
    text = api.transcribe(image)

    # Claude
    api = ClaudeInference(api_key="sk-ant-...")
    text = api.transcribe(image)
"""

import base64
import io
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any
from PIL import Image

# API clients (install with: pip install openai google-generativeai anthropic)
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from google import genai as _google_genai_new
    from google.genai import types as _google_genai_types
    GEMINI_AVAILABLE = True
    GEMINI_NEW_SDK = True
except ImportError:
    GEMINI_NEW_SDK = False
    try:
        import google.generativeai as genai  # legacy fallback
        GEMINI_AVAILABLE = True
    except ImportError:
        GEMINI_AVAILABLE = False

try:
    from anthropic import Anthropic
    CLAUDE_AVAILABLE = True
except ImportError:
    CLAUDE_AVAILABLE = False


class BaseAPIInference(ABC):
    """Base class for commercial API inference."""

    def __init__(self, api_key: str, default_prompt: Optional[str] = None):
        """
        Initialize API client.

        Args:
            api_key: API key for the service
            default_prompt: Default transcription prompt
        """
        self.api_key = api_key
        self.default_prompt = default_prompt or self._get_default_prompt()

    @abstractmethod
    def _get_default_prompt(self) -> str:
        """Get default transcription prompt."""
        pass

    @abstractmethod
    def transcribe(
        self,
        image: Image.Image,
        prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Transcribe a manuscript line image.

        Args:
            image: PIL Image
            prompt: Custom prompt (uses default if None)
            **kwargs: Provider-specific parameters

        Returns:
            Transcribed text
        """
        pass

    @staticmethod
    def encode_image_base64(image: Image.Image, format: str = "PNG") -> str:
        """
        Encode PIL Image to base64 string.

        Args:
            image: PIL Image
            format: Image format (PNG, JPEG, etc.)

        Returns:
            Base64-encoded image string
        """
        buffered = io.BytesIO()
        image.save(buffered, format=format)
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    @staticmethod
    def resize_image_if_needed(
        image: Image.Image,
        max_dimension: int = 2048
    ) -> Image.Image:
        """
        Resize image if larger than max dimension while preserving aspect ratio.

        Args:
            image: PIL Image
            max_dimension: Maximum width or height

        Returns:
            Resized image (or original if already small enough)
        """
        width, height = image.size

        if width <= max_dimension and height <= max_dimension:
            return image

        # Calculate new size preserving aspect ratio
        if width > height:
            new_width = max_dimension
            new_height = int(height * (max_dimension / width))
        else:
            new_height = max_dimension
            new_width = int(width * (max_dimension / height))

        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


class OpenAIInference(BaseAPIInference):
    """OpenAI GPT-4 Vision / GPT-4o inference."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",  # gpt-4o, gpt-4-vision-preview, gpt-4-turbo
        default_prompt: Optional[str] = None
    ):
        """
        Initialize OpenAI inference.

        Args:
            api_key: OpenAI API key
            model: Model name
            default_prompt: Default transcription prompt
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library not installed. Install with: pip install openai")

        super().__init__(api_key, default_prompt)
        self.model = model
        self.client = OpenAI(api_key=api_key)

    def _get_default_prompt(self) -> str:
        return (
            "Transcribe all handwritten text in this manuscript image. "
            "Preserve the original language (Cyrillic, Latin, etc.) and layout. "
            "Output only the transcribed text without any additional commentary."
        )

    def transcribe(
        self,
        image: Image.Image,
        prompt: Optional[str] = None,
        max_tokens: int = 500,
    temperature: float = 1.0,
        **kwargs
    ) -> str:
        """
        Transcribe with OpenAI GPT-4 Vision.

        Args:
            image: PIL Image
            prompt: Custom prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (web default ~1.0). Lower (0-0.3) = deterministic; higher = more variation.
            **kwargs: Additional OpenAI parameters

        Returns:
            Transcribed text
        """
        prompt = prompt or self.default_prompt

        # Resize if needed (GPT-4V supports up to 2048x2048)
        image = self.resize_image_if_needed(image, max_dimension=2048)

        # Encode image
        base64_image = self.encode_image_base64(image, format="PNG")

        # API call
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )

        return response.choices[0].message.content.strip()


class GeminiInference(BaseAPIInference):
    """Google Gemini inference via google-genai SDK (with legacy google-generativeai fallback)."""

    # thinking_mode string -> thinking_budget token count (max tokens for internal reasoning)
    # "low":  8000  — moderate budget; fast enough for most lines
    # "high": None  — no ThinkingConfig passed at all; model decides dynamically (no cap)
    _THINKING_BUDGETS = {"low": 8000, "high": None}

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.0-flash",
        default_prompt: Optional[str] = None,
    ):
        if not GEMINI_AVAILABLE:
            raise ImportError(
                "Google AI library not installed. Install with: pip install google-genai"
            )
        super().__init__(api_key, default_prompt)
        self.model_name = model
        # Populated after each transcribe() call — for UI token display
        self.last_usage: Dict[str, Any] = {}
        self._last_call_usage: Dict[str, Any] = {}

        if GEMINI_NEW_SDK:
            self._client = _google_genai_new.Client(api_key=api_key)
        else:
            # Legacy fallback
            genai.configure(api_key=api_key)
            self._legacy_model = genai.GenerativeModel(model)

    def _get_default_prompt(self) -> str:
        return (
            "Transcribe all handwritten text in this manuscript image. "
            "Preserve the original language (Cyrillic, Latin, etc.) and layout. "
            "Output only the transcribed text without any additional commentary."
        )

    def _build_config(self, temperature, max_output_tokens, thinking_budget, safety_settings):
        """Build GenerateContentConfig for google-genai SDK."""
        kw: Dict[str, Any] = {"temperature": temperature}
        if max_output_tokens:
            kw["max_output_tokens"] = max_output_tokens
        if safety_settings:
            kw["safety_settings"] = safety_settings
        if thinking_budget is not None:
            kw["thinking_config"] = _google_genai_types.ThinkingConfig(
                thinking_budget=thinking_budget
            )
        return _google_genai_types.GenerateContentConfig(**kw)

    def _generate(self, prompt, image, temperature, thinking_budget, safety_settings, verbose):
        """Single generate call. Handles thinking-not-supported gracefully."""
        if not GEMINI_NEW_SDK:
            # Legacy google-generativeai path
            gen_cfg = genai.GenerationConfig(temperature=temperature or 0.0)
            resp = self._legacy_model.generate_content(
                [prompt, image], generation_config=gen_cfg, safety_settings=safety_settings
            )
            self._last_call_usage = {}
            return resp.text.strip()

        config = self._build_config(temperature or 0.0, None, thinking_budget, safety_settings)
        try:
            resp = self._client.models.generate_content(
                model=self.model_name, contents=[prompt, image], config=config
            )
        except Exception as e:
            err = str(e)
            # Non-thinking models reject ThinkingConfig with a 400 error
            if thinking_budget and thinking_budget > 0 and (
                "thinking" in err.lower() or "400" in err
            ):
                if verbose:
                    print(f"Model does not support thinking_budget={thinking_budget}, retrying without.")
                config = self._build_config(temperature or 0.0, None, 0, safety_settings)
                resp = self._client.models.generate_content(
                    model=self.model_name, contents=[prompt, image], config=config
                )
            else:
                raise

        usage = getattr(resp, "usage_metadata", None)
        self._last_call_usage = {
            "prompt_tokens": getattr(usage, "prompt_token_count", None) if usage else None,
            "output_tokens": getattr(usage, "candidates_token_count", None) if usage else None,
            "thinking_tokens": getattr(usage, "thoughts_token_count", None) if usage else None,
            "total_tokens": getattr(usage, "total_token_count", None) if usage else None,
        }
        return resp.text.strip()

    def _maybe_continue(
        self,
        current_text: str,
        prompt: str,
        image,
        thinking_budget,
        safety_settings,
        auto_continue: bool,
        max_auto_continuations: int,
        continuation_min_new_chars: int,
        verbose_block_logging: bool,
    ) -> str:
        if not auto_continue:
            return current_text
        accumulated = current_text
        for pass_idx in range(1, max_auto_continuations + 1):
            continuation_prompt = (
                f"{prompt}\n\nPartial transcription so far (DO NOT repeat it):\n"
                f"{accumulated}\n\nContinue transcribing remaining, previously UNTRANSCRIBED text. "
                "Output ONLY the new continuation without repeating prior characters."
            )
            try:
                new_chunk = self._generate(
                    continuation_prompt, image, None, thinking_budget,
                    safety_settings, verbose_block_logging
                )
            except Exception as e:
                if verbose_block_logging:
                    print(f"Continuation {pass_idx} failed: {e}")
                break
            if not new_chunk:
                if verbose_block_logging:
                    print(f"Continuation {pass_idx}: no new text, stopping.")
                break
            # Guard against repetition
            if accumulated and new_chunk.startswith(accumulated[:200]):
                overlap_pos = new_chunk.find(accumulated[-50:])
                if overlap_pos > 0:
                    new_chunk = new_chunk[overlap_pos + len(accumulated[-50:]):]
            delta = len(new_chunk)
            if delta < continuation_min_new_chars:
                if verbose_block_logging:
                    print(f"Continuation {pass_idx}: only {delta} chars, stopping.")
                break
            accumulated += ("\n" if not accumulated.endswith("\n") else "") + new_chunk
            if verbose_block_logging:
                print(f"Continuation {pass_idx}: +{delta} chars (total {len(accumulated)})")
        return accumulated

    def transcribe(
        self,
        image,
        prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_output_tokens: Optional[int] = None,
        auto_retry_on_block: bool = True,
        safety_relax: bool = True,
        verbose_block_logging: bool = True,
        thinking_mode: Optional[str] = None,
        fast_direct: bool = False,
        fast_direct_early_exit: bool = True,
        auto_continue: bool = False,
        max_auto_continuations: int = 2,
        continuation_min_new_chars: int = 50,
        reasoning_fallback_threshold: float = 1.0,
        record_stats_csv: Optional[str] = None,
        apply_restriction_prompt: bool = False,
        fallback_max_output_tokens: int = 8192,
        **kwargs,
    ) -> str:
        """Transcribe a manuscript image with Google Gemini.

        Args:
            image: PIL Image or numpy array
            prompt: Transcription prompt (uses default if None)
            temperature: Sampling temperature (0.0 = deterministic)
            max_output_tokens: Output token cap (None = model default)
            thinking_mode: None | "low" | "high" -- maps to thinking_budget
            record_stats_csv: Path to append usage CSV row (None to skip)
            auto_continue: Request continuation calls if output seems truncated
        """
        from PIL import Image as _PIL_Image
        import numpy as np
        if isinstance(image, np.ndarray):
            image = _PIL_Image.fromarray(image)
        image = self.resize_image_if_needed(image, max_dimension=3072)
        prompt = prompt or self.default_prompt

        # Map thinking_mode to thinking_budget
        thinking_budget = self._THINKING_BUDGETS.get(thinking_mode)  # None if mode is None/unknown

        # Safety settings
        safety_settings = None
        if safety_relax and GEMINI_NEW_SDK:
            safety_settings = [
                _google_genai_types.SafetySetting(category=cat, threshold="BLOCK_NONE")
                for cat in (
                    "HARM_CATEGORY_HARASSMENT",
                    "HARM_CATEGORY_HATE_SPEECH",
                    "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "HARM_CATEGORY_DANGEROUS_CONTENT",
                )
            ]

        self._last_call_usage = {}

        try:
            result_text = self._generate(
                prompt, image, temperature, thinking_budget, safety_settings, verbose_block_logging
            )
        except Exception as e:
            raise ValueError(f"Gemini transcription failed: {e}") from e

        # Persist usage for callers (e.g. statistics panel, CSV logging)
        self.last_usage = dict(self._last_call_usage)
        u = self.last_usage
        if verbose_block_logging and u.get("total_tokens"):
            print(
                f"[tokens] prompt={u.get('prompt_tokens')} "
                f"output={u.get('output_tokens')} "
                f"thinking={u.get('thinking_tokens')} "
                f"total={u.get('total_tokens')}"
            )

        if record_stats_csv:
            try:
                from datetime import datetime
                with open(record_stats_csv, "a") as f:
                    f.write(
                        f"{datetime.utcnow().isoformat()},"
                        f"{self.model_name},"
                        f"{thinking_mode or 'default'},"
                        f"final_success,"
                        f"{u.get('prompt_tokens')},"
                        f"{u.get('output_tokens')},"
                        f"{u.get('thinking_tokens')},"
                        f"{u.get('total_tokens')},"
                        f"{len(result_text)}\n"
                    )
            except Exception as csv_e:
                if verbose_block_logging:
                    print(f"Stats logging failed: {csv_e}")

        return self._maybe_continue(
            result_text, prompt, image, thinking_budget, safety_settings,
            auto_continue, max_auto_continuations, continuation_min_new_chars,
            verbose_block_logging,
        )

class ClaudeInference(BaseAPIInference):
    """Anthropic Claude 3 inference (Opus, Sonnet, Haiku)."""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-3-5-sonnet-20241022",  # claude-3-5-sonnet-20241022, claude-3-opus-20240229, claude-3-haiku-20240307
        default_prompt: Optional[str] = None
    ):
        """
        Initialize Claude inference.

        Args:
            api_key: Anthropic API key
            model: Model name
            default_prompt: Default transcription prompt
        """
        if not CLAUDE_AVAILABLE:
            raise ImportError("Anthropic library not installed. Install with: pip install anthropic")

        super().__init__(api_key, default_prompt)
        self.model = model
        self.client = Anthropic(api_key=api_key)

    def _get_default_prompt(self) -> str:
        return (
            "Transcribe all handwritten text in this manuscript image. "
            "Preserve the original language (Cyrillic, Latin, etc.) and layout. "
            "Output only the transcribed text without any additional commentary."
        )

    def transcribe(
        self,
        image: Image.Image,
        prompt: Optional[str] = None,
        max_tokens: int = 500,
        temperature: float = 0.0,
        **kwargs
    ) -> str:
        """
        Transcribe with Anthropic Claude.

        Args:
            image: PIL Image
            prompt: Custom prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 = deterministic)
            **kwargs: Additional Claude parameters

        Returns:
            Transcribed text
        """
        prompt = prompt or self.default_prompt

        # Resize if needed (Claude supports up to 1568px on longest side)
        image = self.resize_image_if_needed(image, max_dimension=1568)

        # Encode image
        base64_image = self.encode_image_base64(image, format="PNG")

        # API call
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": base64_image
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ],
            **kwargs
        )

        return response.content[0].text.strip()


# Model availability checks
def check_api_availability() -> Dict[str, bool]:
    """Check which API libraries are installed."""
    return {
        "openai": OPENAI_AVAILABLE,
        "gemini": GEMINI_AVAILABLE,
        "claude": CLAUDE_AVAILABLE,
    }


# Fallback API model lists (used if dynamic fetching fails)
OPENAI_MODELS_FALLBACK = [
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4o-2024-11-20",
    "chatgpt-4o-latest",
    "gpt-4-turbo",
    "gpt-4-vision-preview",
    "o1-preview",
    "o1-mini",
]

GEMINI_MODELS_FALLBACK = [
    # Free tier models (generally available)
    "gemini-1.5-flash",
    "gemini-1.5-flash-002",
    "gemini-1.5-flash-8b",
    "gemini-2.0-flash-exp",
    # Paid/preview models (may require upgrade)
    "gemini-1.5-pro",
    "gemini-1.5-pro-002",
    "gemini-1.5-pro-exp-0827",
    # Experimental (may not be available to all users)
    "gemini-exp-1206",
    "gemini-exp-1121",
    # Gemini 3 preview models (latest, may have restrictions)
    "gemini-3-pro-preview",
]

CLAUDE_MODELS_FALLBACK = [
    "claude-opus-4-6",
    "claude-sonnet-4-6",
    "claude-haiku-4-5-20251001",
    "claude-3-5-sonnet-20241022",
    "claude-3-5-haiku-20241022",
    "claude-3-opus-20240229",
    "claude-3-haiku-20240307",
]


def fetch_openai_models(api_key: str = None) -> list:
    """
    Dynamically fetch available OpenAI models from API.

    Args:
        api_key: OpenAI API key (uses env var if not provided)

    Returns:
        List of vision-capable model IDs, or fallback list if fetch fails
    """
    if not OPENAI_AVAILABLE:
        return OPENAI_MODELS_FALLBACK

    try:
        import os
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            return OPENAI_MODELS_FALLBACK

        client = OpenAI(api_key=api_key)
        models = client.models.list()

        # Filter for vision-capable models (GPT-4 family + o1)
        vision_models = []
        for model in models.data:
            model_id = model.id
            # Include GPT-4 vision models and o1 models
            if any(prefix in model_id for prefix in [
                "gpt-4o", "gpt-4-turbo", "gpt-4-vision",
                "chatgpt-4o", "o1-", "gpt-4.5"  # Include potential GPT-4.5
            ]):
                vision_models.append(model_id)

        # Sort with newest/best models first
        vision_models.sort(reverse=True)

        # Return dynamic list if we found models, otherwise fallback
        return vision_models if vision_models else OPENAI_MODELS_FALLBACK

    except Exception as e:
        print(f"[OpenAI] Could not fetch models dynamically: {e}")
        print(f"[OpenAI] Using fallback model list")
        return OPENAI_MODELS_FALLBACK


def fetch_gemini_models(api_key: str = None) -> list:
    """Dynamically fetch available Gemini models; returns fallback list on failure."""
    if not GEMINI_AVAILABLE:
        return GEMINI_MODELS_FALLBACK
    try:
        import os
        api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            return GEMINI_MODELS_FALLBACK
        if GEMINI_NEW_SDK:
            client = _google_genai_new.Client(api_key=api_key)
            models = [
                m.name.replace("models/", "")
                for m in client.models.list()
                if "generateContent" in (getattr(m, "supported_actions", None) or [])
            ]
        else:
            genai.configure(api_key=api_key)
            models = [
                m.name.replace("models/", "")
                for m in genai.list_models()
                if "generateContent" in m.supported_generation_methods
            ]
        models = [m for m in models if m.startswith("gemini")]
        models.sort(reverse=True)
        return models if models else GEMINI_MODELS_FALLBACK
    except Exception as e:
        print(f"[Gemini] Could not fetch models: {e}")
        return GEMINI_MODELS_FALLBACK

def fetch_claude_models(api_key: str = None) -> list:
    """
    Dynamically fetch available Claude models via Anthropic API.

    Returns:
        List of Claude model IDs (newest first), or fallback list if fetch fails.
    """
    if not CLAUDE_AVAILABLE:
        return CLAUDE_MODELS_FALLBACK

    try:
        import os
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            return CLAUDE_MODELS_FALLBACK

        client = Anthropic(api_key=api_key)
        models_page = client.models.list()
        model_ids = [m.id for m in models_page.data]
        # Sort newest first (IDs contain dates like -20241022 or version numbers)
        model_ids.sort(reverse=True)
        return model_ids if model_ids else CLAUDE_MODELS_FALLBACK

    except Exception as e:
        print(f"[Claude] Could not fetch models dynamically: {e}")
        return CLAUDE_MODELS_FALLBACK


# Initialize model lists (will be updated when API keys are provided)
OPENAI_MODELS = OPENAI_MODELS_FALLBACK.copy()
GEMINI_MODELS = GEMINI_MODELS_FALLBACK.copy()
CLAUDE_MODELS = CLAUDE_MODELS_FALLBACK.copy()


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) < 4:
        print("Usage: python inference_commercial_api.py <provider> <api_key> <image_path>")
        print("Providers: openai, gemini, claude")
        sys.exit(1)

    provider = sys.argv[1].lower()
    api_key = sys.argv[2]
    image_path = sys.argv[3]

    # Load image
    image = Image.open(image_path).convert("RGB")

    # Initialize appropriate inference client
    if provider == "openai":
        api = OpenAIInference(api_key)
    elif provider == "gemini":
        api = GeminiInference(api_key)
    elif provider == "claude":
        api = ClaudeInference(api_key)
    else:
        print(f"Unknown provider: {provider}")
        sys.exit(1)

    # Transcribe
    print(f"Transcribing with {provider}...")
    text = api.transcribe(image)
    print(f"\nResult: {text}")
