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
    import google.generativeai as genai
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
        temperature: float = 0.0,
        **kwargs
    ) -> str:
        """
        Transcribe with OpenAI GPT-4 Vision.

        Args:
            image: PIL Image
            prompt: Custom prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 = deterministic)
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
    """Google Gemini Pro Vision / Flash inference."""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.0-flash",  # gemini-2.0-flash, gemini-1.5-pro-002
        default_prompt: Optional[str] = None
    ):
        """
        Initialize Gemini inference.

        Args:
            api_key: Google API key
            model: Model name
            default_prompt: Default transcription prompt
        """
        if not GEMINI_AVAILABLE:
            raise ImportError("Google Generative AI library not installed. "
                              "Install with: pip install google-generativeai")

        super().__init__(api_key, default_prompt)
        self.model_name = model

        # Configure Gemini
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)

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
        temperature: float = 0.0,
        max_output_tokens: int = 500,
        **kwargs
    ) -> str:
        """
        Transcribe with Google Gemini.

        Args:
            image: PIL Image
            prompt: Custom prompt
            temperature: Sampling temperature (0.0 = deterministic)
            max_output_tokens: Maximum tokens to generate
            **kwargs: Additional Gemini parameters

        Returns:
            Transcribed text
        """
        prompt = prompt or self.default_prompt

        # Resize if needed (Gemini supports up to 3072x3072)
        image = self.resize_image_if_needed(image, max_dimension=3072)

        # Generate
        generation_config = genai.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            **kwargs
        )

        response = self.model.generate_content(
            [prompt, image],
            generation_config=generation_config
        )

        # Handle response with proper error checking
        if not response.parts:
            # Check for safety/block reasons
            if hasattr(response, 'prompt_feedback'):
                raise ValueError(f"Content generation blocked: {response.prompt_feedback}")
            raise ValueError("No response generated. The response might have been blocked by safety filters.")

        # Extract text from response
        try:
            return response.text.strip()
        except ValueError as e:
            # Response might be blocked or incomplete
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'finish_reason'):
                    raise ValueError(f"Content generation issue: {candidate.finish_reason}. "
                                   "This might be due to safety filters or content policy violations.")
            raise ValueError(f"Failed to extract text from response: {e}")


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


# API model lists (for GUI dropdown)
OPENAI_MODELS = [
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-turbo",
    "gpt-4-vision-preview",
]

GEMINI_MODELS = [
    # Free tier models (generally available)
    "gemini-1.5-flash",
    "gemini-1.5-flash-002",
    "gemini-1.5-flash-8b",
    # Paid/preview models (may require upgrade)
    "gemini-1.5-pro",
    "gemini-1.5-pro-002",
    "gemini-2.0-flash-exp",
    # Experimental (may not be available to all users)
    "gemini-exp-1206",
]

CLAUDE_MODELS = [
    "claude-3-5-sonnet-20241022",
    "claude-3-5-haiku-20241022",
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307",
]


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
