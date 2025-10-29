# OpenWebUI Integration

## Overview

The OpenWebUI engine integrates the uni-freiburg.de OpenWebUI API (OpenAI-compatible) into the HTR plugin GUI, providing access to multiple vision-language models for manuscript transcription.

## Features

- **OpenAI-Compatible API**: Uses the standard OpenAI client library
- **Multiple Models**: Supports all models available on the OpenWebUI platform
- **Full-Page Processing**: VLMs process entire pages without line segmentation
- **Dynamic Model Discovery**: Fetch available models from the server
- **Configurable Parameters**: Temperature and max tokens control
- **Custom Prompts**: Optional custom instructions for better transcription
- **API Key Persistence**: Saved locally in `~/.trocr_gui/api_keys.json`

## Installation

The OpenWebUI engine requires the OpenAI Python library:

```bash
pip install openai
```

## Configuration

### 1. Get API Key

Visit https://openwebui.uni-freiburg.de and obtain your API key.

### 2. GUI Setup

1. Launch the plugin GUI:
   ```bash
   python transcription_gui_plugin.py
   ```

2. Select **"OpenWebUI (Uni Freiburg)"** from the engine dropdown

3. Enter your API key in the configuration panel

4. Click **"Refresh Models"** to load available models from the server

5. Select a model from the dropdown (examples: llama, mistral, gemma, qwen)

6. Adjust generation parameters:
   - **Temperature**: 0.0-1.0 (default: 0.1 for consistency)
   - **Max Tokens**: 100-4096 (default: 500)

7. (Optional) Add a custom prompt for specific transcription requirements

8. Click **"Load Model"** to initialize the client

### 3. Usage

1. Load an image (File → Load Image or drag & drop)

2. Click **"Process"** to transcribe
   - No line segmentation required - VLMs process the full page directly
   - Progress bar shows transcription status

3. View results in the transcription panel

4. Export to TXT or CSV as needed

## API Details

### Base URL

```
https://openwebui.uni-freiburg.de/api
```

### Request Format

The engine sends OpenAI-compatible chat completion requests:

```python
response = client.chat.completions.create(
    model="model-name",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Transcribe the text..."},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
            ]
        }
    ],
    temperature=0.1,
    max_tokens=500
)
```

### Response

The API returns standard OpenAI chat completion responses with:
- Transcribed text in `response.choices[0].message.content`
- Token usage statistics in `response.usage`

## Default Prompt

If no custom prompt is specified, the engine uses:

```
Transcribe the text in this historical manuscript line image.
Return only the transcribed text without any explanation or formatting.
```

## Custom Prompts

You can provide custom instructions to improve transcription quality:

**Example 1: Language-Specific**
```
This is a 19th-century Ukrainian manuscript.
Transcribe the text exactly as written, preserving original spelling and punctuation.
```

**Example 2: Format-Specific**
```
Transcribe this manuscript page line by line.
Preserve line breaks and paragraph structure.
Do not add explanations or corrections.
```

**Example 3: Character-Level Detail**
```
Transcribe this text with attention to:
- Old orthography (e.g., і vs и in Ukrainian)
- Abbreviations and superscript characters
- Punctuation marks
Return only the transcription without commentary.
```

## Technical Implementation

### Engine Class

Location: `engines/openwebui_engine.py`

Key methods:
- `load_model()`: Initializes OpenAI client with base URL
- `transcribe_line()`: Processes image (full page for VLMs)
- `requires_line_segmentation()`: Returns `False` (VLMs handle full pages)
- `_refresh_models()`: Fetches available models via API

### Full-Page Processing

Unlike TrOCR/PyLaia which process line-by-line, OpenWebUI VLMs:
1. Receive the entire page image
2. Analyze layout and content holistically
3. Return complete transcription in one API call

The GUI creates a fake "line segment" representing the full page when processing VLM requests.

## Model Recommendations

Different models have different strengths:

- **LLaMA-based**: Good general-purpose transcription
- **Qwen-VL**: Strong for non-Latin scripts (Cyrillic, Arabic)
- **Gemma**: Fast, good for modern handwriting
- **Mistral**: Strong language understanding

Experiment with different models to find the best fit for your manuscripts.

## Performance Considerations

- **API Latency**: ~2-10 seconds per page (network dependent)
- **Cost**: Check with uni-freiburg.de for API pricing/quotas
- **Rate Limits**: May apply depending on your account tier
- **Token Usage**: Displayed in metadata after each transcription

## Comparison with Other Engines

| Feature | OpenWebUI | TrOCR | Qwen3 VLM | PyLaia | Kraken |
|---------|-----------|-------|-----------|--------|--------|
| Segmentation | Not needed | Required | Not needed | Required | Optional |
| Speed | API latency | Fast (GPU) | Medium (GPU) | Fast (GPU) | Medium |
| Accuracy | Variable | High (finetuned) | High | Medium | Medium |
| Cost | API calls | Free (local) | Free (local) | Free (local) | Free |
| Training | No | Yes | Yes | Yes | Yes |
| GPU Required | No | Yes | Yes | Yes | No |

## Troubleshooting

### "Cannot click Process button"

Fixed in current version. Ensure:
1. Model is loaded (status bar shows "Model loaded")
2. Image is loaded (visible in image viewer)
3. Engine is set to OpenWebUI

### "Error fetching models"

- Check API key is correct
- Verify network connection to https://openwebui.uni-freiburg.de
- Check server status

### "API Error: 401 Unauthorized"

- API key is invalid or expired
- Regenerate key from OpenWebUI dashboard

### "API Error: 429 Too Many Requests"

- Rate limit exceeded
- Wait before retrying or contact administrator for quota increase

### Poor transcription quality

1. Try a different model (some work better for specific scripts)
2. Add a custom prompt with language/format instructions
3. Adjust temperature (lower = more consistent, higher = more creative)
4. Consider fine-tuning a local model (TrOCR/Qwen3) for better results

## File Locations

- Engine implementation: `engines/openwebui_engine.py`
- API keys: `~/.trocr_gui/api_keys.json`
- Plugin registry: `htr_engine_base.py` (line 271-276)

## Future Enhancements

Potential improvements:
- Batch processing support
- Streaming responses for real-time feedback
- Model performance caching
- Cost/usage tracking
- Multiple API endpoint support
- Automatic model selection based on script type

## References

- OpenWebUI Platform: https://openwebui.uni-freiburg.de
- OpenAI API Docs: https://platform.openai.com/docs/api-reference
- Plugin System: `Documentation/PLUGIN_SYSTEM.md`
