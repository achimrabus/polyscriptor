# Commercial API Validation Guide

## Problem Fixed

The transcription button was greyed out after entering an API key because the button enable logic wasn't being triggered properly.

## Solutions Implemented

### 1. Automatic Button Enabling

The button now automatically enables when you enter an API key (if an image is already loaded).

**How it works:**
- As you type the API key, the `_on_api_key_changed` handler checks if both the API key and an image are present
- If both conditions are met, the "Transcribe" button is automatically enabled
- Status bar shows helpful messages: "API key entered. Ready to transcribe."

### 2. API Key Validation Button

Added a **"Validate & Check Models"** button next to the API key field.

**Features:**
- Validates the API key with the provider
- Fetches available models from your subscription
- Updates the model dropdown with only models you can access
- Shows visual feedback:
  - ⏳ **Orange**: Validating...
  - ✓ **Green**: Valid! Found X models
  - ✗ **Red**: Invalid

**How to use:**
1. Enter your API key
2. Click **"Validate & Check Models"**
3. Wait for validation (2-5 seconds)
4. Model dropdown will update with your available models

### 3. Provider-Specific Validation

#### OpenAI
- Calls `client.models.list()` to get available models
- Filters for GPT-4 and vision-capable models
- Updates dropdown with models you have access to

#### Gemini
- Calls `genai.list_models()` to get available models
- Filters for vision-capable Gemini models
- Updates dropdown with your available models

#### Claude
- Validates API key format
- Uses default model list (Claude API doesn't expose model listing)
- Key will be fully validated on first transcription

## Usage Guide

### Method 1: Quick Start (No Validation)

1. Load an image (drag & drop or File > Open)
2. Switch to **"Commercial APIs"** tab
3. Select provider (OpenAI / Gemini / Claude)
4. Paste API key
5. Button automatically enables
6. Click **"Transcribe with [Provider]"**

### Method 2: With Validation (Recommended)

1. Switch to **"Commercial APIs"** tab
2. Select provider
3. Paste API key
4. Click **"Validate & Check Models"**
5. Wait for green checkmark (✓)
6. Model dropdown now shows only your available models
7. Load image
8. Click **"Transcribe with [Provider]"**

## Troubleshooting

### Button Still Greyed Out

**Check:**
1. Image is loaded (top panel should show image)
2. API key is entered (not empty)
3. You're on the Commercial API tab (not TrOCR/PyLaia tab)

**If still greyed out:**
- Try clicking "Validate & Check Models" first
- Check status bar for error messages
- Restart GUI if needed

### Validation Fails

**OpenAI:**
```
Error: "Incorrect API key provided"
```
- Check key starts with `sk-...`
- Verify key is copied completely (no spaces)
- Check key is active on platform.openai.com

**Gemini:**
```
Error: "API key not valid"
```
- Check key format (should start with `AIza...`)
- Enable "Generative Language API" on console.cloud.google.com
- Check billing is enabled

**Claude:**
```
Error: "Invalid API key"
```
- Check key starts with `sk-ant-...`
- Verify key on console.anthropic.com
- Check usage limits not exceeded

### No Models Found

If validation succeeds but shows "Using default models":
- Your subscription may not include vision models
- Contact provider support to enable vision API access
- For OpenAI: Upgrade to tier with GPT-4 access
- For Gemini: Check API quotas on Google Cloud Console

## API Key Security Tips

1. **Don't share API keys**: Treat them like passwords
2. **Use the 👁 button**: Toggle visibility only when needed
3. **Rotate keys regularly**: Generate new keys periodically
4. **Set spending limits**: Configure on provider dashboards
5. **Monitor usage**: Check provider dashboards for unexpected usage

## Comparison: Manual Entry vs. Validation

| Feature | Manual Entry | With Validation |
|---------|-------------|-----------------|
| **Speed** | Instant | 2-5 seconds |
| **Model List** | Default (all models) | Your available models only |
| **Key Check** | On first use | Before use |
| **Errors** | During transcription | Before transcription |
| **Recommended For** | Trusted keys | New keys, troubleshooting |

## Example Workflow

### First Time Setup (OpenAI)

```
1. Get API key from platform.openai.com
2. Open transcription_gui_qt.py
3. Switch to "Commercial APIs" tab
4. Select "OpenAI (GPT-4o)"
5. Paste key: sk-proj-abc123...
6. Click "Validate & Check Models"
7. Wait for: "✓ Valid! Found 8 models"
8. Review model dropdown - only shows models you can use
9. Load manuscript image
10. Click "☁️ Transcribe with OpenAI"
```

### Daily Use (After Setup)

```
1. Open GUI
2. Load image
3. Switch to "Commercial APIs" tab
4. Enter saved API key (or paste)
5. Click "☁️ Transcribe with [Provider]"
6. Wait for result (5-15 seconds)
```

## Integration with Existing Features

The API validation works seamlessly with:
- **Image loading**: Drag & drop or file browser
- **Export**: TXT/CSV export includes API transcriptions
- **Statistics**: Shows timing, model used, parameters
- **Comparison mode**: Compare API results with local models

## Cost Estimates After Validation

After validation, you can estimate costs:

**Example: 100 manuscript pages**

| Provider | Model | Est. Cost |
|----------|-------|-----------|
| OpenAI | gpt-4o | $2.50-5.00 |
| Gemini | gemini-2.0-flash | $0.10-0.30 |
| Claude | claude-3-5-sonnet | $3.00-8.00 |

**Note**: Actual costs depend on image size, token usage, and current pricing.

## Advanced: Batch Validation Script

If you need to validate multiple API keys programmatically:

```python
from inference_commercial_api import OpenAIInference, GeminiInference, ClaudeInference

# Test OpenAI
try:
    api = OpenAIInference(api_key="sk-...")
    print("OpenAI key valid!")
except Exception as e:
    print(f"OpenAI key invalid: {e}")

# Test Gemini
try:
    api = GeminiInference(api_key="AIza...")
    print("Gemini key valid!")
except Exception as e:
    print(f"Gemini key invalid: {e}")

# Test Claude
try:
    api = ClaudeInference(api_key="sk-ant-...")
    print("Claude key valid!")
except Exception as e:
    print(f"Claude key invalid: {e}")
```

## FAQ

**Q: Do I need to validate every time?**
A: No, validation is optional. Once you know your key works, you can skip validation.

**Q: Does validation use API credits?**
A: Yes, but minimally. OpenAI/Gemini list calls are free/very cheap. Only actual transcription uses significant credits.

**Q: Can I save API keys?**
A: Currently keys are not saved (for security). Future update may add encrypted key storage.

**Q: Which provider should I use?**
A:
- **Gemini Flash**: Best for cost/speed (recommended for testing)
- **GPT-4o**: Best general accuracy
- **Claude Sonnet**: Best for corrections and difficult handwriting

**Q: Can I use free tier API keys?**
A: Depends on provider:
- OpenAI: Need paid tier for GPT-4 vision
- Gemini: Free tier available (with rate limits)
- Claude: Credit-based, new users get free credits

## Next Steps

1. **Test validation** with your API key
2. **Compare models** - try different providers on same image
3. **Set up billing alerts** on provider dashboards
4. **Monitor usage** - check costs after batch processing
5. **Choose optimal provider** based on your accuracy/cost needs
