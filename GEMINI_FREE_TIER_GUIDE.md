# Gemini Free Tier Usage Guide

## Issues Fixed

### 1. **Response.text Error** ✅
**Problem:** Gemini API was returning a response without accessible `.text` attribute

**Root Cause:**
- Gemini's API can block content for various reasons (safety filters, content policy, rate limits)
- The code wasn't checking if the response had content before accessing `.text`

**Fix Applied:**
```python
# Now checks if response has content
if not response.parts:
    # Provide helpful error message
    raise ValueError("No response generated. Might be blocked by safety filters.")

# Try to access text with error handling
try:
    return response.text.strip()
except ValueError as e:
    # Explain what went wrong
    raise ValueError(f"Content generation issue: {candidate.finish_reason}")
```

### 2. **Free Tier Model Availability** ✅
**Problem:** Gemini 2.5 Pro/Flash models might not be available on free tier

**Fix:** Updated default model list to prioritize free tier models:

**Free Tier Models (should work):**
- `gemini-1.5-flash` ← **Recommended for free tier**
- `gemini-1.5-flash-002`
- `gemini-1.5-flash-8b` ← **Fastest & cheapest**

**Paid/Preview Models (may require upgrade):**
- `gemini-1.5-pro`
- `gemini-1.5-pro-002`
- `gemini-2.0-flash-exp`
- `gemini-exp-1206`

## Free Tier Limitations

### What's Available on Free Tier?

**✅ Available:**
- Gemini 1.5 Flash models
- 15 requests per minute (RPM)
- 1 million tokens per minute (TPM)
- 1,500 requests per day (RPD)

**❌ NOT Available or Limited:**
- Gemini 2.5 models (experimental/preview)
- Gemini 1.5 Pro (may have stricter limits)
- High volume batch processing

### Rate Limits

| Metric | Free Tier |
|--------|-----------|
| RPM | 15 |
| TPM | 1 million |
| RPD | 1,500 |
| Image Size | Up to 3072x3072 |

**Note:** Exceeding these limits will cause the API to return errors.

## Recommended Setup for Free Tier

### 1. Use Gemini 1.5 Flash

```python
# In GUI, select:
Provider: Google Gemini
Model: gemini-1.5-flash  # NOT 2.5-pro
```

**Why?**
- ✅ Available on free tier
- ✅ Fast (2-5 seconds per page)
- ✅ Good accuracy for handwriting
- ✅ No billing required

### 2. Check Your Available Models

Run the test script to see what you can access:
```bash
python test_gemini_models.py YOUR_API_KEY
```

This will show:
- Which models are available with your API key
- Which are vision-capable
- Your actual access level

### 3. Validate API Key Before Use

In the GUI:
1. Enter API key
2. Click **"Validate & Check Models"**
3. Dropdown will show only YOUR available models
4. If empty or shows errors → check API console

## Upgrading from Free Tier

If you need more:

### Option 1: Pay-As-You-Go (Google Cloud)
1. Go to console.cloud.google.com
2. Enable billing
3. Get access to:
   - Gemini 1.5 Pro (better accuracy)
   - Higher rate limits
   - Gemini 2.0 experimental models

**Pricing (approx):**
- Flash: $0.10-0.30 per 1K images
- Pro: $3-7 per 1K images

### Option 2: Use Alternative Providers

**For Free/Low Cost:**
- Use local TrOCR (already integrated, 100% free)
- Use local PyLaia (fast, free, good for bulk)

**For Commercial Projects:**
- OpenAI GPT-4o (best accuracy, ~$2.50-5 per 1K images)
- Claude 3.5 Sonnet (best for corrections, ~$3-8 per 1K images)

## Troubleshooting Free Tier Issues

### Error: "Rate limit exceeded"
**Solution:**
- Wait 1 minute between requests
- Use free tier: max 15 requests/minute
- Consider upgrading for higher limits

### Error: "Model not found" or "Access denied"
**Solution:**
- Use `gemini-1.5-flash` instead of 2.5/2.0
- Click "Validate & Check Models" to see available models
- Check API is enabled: console.cloud.google.com

### Error: "Content blocked by safety filters"
**Solution:**
- Gemini has strict content policies
- Try adjusting prompt to be more neutral
- If manuscript has sensitive content, use local models

### No Response or Empty Response
**Solution:**
- Check image size (should be < 5MB)
- Try lower resolution image
- Ensure image is a valid format (PNG, JPG)
- Check API quota on console.cloud.google.com

## Best Practices for Free Tier

### 1. Batch Processing Strategy

Don't process too many pages at once:
```
✅ Good: 10-15 pages, then wait 1 minute
❌ Bad: 100 pages in a loop (will hit rate limit)
```

### 2. Use Local Models for Bulk

For large projects:
```
1. Use Gemini Flash for 10-20 difficult pages
2. Use local TrOCR/PyLaia for remaining pages
3. Saves API credits, avoids rate limits
```

### 3. Optimize Image Size

Resize before sending:
```python
# Images are auto-resized to 3072x3072 max
# But smaller = faster + cheaper

# Recommended for manuscripts:
Target: 1500-2000px on longest side
Format: PNG or JPG
```

### 4. Monitor Usage

Check your usage:
1. Go to console.cloud.google.com
2. Navigate to "APIs & Services" → "Dashboard"
3. View Gemini API usage statistics

## Comparison: Free vs Paid Gemini

| Feature | Free Tier | Paid Tier |
|---------|-----------|-----------|
| **Models** | 1.5 Flash | All (1.5 Pro, 2.0, 2.5) |
| **RPM** | 15 | 1,000+ |
| **RPD** | 1,500 | Unlimited* |
| **Cost** | $0 | ~$0.10-7 per 1K images |
| **Accuracy** | Good | Better (Pro models) |
| **Speed** | Fast | Same |

*Subject to billing limits

## Testing Your Setup

### Quick Test
```bash
# Test with single image
python inference_commercial_api.py gemini YOUR_KEY test_image.jpg
```

### GUI Test
```
1. Launch: python transcription_gui_qt.py
2. Switch to "Commercial APIs"
3. Select "Google Gemini"
4. Enter API key
5. Click "Validate & Check Models"
6. Should see: "✓ Valid! Found X models"
7. Select: gemini-1.5-flash
8. Load image
9. Click "Transcribe"
```

### Expected Result
- **Success**: Text appears in editor (5-15 seconds)
- **Fail**: Error dialog with clear message

## When to Upgrade

Consider upgrading if you:
- ✅ Process > 50 pages/day regularly
- ✅ Need better accuracy (Pro models)
- ✅ Hit rate limits frequently
- ✅ Need access to experimental models (2.0, 2.5)
- ✅ Running commercial project

Otherwise, **free tier + local models** is a great combination!

## Summary

**For Free Tier Users:**
1. Use `gemini-1.5-flash` (not 2.5-pro)
2. Validate API key to see available models
3. Stay under 15 requests/minute
4. Combine with local models for bulk processing
5. Upgrade only if needed

**The error you saw** was likely because:
- Model wasn't available on free tier (2.5-pro)
- Response was blocked/incomplete
- Code didn't handle the error properly

**Now fixed** with:
- Better error handling
- Correct model list for free tier
- Clear error messages
