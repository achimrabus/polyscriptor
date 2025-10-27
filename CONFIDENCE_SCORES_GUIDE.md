# Confidence Scores Guide

## Overview

Confidence scores indicate how certain the model is about its transcription. They range from 0% (uncertain) to 100% (very confident).

## Model-Specific Confidence Support

### ✅ TrOCR (Reliable Confidence)

**How it works:**
- Uses beam search probabilities
- Character-level confidence from softmax scores
- Line-level average confidence

**When available:**
- Automatically calculated for each transcription
- No special configuration needed
- Always reliable

**Display:**
```
Avg Confidence: 87.3%
```

### ✅ PyLaia (Reliable Confidence)

**How it works:**
- Uses CTC (Connectionist Temporal Classification) probabilities
- Character-level confidence from output probabilities
- Line-level average confidence

**When available:**
- Enable checkbox: "Calculate Confidence Scores" (checked by default)
- Available for all transcriptions
- Very reliable

**Display:**
```
Avg Confidence: 92.1%
```

**Note:** With language model post-correction, confidence is approximated from the beam search scores.

### ⚠️ Qwen3 VLM (Experimental Confidence)

**How it works:**
- Estimates confidence from token probabilities
- Requires "Estimate Confidence" checkbox
- Slower (needs `output_scores=True`)

**When available:**
- Enable checkbox: "Estimate Confidence (slower)"
- Only when explicitly enabled
- Less reliable than TrOCR/PyLaia

**Display:**
```
Avg Confidence: 85.7%  (if enabled and successful)
Avg Confidence: N/A    (if disabled or failed)
```

**Why experimental:**
- VLMs don't naturally provide per-character confidence
- Approximated from token probabilities
- May not reflect actual accuracy

### ❌ Commercial APIs (No Confidence)

**Why not available:**
- OpenAI, Gemini, Claude don't expose confidence scores
- APIs return only final text, no probabilities
- Cannot be calculated client-side

**Display:**
```
Avg Confidence: N/A
```

## Confidence Display Logic

The GUI intelligently shows confidence based on availability:

```python
# Commercial APIs - always N/A
if provider in ["openai", "gemini", "claude"]:
    confidence_display = "N/A"

# Qwen3 - check if confidence was calculated
elif model == "qwen3":
    if result.confidence is not None:
        confidence_display = f"{result.confidence*100:.1f}%"
    else:
        confidence_display = "N/A"

# TrOCR/PyLaia - check if confidences exist
elif model in ["trocr", "pylaia"]:
    if confidences:
        avg = sum(confidences) / len(confidences)
        confidence_display = f"{avg*100:.1f}%"
    else:
        confidence_display = "N/A"
```

## Interpreting Confidence Scores

### High Confidence (> 90%)
- Model is very certain
- Transcription likely accurate
- Can trust without verification

### Medium Confidence (70-90%)
- Model is reasonably certain
- May have some uncertain characters
- Review transcription for quality

### Low Confidence (< 70%)
- Model is uncertain
- Likely transcription errors
- **Manually verify transcription**

### N/A
- Confidence not available
- Does not mean low quality
- Simply means model doesn't provide scores

## Model Comparison

| Model | Confidence | Reliability | Speed Impact |
|-------|-----------|-------------|--------------|
| **TrOCR** | ✅ Always | High | None |
| **PyLaia** | ✅ Always* | Very High | Minimal |
| **Qwen3** | ⚠️ Optional | Medium | +20-30% slower |
| **OpenAI** | ❌ Never | N/A | N/A |
| **Gemini** | ❌ Never | N/A | N/A |
| **Claude** | ❌ Never | N/A | N/A |

*PyLaia: Enable "Calculate Confidence Scores" checkbox (default: on)

## Best Practices

### 1. Use Confidence for Quality Control

**Workflow:**
```
1. Transcribe documents
2. Check average confidence
3. If < 80%: Manually review low-confidence lines
4. If > 90%: High quality, minimal review needed
```

### 2. Model Selection Based on Confidence Needs

**Need reliable confidence?**
- ✅ Use TrOCR or PyLaia
- ❌ Don't use Commercial APIs

**Don't need confidence?**
- ✅ Use any model (choose by accuracy/speed)

### 3. Batch Processing with Confidence Filtering

**Strategy:**
```python
# After transcription
high_confidence = [line for line in results if line.confidence > 0.9]
low_confidence = [line for line in results if line.confidence < 0.8]

# Auto-accept high confidence
# Manually review low confidence
```

### 4. Confidence vs. Accuracy

**Important:** Confidence ≠ Accuracy

- High confidence can still have errors (model is "confidently wrong")
- Low confidence can be correct (model unsure but right)
- Use confidence as guidance, not absolute truth

## FAQ

### Q: Why don't commercial APIs provide confidence?
**A:** API providers don't expose internal probabilities for:
- Security reasons
- Simplicity (users just want text)
- API design philosophy

### Q: Can I calculate confidence for Gemini/OpenAI manually?
**A:** No, the APIs don't return the necessary probability data.

### Q: Is Qwen3 confidence accurate?
**A:** It's an approximation. For critical projects, use TrOCR/PyLaia confidence instead.

### Q: Should I always enable Qwen3 confidence?
**A:** Only if you need it:
- ✅ Enable for quality control workflows
- ❌ Disable for speed (20-30% faster without)

### Q: What if PyLaia shows "N/A" for confidence?
**A:** Check the "Calculate Confidence Scores" checkbox is enabled in the PyLaia tab.

### Q: Can I trust 100% confidence?
**A:** Even 100% confidence can have errors. Always do spot checks on a sample of transcriptions.

## Configuration Examples

### Maximum Accuracy + Confidence
```
Model: TrOCR
Settings:
- Beam Search: 4-8
- Max Length: 128
- Return Confidence: Auto (always on)

Result: Reliable confidence + good accuracy
```

### Maximum Speed + Confidence
```
Model: PyLaia
Settings:
- Calculate Confidence: ✓ (checked)
- Language Model: ✗ (unchecked for speed)

Result: Very fast + reliable confidence
```

### Maximum Accuracy (No Confidence Needed)
```
Model: Claude 3.5 Sonnet (Commercial API)
Settings:
- Temperature: 0.0 (deterministic)

Result: Best accuracy, no confidence scores
```

## Summary

**Confidence is available and reliable for:**
- ✅ TrOCR (always, no performance impact)
- ✅ PyLaia (always, minimal impact, checkbox enabled by default)

**Confidence is experimental for:**
- ⚠️ Qwen3 VLM (optional, slower, less reliable)

**Confidence is NOT available for:**
- ❌ OpenAI GPT-4o
- ❌ Google Gemini
- ❌ Anthropic Claude

**The GUI correctly shows "N/A"** when confidence cannot be calculated, ensuring you always know whether the displayed confidence is reliable.
