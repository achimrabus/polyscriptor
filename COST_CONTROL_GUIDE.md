# Cost Control & Performance Guide

## Problem Identified

Your transcription showed:
```
[tokens] prompt=1147 candidates=0 total=7290
⏱️ Early reasoning fallback triggered: internal=6143 (100% of budget)
Fallback max_output_tokens=12288
✅ Fallback succeeded (527 chars)
Time: 290 seconds (~5 minutes)
```

**Issues**:
1. ❌ **Extremely expensive**: 12,288 token fallback for just 527 characters
2. ❌ **Very slow**: 290 seconds for one page is unsustainable
3. ❌ **gemini-3-pro-preview** burning all tokens on internal reasoning

## Changes Made

### 1. **Capped Fallback at 8192 Tokens**
**Before**: `fallback_tokens = max(8192, max_output_tokens * 2)`
- With 6144 initial → fallback to 12,288 tokens

**After**: `fallback_tokens = 8192` (fixed cap)
- **Saves ~33% tokens** on fallback attempts
- Console shows: `Fallback max_output_tokens=8192 (capped for cost control)`

### 2. **Added Debug Logging**
Now shows:
```
🔧 LOW thinking mode: overriding max_output_tokens to 6144
📊 Final settings: thinking_mode=low, max_output_tokens=6144, temp=1.0
   Using max_output_tokens=6144 (from config)
```
This confirms your LOW-mode token setting is being applied.

### 3. **Restriction Prompt Injection (Replacing Prior Banner)**
Automatic injection for preview models:
```
INSTRUCTION: Provide ONLY the direct diplomatic transcription ... (see code)
```
This replaces the prior GUI warning banner and focuses on reducing hidden reasoning token burn without forcing model switches that are unsuitable for Church Slavonic.

## Recommended Solutions

### 🎯 **Primary Strategy: Preview Model + Restriction Prompt**
Church Slavonic manuscripts require `gemini-3-pro-preview` for acceptable accuracy. Instead of switching models, we now:
1. Inject a restriction instruction to reduce internal reasoning token consumption.
2. Use LOW thinking + fast-direct for early emission.
3. Trigger early fallback if internal reasoning reaches threshold with no output.

### � **Alternate Strategy: High Reasoning Pass (If Low Underproduces)**
If LOW mode still burns tokens without output, switch to HIGH thinking with an 8192 cap and keep restriction prompt. This can yield better completeness at the cost of time.

---

## Cost Comparison

| Model | Time/Page | Tokens/Page | Notes |
|-------|-----------|-------------|-------|
| **gemini-3-pro-preview (LOW + restriction)** | 40-120s | ~4,000–8,000 | Balanced; early fallback + restriction reduce waste |
| **gemini-3-pro-preview (HIGH)** | 90-180s | ~6,000–8,192 | Use if LOW fails to emit; higher completeness |
| *(Other models)* | — | — | Not used (insufficient Church Slavonic fidelity) |

*Approximate, varies by content & API pricing

---

## Debugging Your Current Setup

### Check if LOW-mode override is working:

Look for these lines in console:
```
🔧 LOW thinking mode: overriding max_output_tokens to 6144
📊 Final settings: thinking_mode=low, max_output_tokens=6144, temp=1.0
   Using max_output_tokens=6144 (from config)
```

**If you see**:
```
   Increasing max_output_tokens from 2048 to 4096 for preview model
```
→ Your GUI field is empty or invalid. Check Advanced panel "Low-mode tokens" = `6144`

### Why preview model is slow:

Preview models have internal "reasoning" that:
1. Consumes tokens invisibly (`total - prompt - candidates` = internal)
2. Adds latency (model is "thinking" but not outputting)
3. Doesn't guarantee better output for simple text

Your log: `internal=6143` out of 6143 budget = **100% wasted**

---

## Action Plan

### Immediate (Next Transcription)
1. Ensure restriction prompt injection message appears in console.
2. Use LOW thinking + fast-direct early exit.
3. If MAX_TOKENS hit with no parts → fallback auto-escalates to 8192.
4. If still empty, rerun with HIGH thinking (restriction stays).

### If Output Truncated
- Disable early exit; enable auto continuation (2 passes)
- Raise low-mode tokens (e.g., 7168) within 8192 cap

---

## Technical Notes

### Why Capping Fallback Helps
- Preview model fallback was `max(8192, 6144*2)` = 12,288
- Token budget scales quadratically with reasoning depth
- Capping at 8192 forces model to be concise
- If 8192 fails → likely need different model, not more tokens

### Early Fallback Trigger
Your log shows trigger worked:
```
⏱️ Early reasoning fallback triggered: internal=6143 (100% of budget)
```
This is GOOD - system detected waste early and aborted stream.
Without it, you'd wait even longer before getting fallback result.

### Future Enhancement
Could add **model auto-switching**:
- Try flash first (15s timeout)
- On failure/poor quality → escalate to pro
- On repeated failure → preview as last resort

---

## Summary

✅ **Fallback capped** at 8192 (was 12,288)
✅ **Debug logging** added for transparency
✅ **Restriction prompt** active for preview models
✅ **Removed banner recommending alternative models (not suitable for Church Slavonic)**

**Bottom line**: For Church Slavonic, preview model + restriction prompt + early fallback is the current best-performing path; alternative models underperform in fidelity.
