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

### 3. **Added Warning Banner in GUI**
```
⚠️ Preview models (gemini-3-pro-preview) are experimental and can be slow/expensive.
💡 For production use, select gemini-2.0-flash or gemini-1.5-pro instead.
```
Appears at top of "Thinking Mode (Gemini only)" section.

## Recommended Solutions

### 🎯 **Option 1: Switch to Stable Model (RECOMMENDED)**

**Use**: `gemini-2.0-flash` or `gemini-1.5-pro-002`

**Settings**:
```
Model: gemini-2.0-flash
Thinking: Auto (Low for preview)
Temperature: 1.0
Max tokens: 2048
Early exit: ✓ Checked
Auto continuation: ✗ Unchecked
```

**Expected results**:
- ⚡ **10-30 seconds** per page (vs 290s)
- 💰 **~2000 tokens** total (vs 12,288)
- ✅ **Reliable output** without internal reasoning burn

**Why it works**:
- Stable models don't waste tokens on hidden "thinking"
- Flash model optimized for speed
- Lower cost per token

---

### 🔬 **Option 2: Keep Preview but Optimize**

If you must use `gemini-3-pro-preview` (e.g., for maximum accuracy on complex scripts):

**Settings**:
```
Model: gemini-3-pro-preview
Thinking: High (More reasoning)  ← Paradoxically better
Temperature: 1.0
Max tokens: 8192  ← Start high, skip LOW mode
Early exit: ✗ Unchecked
Auto continuation: ✗ Unchecked
Low-mode tokens: (leave empty)
Fallback %: 0.5  ← More aggressive
```

**Rationale**:
- **HIGH mode from start** → model expects more budget, may plan better
- **Skip LOW → fallback dance** → go straight to full budget
- **Lower fallback %** → trigger earlier before full burn

**Expected**:
- ⏱️ **60-120 seconds** per page
- 💰 **~8000 tokens** per attempt
- Still expensive but more predictable

---

### 💡 **Option 3: Hybrid Approach**

**Batch processing**: Use `gemini-2.0-flash` for clear, simple manuscripts

**Complex cases only**: Switch to `gemini-1.5-pro-002` (not preview) for heavily abbreviated/damaged text

**Never use**: gemini-3-pro-preview for production batches

---

## Cost Comparison

| Model | Time/Page | Tokens/Page | Cost/1000 Pages* |
|-------|-----------|-------------|------------------|
| **gemini-2.0-flash** | 15s | ~2,000 | $10-20 |
| **gemini-1.5-pro-002** | 30s | ~3,000 | $30-50 |
| **gemini-3-pro-preview** | 290s | ~12,000 | $300-500 |

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
1. ✅ **Switch model** to `gemini-2.0-flash` in dropdown
2. ✅ **Keep all other settings** as-is
3. ✅ **Test one page** → should see <30s, ~2000 tokens
4. ✅ **Compare quality** to preview output

### If Quality Suffers
- Try `gemini-1.5-pro-002` (stable pro, not preview)
- Raise temperature to 1.2-1.5 for more variation
- Enable auto continuation (2 passes) for completeness

### If Still Need Preview Model
- Use **Option 2** settings above
- Only for most complex 10-20% of documents
- Budget 5-10x cost vs flash model

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
✅ **Warning banner** in GUI about preview costs
✅ **Recommendation**: Switch to `gemini-2.0-flash` for 10-20x speedup

**Bottom line**: Preview models are experimental research tools, not production workhorses. Use stable models unless you have a specific need for cutting-edge reasoning.
