# Gemini 3 Adjustments & Enhancements

**Branch**: `gemini-3-adjustments`  
**Commit**: 4f62e7f  
**Date**: November 20, 2025

## Overview

This branch implements advanced controls and optimizations for `gemini-3-pro-preview` and other Gemini reasoning models to address internal token consumption ("thinking tokens") that can exhaust the output budget without producing visible transcription.

## Problem Context

Previous testing showed:
- LOW thinking mode with streaming could produce **zero candidate tokens** (candidates=0) while consuming 2000+ internal reasoning tokens.
- First attempt exhausted 4096 output budget on internal planning without emitting text.
- Automatic fallback to HIGH mode + 8192 tokens successfully produced 331 characters.
- Continuation pass found no additional text to append.

**Root cause**: Preview models perform hidden multi-step reasoning; those internal tokens count against `max_output_tokens` but are never surfaced to the user.

## Implemented Enhancements

### 1. Reasoning Token Detection & Logging

**File**: `inference_commercial_api.py`

- Computes **internal reasoning tokens** as:  
  ```
  internal = total_tokens - prompt_tokens - candidates_tokens
  ```
- Logs percentage of budget consumed internally during streaming.
- Displays: `⏱️ Early reasoning fallback triggered: internal=2047 (60% of budget) with no output`

**CSV Stats Logging**:
- Appends to `gemini_runs.csv` on every transcription attempt:
  ```
  timestamp,model,thinking_mode,outcome,prompt_tok,cand_tok,total_tok,internal_tok,emitted_chars
  ```
- Tracks: `stream_early_exit`, `stream_full`, `fallback_success`, `final_success`
- Enables longitudinal analysis of token efficiency across models and settings.

### 2. Early Reasoning Fallback Trigger

**File**: `inference_commercial_api.py`

- During streaming: if internal tokens ≥ **`reasoning_fallback_threshold`** × `max_output_tokens` **and zero emitted text**, abort stream and escalate to standard generation (which may retry with expanded budget).
- Default threshold: **0.6** (60% of budget).
- Prevents wasting the full token allocation on invisible reasoning.

**Behavior**:
- Stream loop checks after each event with `usage_metadata`.
- Sets `reasoning_fallback_triggered = True` and breaks loop.
- Fallback message: `⚠️ Streaming aborted due to excessive internal reasoning; switching to standard generation.`

### 3. GUI Controls for Fine-Tuning

**File**: `engines/commercial_api_engine.py`

#### Advanced Gemini Panel Additions:

| Control | Parameter | Default | Description |
|---------|-----------|---------|-------------|
| **Min new chars** | `continuation_min_new_chars` | 50 | Minimum characters required to accept a continuation chunk. Guards against duplicate/stutter responses. |
| **Low-mode tokens** | `low_thinking_initial_max_tokens` | 6144 | Initial `max_output_tokens` for LOW thinking before fallback escalation. Higher than previous 4096 to give more headroom while still faster than full 8192. |
| **Fallback %** | `reasoning_fallback_threshold` | 0.6 | Fraction (0.0–1.0) of token budget consumed internally (no output) that triggers early fallback. Lower = more aggressive fallback. |

#### Existing Controls Enhanced:
- **Early exit on first chunk**: Checkbox (default: checked) to toggle `fast_direct_early_exit`.
- **Auto continuation passes**: Checkbox + max passes field (e.g., 2).
- **Temperature**: Text input (default: 1.0).
- **Max output tokens**: Text input (default: 4096 for preview / 2048 otherwise).

### 4. Continuation Prompt Adjustments

**File**: `inference_commercial_api.py` → `_maybe_continue` helper

- Continuation prompt now includes:
  ```
  Partial transcription so far (DO NOT repeat it):
  {accumulated}
  
  Continue transcribing remaining, previously UNTRANSCRIBED text.
  Output ONLY the new continuation without repeating prior characters.
  ```
- Detects accidental repetition by checking for overlap with last 50 chars of prior output.
- Only appends chunks meeting `continuation_min_new_chars` threshold.

### 5. Low-Mode Initial Token Boost

**File**: `engines/commercial_api_engine.py`

- When user selects LOW thinking mode **and** provides a value in **"Low-mode tokens"** field:
  - Overrides initial `max_output_tokens` with the specified value (e.g., 6144).
- Gives LOW mode more room to emit text before internal reasoning exhausts budget.
- Fallback (HIGH mode) still escalates to 8192 if initial attempt produces nothing.

### 6. Stats CSV Output

**File**: Writes to `gemini_runs.csv` in workspace root.

**CSV Schema**:
```
timestamp,model,thinking_mode,outcome,prompt_tok,cand_tok,total_tok,internal_tok,emitted_chars
```

**Example Rows**:
```csv
2025-11-20T14:32:10.123456,gemini-3-pro-preview,low,stream_early_exit,1137,18,1180,25,331
2025-11-20T14:32:45.654321,gemini-3-pro-preview,high,fallback_success,,,,,331
```

**Use Cases**:
- Identify models/settings with high internal token waste.
- Compare LOW vs HIGH reasoning efficiency.
- Tune `reasoning_fallback_threshold` based on historical patterns.
- Track correlation between token budgets and transcription completeness.

## Configuration Recommendations

### Profile: Preview Quick (Recommended Default)
```
Model: gemini-3-pro-preview
Thinking mode: Low (Fast)
Temperature: 1.0
Max output tokens: (leave empty; defaults to 4096)
Low-mode tokens: 6144
Early exit: ✓ Checked
Auto continuation: ✓ Checked
Max passes: 2
Min new chars: 50
Fallback %: 0.6
```
**Best for**: Fast transcription of Church Slavonic manuscripts with automatic continuation safety net.

### Profile: Preview Thorough
```
Model: gemini-3-pro-preview
Thinking mode: High (More reasoning)
Temperature: 1.0
Max output tokens: 8192
Low-mode tokens: (n/a)
Early exit: ✗ Unchecked
Auto continuation: ✓ Checked
Max passes: 3
Min new chars: 50
Fallback %: 0.7
```
**Best for**: Complex manuscripts where higher reasoning may improve accuracy; willing to accept longer latency.

### Profile: Stable Low-Latency
```
Model: gemini-2.0-flash
Thinking mode: Auto (Low for preview)
Temperature: 1.0
Max output tokens: 2048
Early exit: ✓ Checked
Auto continuation: ✗ Unchecked
```
**Best for**: High-throughput batch processing on stable models with lower internal reasoning overhead.

## Testing & Validation

**Validation Steps**:
1. Run transcription with `gemini-3-pro-preview` using **Preview Quick** profile.
2. Observe console logs:
   - `⚡ Fast-direct mode enabled: prompting for immediate output`
   - `🧠 Using LOW thinking mode (direct decoding)`
   - `[tokens] prompt=1137 candidates=18 total=1180`
   - `✅ Early streamed output (331 chars) [early-exit]`
   - `ℹ️ Continuation attempt 1 produced no new text; stopping.`
3. Check `gemini_runs.csv` for entry with `outcome=stream_early_exit`.
4. If internal tokens exceed threshold during stream, verify:
   - `⏱️ Early reasoning fallback triggered: internal=2047 (60% of budget)...`
   - Fallback attempt logged with `outcome=fallback_success`.

**Expected Behavior**:
- LOW mode should emit text early (within first 1-3 events).
- If zero candidates after 60% internal tokens, fallback triggers automatically.
- Continuation passes guard against incomplete output but stop when no new text appears.

## Files Modified

1. **`inference_commercial_api.py`**:
   - Added parameters: `reasoning_fallback_threshold`, `record_stats_csv`.
   - Instrumented streaming loop with reasoning detection and early abort.
   - Added CSV logging at success points (early exit, full stream, fallback, final).
   - Enhanced `_maybe_continue` prompt clarity.

2. **`engines/commercial_api_engine.py`**:
   - Added GUI controls: `_min_new_chars_edit`, `_low_initial_tokens_edit`, `_reasoning_fallback_edit`.
   - Extracted user inputs and passed to `transcribe()`.
   - Added low-mode token override logic.
   - Passed `record_stats_csv` parameter.

3. **`.gitignore`**:
   - Already includes `htr_gui/` (venv ignored).
   - CSV logs (`gemini_runs.csv`) will be untracked by default.

## Future Enhancements (Not in This Branch)

- **Dynamic Chunking**: Pre-slice image into logical line crops if persistent reasoning burn remains high.
- **Adaptive Threshold**: Auto-tune `reasoning_fallback_threshold` based on rolling average from CSV.
- **GUI Histogram**: Real-time visualization of reasoning token % over last N runs.
- **Continuation Heuristics**: Track last few characters; if ends mid-word or with ellipsis, force one continuation regardless of length growth.
- **Prompt Library**: Save/load user-defined prompt templates optimized for different manuscript types.

## Rollback Instructions

If issues arise, revert to previous branch:
```bash
git checkout batch-processing-improvements
```

To cherry-pick individual commits from this branch:
```bash
git cherry-pick 4f62e7f
```

## Summary

This branch delivers:
- ✅ Real-time reasoning token detection with early abort.
- ✅ Fine-grained GUI controls for continuation tuning and token budgets.
- ✅ Persistent CSV logging for token usage analysis.
- ✅ Enhanced continuation prompts to reduce duplication.
- ✅ LOW-mode initial token boost for better first-pass success.

All changes are backward-compatible; existing workflows without advanced settings continue to function with defaults.
