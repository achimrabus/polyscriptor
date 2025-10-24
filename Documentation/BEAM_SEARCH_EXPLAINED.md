# Beam Search - Concise Explanation

## What is Beam Search?

Beam search is a **decoding strategy** for generating text sequences from language models. It explores multiple possible outputs simultaneously to find better results than greedy decoding.

---

## How It Works

### Greedy Decoding (`num_beams=1`)
- At each step, pick the **single most likely token**
- Fast but can miss better sequences
- **Example**: "The cat" → picks highest probability next word only

### Beam Search (`num_beams > 1`)
- Keeps **top N candidate sequences** at each step (N = beam width)
- Explores N parallel hypotheses
- Selects final sequence with best overall probability
- **Example with 3 beams**: "The cat" → explores 3 best continuations simultaneously

---

## Beam Width Impact

| Beam Width | Speed | Quality | Use Case |
|------------|-------|---------|----------|
| **1** (greedy) | ⚡⚡⚡ Fastest | ⭐⭐ Good | Default for OCR, real-time |
| **3-4** | ⚡⚡ Moderate | ⭐⭐⭐ Better | Improved accuracy |
| **5-10** | ⚡ Slow | ⭐⭐⭐⭐ Best | Critical documents |

**Speed Formula**: Processing time ≈ `num_beams × base_time`
- Beam width 4 = **4× slower** than greedy
- Beam width 10 = **10× slower**

---

## When Beam Search Helps

✅ **Beneficial for:**
- Ambiguous text (degraded, unclear)
- Long sequences where early mistakes compound
- When small quality improvements matter

❌ **Not worth it when:**
- Text is clear and high quality
- Speed is critical
- Model is already very accurate

---

## Practical Recommendations

### For TrOCR (Line-based OCR):
- **Default**: `num_beams=4` (current setting) - good balance
- **Fast mode**: `num_beams=1` - 4× faster
- **Quality mode**: `num_beams=5-8` - marginally better

### For Qwen3 VLM (Page-based):
- **Default**: `num_beams=1` (greedy) - already very good
- **Only increase** if you see obvious quality issues
- **Max recommended**: `num_beams=3` (3× slower, minimal gain)

---

## Technical Details

**Why it's expensive:**
Each beam maintains:
- Full sequence history
- Probability scores
- Hidden states (model memory)

**Memory usage**: `num_beams × sequence_length × hidden_size`

**Why diminishing returns:**
- Beam 1→3: Noticeable improvement
- Beam 3→5: Small improvement
- Beam 5→10: Minimal improvement

---

## Example: "The quick brown fox"

### Greedy (beam=1):
```
Step 1: "The" → pick "quick" (0.8 prob)
Step 2: "The quick" → pick "brown" (0.7 prob)
Total: 0.8 × 0.7 = 0.56
```

### Beam Search (beam=3):
```
Step 1: Keep top 3: "quick" (0.8), "fast" (0.6), "slow" (0.4)
Step 2: Expand each, keep top 3 overall sequences
Final: "The quick brown" (0.56), "The fast red" (0.48), ...
```

Result: Explores more paths, finds better overall sequence.

---

## Bottom Line

**For your Ukrainian manuscript OCR:**

- **TrOCR**: Keep `num_beams=4` - it's fine
- **Qwen3**: Use `num_beams=1` - greedy is fast and good enough
- **Only increase beams** if you see repeated transcription errors on clear text

**Speed vs Quality Trade-off:**
- Most OCR quality comes from the **model itself**, not beam search
- Beam search helps ~5-10% at most
- Better to use a finetuned model than high beam width
