# Confidence vs Accuracy - Understanding HTR Metrics

## Question

**User**: "How do you calculate error rate if you completely ignore the GT? There is an accuracy information in the verbose GUI window. Where does it come from?"

## Answer: It's Confidence, Not Accuracy

The metric shown in batch processing is **confidence** (model prediction probability), **NOT** accuracy/error rate calculated against ground truth.

---

## What is Confidence?

**Confidence** = Model's self-assessment of prediction quality
- Range: 0.0 to 1.0 (0% to 100%)
- Source: Model's softmax output probabilities
- **Independent of ground truth** - no GT data needed!

---

## How Confidence is Calculated

### PyLaia Example (CTC-based model)

**File**: `inference_pylaia_native.py:270-302`

```python
def decode(self, log_probs):
    """CTC greedy decoding with confidence."""
    # Convert log probabilities to probabilities
    probs = torch.exp(log_probs)
    _, pred_indices = torch.max(probs, dim=2)  # Most likely class at each timestep

    decoded_chars = []
    confidences = []

    for t, idx in enumerate(pred_indices):
        if idx != 0 and idx != prev_idx:  # Not blank, not duplicate
            char = self.idx2char.get(idx, '')
            if char:
                decoded_chars.append(char)
                # Get probability for this character
                char_conf = probs[t, 0, idx].item()  # ← MODEL PROBABILITY
                confidences.append(char_conf)
        prev_idx = idx

    # Average confidence across all characters
    confidence = sum(confidences) / len(confidences) if confidences else 0.0

    return ''.join(decoded_chars), confidence
```

**Key Points**:
- `probs[t, 0, idx]` = probability model assigns to predicted character at timestep `t`
- High probability (e.g., 0.95) → model is confident
- Low probability (e.g., 0.60) → model is uncertain
- Average across all characters in the line

---

## Confidence vs Accuracy Comparison

| Metric | What It Measures | Requires GT? | Source |
|--------|------------------|--------------|---------|
| **Confidence** | Model's prediction probability | ❌ No | Softmax output |
| **Accuracy** | Match with ground truth | ✅ Yes | Compare prediction vs GT |
| **CER** (Character Error Rate) | Edit distance from GT | ✅ Yes | Levenshtein distance |
| **WER** (Word Error Rate) | Word-level errors vs GT | ✅ Yes | Word-level comparison |

---

## Example: PyLaia Prediction

**Input**: Line image of "Слово"

**Model Output**:
```
Predicted: "Слово"
Probabilities for each character:
  С: 0.98 (98% confident)
  л: 0.95 (95% confident)
  о: 0.89 (89% confident)
  в: 0.92 (92% confident)
  о: 0.88 (88% confident)

Average confidence: (0.98 + 0.95 + 0.89 + 0.92 + 0.88) / 5 = 0.924 (92.4%)
```

**Result**: `TranscriptionResult(text="Слово", confidence=0.924)`

**No ground truth needed!** Confidence comes entirely from the model's internal probability distribution.

---

## Why High Confidence ≠ High Accuracy

**Scenario**: Model is poorly trained or domain mismatch

```
Ground Truth: "Слово"
Prediction:   "Слава"
Confidence:   0.95 (95%)
```

The model can be **very confident** (high softmax probabilities) but still **wrong** (doesn't match GT).

**Why?**
- Confidence = "How sure the model is"
- Accuracy = "How correct the model is"
- A poorly trained model can be confidently wrong!

---

## Confidence in Different Engines

### 1. PyLaia (CTC)
```python
# Average of character-level softmax probabilities
confidence = mean([probs[t, 0, predicted_idx] for t in timesteps])
```

### 2. TrOCR (Transformer)
```python
# Average of token-level softmax probabilities
# From beam search or greedy decoding
confidence = mean([softmax_prob for each token])
```

### 3. Qwen3-VL / Party
```python
# Many VLMs don't provide confidence
confidence = 1.0  # Default (no confidence estimation)
```

### 4. Kraken
```python
# CTC-based, similar to PyLaia
confidence = mean([char probabilities])
```

---

## How Batch Processing Uses Confidence

**File**: `batch_processing.py`

**1. Per-Line Confidence** (from engine):
```python
# Engine returns confidence with each transcription
result = engine.transcribe_line(line_image)
# result.confidence = 0.89 (from model softmax)
```

**2. Average Confidence** (across all lines):
```python
def _calculate_avg_confidence(transcriptions):
    confidences = [t.confidence for t in transcriptions
                  if t.confidence is not None]
    return sum(confidences) / len(confidences) if confidences else None
```

**3. Displayed in Summary**:
```
BATCH PROCESSING SUMMARY
============================================================
Total images processed: 42
Total lines transcribed: 168
Total characters: 5432
Average confidence: 89.62%  ← This is MODEL CONFIDENCE, not accuracy!
```

---

## Confidence in CSV/JSON Output

**CSV Output** (`transcriptions/summary.csv`):
```
image,line_count,char_count,avg_confidence,timestamp
page001.jpg,4,156,0.8962,2025-11-12T...
```

**JSON Output** (`batch_results.json`):
```json
{
  "metadata": {
    "total_images": 42,
    "total_errors": 0
  },
  "results": [
    {
      "image": "page001.jpg",
      "line_count": 4,
      "char_count": 156,
      "avg_confidence": 0.8962,  ← Model confidence
      "transcriptions": [
        {
          "text": "Слово божїе",
          "confidence": 0.89  ← Per-line model confidence
        }
      ]
    }
  ]
}
```

---

## When Would We Calculate Accuracy?

**To calculate CER/WER, we would need**:

1. **Ground Truth Data** (manually transcribed)
2. **Prediction Data** (from model)
3. **Comparison Algorithm** (Levenshtein distance)

**Example Code** (NOT CURRENTLY IMPLEMENTED):
```python
def calculate_cer(ground_truth: str, prediction: str) -> float:
    """Calculate Character Error Rate using edit distance."""
    import editdistance
    distance = editdistance.eval(ground_truth, prediction)
    cer = distance / len(ground_truth)
    return cer

# Example
gt = "Слово божїе"
pred = "Слово богїе"  # 1 character wrong
cer = calculate_cer(gt, pred)  # 1/11 = 0.09 (9% error rate)
```

**Why we don't do this in batch processing**:
- Input PAGE XML may contain GT, but we **intentionally ignore it**
- Batch processing is for **inference** (getting predictions), not **evaluation**
- Evaluation should be a separate step (using dedicated tools)

---

## Summary

| Question | Answer |
|----------|--------|
| What is "accuracy" in verbose output? | **It's confidence**, not accuracy! |
| Where does confidence come from? | Model's softmax output probabilities |
| Does it use ground truth? | ❌ No - completely independent of GT |
| Can we trust high confidence? | ⚠️  Not always - model can be confidently wrong |
| How to calculate real accuracy? | Need separate evaluation with GT data |

---

## Recommendations

1. **Terminology**: Use "confidence" (not "accuracy") in UI/logs to avoid confusion
2. **Evaluation**: For real accuracy metrics, use separate evaluation scripts with GT
3. **Interpretation**: High confidence suggests model is sure, but doesn't guarantee correctness
4. **Trust**: Confidence is useful for filtering low-quality predictions, but not a substitute for validation

---

## See Also

- Ground Truth Override: [PAGE_XML_GT_OVERRIDE_ANALYSIS.md](PAGE_XML_GT_OVERRIDE_ANALYSIS.md)
- Batch Processing: [batch_processing.py](batch_processing.py)
- PyLaia Inference: [inference_pylaia_native.py](inference_pylaia_native.py)
