# TrOCR Inference Optimization Guide

## Problem Statement

**Observation**: The same model produces inconsistent results on the same images across different inference runs.

**Why This Matters**:
- Inconsistent results make it hard to evaluate model quality
- Users can't reproduce "good" results reliably
- Unclear which parameters actually improve performance

## Factors Affecting Inference Quality

### 1. **Line Segmentation Parameters**

Line segmentation is the FIRST critical step - bad segmentation = bad OCR results regardless of model quality.

#### Critical Parameters:

| Parameter | Impact | Default | Recommended Range | Test Priority |
|-----------|--------|---------|-------------------|---------------|
| **Threshold** | Determines what counts as "text" in projection | 5% | 0.5-10% | **HIGH** |
| **Min Line Height** | Filters out noise/small artifacts | 10px | 5-30px | **HIGH** |
| **Min Gap** | Spacing between lines for splitting | 5px | 3-15px | **MEDIUM** |
| **Morphological Ops** | Connects broken/faded characters | True | True/False | **HIGH** |
| **Binarization Method** | How image is converted to B&W | Otsu+Adaptive | Various | **MEDIUM** |

#### Why Segmentation Varies:

```
Same model, different segmentation → Different OCR results

Example with 5% threshold:
├─ Detects 15 lines correctly → Good transcription
└─ Image with different contrast → Detects 1 line → Poor transcription

The model didn't change - the INPUT to the model changed!
```

#### Testing Strategy:

1. **Visual Inspection First**: Always check if segmentation detected correct number of lines
2. **Adjust for Document Type**: Aged/faded documents need lower threshold (1-2%)
3. **Script-Specific**: Dense scripts (Arabic, Devanagari) may need morphological ops

---

### 2. **Image Preprocessing**

What happens to the image BEFORE it reaches the model.

#### Critical Parameters:

| Parameter | Impact | Default | Options | Test Priority |
|-----------|--------|---------|---------|---------------|
| **Aspect Ratio Preservation** | Character resolution | False | True/False | **CRITICAL** |
| **Target Height** | Detail vs processing speed | 128px | 96-384px | **HIGH** |
| **Background Normalization** | Handle aged/colored paper | False | True/False | **MEDIUM** |
| **Image Size** | Model input dimensions | 384×384 | Fixed by model | **INFO ONLY** |

#### Why Preprocessing Matters:

```
Ukrainian line: 4077×357 pixels

WITHOUT aspect ratio preservation:
→ Brutally resized to 384×384
→ Character width: 80px → 7px (11.4x downsampling)
→ Model sees blurry mush
→ CER: 23%

WITH aspect ratio preservation (128px height):
→ Resized to 1467×128, then padded
→ Character width: 80px → 29px (2.8x downsampling)
→ Model sees clear text
→ CER: 6% (estimated)
```

#### Training vs Inference Consistency:

**CRITICAL RULE**: Inference preprocessing MUST match training preprocessing!

| Training | Inference | Result |
|----------|-----------|--------|
| With normalization | With normalization | ✅ Good |
| Without normalization | Without normalization | ✅ Good |
| With normalization | **Without** normalization | ❌ **Poor** |
| **Without** normalization | With normalization | ❌ **Poor** |

---

### 3. **Model Inference Parameters**

How the model generates predictions from the processed image.

#### Critical Parameters:

| Parameter | Impact | Default | Recommended Range | Test Priority |
|-----------|--------|---------|-------------------|---------------|
| **Beam Search Width** | Quality vs speed tradeoff | 4 | 1-10 | **HIGH** |
| **Max Length** | Maximum predicted sequence | 128 | 64-256 | **MEDIUM** |
| **Temperature** | Randomness in prediction | 1.0 | 0.7-1.3 | **LOW** |
| **Top-k / Top-p** | Sampling strategy | N/A | Various | **LOW** |

#### Beam Search Explained:

```
Beam Width = 1 (Greedy):
├─ Fastest (1x speed)
├─ Takes most likely token at each step
└─ Can make irrecoverable mistakes
   Result: "The qwick brown fox"

Beam Width = 4 (Default):
├─ Medium speed (1.5x-2x slower)
├─ Keeps 4 candidate sequences
└─ Can recover from early mistakes
   Result: "The quick brown fox"

Beam Width = 10:
├─ Slowest (3x-4x slower)
├─ Keeps 10 candidates
├─ Marginal improvement over beam=4
└─ Diminishing returns
   Result: "The quick brown fox"
```

#### When to Adjust:

- **beam=1**: Quick testing, low quality acceptable
- **beam=4**: Production default, good balance
- **beam=8-10**: Final high-quality transcriptions, willing to wait

---

### 4. **Device & Hardware**

Physical execution environment affects consistency.

#### Parameters:

| Parameter | Impact | Default | Options | Test Priority |
|-----------|--------|---------|---------|---------------|
| **Device** | Speed and memory | CPU | CPU/GPU | **MEDIUM** |
| **Batch Size** | GPU utilization | 1 (GUI) | 1-32 | **LOW** |
| **FP16/FP32** | Speed vs precision | FP32 | FP16/FP32 | **LOW** |
| **Determinism** | Reproducibility | False | True/False | **HIGH for testing** |

#### CPU vs GPU Consistency:

```
CPU Inference:
├─ Deterministic (same input → same output)
├─ Slower but stable
└─ No memory issues

GPU Inference:
├─ Slightly non-deterministic (due to CUDA algorithms)
├─ Faster but can vary
└─ May have OOM errors with large images

For testing/comparison: Use CPU for reproducibility!
```

---

### 5. **Model-Specific Factors**

The model itself has hidden parameters.

#### Factors:

| Factor | Impact | Control | Test Priority |
|--------|--------|---------|---------------|
| **Training Data Characteristics** | What model "expects" | None (model fixed) | **INFO ONLY** |
| **Tokenizer Vocabulary** | Character coverage | None (model fixed) | **INFO ONLY** |
| **Model Checkpoint** | Training iteration | Choose checkpoint | **HIGH** |
| **Fine-tuning** | Domain adaptation | Train new model | **N/A** |

#### Why Different Checkpoints Matter:

```
checkpoint-1000:  CER = 25%
checkpoint-2000:  CER = 15%
checkpoint-3000:  CER = 12%  ← Best
checkpoint-4000:  CER = 14%  ← Overfitting started
checkpoint-5000:  CER = 18%  ← Degraded

Same model architecture, different training iterations!
```

---

## Best Practices for Consistent Results

### 1. **Document Your Pipeline**

Create a config file or document for each inference session:

```yaml
# inference_config.yaml
segmentation:
  threshold: 5.0
  min_line_height: 10
  min_gap: 5
  use_morph: true

preprocessing:
  preserve_aspect_ratio: true
  target_height: 128
  normalize_background: false

model:
  checkpoint: "models/ukrainian_aspect_ratio/checkpoint-3000"
  device: "cpu"

inference:
  num_beams: 4
  max_length: 128
```

### 2. **Standardize Your Workflow**

**Step 1: Inspect Image**
- Check image quality (resolution, contrast, clarity)
- Note any special characteristics (aged paper, faded ink, tight spacing)

**Step 2: Test Segmentation First**
- Run segmentation with defaults
- Visually verify correct number of lines detected
- Adjust threshold/min_height if needed
- Re-segment until correct

**Step 3: Run Inference**
- Use consistent device (CPU for testing, GPU for production)
- Use consistent beam width
- Save both image and text output

**Step 4: Document Results**
- Record all parameters used
- Save example outputs
- Note any issues or special handling

### 3. **Create Reproducible Tests**

```python
# test_inference_consistency.py
def test_consistency(image_path, config, num_runs=5):
    """Test if inference produces consistent results."""
    results = []

    for run in range(num_runs):
        # Segment
        segmenter = LineSegmenter(**config['segmentation'])
        segments = segmenter.segment_lines(image)

        # Infer
        ocr = TrOCRInference(**config['model'])
        transcriptions = []
        for seg in segments:
            text = ocr.transcribe_line(seg.image, **config['inference'])
            transcriptions.append(text)

        results.append(transcriptions)

    # Check consistency
    for i in range(1, num_runs):
        if results[i] != results[0]:
            print(f"⚠️  Run {i} differs from run 0!")
            print_diff(results[0], results[i])

    return results
```

---

## Systematic Testing Plan

### Phase 1: Baseline (Current State)

**Goal**: Document current performance and identify variability sources

1. **Select Test Set**: 10-20 representative images
2. **Run with Defaults**: Current GUI defaults
3. **Run 5 Times Each**: Check consistency
4. **Measure**:
   - Number of lines detected
   - Transcription text
   - Processing time
   - Differences between runs

**Expected Outcome**: Identify which images/settings cause variability

---

### Phase 2: Line Segmentation Optimization

**Goal**: Find optimal segmentation parameters for your document type

#### Test Matrix:

| Threshold | Min Height | Morph Ops | Test Images | Expected Lines |
|-----------|------------|-----------|-------------|----------------|
| 1% | 10px | True | img_001.jpg | 15 |
| 2% | 10px | True | img_001.jpg | 15 |
| 5% | 10px | True | img_001.jpg | 15 |
| 10% | 10px | True | img_001.jpg | 15 |
| 5% | 5px | True | img_001.jpg | 15 |
| 5% | 15px | True | img_001.jpg | 15 |
| 5% | 10px | False | img_001.jpg | 15 |

**Script**:

```python
import itertools
from inference_page import LineSegmenter
from PIL import Image

# Test parameters
thresholds = [0.01, 0.02, 0.05, 0.10]
min_heights = [5, 10, 15, 20]
morph_ops = [True, False]

test_images = ["page1.jpg", "page2.jpg", "page3.jpg"]
expected_lines = [15, 12, 18]

results = []

for img_path, expected in zip(test_images, expected_lines):
    image = Image.open(img_path)

    for threshold, min_height, use_morph in itertools.product(thresholds, min_heights, morph_ops):
        segmenter = LineSegmenter(
            sensitivity=threshold,
            min_line_height=min_height,
            use_morph=use_morph
        )

        segments = segmenter.segment_lines(image)
        detected = len(segments)

        # Score: distance from expected count
        score = abs(detected - expected)

        results.append({
            'image': img_path,
            'threshold': threshold,
            'min_height': min_height,
            'use_morph': use_morph,
            'detected': detected,
            'expected': expected,
            'score': score
        })

# Find best parameters
import pandas as pd
df = pd.DataFrame(results)
best = df.groupby(['threshold', 'min_height', 'use_morph'])['score'].mean().sort_values()
print("Best segmentation parameters:")
print(best.head(10))
```

---

### Phase 3: Preprocessing Optimization

**Goal**: Test impact of aspect ratio preservation and normalization

#### Test Matrix:

| Preserve Aspect | Target Height | Normalize BG | Expected CER |
|----------------|---------------|--------------|--------------|
| False | N/A | False | Baseline |
| True | 96px | False | Better? |
| True | 128px | False | Better? |
| True | 150px | False | Better? |
| False | N/A | True | Better? |
| True | 128px | True | Best? |

**Requirement**: Need ground truth transcriptions to compute CER

**Script**:

```python
from inference_page import TrOCRInference
import Levenshtein

# Load test set with ground truth
test_set = [
    {"image": "line1.png", "ground_truth": "Правильний текст"},
    {"image": "line2.png", "ground_truth": "Інший текст"},
    # ... 50-100 examples
]

configs = [
    {"preserve_aspect_ratio": False, "normalize_bg": False},
    {"preserve_aspect_ratio": True, "target_height": 96, "normalize_bg": False},
    {"preserve_aspect_ratio": True, "target_height": 128, "normalize_bg": False},
    {"preserve_aspect_ratio": True, "target_height": 128, "normalize_bg": True},
]

results = []

for config in configs:
    ocr = TrOCRInference("models/your_model", **config)

    total_cer = 0
    for item in test_set:
        image = Image.open(item['image'])
        prediction = ocr.transcribe_line(image, num_beams=4)

        # Compute CER
        distance = Levenshtein.distance(prediction, item['ground_truth'])
        cer = distance / len(item['ground_truth'])
        total_cer += cer

    avg_cer = total_cer / len(test_set)

    results.append({
        'config': config,
        'cer': avg_cer
    })

    print(f"Config: {config}")
    print(f"Average CER: {avg_cer:.2%}\n")

# Find best config
best_config = min(results, key=lambda x: x['cer'])
print(f"Best configuration: {best_config}")
```

---

### Phase 4: Inference Parameter Optimization

**Goal**: Find optimal beam width and other inference parameters

#### Test Matrix:

| Beam Width | Max Length | Expected Impact |
|------------|------------|-----------------|
| 1 | 128 | Fast, lower quality |
| 2 | 128 | Faster, good quality |
| 4 | 128 | Balanced (default) |
| 8 | 128 | Slower, marginal improvement |
| 4 | 64 | May truncate long lines |
| 4 | 256 | Handles very long lines |

**Script**:

```python
beam_widths = [1, 2, 4, 8]
test_lines = load_test_set()  # Lines with ground truth

results = []

for beam in beam_widths:
    ocr = TrOCRInference("models/your_model", device="cpu")

    import time
    start = time.time()

    total_cer = 0
    for line_image, ground_truth in test_lines:
        prediction = ocr.transcribe_line(line_image, num_beams=beam)
        cer = compute_cer(prediction, ground_truth)
        total_cer += cer

    elapsed = time.time() - start
    avg_cer = total_cer / len(test_lines)

    results.append({
        'beam_width': beam,
        'cer': avg_cer,
        'time_per_line': elapsed / len(test_lines),
        'total_time': elapsed
    })

# Print comparison
import pandas as pd
df = pd.DataFrame(results)
print(df)
```

---

### Phase 5: Model Checkpoint Selection

**Goal**: Find best checkpoint from training run

```python
checkpoints = [
    "models/ukrainian/checkpoint-1000",
    "models/ukrainian/checkpoint-2000",
    "models/ukrainian/checkpoint-3000",
    "models/ukrainian/checkpoint-4000",
]

test_set = load_test_set_with_ground_truth()

for checkpoint in checkpoints:
    ocr = TrOCRInference(checkpoint, device="cpu")

    total_cer = 0
    for image, ground_truth in test_set:
        prediction = ocr.transcribe_line(image, num_beams=4)
        cer = compute_cer(prediction, ground_truth)
        total_cer += cer

    avg_cer = total_cer / len(test_set)
    print(f"{checkpoint}: CER = {avg_cer:.2%}")
```

---

## Quick Test Script

Create `test_inference_params.py`:

```python
"""
Quick script to test inference parameters and find optimal settings.

Usage:
    python test_inference_params.py --test-image page.jpg --ground-truth-file page.txt
"""

import argparse
import itertools
from pathlib import Path
from PIL import Image
import pandas as pd
import Levenshtein

from inference_page import LineSegmenter, TrOCRInference

def compute_cer(prediction: str, ground_truth: str) -> float:
    """Compute Character Error Rate."""
    distance = Levenshtein.distance(prediction, ground_truth)
    return distance / max(len(ground_truth), 1)

def test_segmentation(image_path: Path, expected_lines: int):
    """Test different segmentation parameters."""
    print("=" * 60)
    print("TESTING SEGMENTATION PARAMETERS")
    print("=" * 60)

    image = Image.open(image_path)

    # Parameter grid
    thresholds = [0.01, 0.02, 0.05, 0.10]
    min_heights = [5, 10, 15, 20]
    morph_ops = [True, False]

    results = []

    for threshold, min_height, use_morph in itertools.product(thresholds, min_heights, morph_ops):
        segmenter = LineSegmenter(
            sensitivity=threshold,
            min_line_height=min_height,
            use_morph=use_morph
        )

        segments = segmenter.segment_lines(image)
        detected = len(segments)
        score = abs(detected - expected_lines)

        results.append({
            'threshold': f"{threshold:.2%}",
            'min_height': min_height,
            'morph_ops': use_morph,
            'detected_lines': detected,
            'score': score
        })

    df = pd.DataFrame(results).sort_values('score')
    print("\nTop 10 configurations (closest to expected):")
    print(df.head(10).to_string(index=False))

    return df.iloc[0]

def test_inference(image_path: Path, ground_truth_path: Path, checkpoint: str):
    """Test different inference parameters."""
    print("\n" + "=" * 60)
    print("TESTING INFERENCE PARAMETERS")
    print("=" * 60)

    # Load ground truth lines
    with open(ground_truth_path, 'r', encoding='utf-8') as f:
        ground_truth_lines = [line.strip() for line in f if line.strip()]

    # Segment image
    image = Image.open(image_path)
    segmenter = LineSegmenter()
    segments = segmenter.segment_lines(image)

    if len(segments) != len(ground_truth_lines):
        print(f"⚠️  Warning: Detected {len(segments)} lines but ground truth has {len(ground_truth_lines)}")

    # Test beam widths
    beam_widths = [1, 2, 4, 8]
    results = []

    for beam in beam_widths:
        print(f"\nTesting beam_width={beam}...")
        ocr = TrOCRInference(checkpoint, device="cpu")

        import time
        start = time.time()

        predictions = []
        for seg in segments:
            pred = ocr.transcribe_line(seg.image, num_beams=beam)
            predictions.append(pred)

        elapsed = time.time() - start

        # Compute CER
        total_cer = 0
        for pred, gt in zip(predictions[:len(ground_truth_lines)], ground_truth_lines):
            cer = compute_cer(pred, gt)
            total_cer += cer

        avg_cer = total_cer / len(ground_truth_lines)

        results.append({
            'beam_width': beam,
            'avg_cer': f"{avg_cer:.2%}",
            'time_per_line': f"{elapsed/len(segments):.2f}s",
            'total_time': f"{elapsed:.2f}s"
        })

    df = pd.DataFrame(results)
    print("\nInference Results:")
    print(df.to_string(index=False))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-image', required=True, help='Path to test image')
    parser.add_argument('--ground-truth-file', help='Path to ground truth transcription (one line per text line)')
    parser.add_argument('--expected-lines', type=int, help='Expected number of lines in image')
    parser.add_argument('--checkpoint', default='models/ukrainian_aspect_ratio/checkpoint-3000',
                       help='Model checkpoint to test')

    args = parser.parse_args()

    # Test segmentation
    if args.expected_lines:
        best_seg_config = test_segmentation(Path(args.test_image), args.expected_lines)
        print(f"\nRecommended segmentation: {best_seg_config.to_dict()}")

    # Test inference
    if args.ground_truth_file:
        test_inference(Path(args.test_image), Path(args.ground_truth_file), args.checkpoint)

if __name__ == '__main__':
    main()
```

---

## Summary: Key Recommendations

### 1. **Always Verify Segmentation First**
- Bad segmentation → Bad OCR (regardless of model quality)
- Visually inspect line detection before running inference
- Adjust threshold/min_height until correct line count

### 2. **Match Training Preprocessing**
- If model trained WITH aspect ratio preservation → use it in inference
- If model trained WITH normalization → use it in inference
- Mismatch = Poor results

### 3. **Use Consistent Device for Testing**
- CPU = deterministic (same input → same output)
- GPU = slightly non-deterministic (faster but may vary)
- For comparison tests: use CPU

### 4. **Document Everything**
- Save config files with results
- Record all parameters used
- Enable reproducibility

### 5. **Test Systematically**
1. Segmentation parameters first
2. Preprocessing options second
3. Inference parameters third
4. Model checkpoint selection last

### 6. **Prioritize What Matters**
**Highest Impact**:
- ✅ Line segmentation accuracy
- ✅ Aspect ratio preservation (if applicable)
- ✅ Model checkpoint selection

**Medium Impact**:
- Background normalization (document-dependent)
- Beam width (quality vs speed tradeoff)

**Low Impact**:
- Temperature, top-k/top-p sampling
- FP16 vs FP32 (minimal quality difference)

---

## Next Steps

1. **Create test set**: Select 10-20 representative pages with known line counts
2. **Run baseline**: Current defaults, document results
3. **Run parameter sweep**: Use test scripts above
4. **Document findings**: Update this guide with your optimal parameters
5. **Update GUI defaults**: Set defaults to optimal values found

Good luck with testing! The systematic approach will help you find the optimal configuration for your specific use case.
