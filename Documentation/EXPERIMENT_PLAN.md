# Systematic Experiment Plan: Improving Ukrainian TrOCR

## Current Baseline
- **Model**: kazars24/trocr-base-handwritten-ru
- **CER**: ~26% (best: 25.96% at step 2500)
- **Target**: 6% CER (PyLaia benchmark)

## Hypothesis Analysis

### Key Factors to Test (Priority Order)

#### 1. **Base Model Selection** (HIGHEST IMPACT)
Current model is Russian-focused. Ukrainian has different character distributions.

**Experiment 1a-1d: Base Model Comparison**
- `kazars24/trocr-base-handwritten-ru` (current, Russian)
- `microsoft/trocr-base-handwritten` (multilingual, but Latin-biased?)
- `Yehor-Smorodov/uk-trocr-base-handwritten-full` (Ukrainian-specific if exists)
- Search HuggingFace for other Cyrillic/Slavic models

**Hypothesis**: Ukrainian-specific or multilingual Cyrillic model will perform better than Russian-only

#### 2. **Optimizer & Training** (HIGH IMPACT)
Bullinger uses Adafactor, we used AdamW.

**Experiment 2a: Adafactor (IMPLEMENTED)**
- Switch to `optim: "adafactor"`
- **Status**: ✅ Done in updated config

**Experiment 2b: Learning Rate**
- Current: 4e-5
- Test: 5e-5, 3e-5, 1e-5
- **Hypothesis**: Russian model may need different LR for Ukrainian

#### 3. **Beam Search** (MEDIUM IMPACT)
Bullinger uses 4 beams, we used 1.

**Experiment 3: Beam Search (IMPLEMENTED)**
- Changed from `num_beams: 1` → `num_beams: 4`
- **Status**: ✅ Done in updated config
- **Expected**: Better CER at evaluation (slower but more accurate)

#### 4. **Data Augmentation** (UNCERTAIN)
We use augmentation, Bullinger doesn't.

**Experiment 4a-4c: Augmentation Impact**
- 4a: Current (rotation ±2°, brightness ±30%, contrast ±30%)
- 4b: Disable all augmentation
- 4c: Reduce intensity (rotation ±1°, brightness ±15%, contrast ±15%)

**Hypothesis**: Augmentation may add noise for clear handwriting OR help with varied conditions

#### 5. **Sequence Length** (LOW IMPACT)
**Experiment 5: Max Length**
- Current: 128
- Bullinger: 64
- Test: 64, 96, 128

**Hypothesis**: Shorter sequences may reduce errors but could truncate long lines

#### 6. **Image Preprocessing** (POTENTIAL HIGH IMPACT)
Bullinger converts to white background (`-wbg`). We use raw PAGE XML images.

**Experiment 6a-6c: Background Processing**
- 6a: Current (raw Transkribus images)
- 6b: Convert to white background (like Bullinger)
- 6c: Binarization (threshold-based black/white)

**Hypothesis**: Clean white background reduces noise, improves CER

#### 7. **Warmup Strategy** (LOW-MEDIUM IMPACT)
**Experiment 7: Warmup**
- Current: warmup_ratio=0.1
- Bullinger: warmup_steps=len(data)/batch_size
- Calculate: 18691/6 ≈ 3115 warmup steps

**Hypothesis**: Explicit warmup steps may stabilize early training

---

## Recommended Experiment Sequence

### Phase 1: Quick Wins (Current Run)
**Config**: Updated config_ukrainian.yaml
- ✅ Batch size: 6 (OOM fix)
- ✅ Optimizer: adafactor
- ✅ Num beams: 4
- ⏳ **Action**: Resume training from checkpoint-3000, run to completion

**Expected Outcome**: CER improves from 26% → 22-24% (if lucky)

---

### Phase 2: Base Model Exploration (IF Phase 1 doesn't reach <15% CER)

**Experiment Plan**:
1. Search HuggingFace for Ukrainian/Cyrillic TrOCR models
2. Test top 3 candidates on small subset (1000 samples, 3 epochs)
3. Select best model for full training

**Search queries**:
- "trocr ukrainian"
- "trocr cyrillic"
- "trocr handwritten slavic"
- "trocr base handwritten" (check if multilingual)

**Models to investigate**:
```python
# Potential candidates (need to verify existence):
candidates = [
    "microsoft/trocr-base-handwritten",  # Check Cyrillic support
    "Yehor-Smorodov/uk-trocr-base-handwritten-full",  # Ukrainian if exists
    # Add more from search results
]
```

---

### Phase 3: Systematic Grid Search (IF still >10% CER)

**Variables**:
1. Augmentation: [True, False]
2. Max length: [64, 128]
3. Learning rate: [1e-5, 4e-5]
4. Background preprocessing: [raw, white-bg, binarized]

**Total experiments**: 2×2×2×3 = 24 configs

**Strategy**: Use validation set, train for 5 epochs each, select best 3 for full training

---

### Phase 4: Image Preprocessing (IF preprocessing seems promising)

**Experiment**:
1. Convert all Transkribus images to white background
2. Apply binarization (Otsu threshold)
3. Normalize image sizes
4. Retrain with best config from Phase 1-3

**Implementation**:
```python
# Preprocessing pipeline to add to transkribus_parser.py
from PIL import ImageOps, Image

def preprocess_line_image(image):
    # Convert to white background
    # Option 1: Simple inversion if dark background
    # Option 2: Threshold-based extraction
    # Option 3: Background removal (more complex)
    pass
```

---

## Evaluation Metrics

**Primary**: Character Error Rate (CER)
**Secondary**:
- Word Error Rate (WER)
- Normalized Error Rate (NER)
- Training time
- Inference speed

---

## Implementation Notes

### How to Run Experiments

**Single experiment**:
```bash
# Edit config file with experiment parameters
python optimized_training.py --config config_experiment_X.yaml
```

**Multiple experiments (parallel)**:
```bash
# Create configs: config_exp1.yaml, config_exp2.yaml, ...
# Run on different GPUs or sequentially
```

### Tracking Results

Create results table:
```
| Exp | Model | Optim | Beams | Aug | MaxLen | CER | Notes |
|-----|-------|-------|-------|-----|--------|-----|-------|
| 1   | ru    | adaf  | 4     | yes | 128    | 26% | baseline |
| 2   | ru    | adaf  | 4     | yes | 128    | ?   | current  |
| ... |       |       |       |     |        |     |          |
```

---

## Expected Timeline

- **Phase 1**: 2-3 hours (resume current training)
- **Phase 2**: 1-2 days (model search + small tests)
- **Phase 3**: 3-5 days (grid search)
- **Phase 4**: 1-2 days (preprocessing + retrain)

**Total**: ~1 week for comprehensive investigation

---

## Success Criteria

- **Good**: CER < 15% (significant improvement)
- **Very Good**: CER < 10% (approaching PyLaia)
- **Excellent**: CER < 7% (matching PyLaia)

---

## Questions to Answer

1. ❓ Is Russian model suitable for Ukrainian, or do we need Ukrainian-specific?
2. ❓ Does data augmentation help or hurt for clean handwriting?
3. ❓ Is white background preprocessing necessary?
4. ❓ What's the optimal sequence length for Ukrainian text?
5. ❓ Can we match PyLaia's 6% CER with TrOCR?

---

## Next Steps

1. ✅ Let current training (batch=6, adafactor, beams=4) complete
2. 🔍 Search HuggingFace for better Cyrillic base models
3. 📊 Analyze results and decide on Phase 2 experiments
4. 🧪 Run systematic experiments based on findings
