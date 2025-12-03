# Train CER Logging - Why It Matters

## Changes Made

Added training CER calculation to `train_pylaia.py` to track overfitting and detect data quality issues early.

### Code Changes

1. **History tracking** (line 358): Added `'train_cer': []` to history dict
2. **New method** (lines 468-506): `calculate_train_cer()` - samples 1000 training examples (or 10% of dataset) without augmentation and calculates CER
3. **Training loop** (lines 523-543): Calculate and log train CER after each epoch, with overfitting warning

### What You'll See in Training Logs

```
Epoch 1/250
============================================================
Train loss: 0.8832
Train CER:  45.23%
Val loss:   0.2977
Val CER:    8.71%
LR:         0.000300

Epoch 10/250
============================================================
Train loss: 0.1456
Train CER:  3.12%
Val loss:   0.1562
Val CER:    3.68%
LR:         0.000300

Epoch 20/250
============================================================
Train loss: 0.1210
Train CER:  2.45%
Val loss:   0.1485
Val CER:    3.31%
LR:         0.000300
⚠️  Overfitting gap: 0.86% (Val CER > Train CER)
```

## Why Train CER Matters

### 1. **Overfitting Detection**

**Healthy training**:
- Train CER ≈ Val CER (within 1-2%)
- Both decreasing together
- Example: Train 2.9%, Val 3.0% ✅

**Overfitting**:
- Train CER << Val CER (gap > 3%)
- Train keeps improving, Val plateaus
- Example: Train 1.5%, Val 8.2% ⚠️

**Expected for Prosta Mova V4**:
- Early epochs: Large gap (train learns faster on clean data)
- Later epochs: Gap narrows to 1-2% (healthy generalization)
- Target: Train CER ~4.5%, Val CER ~5.0%

### 2. **⚠️ CRITICAL: Data Quality Issues (How We Would've Caught Rotation Bug Earlier)**

**If V2/V3 had train CER logging**, we would've seen the EXIF rotation bug **immediately**:

```
Epoch 1:  Train CER: 85%, Val CER: 87%  (Both terrible - DATA ISSUE!)
Epoch 10: Train CER: 45%, Val CER: 48%  (Improving but still bad)
Epoch 30: Train CER: 25%, Val CER: 28%  (Model struggling)
Epoch 50: Train CER: 18%, Val CER: 19%  (Stuck at plateau - EXIF bug!)
```

**🚨 CRITICAL INDICATOR**: When BOTH train AND val CER are high (>15-20%) after 10+ epochs, this signals **corrupted training data**, not a model problem.

**Why this signals a bug**:
- Train CER should drop quickly (model memorizes clean training data)
- If train CER stays high (>15-20%), the **training data itself is corrupted**
- This would've flagged: "32% of images have vertical text → model can't learn"

**Compare to Church Slavonic (clean data)**:
```
Epoch 1:  Train CER: 12%, Val CER: 8.7%   (Train higher due to augmentation)
Epoch 10: Train CER: 4.2%, Val CER: 3.6%  (Converging nicely)
Epoch 30: Train CER: 3.1%, Val CER: 3.0%  (Healthy gap)
Epoch 59: Train CER: 2.8%, Val CER: 2.9%  (Best performance)
```

### 3. **Learning Progress Indicator**

**Early epochs** (1-10):
- Train CER drops rapidly (model learns common patterns)
- Val CER drops slower (generalization lag)
- Gap widens temporarily

**Mid-training** (10-40):
- Both CERs decrease
- Gap stabilizes around 1-2%
- Model refining predictions

**Late training** (40+):
- Both CERs plateau
- Gap stays constant
- Early stopping should trigger

### 4. **Architecture/Hyperparameter Diagnosis**

**Too much regularization** (dropout, augmentation):
- Train CER stays higher than val CER
- Example: Train 5%, Val 4% (model not learning training data fully)

**Too little regularization**:
- Train CER much lower than val CER
- Example: Train 1%, Val 10% (severe overfitting)

**Perfect balance** (what we want):
- Train CER slightly lower than val CER
- Gap: 0.5-2%
- Example: Train 4.5%, Val 5.0% ✅

## Performance Impact

**Computation cost**:
- Samples 1000 training images per epoch (or 10% if smaller dataset)
- Runs inference without augmentation
- Adds ~30-60 seconds per epoch

**For Prosta Mova V4**:
- Dataset: 58,843 lines
- Samples: min(1000, 5,884) = 1000 lines
- Time: ~45 seconds per epoch
- Total training time increase: ~5% (acceptable for diagnostic value)

## Expected Results for V4

### Best Case (No Overfitting, Clean Data)
```
Epoch 50 (best):
  Train CER:  4.5%
  Val CER:    5.0%
  Gap:        0.5% ✅ Healthy generalization
```

### If We Still Had Bugs
```
Epoch 50:
  Train CER: 16.0%
  Val CER:   19.0%
  Gap:       3.0% → Both too high, data quality issue!
```

### If Overfitting Occurred
```
Epoch 80:
  Train CER:  2.0%
  Val CER:    8.5%
  Gap:        6.5% ⚠️ Severe overfitting, need more data/regularization
```

## Historical Context

**Why Church Slavonic succeeded** (retrospective):
- 309,959 clean training lines
- Tight line segmentation (42.8px)
- No rotation bugs
- **If we had train CER**: Would've confirmed Train ~2.8%, Val ~2.9% (healthy gap)

**Why Prosta Mova V2/V3 failed** (retrospective):
- 32% rotated images in training
- Loose segmentation (67.2px)
- **If we had train CER**: Would've shown Train 18%, Val 19% → both high → data bug!

**Why Prosta Mova V4 should succeed**:
- 0% rotated images (EXIF fixed)
- Tighter segmentation (64.0px)
- **With train CER**: Will confirm Train ~4.5%, Val ~5.0% → healthy gap → clean data!

## 🚨 EARLY WARNING SYSTEM: Monitoring During Training

Watch for these patterns to catch bugs EARLY:

### 1. **First 5 epochs**: Train CER should drop rapidly to <10%

**Healthy training** (clean data):
```
Epoch 1: Train 45% → Val 48%
Epoch 5: Train  8% → Val 10%  ✅ Data is clean
```

**🚨 RED FLAG** (data quality issue - possibly EXIF rotation bug):
```
Epoch 1: Train 85% → Val 87%
Epoch 5: Train 35% → Val 38%  ❌ BOTH TOO HIGH - CHECK DATA
```

**If train CER stuck above 20% after 5 epochs**:
- ❌ Check for EXIF rotation bug (vertical text)
- ❌ Check vocabulary mismatch
- ❌ Check for corrupted/blank images
- ❌ Visual inspection of training samples

### 2. **Epochs 10-30**: Gap should stabilize at 1-3%

**Healthy gap**:
```
Epoch 10: Train 5.2% → Val 6.1% (gap: 0.9%)  ✅
Epoch 20: Train 4.8% → Val 5.5% (gap: 0.7%)  ✅
```

**Overfitting** (gap >5%):
```
Epoch 20: Train 2.0% → Val 8.5% (gap: 6.5%)  ⚠️ Need more data/augmentation
```

### 3. **Epochs 30+**: Both should plateau together

**Healthy plateau** (early stopping should trigger):
```
Epoch 30: Train 4.5% → Val 5.0%
Epoch 40: Train 4.4% → Val 4.9%
Epoch 50: Train 4.5% → Val 5.1%  ✅ Early stopping triggered
```

**Continued divergence** (overfitting):
```
Epoch 30: Train 3.0% → Val 5.0%
Epoch 40: Train 2.0% → Val 6.0%
Epoch 50: Train 1.5% → Val 7.0%  ⚠️ Stop training, add regularization
```

## Summary

✅ **Added train CER logging to detect**:
- Overfitting (train << val)
- Data quality issues (both train and val high)
- Learning progress (both decreasing healthily)

✅ **Performance cost**: ~5% slower training (worth it for diagnostics)

✅ **Expected for V4**: Train ~4.5%, Val ~5.0% with 0.5-1% healthy gap

✅ **Historical lesson**: Train CER would've caught V2/V3 rotation bug immediately (train CER stuck at 18-20%)
