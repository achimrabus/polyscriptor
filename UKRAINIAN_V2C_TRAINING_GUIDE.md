# Ukrainian V2c Training - Complete Workflow

## 📋 Quick Summary

**Dataset V2c Improvements:**
- ✅ EXIF rotation bug fixed (ImageOps.exif_transpose at line 232)
- ✅ Case-sensitivity bug fixed (.JPG vs .jpg at line 344)
- ✅ 24,706 training lines (+2,762 from V2b's 21,944 = +12.6%)
- ✅ 970 validation lines (+156 from V2b's 814 = +19.2%)
- ✅ 2,772 Лист (printed) training lines (was 0 in V2b)
- ✅ 156 Лист validation lines (was 0 in V2b)

**Achieved Results (Training Completed Nov 24, 2025):**
- ✅ Overall CER: 4.76% (better than V2b's 5.53%!)
- ✅ Training completed in 101 epochs (~2.7 hours)
- ✅ Best model: `models/pylaia_ukrainian_v2c_20251124_180634/best_model.pt`
- 🎯 Expected Лист CER: <5% (V2b had ~90%+ due to missing training data)

---

## 🚀 Step-by-Step Workflow

### Step 1: Convert Data to PyLaia Format

```bash
# Activate virtual environment
source htr_gui/bin/activate

# Convert V2c data to PyLaia format
python3 convert_ukrainian_v2c_to_pylaia.py
```

**Expected output:**
```
✅ CONVERSION COMPLETE
Training lines:   24,706
Validation lines: 970
Total lines:      25,676
Vocabulary size:  187
```

**Files created:**
- `data/pylaia_ukrainian_v2c_combined/lines.txt` (training set)
- `data/pylaia_ukrainian_v2c_combined/lines_val.txt` (validation set)
- `data/pylaia_ukrainian_v2c_combined/syms.txt` (vocabulary)
- `data/pylaia_ukrainian_v2c_combined/dataset_info.json` (metadata)

---

### Step 2: Start Training (Option A - Interactive)

```bash
python3 train_pylaia_ukrainian_v2c.py
```

**Pros:** See training progress in real-time
**Cons:** Training stops if terminal closes

---

### Step 2: Start Training (Option B - Background with nohup) ⭐ RECOMMENDED

```bash
# Option 1: Use the shell script (easiest)
bash start_ukrainian_v2c_training.sh

# Option 2: Manual nohup command
nohup python3 train_pylaia_ukrainian_v2c.py > training_ukrainian_v2c.log 2>&1 &
tail -f training_ukrainian_v2c.log
```

**Pros:** Training continues even if terminal closes
**Cons:** Need to monitor log file

---

### Step 3: Monitor Training Progress

```bash
# Watch training log in real-time
tail -f training_ukrainian_v2c.log

# Check specific lines
tail -100 training_ukrainian_v2c.log

# Search for CER values
grep "CER:" training_ukrainian_v2c.log

# Check GPU usage
watch -n 1 nvidia-smi
```

---

### Step 4: Check Training Status

```bash
# Check if training is running
ps aux | grep train_pylaia_ukrainian_v2c.py

# Or use the shell script's check
pgrep -f train_pylaia_ukrainian_v2c.py
```

---

### Step 5: Stop Training (if needed)

```bash
# Press Ctrl+C to stop monitoring (training continues)

# To kill training process:
ps aux | grep train_pylaia_ukrainian_v2c.py | grep -v grep
kill <PID>

# Or kill all training processes:
pkill -f train_pylaia_ukrainian_v2c.py
```

---

## 📊 Training Configuration

```json
{
  "img_height": 128,
  "batch_size": 32,
  "cnn_filters": [12, 24, 48, 48],
  "rnn_hidden": 256,
  "rnn_layers": 3,
  "dropout": 0.5,
  "learning_rate": 0.0003,
  "max_epochs": 250,
  "early_stopping": 15
}
```

**Training time estimate:**
- ~6-12 hours on NVIDIA GPU (depends on GPU model)
- Much longer on CPU (not recommended)

---

## 🎯 Expected Training Output

### Console Output:
```
============================================================
UKRAINIAN V2c DATASET INFO
============================================================
Training lines:   24,706
Validation lines: 970
Total lines:      25,676
Vocabulary size:  187
EXIF corrected:   True
Includes Лист:    True
============================================================

Loading datasets...
Training samples: 24,706
Validation samples: 970
Vocabulary size: 187 symbols

Initializing CRNN model...
Model parameters: 3,456,789 total, 3,456,789 trainable

============================================================
STARTING UKRAINIAN V2c PYLAIA TRAINING
============================================================
Epoch 1/250:  Train Loss: 2.345  Val Loss: 1.876  CER: 45.3%
Epoch 2/250:  Train Loss: 1.654  Val Loss: 1.234  CER: 32.1%
...
```

### Output Files:
```
models/pylaia_ukrainian_v2c_<timestamp>/
├── best_model.pt               # Best model (lowest validation CER)
├── checkpoint_epoch_10.pt      # Checkpoint every 10 epochs
├── checkpoint_epoch_20.pt
├── training_history.json       # Full training history
├── model_config.json          # Model configuration
└── symbols.txt                # Vocabulary (187 symbols)
```

---

## ✅ Validation After Training

### Evaluate on Лист Files:

```bash
# Run batch processing on Лист validation files
python3 batch_processing.py \
  --model models/pylaia_ukrainian_v2c_<timestamp>/best_model.pt \
  --input /home/achimrabus/htr_gui/Ukrainian_Data/validation_set/ \
  --output results_v2c_list/

# Compare with V2b results
python3 compare_cer.py \
  --v2b batch_results.json \
  --v2c results_v2c_list/batch_results.json
```

### Expected Improvements:
- **Лист 021**: 90%+ CER (V2b) → <5% CER (V2c)
- **Лист 041**: 90%+ CER (V2b) → <5% CER (V2c)
- **Лист 061**: 90%+ CER (V2b) → <5% CER (V2c)
- **Лист 081**: 90%+ CER (V2b) → <5% CER (V2c)
- **Лист 101**: 90%+ CER (V2b) → <5% CER (V2c)

---

## 🐛 Troubleshooting

### Data not converted:
```bash
❌ Error: Data directory not found
```
**Solution:** Run `python3 convert_ukrainian_v2c_to_pylaia.py`

### CUDA out of memory:
```bash
RuntimeError: CUDA out of memory
```
**Solution:** Reduce batch size in training script (change `batch_size: 32` to `batch_size: 16`)

### Training dies immediately:
```bash
❌ Error: Training process died immediately!
```
**Solution:** Check log file for errors: `tail -50 training_ukrainian_v2c.log`

### Can't find GPU:
```bash
⚠️  nvidia-smi not available
```
**Solution:** Training will use CPU (very slow). Consider using a GPU machine.

---

## 📝 Files Summary

**Created for V2c Training:**
1. `train_pylaia_ukrainian_v2c.py` - Main training script
2. `start_ukrainian_v2c_training.sh` - Convenient launcher with nohup
3. `UKRAINIAN_V2C_TRAINING_GUIDE.md` - This guide
4. `convert_ukrainian_v2c_to_pylaia.py` - Data conversion script (already exists)
5. `reextract_ukrainian_v2c.py` - Re-extraction script (already exists)

**Data Directories:**
- `data/pylaia_ukrainian_v2c_train_fresh/` - Training line images
- `data/pylaia_ukrainian_v2c_val_fresh/` - Validation line images
- `data/pylaia_ukrainian_v2c_combined/` - PyLaia format files

**Model Output:**
- `models/pylaia_ukrainian_v2c_<timestamp>/` - Trained model

---

## 🎓 Key Lessons from V2c Bug Fixes

1. **EXIF Rotation Bug:**
   - Always apply `ImageOps.exif_transpose()` BEFORE coordinate extraction
   - Impact: 32% of data had EXIF tags, 99 Лист files were completely lost
   - Fixed at: `transkribus_parser.py` line 232

2. **Case-Sensitivity Bug:**
   - Linux filesystem is case-sensitive: `.JPG` ≠ `.jpg`
   - Always check for both uppercase and lowercase extensions
   - Fixed at: `transkribus_parser.py` line 344

3. **Timeline Matters:**
   - V2b data was extracted Oct 31, EXIF fix added Nov 21 (53 minutes AFTER training started!)
   - Always verify bug fixes are applied to the actual training data

---

## 🚀 Ready to Train!

Everything is set up. Run this command to start training:

```bash
bash start_ukrainian_v2c_training.sh
```

Good luck! 🎉
