# TrOCR for Cyrillic Handwriting Recognition

Fine-tuning TrOCR (Transformer-based OCR) for recognizing handwritten Cyrillic text in Russian, Ukrainian, and Church Slavonic manuscripts.

## 🚀 Quick Start

See [USAGE_GUIDE.md](USAGE_GUIDE.md) for detailed instructions.

### Installation

```bash
pip install -r requirements.txt
```

### Prepare Data from Transkribus

```bash
python transkribus_parser.py \
    --input_dir /path/to/transkribus_export \
    --output_dir ./processed_data
```

### Train Model

```bash
cp example_config.yaml config.yaml
# Edit config.yaml with your paths
python optimized_training.py --config config.yaml
```

## 📊 Performance

**Optimized Pipeline Benefits:**
- **10-50x faster** data loading (image caching)
- **16x larger** effective batch size (64 vs 4)
- **3-4x faster** overall training time
- **Better generalization** (augmentation enabled)

**Current Best Model:**
- Model: `kazars24/trocr-base-handwritten-ru`
- CER: 0.253 (75% character accuracy)
- Languages: Mostly Russian

## 📁 Repository Structure

```
.
├── transkribus_parser.py      # Parse Transkribus PAGE XML exports
├── optimized_training.py      # Fast training with image caching
├── example_config.yaml        # Training configuration template
├── USAGE_GUIDE.md            # Detailed usage instructions
├── requirements.txt          # Python dependencies
│
├── Fine_Tune_TrOCR.ipynb    # Legacy notebook (deprecated)
├── lineSegmentation.ipynb    # Legacy segmentation (deprecated)
└── party/
    └── divideInTestTrainLst.py  # Dataset splitting utility
```

**⚠️ Deprecated:** The old notebooks (`Fine_Tune_TrOCR.ipynb`, `lineSegmentation.ipynb`) are kept for reference but are **significantly slower** than the new pipeline.

## 🎯 Project Goals

1. ✅ Fine-tune TrOCR for Cyrillic handwriting (Russian, Ukrainian, Church Slavonic)
2. ⏳ Deploy eScriptorium instance on lab server
3. ✅ Create optimized training pipeline with Transkribus integration

## 🔧 Troubleshooting

### Known Issues (Resolved)

**PyTorch compatibility:**
```bash
pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**Transformers API changes:**
- Fixed in transformers==4.45.2
- See: https://github.com/huggingface/transformers/issues/36074

### Common Problems

See [USAGE_GUIDE.md](USAGE_GUIDE.md#troubleshooting) for:
- Out of memory errors
- Slow data loading
- Poor line segmentation
- Training not converging

## 📈 Results

### Best Model: Combined Cyrillic Dataset

- **Training data**: Russian (1365312) + Ukrainian (6470048) + Church Slavonic
- **CER**: 0.253 (75% character accuracy)
- **Model**: Based on `kazars24/trocr-base-handwritten-ru`
- **Download**: [Dropbox Link](https://www.dropbox.com/scl/fi/umth7s1l619ok693l9xcy/seq2seq_mixed.zip)

### Training History

1. ✅ Preprocessed Russian handwritten dataset (split into halves)
2. ✅ Implemented augmentation (RandomRotation, RandomAffine)
3. ✅ Trained on combined Russian + Ukrainian + Church Slavonic
4. ✅ Created optimized training pipeline (10-50x faster)
5. ✅ Integrated Transkribus PAGE XML parser

## 🔬 Future Work

1. Test alternative training approaches:
   - [wjbmattingly/trocr-train](https://github.com/wjbmattingly/trocr-train)
   - [QWEN VLM fine-tuning](https://huggingface.co/learn/cookbook/fine_tuning_vlm_trl)

2. Latin script training (CATMos dataset):
   - Prepare dataset splits (train/val/test)
   - Run large-scale training

3. Deploy to production:
   - eScriptorium integration
   - Batch processing scripts