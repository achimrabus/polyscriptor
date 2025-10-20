# Training on Efendiev_3 Dataset

Quick reference for training on your Efendiev manuscript data.

## Dataset Info

- **Source**: Transkribus export with PAGE XML
- **Language**: Russian (Cyrillic)
- **Pages**: 44 high-resolution scans (17800×11866px each)
- **Location**: `C:\Users\Achim\Downloads\Copy_of_HTR_Train_Set_'Efendiev_3'\Copy_of_HTR_Train_Set_'Efendiev_3'`

## Step 1: Data Parsing (RUNNING)

```bash
python transkribus_parser.py \
    --input_dir "C:\Users\Achim\Downloads\Copy_of_HTR_Train_Set_'Efendiev_3'\Copy_of_HTR_Train_Set_'Efendiev_3'" \
    --output_dir "./data/efendiev_3" \
    --train_ratio 0.8
```

**Status**: Currently processing (~40 seconds per page due to huge images)
**Output**:
- `data/efendiev_3/line_images/` - Cropped text line images
- `data/efendiev_3/train.csv` - Training data
- `data/efendiev_3/val.csv` - Validation data
- `data/efendiev_3/dataset_info.json` - Metadata

## Step 2: Training

Once parsing completes:

```bash
python optimized_training.py --config config_efendiev.yaml
```

Monitor with TensorBoard:
```bash
tensorboard --logdir ./models/efendiev_3_model
```

## Configuration

The config file `config_efendiev.yaml` has been prepared with:
- Base model: `kazars24/trocr-base-handwritten-ru` (Cyrillic pre-trained)
- Batch size: 16 × 4 accumulation = 64 effective
- Epochs: 15 (since this is a smaller dataset)
- Image caching: Enabled (line images are small)
- Augmentation: Enabled
- Evaluation: Every 100 steps

## Expected Results

With 44 pages and typical line density:
- **Estimated lines**: ~1,500-2,500 text lines
- **Training data**: ~1,200-2,000 lines (80%)
- **Validation data**: ~300-500 lines (20%)
- **Training time**: 1-3 hours (depends on line count)

## Adjustments for Your GPU

### If you get OOM (Out of Memory) errors:

Edit `config_efendiev.yaml`:
```yaml
batch_size: 8                    # Reduce from 16
gradient_accumulation_steps: 8   # Increase to maintain effective batch size
cache_images: false              # If still OOM
gradient_checkpointing: true     # Last resort (slower but less memory)
```

### If training is slow:

```yaml
cache_images: true               # Make sure this is enabled
dataloader_num_workers: 8        # Increase if you have many CPU cores
fp16: true                       # Make sure mixed precision is on
```

## After Training

Your trained model will be in: `./models/efendiev_3_model/`

To use it for inference:
```python
from transformers import VisionEncoderDecoderModel, TrOCRProcessor
from PIL import Image

model = VisionEncoderDecoderModel.from_pretrained("./models/efendiev_3_model")
processor = TrOCRProcessor.from_pretrained("./models/efendiev_3_model")

image = Image.open("line_image.png")
pixel_values = processor(image, return_tensors="pt").pixel_values
generated_ids = model.generate(pixel_values)
text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(text)
```

## Next Steps

1. ✓ Parse data (in progress)
2. ⏳ Wait for parsing to complete (~20-25 more minutes)
3. ⏳ Start training
4. ⏳ Monitor progress with TensorBoard
5. ⏳ Evaluate results

---

**Note**: The large image size (17800×11866px) makes parsing slow but only happens once. The cropped line images will be much smaller and training will be fast with caching enabled.
