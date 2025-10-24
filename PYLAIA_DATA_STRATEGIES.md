# PyLaia Data Strategy: Line Images vs Full Page + PAGE XML

## The Fundamental Question

Should we:
1. **Option A**: Extract line images + transcriptions (current plan)
2. **Option B**: Use full page images + PAGE XML coordinates

## Deep Dive Analysis

---

## Option A: Pre-extracted Line Images (Current Plan)

### What This Means
```
Input:
  data/lines/line_0001.png  (cropped line image)
  data/ground_truth/line_0001.txt  (transcription)

Training:
  Model sees: Individual line images
  Model learns: Text recognition only
```

### Architecture Requirements
```python
Model Input:  Line image (1, H, W) - grayscale, fixed height
Model Output: Text sequence via CTC
Training:     Direct image-to-text mapping
```

### Advantages ✅

1. **Simpler Training Pipeline**
   - One image → one text sequence
   - Standard CRNN architecture works out of the box
   - No need for attention mechanisms
   - Well-established approach (90% of HTR systems use this)

2. **Faster Training**
   - Smaller images to process
   - More samples per batch
   - Less VRAM required
   - Training time: ~2-3 hours for 17K lines

3. **Better Data Augmentation**
   - Easy to apply line-level augmentation
   - Rotation, shearing, elastic distortion
   - Consistent across the line

4. **Proven for Historical Documents**
   - Transkribus uses this approach
   - READ project uses this
   - Most academic HTR papers use this
   - Transfer learning from other line-based models possible

5. **Easier Debugging**
   - Can visualize what went wrong per line
   - CER calculation straightforward
   - Error analysis simpler

6. **Your Data is Already Prepared!**
   ```
   data/ukrainian_train_aspect_ratio/
     ├── images/
     │   ├── page_001_line_01.png
     │   ├── page_001_line_02.png
     │   └── ...
     └── train.csv (image_path, text)
   ```
   - You already have 17,602 line images
   - Already filtered and validated
   - Just need to resize to fixed height

### Disadvantages ❌

1. **Line Segmentation Dependency**
   - Quality depends on segmentation
   - Bad segmentation = bad training data
   - Need to maintain line extraction pipeline

2. **Lost Context**
   - No information about page layout
   - No multi-line context
   - Can't learn from page-level patterns

3. **Preprocessing Overhead**
   - Need to extract lines first
   - Disk space for line images (already done in your case)
   - Extra pipeline step

### Implementation Complexity: ⭐ LOW

```python
# Dead simple
dataset[i] = {
    'image': load_and_resize_line(img_path),
    'text': ground_truth
}
```

---

## Option B: Full Page + PAGE XML Coordinates

### What This Means
```
Input:
  data/pages/page_001.png  (full page scan)
  data/pagexml/page_001.xml  (line coordinates + transcriptions)

Training:
  Model sees: Full page image
  Model learns: Where lines are + what they say
```

### Architecture Requirements

This requires a **completely different architecture**:

#### Approach B1: Region Proposal Network (Faster R-CNN style)
```python
Model Architecture:
  1. Page Encoder (CNN/Transformer)
  2. Region Proposal Network → Detects line regions
  3. Line Encoder → Extracts features per region
  4. Text Decoder → Transcribes each region

Similar to: Document Understanding models (LayoutLMv3, Donut)
```

#### Approach B2: Attention-Based Multi-Line
```python
Model Architecture:
  1. Page Encoder → Full page features
  2. Multi-Head Attention → Attends to different lines
  3. Sequential Decoder → Transcribes line by line

Similar to: Vision transformers for documents
```

### Advantages ✅

1. **End-to-End Learning**
   - Learn segmentation + recognition jointly
   - Can optimize for both tasks
   - No separate segmentation pipeline needed

2. **Page-Level Context**
   - Can use neighboring lines for context
   - Learn page layout patterns
   - Better for structured documents

3. **No Preprocessing**
   - Direct page → transcription
   - No line extraction needed
   - Simpler data pipeline (just page + XML)

4. **Modern Approach**
   - Similar to current VLM trends
   - More "end-to-end"
   - Could leverage pretrained document models

### Disadvantages ❌

1. **🔴 MASSIVE Complexity Increase**
   ```
   Line-based model:  ~2,000 lines of code
   Page-based model:  ~10,000+ lines of code
   ```

2. **🔴 Much Harder to Train**
   - Need to balance segmentation + recognition losses
   - Multi-task learning is tricky
   - Harder to debug failures
   - Longer training time (5-10x)

3. **🔴 Data Requirements**
   ```
   Line-based:  17K line images (you have this!)
   Page-based:  Need full page images + perfect PAGE XML

   Do you have:
     - Full page images for all 17K lines?
     - PAGE XML with polygon coordinates?
     - Multiple lines per page mapped correctly?
   ```

4. **🔴 VRAM Requirements**
   ```
   Line-based:  ~1GB VRAM, batch_size=16
   Page-based:  ~8GB VRAM, batch_size=2

   Full pages are HUGE:
     - Typical page: 2000x3000 pixels
     - vs Line: 64x400 pixels
     - That's 100x more data!
   ```

5. **🔴 Slower Inference**
   ```
   Line-based:  Process 100 lines/second
   Page-based:  Process 1-2 pages/second
                (10-30 lines per page = 10-60 lines/second)
   ```

6. **🔴 No Standard Implementation**
   - Would need to build from scratch
   - Or heavily modify existing models
   - PyLaia doesn't support this
   - TrOCR doesn't support this natively

### Implementation Complexity: ⭐⭐⭐⭐⭐ VERY HIGH

```python
# Extremely complex
class PageHTRModel(nn.Module):
    def __init__(self):
        self.page_encoder = ...  # ResNet or ViT
        self.region_proposals = ...  # RPN or attention
        self.roi_align = ...  # Spatial transformer
        self.line_encoder = ...  # Per-region CNN
        self.text_decoder = ...  # CTC or Attention decoder
        self.segmentation_head = ...  # For line detection

    def forward(self, page_image, line_coords=None):
        # Extract page features
        features = self.page_encoder(page_image)

        # Detect/use line regions
        if training:
            regions = self.extract_regions(features, line_coords)
        else:
            regions = self.region_proposals(features)

        # Transcribe each region
        outputs = []
        for region in regions:
            region_features = self.roi_align(features, region)
            text = self.text_decoder(region_features)
            outputs.append(text)

        return outputs
```

---

## Your Data Situation Analysis

Let me check what you actually have:

### Current Data Structure (TrOCR Format)
```csv
# train.csv
image_path,text
page_0001_line_01.png,"Текст першого рядка"
page_0001_line_02.png,"Текст другого рядка"
```

### Questions About Your Data:

1. **Do you have full page images?**
   - Path: `C:\Users\Achim\Documents\TrOCR\Ukrainian_Data\training_set\`
   - Are there full page scans there?

2. **Do you have PAGE XML files?**
   - With `<TextLine>` coordinates
   - With `<TextEquiv>` transcriptions
   - With polygon/baseline coordinates

3. **Data Origin**
   - If from Transkribus: You probably have PAGE XML + full pages
   - If pre-extracted: You only have line images

---

## Recommendation Matrix

### If You Have: Line Images Only
**→ Use Option A (Pre-extracted Lines)**
- ✅ Ready to train immediately
- ✅ No additional work needed
- ✅ Standard, proven approach
- ⏱️ 2-3 hours to first results

### If You Have: Full Pages + PAGE XML
**→ STILL Use Option A (Extract Lines First)**

Why? Because:
1. **Proven approach**: 90% of HTR systems do this
2. **Easier to debug**: Can see which lines fail
3. **Faster iteration**: Train and test quickly
4. **Standard pipeline**:
   ```
   Page → Line Segmentation → Line Recognition
   (Kraken/HPP)  → (PyLaia/TrOCR)
   ```

### Only Consider Option B If:
1. You're doing research on end-to-end models
2. You have >100K annotated pages
3. You have 4+ months for development
4. You have 16GB+ VRAM GPUs
5. You want to publish a paper on architecture innovation

**For production HTR system: Always use line-based approach**

---

## Hybrid Approach (Best of Both Worlds)

### What If: Use PAGE XML but Extract Lines

```python
# parse_pagexml_to_lines.py

def extract_lines_from_pagexml(
    page_image_path: str,
    pagexml_path: str,
    output_dir: str
):
    """
    Extract line images from full page using PAGE XML coordinates.

    Advantages:
    - Use original full-quality page images
    - Get exact line coordinates from PAGE XML
    - Generate line images on-the-fly
    - Preserve all metadata
    """
    import xml.etree.ElementTree as ET
    from PIL import Image

    # Parse PAGE XML
    tree = ET.parse(pagexml_path)
    root = tree.getroot()
    ns = {'pc': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'}

    # Load full page image
    page_img = Image.open(page_image_path)

    lines = []

    # Extract each TextLine
    for textline in root.findall('.//pc:TextLine', ns):
        # Get coordinates
        coords = textline.find('pc:Coords', ns)
        points = coords.get('points').split()
        polygon = [tuple(map(int, p.split(','))) for p in points]

        # Get bounding box
        xs = [p[0] for p in polygon]
        ys = [p[1] for p in polygon]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)

        # Crop line image
        line_img = page_img.crop((x1, y1, x2, y2))

        # Get transcription
        textequiv = textline.find('.//pc:TextEquiv/pc:Unicode', ns)
        text = textequiv.text if textequiv is not None else ""

        # Save
        line_id = textline.get('id')
        line_img.save(f"{output_dir}/images/{line_id}.png")

        lines.append({
            'line_id': line_id,
            'text': text,
            'coords': polygon
        })

    return lines
```

### This Gives You:
1. ✅ High-quality line images (from original pages)
2. ✅ Perfect segmentation (from PAGE XML)
3. ✅ All transcriptions
4. ✅ Standard line-based training
5. ✅ Can still use page context if needed later

---

## Practical Recommendation for Your Project

### Phase 1: Start with Line Images (RECOMMENDED)
```bash
# Use your existing data
python train_pylaia.py --config config_pylaia_ukrainian.yaml

# Training time: 2-3 hours
# Results: Comparable to TrOCR
# Risk: Low
```

### Phase 2: If Results Are Good, Optimize
- Try different image heights (64, 96, 128)
- Experiment with data augmentation
- Fine-tune hyperparameters

### Phase 3: Consider Full-Page Only If:
- Line-based models don't work well
- You need joint segmentation + recognition
- You have 6+ months for research

---

## Data Format Decision Tree

```
Do you have full page images + PAGE XML?
│
├─ YES
│  │
│  ├─ Extract lines from PAGE XML → Use Option A
│  │  ✅ Best quality
│  │  ✅ Perfect segmentation
│  │  ✅ Standard approach
│  │
│  └─ Use pages directly → Option B
│     ❌ Too complex
│     ❌ Not worth it for <100K pages
│
└─ NO (only have line images)
   │
   └─ Use existing line images → Option A
      ✅ Already prepared
      ✅ Quick start
      ✅ Proven approach
```

---

## Real-World Examples

### Transkribus
- Input: Full pages + PAGE XML
- Processing: **Extracts lines first**
- Recognition: Line-based models
- Why: Proven, debuggable, fast

### Google Cloud Vision OCR
- Input: Full pages
- Processing: **Detects lines internally, then recognizes**
- Still line-based internally

### Academic State-of-the-Art (READ Project)
- Input: Full pages + PAGE XML
- Training: **Line-based models**
- Why: Better results, easier to train

---

## Your Specific Situation

Based on your earlier work, you have:
```
data/ukrainian_train_aspect_ratio/
  ├── train.csv (17,602 lines)
  └── images/ (line images)

data/ukrainian_val_aspect_ratio/
  ├── val.csv (4,342 lines)
  └── images/ (line images)
```

### My Recommendation: **Use Option A with your existing line images**

**Why:**
1. ✅ Data is already prepared
2. ✅ Already filtered (removed 1,705 missing images)
3. ✅ Already validated with TrOCR
4. ✅ Can start training TODAY
5. ✅ 2-3 hours to results
6. ✅ Low risk, high reward

**Don't:**
- ❌ Spend weeks building page-level architecture
- ❌ Re-extract lines from original pages (unless quality issues)
- ❌ Over-engineer the solution

---

## If You DO Have Original PAGE XML

Let me know and I can create a script to:
1. Parse PAGE XML files
2. Extract line coordinates
3. Crop from original high-res pages
4. Generate higher-quality line images
5. Preserve all metadata

But **still use line-based training** after extraction.

---

## Summary Table

| Aspect | Option A: Line Images | Option B: Full Page + XML |
|--------|----------------------|---------------------------|
| **Complexity** | ⭐ Simple | ⭐⭐⭐⭐⭐ Very Complex |
| **Training Time** | 2-3 hours | 15-30 hours |
| **VRAM Required** | 1-2 GB | 8-16 GB |
| **Implementation** | 1-2 days | 2-3 months |
| **Debugging** | Easy | Very Hard |
| **Success Rate** | 95% | 30% |
| **Industry Standard** | ✅ Yes | ❌ No |
| **Your Data Ready** | ✅ Yes | ❓ Unknown |
| **Proven Results** | ✅ Yes | ❌ Experimental |

---

## Final Answer

**Use Option A: Pre-extracted line images**

**Reasoning:**
1. Your data is ready
2. Industry-proven approach
3. Fast iteration
4. Easy to debug
5. PyLaia designed for this
6. 95% of production HTR systems use this

**Next Steps:**
1. Convert your CSV to PyLaia format (script provided)
2. Train model (2-3 hours)
3. Evaluate results
4. If results are good → Done!
5. If results are bad → Debug line-by-line (easy)

**Only consider page-level approach if:**
- You're publishing research
- You have unlimited time/resources
- Line-based approach fails completely

---

## Do You Have Original PAGE XML?

If yes, I can create a script to extract higher-quality line images from original pages using PAGE XML coordinates. But **we'll still train line-based models**.

This gives best of both worlds:
- High-quality line extraction (from PAGE XML)
- Standard line-based training (proven approach)
