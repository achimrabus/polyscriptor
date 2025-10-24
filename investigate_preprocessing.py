"""
Investigate TrOCR preprocessing to understand CER gap with PyLaia
Phase 1: Check image resizing and token lengths
"""
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
from transformers import TrOCRProcessor

def check_image_preprocessing():
    """Check how TrOCR processes Ukrainian line images."""
    print("=" * 80)
    print("PHASE 1A: Image Preprocessing Analysis")
    print("=" * 80)

    # Load processor
    processor = TrOCRProcessor.from_pretrained("kazars24/trocr-base-handwritten-ru")

    # Sample images from training data
    img_dir = Path("data/ukrainian_train_normalized/line_images")
    sample_images = list(img_dir.glob("*.png"))[:10]

    print(f"\nAnalyzing {len(sample_images)} sample images...\n")

    widths = []
    heights = []
    aspect_ratios = []

    for img_path in sample_images:
        img = Image.open(img_path)
        w, h = img.size
        widths.append(w)
        heights.append(h)
        aspect_ratios.append(w/h)

        print(f"Image: {img_path.name}")
        print(f"  Original size: {w}×{h} (aspect ratio: {w/h:.2f})")

        # Process through TrOCR
        pixel_values = processor(images=img, return_tensors="pt").pixel_values
        print(f"  Processed shape: {pixel_values.shape}")
        print(f"  -> Resized to: {pixel_values.shape[2]}x{pixel_values.shape[3]}")
        print()

    print("\nSummary Statistics:")
    print(f"  Width:  min={min(widths)}, max={max(widths)}, mean={np.mean(widths):.0f}, median={np.median(widths):.0f}")
    print(f"  Height: min={min(heights)}, max={max(heights)}, mean={np.mean(heights):.0f}, median={np.median(heights):.0f}")
    print(f"  Aspect ratio: min={min(aspect_ratios):.1f}, max={max(aspect_ratios):.1f}, mean={np.mean(aspect_ratios):.1f}")

    print("\n[WARNING] CRITICAL FINDING:")
    print(f"  Average line is {np.mean(widths):.0f}x{np.mean(heights):.0f}")
    print(f"  TrOCR resizes to 384x384 (square)")
    print(f"  -> Width downsampling factor: {np.mean(widths)/384:.1f}x")
    print(f"  -> Potential resolution loss!")

def check_token_lengths():
    """Check if max_length=128 is sufficient."""
    print("\n" + "=" * 80)
    print("PHASE 1B: Token Length Analysis")
    print("=" * 80)

    # Load processor
    processor = TrOCRProcessor.from_pretrained("kazars24/trocr-base-handwritten-ru")
    tokenizer = processor.tokenizer

    # Load training data
    df = pd.read_csv("data/ukrainian_train_normalized/train.csv", header=None, names=['image', 'text'])

    print(f"\nAnalyzing {len(df)} transcriptions...\n")

    token_lengths = []
    char_lengths = []

    for text in df['text']:
        # Tokenize
        tokens = tokenizer(text, return_tensors="pt").input_ids
        token_len = tokens.shape[1]
        char_len = len(text)

        token_lengths.append(token_len)
        char_lengths.append(char_len)

    token_lengths = np.array(token_lengths)
    char_lengths = np.array(char_lengths)

    print("Token Length Statistics:")
    print(f"  Min:  {token_lengths.min()}")
    print(f"  Max:  {token_lengths.max()}")
    print(f"  Mean: {token_lengths.mean():.1f}")
    print(f"  Median: {np.median(token_lengths):.1f}")
    print(f"  95th percentile: {np.percentile(token_lengths, 95):.0f}")
    print(f"  99th percentile: {np.percentile(token_lengths, 99):.0f}")

    print("\nCharacter Length Statistics:")
    print(f"  Min:  {char_lengths.min()}")
    print(f"  Max:  {char_lengths.max()}")
    print(f"  Mean: {char_lengths.mean():.1f}")
    print(f"  Median: {np.median(char_lengths):.1f}")
    print(f"  95th percentile: {np.percentile(char_lengths, 95):.0f}")
    print(f"  99th percentile: {np.percentile(char_lengths, 99):.0f}")

    # Check how many exceed max_length=128
    exceed_128 = (token_lengths > 128).sum()
    exceed_pct = exceed_128 / len(token_lengths) * 100

    print(f"\n[WARNING] CRITICAL FINDING:")
    print(f"  Current max_length: 128 tokens")
    print(f"  Transcriptions exceeding 128: {exceed_128} ({exceed_pct:.1f}%)")

    if exceed_128 > 0:
        print(f"  -> These get TRUNCATED during training!")
        print(f"  -> Model never learns full sequences")

        # Show some examples
        print(f"\nExamples of long transcriptions:")
        long_indices = np.where(token_lengths > 128)[0][:5]
        for idx in long_indices:
            print(f"\n  [{token_lengths[idx]} tokens, {char_lengths[idx]} chars]")
            print(f"  {df.iloc[idx]['text'][:100]}...")

    # Recommendation
    print("\n📊 RECOMMENDATION:")
    if exceed_pct > 5:
        recommended = int(np.percentile(token_lengths, 99))
        print(f"  Increase max_length to {recommended} (covers 99% of data)")
    elif exceed_pct > 1:
        recommended = int(np.percentile(token_lengths, 95))
        print(f"  Increase max_length to {recommended} (covers 95% of data)")
    else:
        print(f"  max_length=128 is adequate (covers {100-exceed_pct:.1f}% of data)")

def check_polygon_vs_rectangle():
    """Compare polygon vs rectangle segmentation."""
    print("\n" + "=" * 80)
    print("PHASE 1C: Segmentation Method Check")
    print("=" * 80)

    # Check if we have polygon data
    xml_files = list(Path("C:/Users/Achim/Documents/TrOCR/Ukrainian_Data/training_set/page").glob("*.xml"))

    if xml_files:
        print(f"\nFound {len(xml_files)} PAGE XML files")
        print("  -> Polygon coordinates available")
        print("  -> Currently using: RECTANGULAR bounding boxes")
        print("  -> PyLaia likely uses: POLYGON masks")
        print("\n[WARNING] CRITICAL FINDING:")
        print("  We can re-run preprocessing with --use-polygon-mask")
        print("  This will extract exact line shapes (not rectangles)")
    else:
        print("\n[ERROR] No PAGE XML files found")
        print("  Cannot use polygon masks")

if __name__ == "__main__":
    print("\n" + "="*80)
    print("TrOCR vs PyLaia Gap Investigation")
    print("Current: TrOCR 23% CER vs PyLaia 6% CER")
    print("="*80)

    try:
        check_image_preprocessing()
    except Exception as e:
        print(f"\n[ERROR] Image preprocessing check failed: {e}")

    try:
        check_token_lengths()
    except Exception as e:
        print(f"\n[ERROR] Token length check failed: {e}")

    try:
        check_polygon_vs_rectangle()
    except Exception as e:
        print(f"\n[ERROR] Segmentation check failed: {e}")

    print("\n" + "="*80)
    print("Investigation complete! See CER_GAP_ANALYSIS.md for full analysis.")
    print("="*80)
