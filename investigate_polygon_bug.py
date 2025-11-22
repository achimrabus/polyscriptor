#!/usr/bin/env python3
"""
Investigate polygon extraction bug by comparing V2 bbox vs V3 polygon images.
"""

import os
import random
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Dataset paths
V2_DIR = Path("/home/achimrabus/htr_gui/dhlab-slavistik/data/pylaia_prosta_mova_train/line_images")
V3_DIR = Path("/home/achimrabus/htr_gui/dhlab-slavistik/data/pylaia_prosta_mova_v3_train/line_images")

# Get all image filenames (should be identical)
v2_images = sorted(os.listdir(V2_DIR))
v3_images = sorted(os.listdir(V3_DIR))

print(f"V2 bbox images: {len(v2_images)}")
print(f"V3 polygon images: {len(v3_images)}")
print(f"Images match: {v2_images == v3_images}")

# Sample random images for comparison
random.seed(42)
sample_indices = random.sample(range(len(v2_images)), min(20, len(v2_images)))

# Compare dimensions and identify anomalies
dimension_mismatches = []
visual_differences = []

print("\n" + "="*80)
print("DIMENSION COMPARISON")
print("="*80)

for i in sample_indices[:20]:
    img_name = v2_images[i]

    v2_img = Image.open(V2_DIR / img_name)
    v3_img = Image.open(V3_DIR / img_name)

    v2_size = v2_img.size
    v3_size = v3_img.size

    size_match = v2_size == v3_size

    # Calculate pixel difference
    if size_match:
        v2_arr = np.array(v2_img)
        v3_arr = np.array(v3_img)
        pixel_diff = np.mean(np.abs(v2_arr.astype(float) - v3_arr.astype(float)))
    else:
        pixel_diff = None

    print(f"\n{i+1}. {img_name}")
    print(f"   V2 bbox:    {v2_size[0]:4d} x {v2_size[1]:3d}")
    print(f"   V3 polygon: {v3_size[0]:4d} x {v3_size[1]:3d}")
    print(f"   Match: {size_match}")
    if pixel_diff is not None:
        print(f"   Pixel diff: {pixel_diff:.2f}")
        if pixel_diff > 50:
            visual_differences.append((img_name, pixel_diff))

    if not size_match:
        dimension_mismatches.append((img_name, v2_size, v3_size))

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Dimension mismatches: {len(dimension_mismatches)}/{len(sample_indices)}")
print(f"Visual differences (>50 pixel avg): {len(visual_differences)}/{len(sample_indices)}")

# Create visual comparison grid
print("\n" + "="*80)
print("CREATING VISUAL COMPARISON...")
print("="*80)

fig, axes = plt.subplots(10, 2, figsize=(16, 40))
fig.suptitle('V2 Bbox (left) vs V3 Polygon (right) - Sample Comparison', fontsize=16)

for idx, i in enumerate(sample_indices[:10]):
    img_name = v2_images[i]

    v2_img = Image.open(V2_DIR / img_name)
    v3_img = Image.open(V3_DIR / img_name)

    # V2 bbox
    axes[idx, 0].imshow(v2_img)
    axes[idx, 0].set_title(f'V2 BBOX: {img_name[:50]}...', fontsize=8)
    axes[idx, 0].axis('off')

    # V3 polygon
    axes[idx, 1].imshow(v3_img)
    axes[idx, 1].set_title(f'V3 POLYGON: {v3_img.size}', fontsize=8)
    axes[idx, 1].axis('off')

    # Add border to highlight if dimensions differ
    v2_size = v2_img.size
    v3_size = v3_img.size
    if v2_size != v3_size:
        for ax in [axes[idx, 0], axes[idx, 1]]:
            ax.add_patch(mpatches.Rectangle((0, 0), 1, 1,
                                           transform=ax.transAxes,
                                           fill=False, edgecolor='red', linewidth=3))

plt.tight_layout()
plt.savefig('/home/achimrabus/htr_gui/dhlab-slavistik/polygon_comparison.png', dpi=150)
print("Saved visual comparison to: polygon_comparison.png")

# Check for specific patterns in corrupted images
print("\n" + "="*80)
print("ANALYZING CORRUPTION PATTERNS...")
print("="*80)

# Check if corrupted images have certain characteristics
if dimension_mismatches:
    print(f"\nDimension mismatches found:")
    for img_name, v2_size, v3_size in dimension_mismatches:
        print(f"  {img_name}: V2={v2_size}, V3={v3_size}")
