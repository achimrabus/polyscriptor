#!/usr/bin/env python3
"""
Visualize line image comparison between Church Slavonic and Prosta Mova.
"""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def analyze_and_visualize_line(image_path: Path):
    """Analyze and visualize a line image with margin annotations."""
    img = Image.open(image_path).convert('L')
    img_array = np.array(img)

    # Calculate vertical projection
    vertical_proj = np.sum(img_array < 200, axis=1)  # Count dark pixels

    # Find first and last rows with significant ink
    ink_rows = np.where(vertical_proj > img_array.shape[1] * 0.05)[0]
    if len(ink_rows) > 0:
        first_ink_row = ink_rows[0]
        last_ink_row = ink_rows[-1]
    else:
        first_ink_row = 0
        last_ink_row = img_array.shape[0] - 1

    # Create visualization
    vis_img = Image.fromarray(img_array).convert('RGB')
    draw = ImageDraw.Draw(vis_img)

    # Draw ink boundaries
    draw.line([(0, first_ink_row), (img_array.shape[1], first_ink_row)], fill='green', width=2)
    draw.line([(0, last_ink_row), (img_array.shape[1], last_ink_row)], fill='green', width=2)

    # Add text annotations
    top_margin = first_ink_row
    bottom_margin = img_array.shape[0] - last_ink_row - 1
    ink_height = last_ink_row - first_ink_row + 1

    return vis_img, {
        'top_margin': top_margin,
        'bottom_margin': bottom_margin,
        'ink_height': ink_height,
        'total_height': img_array.shape[0],
        'margin_ratio': (top_margin + bottom_margin) / img_array.shape[0],
    }

def create_comparison_grid(church_slavonic_images: list, prosta_mova_images: list, output_path: Path):
    """Create a comparison grid of Church Slavonic vs Prosta Mova line images."""
    num_samples = min(5, len(church_slavonic_images), len(prosta_mova_images))

    # Analyze images
    church_slavonic_results = []
    prosta_mova_results = []

    for i in range(num_samples):
        cs_vis, cs_stats = analyze_and_visualize_line(church_slavonic_images[i])
        pm_vis, pm_stats = analyze_and_visualize_line(prosta_mova_images[i])

        church_slavonic_results.append((cs_vis, cs_stats))
        prosta_mova_results.append((pm_vis, pm_stats))

    # Create composite image
    max_width = max(
        max(vis.width for vis, _ in church_slavonic_results),
        max(vis.width for vis, _ in prosta_mova_results)
    )

    row_height = 128 + 60  # Image height + text space
    total_height = row_height * num_samples

    composite = Image.new('RGB', (max_width * 2 + 40, total_height + 100), color='white')
    draw = ImageDraw.Draw(composite)

    # Title
    draw.text((20, 10), "Church Slavonic (Left) vs Prosta Mova (Right)", fill='black')
    draw.text((20, 30), "Green lines show ink boundaries", fill='green')

    y_offset = 80

    for i in range(num_samples):
        cs_vis, cs_stats = church_slavonic_results[i]
        pm_vis, pm_stats = prosta_mova_results[i]

        # Paste images
        composite.paste(cs_vis, (20, y_offset))
        composite.paste(pm_vis, (max_width + 40, y_offset))

        # Add statistics
        cs_text = f"CS: Top={cs_stats['top_margin']}px, Bottom={cs_stats['bottom_margin']}px, Ink={cs_stats['ink_height']}px, Margin={cs_stats['margin_ratio']:.1%}"
        pm_text = f"PM: Top={pm_stats['top_margin']}px, Bottom={pm_stats['bottom_margin']}px, Ink={pm_stats['ink_height']}px, Margin={pm_stats['margin_ratio']:.1%}"

        draw.text((20, y_offset + 130), cs_text, fill='black')
        draw.text((max_width + 40, y_offset + 130), pm_text, fill='black')

        y_offset += row_height

    composite.save(output_path)
    print(f"Saved comparison grid to {output_path}")

def main():
    church_slavonic_images_dir = Path('/home/achimrabus/htr_gui/dhlab-slavistik/data/pylaia_church_slavonic_train/line_images')
    prosta_mova_images_dir = Path('/home/achimrabus/htr_gui/dhlab-slavistik/data/pylaia_prosta_mova_train/line_images')

    # Get first 10 images
    church_slavonic_images = sorted(list(church_slavonic_images_dir.glob('*.png')))[:10]
    prosta_mova_images = sorted(list(prosta_mova_images_dir.glob('*.png')))[:10]

    print(f"Found {len(church_slavonic_images)} Church Slavonic images")
    print(f"Found {len(prosta_mova_images)} Prosta Mova images")

    output_path = Path('/home/achimrabus/htr_gui/dhlab-slavistik/line_comparison_grid.png')
    create_comparison_grid(church_slavonic_images, prosta_mova_images, output_path)

    # Print detailed statistics
    print("\n" + "="*80)
    print("DETAILED STATISTICS")
    print("="*80)

    print("\nChurch Slavonic (first 10 images):")
    for img_path in church_slavonic_images[:10]:
        _, stats = analyze_and_visualize_line(img_path)
        print(f"  {img_path.name}: Top={stats['top_margin']:2d}px, Bottom={stats['bottom_margin']:2d}px, Ink={stats['ink_height']:3d}px, Margin={stats['margin_ratio']:5.1%}")

    print("\nProsta Mova (first 10 images):")
    for img_path in prosta_mova_images[:10]:
        _, stats = analyze_and_visualize_line(img_path)
        print(f"  {img_path.name}: Top={stats['top_margin']:2d}px, Bottom={stats['bottom_margin']:2d}px, Ink={stats['ink_height']:3d}px, Margin={stats['margin_ratio']:5.1%}")

if __name__ == '__main__':
    main()
