"""
Test script to visualize background normalization effect.

This script loads sample Ukrainian images and shows before/after normalization.
"""

from PIL import Image
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt


def normalize_background(image: Image.Image) -> Image.Image:
    """
    Normalize background to light gray (same as in transkribus_parser.py).
    """
    # Convert PIL to OpenCV format
    img_array = np.array(image)

    # Convert to LAB color space for better lighting normalization
    lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_normalized = clahe.apply(l)

    # Merge back and convert to RGB
    lab_normalized = cv2.merge([l_normalized, a, b])
    rgb_normalized = cv2.cvtColor(lab_normalized, cv2.COLOR_LAB2RGB)

    # Convert to grayscale to remove color variations (aged paper tones)
    gray = cv2.cvtColor(rgb_normalized, cv2.COLOR_RGB2GRAY)

    # Convert back to RGB with uniform background
    normalized_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    return Image.fromarray(normalized_rgb)


def test_samples():
    """Test normalization on sample images."""

    # Find sample Ukrainian images
    ukrainian_dir = Path(r"c:\Users\Achim\Documents\TrOCR\dhlab-slavistik\data\ukrainian_train\line_images")
    efendiev_dir = Path(r"c:\Users\Achim\Documents\TrOCR\dhlab-slavistik\data\efendiev_3\line_images")

    # Get 3 Ukrainian samples
    ukrainian_samples = list(ukrainian_dir.glob("*.png"))[:3]

    # Get 1 Efendiev sample for comparison
    efendiev_samples = list(efendiev_dir.glob("*.png"))[:1]

    if not ukrainian_samples:
        print("ERROR: No Ukrainian samples found!")
        return

    # Create comparison figure
    fig, axes = plt.subplots(len(ukrainian_samples) + 1, 3, figsize=(15, 12))
    fig.suptitle('Background Normalization Effect: Ukrainian vs Efendiev', fontsize=16)

    # Process Ukrainian samples
    for i, img_path in enumerate(ukrainian_samples):
        original = Image.open(img_path).convert('RGB')
        normalized = normalize_background(original)

        axes[i, 0].imshow(np.array(original))
        axes[i, 0].set_title(f'Ukrainian Original {i+1}')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(np.array(normalized))
        axes[i, 1].set_title(f'Ukrainian Normalized {i+1}')
        axes[i, 1].axis('off')

        # Calculate mean background color
        orig_mean = np.array(original).mean(axis=(0, 1))
        norm_mean = np.array(normalized).mean(axis=(0, 1))

        axes[i, 2].text(0.1, 0.7, f'Original mean:\nR={orig_mean[0]:.1f}, G={orig_mean[1]:.1f}, B={orig_mean[2]:.1f}',
                       fontsize=10, verticalalignment='top')
        axes[i, 2].text(0.1, 0.3, f'Normalized mean:\nR={norm_mean[0]:.1f}, G={norm_mean[1]:.1f}, B={norm_mean[2]:.1f}',
                       fontsize=10, verticalalignment='top')
        axes[i, 2].axis('off')

    # Show Efendiev reference
    if efendiev_samples:
        efendiev_img = Image.open(efendiev_samples[0]).convert('RGB')
        efendiev_mean = np.array(efendiev_img).mean(axis=(0, 1))

        axes[-1, 0].imshow(np.array(efendiev_img))
        axes[-1, 0].set_title('Efendiev Reference (6% CER)')
        axes[-1, 0].axis('off')

        axes[-1, 1].text(0.1, 0.5,
                        f'Efendiev background:\nR={efendiev_mean[0]:.1f}, G={efendiev_mean[1]:.1f}, B={efendiev_mean[2]:.1f}\n\n' +
                        'Goal: Normalized Ukrainian\nshould match this gray tone',
                        fontsize=10, verticalalignment='center')
        axes[-1, 1].axis('off')

        axes[-1, 2].text(0.1, 0.5,
                        'Key Insight:\n\n' +
                        'Efendiev has uniform\ngray background.\n\n' +
                        'Ukrainian has aged\ntan/beige paper.\n\n' +
                        'Normalization converts\nUkrainian to match Efendiev.',
                        fontsize=9, verticalalignment='center')
        axes[-1, 2].axis('off')
    else:
        for j in range(3):
            axes[-1, j].text(0.5, 0.5, 'Efendiev samples not found', ha='center', va='center')
            axes[-1, j].axis('off')

    plt.tight_layout()

    # Save figure
    output_path = Path(r"c:\Users\Achim\Documents\TrOCR\dhlab-slavistik\normalization_comparison.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n[OK] Comparison saved to: {output_path}")
    print("\nYou can view the comparison image to see the normalization effect.")

    # Don't show in headless mode
    # plt.show()


if __name__ == '__main__':
    print("=" * 80)
    print("Background Normalization Test")
    print("=" * 80)
    print("\nThis script demonstrates why normalization is critical:")
    print("1. Ukrainian data: aged tan/beige paper backgrounds")
    print("2. Efendiev data: clean gray backgrounds (6-7% CER)")
    print("3. Normalization converts Ukrainian to match Efendiev")
    print("\nGenerating comparison...")
    print("=" * 80)

    test_samples()
