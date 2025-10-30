"""
Resize PyLaia images to fixed height while preserving aspect ratio.

PyLaia requires all images to have the same height. This script resizes
images to a target height while preserving aspect ratio.
"""

from PIL import Image
from pathlib import Path
from tqdm import tqdm
import argparse


def resize_to_fixed_height(image: Image.Image, target_height: int) -> Image.Image:
    """
    Resize image to target height while preserving aspect ratio.

    Args:
        image: PIL Image
        target_height: Target height in pixels

    Returns:
        Resized PIL Image
    """
    width, height = image.size

    if height == target_height:
        return image

    # Calculate new width to preserve aspect ratio
    aspect_ratio = width / height
    new_width = int(target_height * aspect_ratio)

    # Resize with high-quality resampling
    resized = image.resize((new_width, target_height), Image.Resampling.LANCZOS)

    return resized


def main():
    parser = argparse.ArgumentParser(description="Resize PyLaia images to fixed height")
    parser.add_argument("--input-dir", type=str, required=True,
                       help="Directory containing input images")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Directory for resized images (default: overwrite input)")
    parser.add_argument("--height", type=int, default=128,
                       help="Target height in pixels (default: 128)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be done without actually resizing")

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir

    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        return

    # Create output directory if needed
    if output_dir != input_dir and not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Find all images
    image_files = list(input_dir.glob("*.png")) + list(input_dir.glob("*.jpg"))

    if not image_files:
        print(f"No images found in {input_dir}")
        return

    print(f"Found {len(image_files)} images")
    print(f"Target height: {args.height}px")
    print(f"Output directory: {output_dir}")

    if args.dry_run:
        print("\nDRY RUN - No files will be modified\n")

    # Resize images
    heights_before = []
    heights_after = []

    for img_path in tqdm(image_files, desc="Resizing images"):
        img = Image.open(img_path)
        heights_before.append(img.height)

        if not args.dry_run:
            resized = resize_to_fixed_height(img, args.height)
            heights_after.append(resized.height)

            # Save to output directory
            output_path = output_dir / img_path.name
            resized.save(output_path)
        else:
            # Just calculate what the new height would be
            heights_after.append(args.height)

    # Statistics
    print(f"\nStatistics:")
    print(f"  Images processed: {len(image_files)}")
    print(f"  Heights before: min={min(heights_before)}, max={max(heights_before)}")
    print(f"  Heights after: min={min(heights_after)}, max={max(heights_after)}")

    if not args.dry_run:
        print(f"\n✓ Images resized successfully to {output_dir}")
    else:
        print(f"\nRun without --dry-run to apply changes")


if __name__ == "__main__":
    main()
