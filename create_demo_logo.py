#!/usr/bin/env python3
"""
Demo Logo Creator for Polyscriptor

Creates a simple text-based logo as a placeholder until a professional
logo is designed. Run this script to generate assets/logo.png.

Usage:
    python3 create_demo_logo.py
"""

from PIL import Image, ImageDraw, ImageFont
from pathlib import Path


def create_demo_logo(output_path: str = "assets/logo.png"):
    """
    Create a demo logo with Cyrillic and Latin text.

    Args:
        output_path: Where to save the logo PNG
    """
    # Logo dimensions
    width, height = 400, 120

    # Create image with transparent background
    img = Image.new('RGBA', (width, height), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)

    # Try to load fonts, fall back to default if not available
    try:
        # Try system fonts (Ubuntu/Debian path)
        font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 48)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    except:
        try:
            # Try alternative path (Red Hat/Fedora)
            font_large = ImageFont.truetype("/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf", 48)
            font_small = ImageFont.truetype("/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf", 18)
        except:
            try:
                # Try Windows path
                font_large = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 48)
                font_small = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 18)
            except:
                # Fall back to default font
                print("Warning: Could not load custom font, using default")
                font_large = ImageFont.load_default()
                font_small = ImageFont.load_default()

    # Main text: "Polyscriptor"
    main_text = "Polyscriptor"
    main_color = (41, 128, 185)  # Blue

    # Get text size for centering
    bbox = draw.textbbox((0, 0), main_text, font=font_large)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = (width - text_width) // 2
    y = (height - text_height) // 2 - 15

    draw.text((x, y), main_text, fill=main_color, font=font_large)

    # Subtitle: "Multi-Engine HTR"
    subtitle = "Multi-Engine HTR"
    subtitle_color = (127, 140, 141)  # Gray

    bbox_sub = draw.textbbox((0, 0), subtitle, font=font_small)
    sub_width = bbox_sub[2] - bbox_sub[0]
    x_sub = (width - sub_width) // 2
    y_sub = y + text_height + 5

    draw.text((x_sub, y_sub), subtitle, fill=subtitle_color, font=font_small)

    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save PNG
    img.save(output_path)
    print(f"✓ Demo logo saved to: {output_path}")
    print(f"  Size: {width}×{height}px")
    print(f"  Replace this with a professional logo for production!")


def create_icon(output_path: str = "assets/icon.png"):
    """
    Create a simple icon (circular badge with 'П').

    Args:
        output_path: Where to save the icon PNG
    """
    size = 256
    img = Image.new('RGBA', (size, size), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)

    # Background circle
    margin = 10
    draw.ellipse([margin, margin, size - margin, size - margin],
                 fill=(41, 128, 185))  # Blue

    # Try to load font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 150)
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf", 150)
        except:
            try:
                font = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 150)
            except:
                font = ImageFont.load_default()

    # Draw Cyrillic 'П' (P)
    text = "П"
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = (size - text_width) // 2
    y = (size - text_height) // 2 - 10

    draw.text((x, y), text, fill=(255, 255, 255), font=font)

    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save PNG
    img.save(output_path)
    print(f"✓ Demo icon saved to: {output_path}")
    print(f"  Size: {size}×{size}px")


if __name__ == "__main__":
    print("Creating demo logo and icon for Polyscriptor...")
    print("=" * 60)

    create_demo_logo()
    create_icon()

    print("=" * 60)
    print("Done! You can now run the GUI:")
    print("  python3 transcription_gui_plugin.py")
    print()
    print("For a professional logo, consider:")
    print("  - AI generation (DALL-E, Midjourney, Stable Diffusion)")
    print("  - Design tools (Canva, Figma, Inkscape)")
    print("  - Professional designers (Fiverr, 99designs)")
    print()
    print("See assets/README.md for more details.")
