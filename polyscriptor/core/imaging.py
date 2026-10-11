"""Shared image helpers for dataset preparation and inference."""

from typing import List, Optional, Tuple

import cv2  # type: ignore
import numpy as np
from PIL import Image, ImageDraw


def normalize_background_image(image: Image.Image) -> Image.Image:
    """Normalize background to uniform light gray (CLAHE on LAB L-channel, then grayscale).

    Addresses aged paper, color cast and lighting inconsistencies while keeping
    text contrast.
    """
    img_array = np.array(image)
    lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_normalized = clahe.apply(l)
    lab_normalized = cv2.merge([l_normalized, a, b])
    rgb_normalized = cv2.cvtColor(lab_normalized, cv2.COLOR_LAB2RGB)
    gray = cv2.cvtColor(rgb_normalized, cv2.COLOR_RGB2GRAY)
    normalized_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(normalized_rgb)


def resize_with_aspect_ratio(image: Image.Image, target_height: int) -> Image.Image:
    """Resize to *target_height* preserving aspect ratio (LANCZOS).

    TrOCR's ViTImageProcessor resizes to 384x384 no matter the input shape; for
    long manuscript lines that crushes characters (~10x width compression).
    Pre-resizing to a sane height keeps character strokes resolvable.
    """
    width, height = image.size
    new_width = int(target_height * (width / height))
    return image.resize((new_width, target_height), Image.Resampling.LANCZOS)


def crop_line(
    image: Image.Image,
    coords: List[Tuple[int, int]],
    *,
    padding: int = 5,
    normalize_background: bool = False,
    use_polygon_mask: bool = True,
    target_height: Optional[int] = None,
) -> Image.Image:
    """Crop one text line given its polygon coordinates.

    Order of operations: bbox crop (+padding) -> background normalization ->
    optional polygon mask (outside set to white) -> optional aspect-preserving
    resize.
    """
    xs = [p[0] for p in coords]
    ys = [p[1] for p in coords]
    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)

    x1_pad = max(0, x1 - padding)
    y1_pad = max(0, y1 - padding)
    x2_pad = min(image.width, x2 + padding)
    y2_pad = min(image.height, y2 + padding)

    cropped = image.crop((x1_pad, y1_pad, x2_pad, y2_pad))

    if normalize_background:
        cropped = normalize_background_image(cropped)

    if not use_polygon_mask:
        result = cropped
    else:
        mask = Image.new("L", (x2_pad - x1_pad, y2_pad - y1_pad), 0)
        draw = ImageDraw.Draw(mask)
        adjusted = [(x - x1_pad, y - y1_pad) for x, y in coords]
        draw.polygon(adjusted, fill=255)
        result = Image.new("RGB", cropped.size, (255, 255, 255))
        result.paste(cropped, mask=mask)

    if target_height is not None:
        result = resize_with_aspect_ratio(result, target_height)
    return result
