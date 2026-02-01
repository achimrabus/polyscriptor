"""
Parse ALTO XML exports to create training datasets for TrOCR.

Usage:
    python alto_parser.py --input_dir /path/to/alto/export --output_dir /path/to/output

Expected structure:
    input_dir/
        page-000.xml
        page-000.jpg
        page-001.xml
        page-001.jpg
        ...

Crops are always made from polygon masks (Shape/Polygon in ALTO), not bounding boxes.
"""

import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple, Optional
import argparse
import pandas as pd
from PIL import Image, ImageDraw, ImageOps
import numpy as np
from tqdm import tqdm
import json
import csv
import cv2  # type: ignore
from multiprocessing import Pool, cpu_count


class AltoParser:
    """Parse ALTO XML files. Crops use polygon masks from Shape/Polygon (not bounding boxes)."""

    # ALTO XML namespace (v4)
    NS = {'alto': 'http://www.loc.gov/standards/alto/ns-v4#'}

    def __init__(self, input_dir: str, output_dir: str, min_line_width: int = 20,
                 normalize_background: bool = False,
                 preserve_aspect_ratio: bool = False, target_height: int = 128,
                 num_workers: int = None):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.min_line_width = min_line_width
        # ALTO parser always uses polygon mask (crops from polygons, not bboxes)
        self.use_polygon_mask = True
        self.normalize_background = normalize_background
        self.preserve_aspect_ratio = preserve_aspect_ratio
        self.target_height = target_height

        if num_workers is not None:
            self.num_workers = num_workers
        else:
            default_workers = max(1, int(cpu_count() * 0.625))
            self.num_workers = default_workers

        self.images_dir = self.output_dir / "line_images"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.metadata = []

    def parse_coords(self, points_str: str) -> List[Tuple[int, int]]:
        """Parse POINTS string from ALTO XML (space-separated 'x y x y ...')."""
        values = points_str.split()
        return [(int(values[i]), int(values[i + 1])) for i in range(0, len(values), 2)]

    def get_bounding_box(self, coords: List[Tuple[int, int]]) -> Tuple[int, int, int, int]:
        """Get bounding box (x1, y1, x2, y2) from polygon coordinates."""
        xs = [p[0] for p in coords]
        ys = [p[1] for p in coords]
        return min(xs), min(ys), max(xs), max(ys)

    def normalize_background_image(self, image: Image.Image) -> Image.Image:
        """Normalize background to light gray (similar to Efendiev dataset)."""
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

    def resize_with_aspect_ratio(self, image: Image.Image) -> Image.Image:
        """Resize to target height preserving aspect ratio."""
        width, height = image.size
        aspect_ratio = width / height
        new_height = self.target_height
        new_width = int(new_height * aspect_ratio)
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    def crop_polygon(self, image: Image.Image, coords: List[Tuple[int, int]]) -> Image.Image:
        """Crop image to polygon shape with masking (always uses polygon mask for ALTO)."""
        x1, y1, x2, y2 = self.get_bounding_box(coords)
        padding = 5
        x1_pad = max(0, x1 - padding)
        y1_pad = max(0, y1 - padding)
        x2_pad = min(image.width, x2 + padding)
        y2_pad = min(image.height, y2 + padding)

        cropped = image.crop((x1_pad, y1_pad, x2_pad, y2_pad))

        if self.normalize_background:
            cropped = self.normalize_background_image(cropped)

        # ALTO: always apply polygon mask (crops from polygons, not bboxes)
        mask = Image.new('L', (x2_pad - x1_pad, y2_pad - y1_pad), 0)
        draw = ImageDraw.Draw(mask)
        adjusted_coords = [(x - x1_pad, y - y1_pad) for x, y in coords]
        draw.polygon(adjusted_coords, fill=255)
        result = Image.new('RGB', cropped.size, (255, 255, 255))
        result.paste(cropped, mask=mask)

        if self.preserve_aspect_ratio:
            result = self.resize_with_aspect_ratio(result)
        return result

    def extract_lines_from_page(self, xml_path: Path, image_path: Path) -> List[dict]:
        """Extract all text lines from an ALTO XML file."""
        tree = ET.parse(xml_path)
        root = tree.getroot()

        lines_data = []

        Image.MAX_IMAGE_PIXELS = None
        try:
            page_image = Image.open(image_path)
            page_image = ImageOps.exif_transpose(page_image)
            page_image = page_image.convert('RGB')
        except Exception as e:
            print(f"Error opening image {image_path}: {e}")
            return lines_data

        # ALTO: TextBlock -> TextLine; each TextLine has Shape/Polygon and String
        for text_block in root.findall('.//alto:TextBlock', self.NS):
            block_id = text_block.get('id', 'unknown')

            for idx, text_line in enumerate(text_block.findall('alto:TextLine', self.NS)):
                line_id = text_line.get('id', f'{block_id}_line_{idx}')

                # ALTO: polygon in Shape/Polygon POINTS="x1 y1 x2 y2 ..."
                shape = text_line.find('alto:Shape', self.NS)
                polygon = shape.find('alto:Polygon', self.NS) if shape is not None else None
                if polygon is None:
                    polygon = text_line.find('alto:Shape/alto:Polygon', self.NS)
                if polygon is None:
                    continue

                points_str = polygon.get('POINTS')
                if not points_str:
                    continue

                coords = self.parse_coords(points_str)
                if len(coords) < 3:
                    continue

                x1, y1, x2, y2 = self.get_bounding_box(coords)

                # ALTO: text in String CONTENT="..."
                string_elem = text_line.find('alto:String', self.NS)
                if string_elem is None:
                    continue
                text = string_elem.get('CONTENT')
                if text is None or not text.strip():
                    continue
                text = text.strip()

                if (x2 - x1) < self.min_line_width:
                    continue

                try:
                    line_image = self.crop_polygon(page_image, coords)
                    page_name = image_path.stem
                    line_filename = f"{page_name}_{line_id}.png"
                    line_image_path = self.images_dir / line_filename
                    line_image.save(line_image_path)

                    lines_data.append({
                        'image_path': str(line_image_path.relative_to(self.output_dir)),
                        'text': text,
                        'page': page_name,
                        'line_id': line_id,
                        'bbox': (x1, y1, x2, y2),
                        'width': x2 - x1,
                        'height': y2 - y1
                    })
                except Exception as e:
                    try:
                        print(f"Error cropping line {line_id} from {image_path.name}: {e}")
                    except Exception:
                        print("Error cropping line (Unicode error in path)")
                    continue

        return lines_data

    def _process_single_page_wrapper(self, xml_path: Path) -> List[dict]:
        """Wrapper for multiprocessing."""
        return AltoParser._process_single_page_static(
            xml_path=xml_path,
            output_dir=self.output_dir,
            images_dir=self.images_dir,
            min_line_width=self.min_line_width,
            normalize_background=self.normalize_background,
            preserve_aspect_ratio=self.preserve_aspect_ratio,
            target_height=self.target_height
        )

    @staticmethod
    def _process_single_page_static(xml_path: Path, output_dir: Path, images_dir: Path,
                                     min_line_width: int,
                                     normalize_background: bool, preserve_aspect_ratio: bool,
                                     target_height: int) -> List[dict]:
        """Process a single page (static method for parallel processing)."""
        temp_parser = AltoParser(
            input_dir=xml_path.parent if xml_path.parent.name != "page" else xml_path.parent.parent,
            output_dir=output_dir,
            min_line_width=min_line_width,
            normalize_background=normalize_background,
            preserve_aspect_ratio=preserve_aspect_ratio,
            target_height=target_height,
            num_workers=1
        )
        temp_parser.images_dir = images_dir

        image_extensions = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.tif', '.TIF', '.tiff', '.TIFF']
        image_path = None

        # Optional: read image filename from ALTO Description/sourceImageInformation/fileName
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            fn_elem = root.find('.//alto:fileName', temp_parser.NS)
            if fn_elem is not None and fn_elem.text:
                alt_name = Path(fn_elem.text.strip()).stem
                for ext in image_extensions:
                    p = xml_path.parent / (alt_name + ext)
                    if p.exists():
                        image_path = p
                        break
        except Exception:
            pass

        if image_path is None:
            for ext in image_extensions:
                potential_path = xml_path.with_suffix(ext)
                if potential_path.exists():
                    image_path = potential_path
                    break

        if image_path is None and xml_path.parent.name == "page":
            for ext in image_extensions:
                potential_path = xml_path.parent.parent / (xml_path.stem + ext)
                if potential_path.exists():
                    image_path = potential_path
                    break

        if image_path is None:
            print(f"Warning: No image found for {xml_path.name}")
            return []

        return temp_parser.extract_lines_from_page(xml_path, image_path)

    def process_all(self) -> pd.DataFrame:
        """Process all ALTO XML files in the input directory and all subdirectories using parallel processing."""
        # Recursively find all *.xml (parent and subdirs, e.g. input_dir/mss-001/page-000.xml)
        xml_files = list(self.input_dir.rglob("*.xml"))
        xml_files = [f for f in xml_files if f.stem not in ['metadata', 'mets']]
        # Sort for stable ordering
        xml_files.sort(key=lambda p: (p.relative_to(self.input_dir),))

        if not xml_files:
            print(f"No ALTO XML files found in {self.input_dir} (searched recursively)")
            return pd.DataFrame()

        print(f"Found {len(xml_files)} ALTO XML files")
        print(f"Using {self.num_workers} CPU cores for parallel processing")
        print("Crops: polygon mask (from Shape/Polygon)")

        temp_csv_path = self.output_dir / "temp_lines.csv"
        csv_file = open(temp_csv_path, 'w', encoding='utf-8', newline='')
        csv_writer = csv.writer(csv_file)

        total_lines = 0
        total_width = 0
        total_height = 0
        unique_pages = set()

        if self.num_workers > 1:
            with Pool(processes=self.num_workers) as pool:
                chunksize = max(1, len(xml_files) // (self.num_workers * 4))
                for lines in tqdm(
                    pool.imap(self._process_single_page_wrapper, xml_files, chunksize=chunksize),
                    total=len(xml_files),
                    desc="Processing pages"
                ):
                    for line in lines:
                        csv_writer.writerow([line['image_path'], line['text'], line['page'], line['width'], line['height']])
                        total_lines += 1
                        total_width += line['width']
                        total_height += line['height']
                        unique_pages.add(line['page'])
        else:
            for xml_path in tqdm(xml_files, desc="Processing pages"):
                lines = self._process_single_page_wrapper(xml_path)
                for line in lines:
                    csv_writer.writerow([line['image_path'], line['text'], line['page'], line['width'], line['height']])
                    total_lines += 1
                    total_width += line['width']
                    total_height += line['height']
                    unique_pages.add(line['page'])

        csv_file.close()

        if total_lines == 0:
            print("No lines extracted!")
            temp_csv_path.unlink(missing_ok=True)
            return pd.DataFrame()

        df = pd.read_csv(temp_csv_path, names=['image_path', 'text', 'page', 'width', 'height'], encoding='utf-8')
        temp_csv_path.unlink()

        print(f"\nExtracted {total_lines} text lines from {len(unique_pages)} pages")
        print(f"Average line width: {total_width/total_lines:.1f}px")
        print(f"Average line height: {total_height/total_lines:.1f}px")
        return df

    def save_dataset(self, df: pd.DataFrame, train_ratio: float = 0.8):
        """Save dataset in TrOCR-compatible format."""
        if df.empty:
            return

        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        split_idx = int(len(df) * train_ratio)
        train_df = df[:split_idx]
        val_df = df[split_idx:]

        train_csv_path = self.output_dir / "train.csv"
        val_csv_path = self.output_dir / "val.csv"

        train_df[['image_path', 'text']].to_csv(
            train_csv_path, index=False, header=False, encoding='utf-8'
        )
        val_df[['image_path', 'text']].to_csv(
            val_csv_path, index=False, header=False, encoding='utf-8'
        )

        metadata = {
            'num_train': len(train_df),
            'num_val': len(val_df),
            'total_lines': len(df),
            'avg_line_width': float(df['width'].mean()),
            'avg_line_height': float(df['height'].mean()),
            'pages_processed': df['page'].nunique(),
            'background_normalized': self.normalize_background,
            'preserve_aspect_ratio': self.preserve_aspect_ratio,
            'target_height': self.target_height if self.preserve_aspect_ratio else None,
            'source_format': 'ALTO',
            'crop_mode': 'polygon'
        }

        metadata_path = self.output_dir / "dataset_info.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        print(f"\nDataset saved:")
        print(f"  Train: {len(train_df)} lines -> {train_csv_path}")
        print(f"  Val:   {len(val_df)} lines -> {val_csv_path}")
        print(f"  Metadata: {metadata_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Parse ALTO XML exports for TrOCR training (polygon crops)"
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help='Directory containing ALTO XML and image files'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory for processed dataset'
    )
    parser.add_argument(
        '--min_line_width',
        type=int,
        default=20,
        help='Minimum line width in pixels (default: 20)'
    )
    parser.add_argument(
        '--train_ratio',
        type=float,
        default=0.8,
        help='Ratio of training data (default: 0.8)'
    )
    parser.add_argument(
        '--normalize-background',
        action='store_true',
        help='Normalize image backgrounds to light gray'
    )
    parser.add_argument(
        '--preserve-aspect-ratio',
        action='store_true',
        help='Resize to target height while preserving aspect ratio (recommended for TrOCR)'
    )
    parser.add_argument(
        '--target-height',
        type=int,
        default=128,
        help='Target height for aspect-ratio-preserving resize (default: 128px)'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=None,
        help=f'Number of CPU cores for parallel processing (default: auto, {cpu_count()} available)'
    )

    args = parser.parse_args()

    print("=" * 60)
    print("ALTO XML Parser for TrOCR (polygon crops)")
    print("=" * 60)
    print(f"Input directory:         {args.input_dir}")
    print(f"Output directory:        {args.output_dir}")
    print(f"Crop mode:               polygon (from Shape/Polygon)")
    print(f"Background normalize:    {'Enabled' if args.normalize_background else 'Disabled'}")
    print(f"Preserve aspect ratio:   {'Enabled' if args.preserve_aspect_ratio else 'Disabled'}")
    if args.preserve_aspect_ratio:
        print(f"  Target height:         {args.target_height}px")
    print("=" * 60)

    parser_obj = AltoParser(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        min_line_width=args.min_line_width,
        normalize_background=args.normalize_background,
        preserve_aspect_ratio=args.preserve_aspect_ratio,
        target_height=args.target_height,
        num_workers=args.num_workers
    )

    df = parser_obj.process_all()

    if not df.empty:
        parser_obj.save_dataset(df, train_ratio=args.train_ratio)
        print("\n[SUCCESS] Dataset creation complete!")
    else:
        print("\n[FAILED] Failed to create dataset")


if __name__ == '__main__':
    main()
