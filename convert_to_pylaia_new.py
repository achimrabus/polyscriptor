# Usage: python convert_to_pylaia_new.py --input_train_csv output_from_transkribus_parser\train.csv --input_val_csv output_from_transkribus_parser\val.csv --output_dir output_dir\ --train_img_root output_from_transkribus_parser\ --val_img_root output_from_transkribus_parse\ --height 96 --process_images_from train,val
# --process_images_from must contain train or val or all together, but with no whitespaces inbetween

# convert to txt
import csv
import os
# get symbols
import pandas as pd
from typing import Set
from tqdm import tqdm
import argparse
from pathlib import Path
from PIL import Image

failed = []

def normalize_height(image: Image.Image, target_height: int = 64) -> Image.Image:
    """
    Normalize image height while preserving aspect ratio.
    
    Args:
        image: Input PIL Image
        target_height: Target height in pixels
        
    Returns:
        Resized image
    """
    width, height = image.size
    if height == 0:
        return image
    new_width = int(width * target_height / height)
    if new_width == 0:
        new_width = 1
    return image.resize((new_width, target_height), Image.Resampling.LANCZOS)


def convert_csv_to_txt(csv_path: str, image_dir: str):
    '''
    Copy filename,text from csv to a txt file of existing images only.
    
    csv_path: path to the csv
    image_dir: path to the image directory
    output_dir: is image_dir
    '''
    image_files = set(os.listdir(image_dir))
    with open(csv_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        with open(os.path.join(image_dir, "lines.txt"), 'w', encoding='utf-8') as t:
            for row in reader:
                filename = row[0]
                if filename in image_files:
                    t.write(','.join(row) + '\n')
                else:
                    print(filename, "does not exist in", image_dir)


def process_images(root_dir: Path, output_dir: Path, df, grayscale: bool, normalize_images: bool, target_height: int):
    global failed
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        img_path = Path(root_dir) / row['image_path']
        if not img_path.exists():
            print(f"{idx}: Image not found: {img_path}")
            failed.append(img_path)
            continue

        # Load image
        img = Image.open(img_path)

        # Convert to grayscale if requested
        if grayscale:
            img = img.convert('L')
        else:
            img = img.convert('RGB')

        # Normalize height if requested
        if normalize_images:
            img = normalize_height(img, target_height)

        output_img_path = output_dir / img_path.name

        # Save image
        img.save(output_img_path, "PNG")


def convert_dataset(
    train_csv_path: str,
    val_csv_path: str,
    output_dir: str,
    train_img_root: str,
    val_img_root: str,
    process_images_from: str,
    grayscale: bool = True,
    normalize_images: bool = True,
    target_height: int = 64
):
    '''
    Create symbols.txt, train.txt, val.txt
    and edit images (if needed) for pylaia training.
    
    train_csv_path: path to the train csv
    val_csv_path: path to the val csv
    output_dir: path to the output directory
    train_img_root: path to the train image directory
    val_img_root: path to the val image directory
    '''
    print("normalize: ", normalize_images)
    print("grayscale: ", grayscale)
    print("height: ", target_height)
    
    global failed

    output_path = Path(output_dir)
    train_img_root = Path(train_img_root)
    val_img_root = Path(val_img_root)
    train_df = pd.read_csv(train_csv_path, names=['image_path', 'text'], header=None, encoding='utf-8')
    val_df = pd.read_csv(val_csv_path, names=['image_path', 'text'], header=None, encoding='utf-8')
    char_set: Set[str] = set()
    # failed = []
    new_train_dir = output_path / "train"
    new_val_dir = output_path / "val"
    new_train_dir.mkdir(parents=True, exist_ok=True)
    new_val_dir.mkdir(parents=True, exist_ok=True)

    convert_csv_to_txt(train_csv_path, new_train_dir)
    convert_csv_to_txt(train_csv_path, new_val_dir)

    for idx, row in tqdm(train_df.iterrows(), total=len(train_df)):
        # Get ground truth text
        text = str(row['text']).strip()
        if not text:
            failed.append(f"{idx}: Empty text")
            continue
        # Update character set (preserve spaces)
        char_set.update(text)

        # Manipulate images only if needed
        if process_images_from == "train":
            process_images(train_img_root, new_train_dir, train_df, grayscale, normalize_images, target_height)

        elif process_images_from == "val":
            process_images(val_img_root, new_val_dir, val_df, grayscale, normalize_images, target_height)
        
        elif "train" in process_images_from and "val" in process_images_from:
            process_images(train_img_root, new_train_dir, train_df, grayscale, normalize_images, target_height)
            process_images(val_img_root, new_val_dir, val_df, grayscale, normalize_images, target_height)
        else:
            # In case no image processing needed
            # TODO: then save symbols and txt in the train|val_img_root?
            continue

    symbols = ['<SPACE>']
    regular_chars = sorted(char_set - {' '})
    symbols.extend(regular_chars)
    symbols_file_train = new_train_dir / "symbols.txt"
    with open(symbols_file_train, 'w', encoding='utf-8') as f:
        for symbol in symbols:
            f.write(f"{symbol}\n")
    symbols_file_val = new_val_dir / "symbols.txt"
    with open(symbols_file_val, 'w', encoding='utf-8') as f:
        for symbol in symbols:
            f.write(f"{symbol}\n")

    summary_file = output_path / "conversion_summary.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f"PyLaia Dataset Conversion Summary\n")
        f.write(f"=" * 60 + "\n\n")
        f.write(f"Train CSV: {train_csv_path}\n")
        f.write(f"Val CSV: {val_csv_path}\n")
        f.write(f"Output directory: {output_dir}\n")
        f.write(f"Target height: {target_height}px\n")
        f.write(f"Grayscale: {grayscale}\n")
        f.write(f"Normalize heights: {normalize_images}\n\n")
        f.write(f"Converted train samples: {len(train_df)}\n")
        f.write(f"Converted val samples: {len(val_df)}\n")
        f.write(f"Failed samples: {len(failed)}\n")
        f.write(f"Vocabulary size: {len(symbols)} characters\n\n")
        f.write(f"Files created:\n")
        f.write(f"  - lines.txt (image.png,text)\n")
        f.write(f"  - symbols.txt (vocabulary)\n")
    
    failed = []

def main():
    parser = argparse.ArgumentParser(
        description="Convert TrOCR dataset to PyLaia format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=
        """
        Examples:
        # Convert training set
        python convert_to_pylaia_new.py --input_train_csv \train\train.csv --input_val_csv val\val.csv --output_dir \output_dir --train_img_root \train --val_img_root \val
        
        # Convert with custom height and keep color
        python convert_to_pylaia_new.py --input_train_csv \train\train.csv --input_val_csv val\val.csv --output_dir \output_dir --train_img_root \train --val_img_root \val --height 96 --no-grayscale
        """
    )

    parser.add_argument(
        '--input_train_csv',
        type=str,
        required=True,
        help='Input train CSV file (image_path,text format)'
    )

    parser.add_argument(
        '--input_val_csv',
        type=str,
        required=True,
        help='Input val CSV file (image_path,text format)'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory for PyLaia dataset'
    )

    parser.add_argument(
        '--train_img_root',
        type=str,
        default='/train',
        help='Root directory containing train line images'
    )

    parser.add_argument(
        '--val_img_root',
        type=str,
        default='/train',
        help='Root directory containing val line images'
    )

    parser.add_argument(
        '--height',
        type=int,
        default=64,
        help='Target image height in pixels (default: 64)'
    )

    parser.add_argument(
        '--no_normalize',
        action='store_true',
        help='Do not normalize image heights (keep original sizes)'
    )

    parser.add_argument(
        '--no_grayscale',
        action='store_true',
        help='Keep RGB images instead of converting to grayscale'
    )
    
    parser.add_argument(
        '--process_images_from',
        type=str,
        default=None,
        help='Choose whether images from this dataset shall be edited, no whitespaces between train,val'
    )

    args = parser.parse_args()

    convert_dataset(
        train_csv_path=args.input_train_csv,
        val_csv_path=args.input_val_csv,
        output_dir=args.output_dir,
        train_img_root=args.train_img_root,
        val_img_root=args.val_img_root,
        grayscale=not args.no_grayscale,
        normalize_images=not args.no_normalize,
        target_height=args.height,
        process_images_from=args.process_images_from
    )

if __name__ == '__main__':
    main()
