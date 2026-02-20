# Usage: python convert_to_pylaia_format.py --input_train_csv output_from_transkribus_parser\train.csv --input_val_csv output_from_transkribus_parser\val.csv --output_dir output_dir\ --train_img_root output_from_transkribus_parser\ --val_img_root output_from_transkribus_parse\ --height 96 --process_images_from train,val
# --process_images_from must contain train or val or all together, but with no whitespaces inbetween

import shutil
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


def norm_path(path: str) -> str:
    """
    Remove directories and file extension:
    line_images/Akh'ka_r1l9.png -> Akh'ka_r1l9
    """
    return Path(path).stem


def tokenize_text(text: str) -> str:
    """Character-level tokenization for PyLaia"""
    tokens = []
    for ch in text:
        if ch == " ":
            tokens.append("<SPACE>")
        else:
            tokens.append(ch)
    return " ".join(tokens)


def write_pylaia_txts(csv_path: str,
                      out_dir: Path,
                      split: str,
                      char_set: set):

    df = pd.read_csv(csv_path, names=["image_path", "text"], header=None, encoding="utf-8")

    ids_f = out_dir / f"{split}_ids.txt"
    text_f = out_dir / f"{split}_text.txt"
    tok_f = out_dir / f"{split}.txt"

    with open(ids_f, "w", encoding="utf-8") as f_ids, \
         open(text_f, "w", encoding="utf-8") as f_text, \
         open(tok_f, "w", encoding="utf-8") as f_tok:

        for _, row in df.iterrows():
            img_id = norm_path(row["image_path"])
            raw = str(row["text"]).strip()
            if not raw:
                continue

            char_set.update(raw)

            f_ids.write(f"{img_id}\n")
            f_text.write(f"{img_id} {raw}\n")
            f_tok.write(f"{img_id} {tokenize_text(raw)}\n")


def process_images(root_dir: Path,
                   output_dir: Path,
                   df,
                   do_process: bool,
                   grayscale: bool,
                   normalize_images: bool,
                   target_height: int):
    global failed
    for _, row in tqdm(df.iterrows(), total=len(df)):
        src = root_dir / Path(row["image_path"]).name
        if not src.exists():
            print("Missing image:", src)
            failed.append(src)
            continue

        dst = output_dir / src.name

        if not do_process:
            shutil.copy2(src, dst)
            continue

        img = Image.open(src)

        if grayscale:
            img = img.convert("L")
        else:
            img = img.convert("RGB")

        if normalize_images:
            img = normalize_height(img, target_height)

        img.save(dst, "PNG")


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
    train_img_path = Path(train_img_root)
    val_img_path = Path(val_img_root)
    train_df = pd.read_csv(train_csv_path, names=['image_path', 'text'], header=None, encoding='utf-8-sig')
    val_df = pd.read_csv(val_csv_path, names=['image_path', 'text'], header=None, encoding='utf-8-sig')
    char_set: Set[str] = set()

    new_train_dir = output_path / "pylaia_train"
    images_dir_train = new_train_dir / "images"
    images_dir_train.mkdir(parents=True, exist_ok=True)

    new_val_dir = output_path / "pylaia_val"
    images_dir_val = new_val_dir / "images"
    images_dir_val.mkdir(parents=True, exist_ok=True)

    write_pylaia_txts(train_csv_path, new_train_dir, "train", char_set)
    write_pylaia_txts(val_csv_path, new_val_dir, "val", char_set)

    process_images(train_img_path,
                   images_dir_train,
                   train_df,
                   "train" in process_images_from,
                   grayscale,
                   normalize_images,
                   target_height)

    process_images(val_img_path,
                   images_dir_val,
                   val_df,
                   "val" in process_images_from,
                   grayscale,
                   normalize_images,
                   target_height)

    symbols = ['<SPACE>']
    regular_chars = sorted(char_set - {' '})
    symbols.extend(regular_chars)
    for d in [new_train_dir, new_val_dir]:
        with open(d / "symbols.txt", "w", encoding="utf-8") as f:
            for s in symbols:
                f.write(s + "\n")

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
