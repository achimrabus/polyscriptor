"""
Preprocess Ukrainian dataset for PyLaia training.

Converts from CSV format (train.csv/val.csv + line_images/) to PyLaia format
(images/, gt/, lines.txt, symbols.txt).

Usage:
    python preprocess_ukrainian_pylaia.py
"""

import csv
import shutil
from pathlib import Path
from typing import Set, List, Tuple
import logging
from tqdm import tqdm
from collections import Counter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def collect_symbols(texts: List[str]) -> List[str]:
    """
    Collect unique characters from all texts.
    
    Args:
        texts: List of text strings
    
    Returns:
        Sorted list of unique characters
    """
    char_counter = Counter()
    
    for text in texts:
        for char in text:
            if char == ' ':
                char_counter['<SPACE>'] += 1
            else:
                char_counter[char] += 1
    
    # Sort by frequency (most common first)
    symbols = [char for char, _ in char_counter.most_common()]
    
    logger.info(f"Found {len(symbols)} unique characters")
    logger.info(f"Most common: {symbols[:20]}")
    
    return symbols


def preprocess_dataset(
    input_dir: str,
    output_dir: str,
    csv_filename: str = None
) -> Tuple[int, int]:
    """
    Preprocess dataset from CSV format to PyLaia format.
    
    Args:
        input_dir: Directory with train.csv/val.csv and line_images/
        output_dir: Output directory for PyLaia format
        csv_filename: Name of CSV file (auto-detect if None)
    
    Returns:
        (num_samples, num_symbols)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Check input exists
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_path}")
    
    logger.info(f"Processing {input_dir} -> {output_dir}")
    
    # Find CSV file
    if csv_filename is None:
        # Auto-detect: prefer val.csv for validation, train.csv for training
        if 'val' in input_dir.lower():
            csv_filename = 'val.csv'
        else:
            csv_filename = 'train.csv'
    
    csv_path = input_path / csv_filename
    if not csv_path.exists():
        # Try other options
        csv_candidates = list(input_path.glob('*.csv'))
        if csv_candidates:
            csv_path = csv_candidates[0]
            logger.info(f"Using CSV file: {csv_path}")
        else:
            raise FileNotFoundError(f"No CSV file found in {input_path}")
    
    logger.info(f"Loading data from {csv_path}")
    
    # Find images directory
    images_dir = input_path / "line_images"
    if not images_dir.exists():
        images_dir = input_path / "images"
        if not images_dir.exists():
            raise FileNotFoundError(f"No images directory found in {input_path}")
    
    logger.info(f"Images directory: {images_dir}")
    
    # Create output directories
    output_path.mkdir(parents=True, exist_ok=True)
    output_images_dir = output_path / "images"
    output_gt_dir = output_path / "gt"
    output_images_dir.mkdir(exist_ok=True)
    output_gt_dir.mkdir(exist_ok=True)
    
    # Load CSV data
    # Expected format: file_name, text (or similar columns)
    rows = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    logger.info(f"Loaded {len(rows)} rows from CSV")
    
    if len(rows) == 0:
        raise ValueError(f"CSV file is empty: {csv_path}")
    
    # Check column names
    columns = rows[0].keys()
    logger.info(f"CSV columns: {list(columns)}")
    
    # Determine column names for filename and text
    filename_col = None
    text_col = None
    
    for col in columns:
        col_lower = col.lower()
        if 'file' in col_lower or 'image' in col_lower or 'name' in col_lower:
            filename_col = col
        if 'text' in col_lower or 'transcription' in col_lower or 'label' in col_lower or 'gt' in col_lower:
            text_col = col
    
    if filename_col is None:
        # Use first column as filename
        filename_col = list(columns)[0]
        logger.warning(f"Could not auto-detect filename column, using: {filename_col}")
    
    if text_col is None:
        # Use second column as text
        if len(columns) > 1:
            text_col = list(columns)[1]
        else:
            raise ValueError("Could not find text column in CSV")
        logger.warning(f"Could not auto-detect text column, using: {text_col}")
    
    logger.info(f"Using columns: filename='{filename_col}', text='{text_col}'")
    
    # Process each sample
    sample_ids = []
    all_texts = []
    skipped = 0
    
    for row in tqdm(rows, desc="Processing samples"):
        filename = row.get(filename_col, '').strip()
        text = row.get(text_col, '').strip()
        
        if not filename:
            logger.warning(f"Skipping row with empty filename")
            skipped += 1
            continue
        
        if not text:
            logger.warning(f"Skipping {filename}: empty text")
            skipped += 1
            continue
        
        # Get sample ID (remove extension)
        sample_id = Path(filename).stem
        
        # Check if image exists
        image_found = False
        for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']:
            src_image = images_dir / f"{sample_id}{ext}"
            if not src_image.exists():
                # Try with original filename
                src_image = images_dir / filename
            
            if src_image.exists():
                # Copy image
                dst_image = output_images_dir / f"{sample_id}.png"
                shutil.copy2(src_image, dst_image)
                image_found = True
                break
        
        if not image_found:
            logger.warning(f"Skipping {sample_id}: image not found")
            skipped += 1
            continue
        
        # Write ground truth
        gt_file = output_gt_dir / f"{sample_id}.txt"
        with open(gt_file, 'w', encoding='utf-8') as f:
            f.write(text)
        
        sample_ids.append(sample_id)
        all_texts.append(text)
    
    logger.info(f"Processed {len(sample_ids)} samples")
    if skipped > 0:
        logger.warning(f"Skipped {skipped} samples")
    
    if len(sample_ids) == 0:
        raise ValueError(f"No valid samples found in {input_dir}")
    
    # Write lines.txt
    lines_file = output_path / "lines.txt"
    with open(lines_file, 'w', encoding='utf-8') as f:
        for sample_id in sample_ids:
            f.write(f"{sample_id}\n")
    
    logger.info(f"Wrote {len(sample_ids)} sample IDs to {lines_file}")
    
    # Collect and write symbols
    symbols = collect_symbols(all_texts)
    symbols_file = output_path / "symbols.txt"
    with open(symbols_file, 'w', encoding='utf-8') as f:
        for symbol in symbols:
            f.write(f"{symbol}\n")
    
    logger.info(f"Wrote {len(symbols)} symbols to {symbols_file}")
    
    # Print statistics
    logger.info("\n" + "="*60)
    logger.info("PREPROCESSING SUMMARY")
    logger.info("="*60)
    logger.info(f"Input directory:  {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"CSV file:         {csv_path}")
    logger.info(f"Total samples:    {len(sample_ids)}")
    logger.info(f"Skipped samples:  {skipped}")
    logger.info(f"Vocabulary size:  {len(symbols)}")
    logger.info(f"Images saved to:  {output_images_dir}")
    logger.info(f"GT saved to:      {output_gt_dir}")
    logger.info(f"Lines list:       {lines_file}")
    logger.info(f"Vocabulary:       {symbols_file}")
    
    # Sample texts
    logger.info("\nSample texts:")
    for i, text in enumerate(all_texts[:5], 1):
        logger.info(f"  {i}: {text[:80]}{'...' if len(text) > 80 else ''}")
    
    # Character distribution
    char_counts = Counter()
    for text in all_texts:
        char_counts.update(text)
    
    logger.info("\nTop 20 characters:")
    for char, count in char_counts.most_common(20):
        display_char = '<SPACE>' if char == ' ' else char
        logger.info(f"  '{display_char}': {count}")
    
    return len(sample_ids), len(symbols)


def main():
    """Preprocess both train and validation datasets."""
    
    # Configuration
    datasets = [
        {
            'input': 'data/ukrainian_train_aspect_ratio',
            'output': 'data/pylaia_ukrainian_train',
            'csv': 'train.csv'
        },
        {
            'input': 'data/ukrainian_val_aspect_ratio',
            'output': 'data/pylaia_ukrainian_val',
            'csv': 'val.csv'
        }
    ]
    
    total_train = 0
    total_val = 0
    all_symbols = set()
    
    # Process each dataset
    for i, dataset in enumerate(datasets, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"DATASET {i}/{len(datasets)}")
        logger.info(f"{'='*60}\n")
        
        try:
            num_samples, num_symbols = preprocess_dataset(
                input_dir=dataset['input'],
                output_dir=dataset['output'],
                csv_filename=dataset['csv']
            )
            
            # Load symbols
            symbols_file = Path(dataset['output']) / 'symbols.txt'
            with open(symbols_file, 'r', encoding='utf-8') as f:
                symbols = set(line.strip() for line in f)
            all_symbols.update(symbols)
            
            if 'train' in dataset['input'].lower():
                total_train = num_samples
            else:
                total_val = num_samples
                
        except Exception as e:
            logger.error(f"Failed to process {dataset['input']}: {e}")
            raise
    
    # Create unified vocabulary from both train and val
    logger.info("\n" + "="*60)
    logger.info("CREATING UNIFIED VOCABULARY")
    logger.info("="*60)
    
    # Sort symbols by category: letters, digits, punctuation, special
    def char_category(char):
        if char == '<SPACE>':
            return (0, char)
        elif char.isalpha():
            return (1, char.lower())
        elif char.isdigit():
            return (2, char)
        else:
            return (3, char)
    
    unified_symbols = sorted(all_symbols, key=char_category)
    
    logger.info(f"Total unique characters across both datasets: {len(unified_symbols)}")
    
    # Write unified vocabulary to both datasets
    for dataset in datasets:
        symbols_file = Path(dataset['output']) / 'symbols.txt'
        with open(symbols_file, 'w', encoding='utf-8') as f:
            for symbol in unified_symbols:
                f.write(f"{symbol}\n")
        logger.info(f"Updated {symbols_file} with unified vocabulary")
    
    # Final summary
    logger.info("\n" + "="*60)
    logger.info("FINAL SUMMARY")
    logger.info("="*60)
    logger.info(f"Training samples:   {total_train}")
    logger.info(f"Validation samples: {total_val}")
    logger.info(f"Total samples:      {total_train + total_val}")
    logger.info(f"Vocabulary size:    {len(unified_symbols)}")
    logger.info("\nReady for PyLaia training!")
    logger.info("Next step: python train_pylaia_ukrainian.py")


if __name__ == '__main__':
    main()