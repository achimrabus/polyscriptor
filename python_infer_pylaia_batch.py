"""
Batch PyLaia inference for multiple page images.

Usage:
    python infer_pylaia_batch.py --input_dir pages/ --output_dir transcriptions/
"""

import argparse
from pathlib import Path
from tqdm import tqdm
import logging

from infer_pylaia import PyLaiaInference, LineSegmenter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Batch PyLaia inference")
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing page images')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory for output text files')
    parser.add_argument('--model', type=str, default='models/pylaia_efendiev/best_model.pt',
                        help='Path to trained model checkpoint')
    parser.add_argument('--visualize', action='store_true',
                        help='Save segmentation visualizations')
    parser.add_argument('--extensions', nargs='+', default=['.jpg', '.jpeg', '.png'],
                        help='Image file extensions to process')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all image files
    image_files = []
    for ext in args.extensions:
        image_files.extend(input_dir.glob(f'*{ext}'))
        image_files.extend(input_dir.glob(f'*{ext.upper()}'))
    
    image_files = sorted(set(image_files))
    logger.info(f"Found {len(image_files)} images in {input_dir}")
    
    if len(image_files) == 0:
        logger.error(f"No images found with extensions {args.extensions}")
        return
    
    # Initialize models
    segmenter = LineSegmenter()
    inference = PyLaiaInference(model_path=args.model)
    
    # Process each image
    for image_path in tqdm(image_files, desc="Processing pages"):
        try:
            # Transcribe
            transcriptions = inference.transcribe_page(
                str(image_path),
                segmenter=segmenter,
                visualize_segmentation=args.visualize
            )
            
            # Save
            output_path = output_dir / image_path.with_suffix('.txt').name
            with open(output_path, 'w', encoding='utf-8') as f:
                for line in transcriptions:
                    f.write(line + '\n')
            
            logger.info(f"✓ {image_path.name}: {len(transcriptions)} lines")
            
        except Exception as e:
            logger.error(f"✗ {image_path.name}: {e}")
    
    logger.info(f"\nBatch processing complete!")
    logger.info(f"Transcriptions saved to {output_dir}")


if __name__ == '__main__':
    main()