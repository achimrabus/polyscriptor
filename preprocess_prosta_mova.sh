#!/bin/bash
# Prosta Mova Preprocessing Script
# Parses Transkribus PAGE XML exports for PyLaia training
# Based on successful Church Slavonic preprocessing

set -e  # Exit on error

# CRITICAL: Always activate htr_gui virtual environment
source /home/achimrabus/htr_gui/dhlab-slavistik/htr_gui/bin/activate

echo "=========================================="
echo "PROSTA MOVA PREPROCESSING PIPELINE"
echo "=========================================="
echo ""

# Configuration
INPUT_TRAIN="/home/achimrabus/htr_gui/Prosta_Mova/Prosta_Mova_train"
INPUT_VAL="/home/achimrabus/htr_gui/Prosta_Mova/Prosta_Mova_val"
OUTPUT_TRAIN="./data/pylaia_prosta_mova_train"
OUTPUT_VAL="./data/pylaia_prosta_mova_val"
OUTPUT_MERGED="./data/pylaia_prosta_mova"
TARGET_HEIGHT=128
NUM_WORKERS=16

echo "Configuration:"
echo "  Training input: $INPUT_TRAIN"
echo "  Validation input: $INPUT_VAL"
echo "  Target height: ${TARGET_HEIGHT}px"
echo "  Workers: $NUM_WORKERS"
echo ""

# Step 1: Parse training set
echo "Step 1/4: Parsing training set..."
echo "----------------------------------------"
python transkribus_parser.py \
    --input_dir "$INPUT_TRAIN" \
    --output_dir "$OUTPUT_TRAIN" \
    --preserve-aspect-ratio \
    --target-height "$TARGET_HEIGHT" \
    --num-workers "$NUM_WORKERS"

echo ""
echo "Training set parsed successfully!"
echo ""

# Step 2: Parse validation set
echo "Step 2/4: Parsing validation set..."
echo "----------------------------------------"
python transkribus_parser.py \
    --input_dir "$INPUT_VAL" \
    --output_dir "$OUTPUT_VAL" \
    --preserve-aspect-ratio \
    --target-height "$TARGET_HEIGHT" \
    --num-workers "$NUM_WORKERS"

echo ""
echo "Validation set parsed successfully!"
echo ""

# Step 3: Convert CSV to PyLaia format
echo "Step 3/5: Converting CSV to PyLaia format..."
echo "----------------------------------------"

# Convert training set
echo "Converting training set..."
python convert_to_pylaia.py \
    --input_csv "$OUTPUT_TRAIN/train.csv" \
    --output_dir "${OUTPUT_MERGED}_train_tmp" \
    --data_root "$OUTPUT_TRAIN" \
    --target_height "$TARGET_HEIGHT" \
    --no-normalize \
    --no-grayscale

# Convert validation set
echo "Converting validation set..."
python convert_to_pylaia.py \
    --input_csv "$OUTPUT_VAL/train.csv" \
    --output_dir "${OUTPUT_MERGED}_val_tmp" \
    --data_root "$OUTPUT_VAL" \
    --target_height "$TARGET_HEIGHT" \
    --no-normalize \
    --no-grayscale

echo ""
echo "CSV conversion complete!"
echo ""

# Step 4: Merge PyLaia datasets
echo "Step 4/5: Merging PyLaia datasets..."
echo "----------------------------------------"

# Create merged directory
mkdir -p "$OUTPUT_MERGED/images"
mkdir -p "$OUTPUT_MERGED/gt"

# Copy training data
echo "Merging training data..."
rsync -a "${OUTPUT_MERGED}_train_tmp/images/" "$OUTPUT_MERGED/images/"
rsync -a "${OUTPUT_MERGED}_train_tmp/gt/" "$OUTPUT_MERGED/gt/"

# Copy validation data
echo "Merging validation data..."
rsync -a "${OUTPUT_MERGED}_val_tmp/images/" "$OUTPUT_MERGED/images/"
rsync -a "${OUTPUT_MERGED}_val_tmp/gt/" "$OUTPUT_MERGED/gt/"

# Copy list files
cp "${OUTPUT_MERGED}_train_tmp/lines.txt" "$OUTPUT_MERGED/lines.txt"
cp "${OUTPUT_MERGED}_val_tmp/lines.txt" "$OUTPUT_MERGED/lines_val.txt"

# Copy vocabulary (from training set)
cp "${OUTPUT_MERGED}_train_tmp/syms.txt" "$OUTPUT_MERGED/syms.txt"

# Clean up temporary directories
rm -rf "${OUTPUT_MERGED}_train_tmp"
rm -rf "${OUTPUT_MERGED}_val_tmp"

echo ""
echo "Dataset merge complete!"
echo ""

# Step 5: Verification
echo "Step 5/5: Verification..."
echo "----------------------------------------"

# Count lines
TRAIN_LINES=$(wc -l < "$OUTPUT_MERGED/lines.txt")
VAL_LINES=$(wc -l < "$OUTPUT_MERGED/lines_val.txt")
TOTAL_LINES=$((TRAIN_LINES + VAL_LINES))

echo "Training lines: $TRAIN_LINES"
echo "Validation lines: $VAL_LINES"
echo "Total lines: $TOTAL_LINES"
echo ""

# Check vocabulary format
echo "Vocabulary format (first 10 lines):"
head -10 "$OUTPUT_MERGED/syms.txt"
echo ""

# Check sample line image
SAMPLE_IMAGE=$(ls "$OUTPUT_MERGED/line_images/" | head -1)
if [ -n "$SAMPLE_IMAGE" ]; then
    echo "Sample line image: $SAMPLE_IMAGE"
    file "$OUTPUT_MERGED/line_images/$SAMPLE_IMAGE"
    echo ""
fi

# Check sample ground truth
SAMPLE_GT=$(ls "$OUTPUT_MERGED/gt/" | head -1)
if [ -n "$SAMPLE_GT" ]; then
    echo "Sample ground truth: $SAMPLE_GT"
    cat "$OUTPUT_MERGED/gt/$SAMPLE_GT"
    echo ""
fi

echo "=========================================="
echo "PREPROCESSING COMPLETE!"
echo "=========================================="
echo ""
echo "Output directory: $OUTPUT_MERGED"
echo "Ready for training with start_pylaia_prosta_mova_training.py"
echo ""
