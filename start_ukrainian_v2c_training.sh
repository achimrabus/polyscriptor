#!/bin/bash
#
# Start Ukrainian V2c PyLaia training with nohup
#
# This script will:
# 1. Check if data has been converted to PyLaia format
# 2. Start training in background with nohup
# 3. Monitor training progress with tail -f
#
# Usage:
#   bash start_ukrainian_v2c_training.sh
#
# To stop training:
#   Press Ctrl+C to stop monitoring (training continues in background)
#   To kill training: ps aux | grep train_pylaia_ukrainian_v2c.py
#                     kill <PID>
#

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Ukrainian V2c PyLaia Training Launcher${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Check if virtual environment is activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo -e "${YELLOW}⚠️  Virtual environment not activated${NC}"
    echo "Activating htr_gui virtual environment..."
    source /home/achimrabus/htr_gui/dhlab-slavistik/htr_gui/bin/activate
    echo -e "${GREEN}✅ Virtual environment activated${NC}"
fi

# Check if data directory exists
DATA_DIR="/home/achimrabus/htr_gui/dhlab-slavistik/data/pylaia_ukrainian_v2c_combined"
if [ ! -d "$DATA_DIR" ]; then
    echo -e "${RED}❌ Error: Data directory not found: $DATA_DIR${NC}"
    echo ""
    echo "Please convert the data first:"
    echo "  python3 convert_ukrainian_v2c_to_pylaia.py"
    exit 1
fi

# Check if required files exist
if [ ! -f "$DATA_DIR/lines.txt" ] || [ ! -f "$DATA_DIR/lines_val.txt" ] || [ ! -f "$DATA_DIR/syms.txt" ]; then
    echo -e "${RED}❌ Error: Missing required files in $DATA_DIR${NC}"
    echo ""
    echo "Please convert the data first:"
    echo "  python3 convert_ukrainian_v2c_to_pylaia.py"
    exit 1
fi

echo -e "${GREEN}✅ Data directory found: $DATA_DIR${NC}"

# Check if training is already running
if pgrep -f "train_pylaia_ukrainian_v2c.py" > /dev/null; then
    echo -e "${YELLOW}⚠️  Training appears to be already running!${NC}"
    echo ""
    echo "Running processes:"
    ps aux | grep train_pylaia_ukrainian_v2c.py | grep -v grep
    echo ""
    read -p "Do you want to start another training session? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 1
    fi
fi

# Check GPU availability
echo ""
echo "Checking GPU availability..."
if nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader
    echo -e "${GREEN}✅ GPU detected${NC}"
else
    echo -e "${YELLOW}⚠️  nvidia-smi not available. Training may use CPU (very slow).${NC}"
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Starting Training${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Command: nohup python3 train_pylaia_ukrainian_v2c.py > training_ukrainian_v2c.log 2>&1 &"
echo ""
echo "Training will run in background. Logs will be saved to:"
echo "  - training_ukrainian_v2c.log (main log file)"
echo ""
echo "Press Ctrl+C to stop monitoring (training continues in background)"
echo ""

# Start training with nohup
nohup python3 train_pylaia_ukrainian_v2c.py > training_ukrainian_v2c.log 2>&1 &
TRAINING_PID=$!

echo -e "${GREEN}✅ Training started with PID: $TRAINING_PID${NC}"
echo ""
echo "To check training status:"
echo "  tail -f training_ukrainian_v2c.log"
echo ""
echo "To stop training:"
echo "  kill $TRAINING_PID"
echo ""

# Wait a moment to let training start
sleep 2

# Check if process is still running
if ! ps -p $TRAINING_PID > /dev/null; then
    echo -e "${RED}❌ Error: Training process died immediately!${NC}"
    echo ""
    echo "Last 20 lines of log:"
    tail -20 training_ukrainian_v2c.log
    exit 1
fi

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Monitoring Training Log${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Monitor log file
tail -f training_ukrainian_v2c.log
