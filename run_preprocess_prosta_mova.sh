#!/bin/bash
# Wrapper to run Prosta Mova preprocessing with nohup
# Output will be logged to preprocess_prosta_mova.log

echo "Starting Prosta Mova preprocessing in background with nohup..."
echo "Log file: preprocess_prosta_mova.log"
echo ""

nohup ./preprocess_prosta_mova.sh > preprocess_prosta_mova.log 2>&1 &

PID=$!
echo "Process started with PID: $PID"
echo "Monitor progress with: tail -f preprocess_prosta_mova.log"
echo "Check if running with: ps aux | grep $PID"
