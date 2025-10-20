@echo off
cd "c:\Users\Achim\Documents\TrOCR\dhlab-slavistik"
python optimized_training.py --config config_efendiev.yaml > training_output.log 2>&1
echo Exit code: %ERRORLEVEL% >> training_output.log
type training_output.log
