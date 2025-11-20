# CRITICAL: Virtual Environment Requirement

## Always Use htr_gui Virtual Environment

**MANDATORY FOR ALL PYTHON SCRIPTS**:
```bash
source /home/achimrabus/htr_gui/dhlab-slavistik/htr_gui/bin/activate
```

**CORRECT PATH**: `/home/achimrabus/htr_gui/dhlab-slavistik/htr_gui/bin/activate`

## Why This Matters

- All dependencies (PyTorch, PIL, lxml, etc.) are installed in this venv
- System Python may not have required packages
- Different Python versions may cause compatibility issues

## Usage Patterns

### Direct Script Execution
```bash
# WRONG - will fail with missing dependencies
python3 script.py

# CORRECT - activate venv first
source /home/achimrabus/htr_gui/htr_gui/bin/activate
python3 script.py
```

### Shell Scripts
Always add at the top of any shell script:
```bash
#!/bin/bash
# CRITICAL: Always activate htr_gui virtual environment
source /home/achimrabus/htr_gui/htr_gui/bin/activate

# ... rest of script
```

### Long-Running Processes (Preprocessing, Training)
Use `nohup` to prevent disconnection from stopping the process:
```bash
#!/bin/bash
source /home/achimrabus/htr_gui/htr_gui/bin/activate

nohup python3 training_script.py > training.log 2>&1 &
echo "Process started with PID: $!"
```

## Current Scripts Updated

All Prosta Mova scripts now include venv activation:
- ✅ `preprocess_prosta_mova.sh` - includes venv activation
- ✅ `run_preprocess_prosta_mova.sh` - nohup wrapper with venv
- ✅ `run_pylaia_prosta_mova_training.sh` - nohup wrapper with venv
- ✅ `start_pylaia_prosta_mova_training.py` - run via shell wrapper

## Monitoring Long-Running Processes

```bash
# Start preprocessing in background
./run_preprocess_prosta_mova.sh

# Monitor progress
tail -f preprocess_prosta_mova.log

# Start training in background (after preprocessing completes)
./run_pylaia_prosta_mova_training.sh

# Monitor training
tail -f training_prosta_mova.log
```

## Reference Scripts

See successful examples:
- Church Slavonic training: Used venv + nohup
- Ukrainian PyLaia training: Used venv + nohup
- Glagolitic PyLaia training: Used venv + nohup

**NEVER forget to activate the venv - it will save hours of debugging!**
