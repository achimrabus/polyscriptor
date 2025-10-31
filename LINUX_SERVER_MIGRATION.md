# Linux Server Migration Guide

This document provides comprehensive instructions for migrating the dhlab-slavistik project from Windows to a Linux server.

## Table of Contents
1. [Initial Setup](#initial-setup)
2. [Configuration Changes](#configuration-changes)
3. [Required Packages & Dependencies](#required-packages--dependencies)
4. [Model Files & API Keys](#model-files--api-keys)
5. [GUI Application Setup (X11)](#gui-application-setup-x11)
6. [Data Migration](#data-migration)
7. [Claude Code Conversations](#claude-code-conversations)
8. [Testing & Verification](#testing--verification)

---

## 1. Initial Setup

### Switch from GitLab to GitHub

```bash
# On the server
cd ~/dhlab-slavistik

# Check current remote
git remote -v

# Remove old remote (if GitLab)
git remote remove origin

# Add GitHub remote
git remote add origin https://github.com/YOUR_USERNAME/dhlab-slavistik.git

# Or with SSH (recommended)
git remote add origin git@github.com:YOUR_USERNAME/dhlab-slavistik.git

# Verify
git remote -v

# Pull latest from GitHub
git fetch origin
git branch --set-upstream-to=origin/ar-exp/claude ar-exp/claude
git pull
```

### Create Virtual Environments

```bash
cd ~/dhlab-slavistik

# Main environment (TrOCR, Qwen, GUIs)
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip

# PyLaia environment (separate due to dependency conflicts)
python3 -m venv venv_pylaia
source venv_pylaia/bin/activate
pip install --upgrade pip

# Party environment (separate due to dependency conflicts)
python3 -m venv venv_party
source venv_party/bin/activate
pip install --upgrade pip
```

---

## 2. Configuration Changes

### A. Remove Windows-Specific Code

**No WSL paths needed!** All paths are now native Linux paths.

**Changes in `transcription_gui_party.py`:**

```python
# OLD (Windows):
self.wsl_project_root = "/mnt/c/Users/Achim/Documents/TrOCR/dhlab-slavistik"

def _windows_to_wsl_path(self, windows_path: str) -> str:
    # ... conversion code ...

# NEW (Linux):
self.project_root = os.path.expanduser("~/dhlab-slavistik")

# No path conversion needed!
# Just use paths directly
```

**Update Party worker command:**

```python
# OLD (Windows with WSL):
cmd = (
    f"cd {wsl_image_dir} && "
    f"source {self.wsl_project_root}/venv_party_wsl/bin/activate && "
    f"party -d cuda:0 ocr ..."
)

result = subprocess.run(["wsl", "bash", "-c", cmd], ...)

# NEW (Linux):
cmd = (
    f"cd {image_dir} && "
    f"source {self.project_root}/venv_party/bin/activate && "
    f"party -d cuda:0 ocr ..."
)

result = subprocess.run(["bash", "-c", cmd], ...)
```

### B. Path Separators

**Windows uses `\`, Linux uses `/`**

Most Python code uses `pathlib.Path` which handles this automatically, but check any string paths:

```python
# Good (cross-platform)
from pathlib import Path
path = Path("data") / "images" / "file.png"

# Bad (Windows-specific)
path = "data\\images\\file.png"

# Good alternative
path = os.path.join("data", "images", "file.png")
```

### C. Line Endings

```bash
# Convert CRLF (Windows) to LF (Linux) if needed
dos2unix *.py
# Or:
find . -name "*.py" -exec dos2unix {} \;
```

---

## 3. Required Packages & Dependencies

### A. System Libraries (Ubuntu/Debian)

```bash
sudo apt update && sudo apt install -y \
    python3-venv \
    python3-pip \
    python3-dev \
    build-essential \
    git \
    curl \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libgtk-3-dev \
    dos2unix
```

### B. GPU Support (NVIDIA)

```bash
# Check NVIDIA driver
nvidia-smi

# Install CUDA Toolkit (if not already installed)
# Check CUDA version:
nvcc --version

# For CUDA 12.1:
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda-repo-ubuntu2204-12-1-local_12.1.0-530.30.02-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-12-1-local_12.1.0-530.30.02-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-12-1-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda
```

### C. X11 Support for GUIs

```bash
# Required for PyQt6 GUIs
sudo apt install -y \
    python3-pyqt6 \
    libxcb-xinerama0 \
    libxcb-cursor0 \
    libxkbcommon-x11-0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-shape0 \
    x11-apps \
    xauth

# Test X11 (requires X forwarding from client)
xclock
```

### D. Python Packages

#### Main Environment (venv)

```bash
source venv/bin/activate

# Core dependencies
pip install -r requirements.txt

# PyTorch with CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Additional packages for GUIs
pip install PyQt6 PyQt6-Qt6

# Qwen3-VL dependencies
pip install qwen-vl-utils transformers accelerate

# Commercial APIs (if using)
pip install anthropic openai google-generativeai

# Kraken for segmentation
pip install kraken

# Jupyter for interactive work
pip install jupyter ipywidgets matplotlib
```

#### PyLaia Environment

```bash
source venv_pylaia/bin/activate

pip install pylaia
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

#### Party Environment

```bash
source venv_party/bin/activate

# Install Party from git
cd ~/
git clone https://github.com/jahtz/party.git party_repo
cd party_repo
pip install -e .

# Or from local copy if you have party_repo in dhlab-slavistik:
cd ~/dhlab-slavistik/party_repo
pip install -e .

# Don't forget to apply the tokenizer fix!
# Edit party/tokenizer.py line 236:
# Change: id.to_bytes()
# To: id.to_bytes(1, 'big')
```

---

## 4. Model Files & API Keys

### A. Model Files to Transfer

These files are **NOT** in git and must be transferred manually:

#### TrOCR Models (Checkpoints)

```bash
# From Windows: C:\Users\Achim\Documents\TrOCR\dhlab-slavistik\models\
# To Linux: ~/dhlab-slavistik/models/

# Transfer via scp (from Windows):
scp -r "C:\Users\Achim\Documents\TrOCR\dhlab-slavistik\models" user@server:~/dhlab-slavistik/

# Or use VSCode's file explorer to drag & drop

# Expected models:
models/
├── ukrainian_model/
│   ├── checkpoint-3000/
│   └── training_config.yaml
├── efendiev_3_model/
│   └── checkpoint-*/
└── party_models/
    └── party_european_langs.safetensors
```

#### Qwen3-VL Models

Qwen models are typically downloaded from HuggingFace Hub automatically, but you can also transfer cached models:

```bash
# Option 1: Let it download on first use (recommended)
# The GUI will auto-download from HuggingFace

# Option 2: Transfer cached models
# From Windows: C:\Users\Achim\.cache\huggingface\hub\
# To Linux: ~/.cache/huggingface/hub/

# Expected models:
~/.cache/huggingface/hub/
├── models--Qwen--Qwen2-VL-2B-Instruct/
├── models--Qwen--Qwen2-VL-7B-Instruct/
└── models--Qwen--Qwen2-VL-8B-Instruct/
```

#### PyLaia Models

```bash
# Transfer PyLaia trained models
models/
└── pylaia_glagolitic/
    └── model
```

### B. API Keys (CRITICAL - Not in Git!)

**Create `.env` file** in project root:

```bash
cd ~/dhlab-slavistik
nano .env
```

Add your API keys:

```bash
# .env file
ANTHROPIC_API_KEY=your_anthropic_key_here
OPENAI_API_KEY=your_openai_key_here
GOOGLE_API_KEY=your_google_key_here
FREIBURG_API_KEY=your_freiburg_key_here
```

**Verify `.env` is in `.gitignore`:**

```bash
grep -q "^\.env$" .gitignore || echo ".env" >> .gitignore
```

**Load API keys in Python:**

```python
# In transcription_gui_plugin.py
import os
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# etc.
```

### C. Data Directories

Transfer your datasets:

```bash
# From Windows to Linux
scp -r "C:\Users\Achim\Documents\TrOCR\Ukrainian_Data" user@server:~/dhlab-slavistik/data/
scp -r "C:\Users\Achim\Documents\TrOCR\Glagolitic" user@server:~/dhlab-slavistik/data/

# Expected structure:
data/
├── ukrainian_train_aspect_ratio/
├── ukrainian_val_aspect_ratio/
├── pylaia_glagolitic/
└── pylaia_ukrainian_train/
```

---

## 5. GUI Application Setup (X11)

### Using MobaXterm from Windows

**On Windows (MobaXterm):**
1. Create SSH session to server
2. Check ✅ "X11-Forwarding" in Advanced SSH settings
3. Connect

**On Server:**
```bash
# Verify X11 forwarding
echo $DISPLAY
# Should show: localhost:10.0 or similar

# Test with xclock
xclock
# Clock should appear in MobaXterm window

# Run GUIs
source venv/bin/activate
python transcription_gui_qt.py
```

### Why Some Engines Are Missing in GUI

The GUI engines depend on installed packages:

| Engine | Required Package | Install Command |
|--------|-----------------|-----------------|
| TrOCR | transformers | `pip install transformers` (in venv) |
| Qwen3-VL | qwen-vl-utils | `pip install qwen-vl-utils transformers` |
| PyLaia | pylaia | Use venv_pylaia separately |
| Claude (Anthropic) | anthropic | `pip install anthropic` + API key in .env |
| ChatGPT (OpenAI) | openai | `pip install openai` + API key in .env |
| Gemini (Google) | google-generativeai | `pip install google-generativeai` + API key |
| Uni Freiburg | requests | `pip install requests` + API key in .env |

**To enable all engines:**

```bash
source venv/bin/activate

# Install commercial API packages
pip install anthropic openai google-generativeai requests

# Install Qwen
pip install qwen-vl-utils

# Create .env with API keys (see section 4.B)
```

---

## 6. Data Migration

### Transfer Data from Windows

**Method 1: SCP (Command Line)**

```bash
# From Windows PowerShell or Command Prompt
scp -r "C:\Users\Achim\Documents\TrOCR\Ukrainian_Data" user@server:~/dhlab-slavistik/data/
scp -r "C:\Users\Achim\Documents\TrOCR\dhlab-slavistik\models" user@server:~/dhlab-slavistik/
```

**Method 2: VSCode File Explorer**

1. Connect to server via Remote-SSH in VSCode
2. Open Explorer
3. Drag and drop folders from Windows to server

**Method 3: rsync (Faster for Large Data)**

```bash
# From Windows with WSL or Git Bash
rsync -avz --progress /mnt/c/Users/Achim/Documents/TrOCR/Ukrainian_Data user@server:~/dhlab-slavistik/data/
```

### Verify Data Integrity

```bash
# Check file counts
find data/ukrainian_train_aspect_ratio/line_images -type f | wc -l
find data/ukrainian_val_aspect_ratio/line_images -type f | wc -l

# Check model files
ls -lh models/ukrainian_model/checkpoint-3000/
ls -lh models/party_models/
```

---

## 7. Claude Code Conversations

### Can You Sync Claude Code Conversations?

**Short answer: No direct sync, but there are workarounds.**

Claude Code conversations are stored locally and **cannot be directly synced** between Windows and Linux. However:

### Option 1: Export Conversation Context (Recommended)

**On Windows (before switching):**

1. Save important context to markdown files:
   - `CLAUDE.md` (project instructions) ✅ Already committed
   - `PARTY_GUI_INTEGRATION_PLAN.md` ✅ Already committed
   - `LINUX_SERVER_MIGRATION.md` ✅ This file

2. Commit knowledge to git:
```bash
git add CLAUDE.md PARTY_GUI_INTEGRATION_PLAN.md LINUX_SERVER_MIGRATION.md
git commit -m "Add comprehensive documentation for server migration"
git push
```

**On Linux Server:**
```bash
git pull
# Claude can read these files to understand project context
```

### Option 2: Manual Context Transfer

Create a `CONVERSATION_SUMMARY.md` with key decisions:

```markdown
# Conversation Summary

## Key Decisions Made
- Party tokenizer fix: line 236, add (1, 'big') to to_bytes()
- PyLaia image height: 128px for proper pooling
- Aspect ratio preservation: Critical for TrOCR quality
- GUI architecture: Plugin-based with HTREngine base class

## Current Status
- Party OCR: ✅ Working (with tokenizer fix)
- TrOCR training: ✅ Working (optimized pipeline)
- PyLaia training: ⚠️ Needs cache clear on Linux
- Qwen3-VL: ✅ Working (8B model)

## Configuration Details
- CUDA version: 12.1
- PyTorch version: 2.x
- Multi-GPU training: DDP with 2x RTX 4090
```

### Option 3: Start Fresh with Context

On the Linux server, start a new Claude Code session and provide context:

```
"Hi Claude, I'm migrating the dhlab-slavistik project from Windows to Linux.
Please read:
- CLAUDE.md (project overview)
- LINUX_SERVER_MIGRATION.md (migration guide)
- PARTY_GUI_INTEGRATION_PLAN.md (Party integration details)

Key things to know:
1. Party tokenizer bug fixed (line 236)
2. No WSL needed on Linux
3. GUIs work via X11 forwarding
4. Multi-GPU training setup
"
```

### Conversation Export Workaround

**Before leaving Windows**, create a summary:

```bash
# Create a comprehensive summary
cat > PREVIOUS_SESSION_SUMMARY.md << 'EOF'
# Previous Session Summary

## Tasks Completed
1. ✅ Added Party OCR integration
2. ✅ Fixed Party tokenizer bug (Python 3.10+ compatibility)
3. ✅ Created PAGE XML exporter
4. ✅ Added PyLaia image resizing utility
5. ✅ Fixed GUI PAGE XML export
6. ✅ Created comprehensive documentation

## Current Issues
- PyLaia training: Height mismatch due to cached data
- Need to clear cache on fresh Linux environment

## Important Files Modified
- transcription_gui_party.py: Party PoC GUI
- page_xml_exporter.py: PAGE XML export utility
- party_repo/party/tokenizer.py: Bug fix (NOT committed, apply manually)
- CLAUDE.md: Updated with Party documentation

## Configuration Notes
- Party model: models/party_models/party_european_langs.safetensors
- Party language code: chu (Church Slavonic)
- PyLaia target height: 128px
- TrOCR aspect ratio preservation: CRITICAL
EOF

git add PREVIOUS_SESSION_SUMMARY.md
git commit -m "Add previous session summary for context transfer"
git push
```

---

## 8. Testing & Verification

### A. Environment Test

```bash
# Test main environment
source venv/bin/activate
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
python -c "import transformers; print('Transformers OK')"

# Test PyLaia environment
source venv_pylaia/bin/activate
python -c "import laia; print('PyLaia OK')"

# Test Party environment
source venv_party/bin/activate
python -c "import party; print('Party OK')"
```

### B. GUI Test (X11)

```bash
source venv/bin/activate

# Test X11 forwarding first
xclock

# Test GUI (requires X11 forwarding)
python transcription_gui_qt.py
```

### C. Training Test

```bash
source venv/bin/activate

# Quick training test (1 epoch)
python optimized_training.py --config config_efendiev.yaml --epochs 1
```

### D. Inference Test

```bash
# TrOCR inference
python inference_page.py \
    --image test_image.jpg \
    --checkpoint models/ukrainian_model/checkpoint-3000

# Party inference (native Linux, no WSL!)
cd ~/dhlab-slavistik
source venv_party/bin/activate
party -d cuda:0 ocr -i test.xml test_output.xml \
-mi models/party_models/party_european_langs.safetensors --language chu
```

---

## Quick Setup Script

Save this as `setup_linux_server.sh`:

```bash
#!/bin/bash
set -e

echo "=== Setting up dhlab-slavistik on Linux Server ==="

# 1. System packages
echo "Installing system packages..."
sudo apt update
sudo apt install -y python3-venv python3-pip build-essential \
    libgl1-mesa-glx python3-pyqt6 x11-apps git dos2unix

# 2. Create virtual environments
echo "Creating virtual environments..."
python3 -m venv venv
python3 -m venv venv_pylaia
python3 -m venv venv_party

# 3. Install main environment
echo "Installing main environment..."
source venv/bin/activate
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install PyQt6 anthropic openai google-generativeai qwen-vl-utils kraken
deactivate

# 4. Install PyLaia
echo "Installing PyLaia..."
source venv_pylaia/bin/activate
pip install --upgrade pip
pip install pylaia torch torchvision --index-url https://download.pytorch.org/whl/cu121
deactivate

# 5. Install Party
echo "Installing Party..."
source venv_party/bin/activate
pip install --upgrade pip
cd party_repo && pip install -e .
deactivate

# 6. Convert line endings
echo "Converting line endings..."
find . -name "*.py" -exec dos2unix {} \;

# 7. Test
echo "Testing installations..."
source venv/bin/activate
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

echo ""
echo "=== Setup Complete! ==="
echo ""
echo "Next steps:"
echo "1. Transfer model files to models/"
echo "2. Create .env file with API keys"
echo "3. Transfer data to data/"
echo "4. Test X11: xclock"
echo "5. Run GUI: python transcription_gui_qt.py"
```

Make it executable and run:
```bash
chmod +x setup_linux_server.sh
./setup_linux_server.sh
```

---

## Summary Checklist

- [ ] Switch git remote from GitLab to GitHub
- [ ] Create virtual environments (venv, venv_pylaia, venv_party)
- [ ] Install system packages (X11, build tools)
- [ ] Install Python packages in each environment
- [ ] Transfer model files from Windows
- [ ] Create .env file with API keys
- [ ] Transfer dataset directories
- [ ] Apply Party tokenizer fix manually
- [ ] Convert line endings (dos2unix)
- [ ] Update paths in transcription_gui_party.py (remove WSL code)
- [ ] Test X11 forwarding (xclock)
- [ ] Test GPU access (nvidia-smi, torch.cuda)
- [ ] Test GUIs (transcription_gui_qt.py)
- [ ] Test training (quick 1-epoch test)
- [ ] Test inference (TrOCR, Party)
- [ ] Pull git repo documentation for Claude context

---

## Getting Help

If you encounter issues:

1. **Check logs**: Most Python errors will show in terminal
2. **Verify CUDA**: `nvidia-smi` and `torch.cuda.is_available()`
3. **Check X11**: `echo $DISPLAY` and `xclock`
4. **Verify packages**: `pip list` in each venv
5. **Check paths**: Ensure no Windows paths (C:\, \\) remain
6. **Refer to CLAUDE.md**: Project-specific documentation
7. **Start new Claude session**: With LINUX_SERVER_MIGRATION.md context
