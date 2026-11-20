#!/usr/bin/env bash
# Create the isolated venvs for the GLM-OCR, dots.mocr and Kraken 7 (PP-OCR) engines.
# Same convention as venv_deepseek / venv_chandra: one venv per engine family,
# so conflicting transformers versions never touch the main venv.
#
# Usage:
#   ./setup_ocr_vlm_venvs.sh            # glm and dots
#   ./setup_ocr_vlm_venvs.sh glm        # one
#   ./setup_ocr_vlm_venvs.sh kraken7    # Kraken 7 (PP-OCR): venv + base models
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGETS="${@:-glm dots}"

create_venv() {
    local name="$1"; shift
    local venv="$PROJECT_ROOT/$name"
    if [ -d "$venv" ]; then
        echo "== $name already exists at $venv — skipping creation =="
    else
        echo "== Creating $name at $venv =="
        python3 -m venv "$venv"
    fi
    echo "-- installing deps into $name"
    "$venv/bin/pip" install --upgrade pip
    "$venv/bin/pip" install "$@"
    echo "== $name ready =="
}

for t in $TARGETS; do
    case "$t" in
        glm)
            # GLM-OCR (zai-org/GLM-OCR, ~1.3B): transformers >= 5.3.0, needs torch >= 2.6
            create_venv venv_glm \
                torch torchvision --index-url https://download.pytorch.org/whl/cu126
            create_venv venv_glm "transformers>=5.3.0" accelerate pillow
            ;;
        dots)
            # dots.mocr (rednote-hilab/dots.mocr, ~3B): transformers == 4.56.1 + qwen-vl-utils
            create_venv venv_dots \
                torch torchvision --index-url https://download.pytorch.org/whl/cu121
            create_venv venv_dots "transformers==4.56.1" accelerate qwen-vl-utils pillow
            # The dots.mocr model code imports flash_attn unconditionally — install the
            # prebuilt wheel matching torch 2.5.x + cu12 + cp312 (no compile):
            "$PROJECT_ROOT/venv_dots/bin/pip" install \
                "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"
            ;;
        kraken7)
            # Kraken 7 (PP-OCRv6 line recognition): kraken 7.1 cannot share the main
            # venv, which needs kraken 6 for blla and the Kraken engine.
            create_venv venv_kraken7 \
                torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128
            create_venv venv_kraken7 kraken==7.1.1
            # Base models (Apache-2.0); kraken get stores them under
            # ~/.local/share/htrmopo/<uuid>/, the engine expects models/kraken7/ppocr_*.
            mkdir -p "$PROJECT_ROOT/models/kraken7"
            for spec in medium:21788410 small:21788405 tiny:21788403; do
                size="${spec%%:*}"
                "$PROJECT_ROOT/venv_kraken7/bin/kraken" get "10.5281/zenodo.${spec##*:}"
                src=$(ls -t "$HOME"/.local/share/htrmopo/*/"$size".safetensors | head -1)
                ln -sfn "$src" "$PROJECT_ROOT/models/kraken7/ppocr_$size.safetensors"
            done
            ;;
        *)
            echo "Unknown target: $t (use: glm, dots, kraken7)" >&2
            exit 1
            ;;
    esac
done

echo
echo "Done. Download the dots.mocr weights (name without periods!):"
echo "  huggingface-cli download rednote-hilab/dots.mocr --local-dir $PROJECT_ROOT/models/DotsMOCR"
