# Qwen3-VL Fine-tuning & Commercial API Integration Guide

## Overview

This guide covers two major new features added to the TrOCR transcription pipeline:

1. **Qwen3-VL Fine-tuning**: Adapt Qwen3 vision-language models to Ukrainian manuscripts
2. **Commercial API Integration**: Use OpenAI GPT-4o, Google Gemini, or Anthropic Claude for transcription

## 1. Qwen3-VL Fine-tuning

### Purpose

Fine-tune Qwen3-VL models on Ukrainian/Cyrillic manuscripts for improved accuracy on your specific dataset.

### Key Features

- **LoRA (Low-Rank Adaptation)**: Memory-efficient fine-tuning
- **Optimized for 2x RTX 4090**: Hyperparameters tuned for dual-GPU setup
- **Works with Transkribus exports**: Uses `transkribus_parser.py` output
- **Multi-GPU support**: Automatic DDP (DistributedDataParallel)

### Files Created

- [finetune_qwen_ukrainian.py](c:\Users\Achim\Documents\TrOCR\dhlab-slavistik\finetune_qwen_ukrainian.py) - Main training script
- [inference_qwen.py](c:\Users\Achim\Documents\TrOCR\dhlab-slavistik\inference_qwen.py) - Inference script for fine-tuned models

### Usage

#### Prepare Dataset

First, prepare your dataset using the existing `transkribus_parser.py`:

```bash
python transkribus_parser.py \
    --input_dir /path/to/transkribus_export \
    --output_dir ./processed_data \
    --train_ratio 0.8 \
    --preserve-aspect-ratio \
    --target-height 128
```

#### Train Model (Single GPU)

```bash
python finetune_qwen_ukrainian.py \
    --data_dir ./processed_data \
    --output_dir ./results-ukrainian-qwen \
    --epochs 10 \
    --batch_size 4 \
    --gradient_accumulation 8
```

#### Train Model (Multi-GPU)

```bash
torchrun --nproc_per_node=2 finetune_qwen_ukrainian.py \
    --data_dir ./processed_data \
    --output_dir ./results-ukrainian-qwen \
    --epochs 10 \
    --batch_size 4 \
    --gradient_accumulation 8 \
    --multi_gpu
```

**Effective batch size**: `batch_size * gradient_accumulation * num_gpus = 4 * 8 * 2 = 64`

#### Key Hyperparameters

- **Epochs**: 10 (default) - Increase for larger datasets
- **Batch size**: 4 (default for 2B model on 4090) - Reduce if OOM
- **Gradient accumulation**: 8 (default) - Increase for larger effective batch
- **Learning rate**: 5e-5 (default)
- **LoRA rank**: 16 (default), alpha: 32

#### Monitor Training

```bash
tensorboard --logdir ./results-ukrainian-qwen
```

#### Inference

```bash
# Single image
python inference_qwen.py \
    --checkpoint ./results-ukrainian-qwen/final_model \
    --image line.jpg

# Batch inference
python inference_qwen.py \
    --checkpoint ./results-ukrainian-qwen/final_model \
    --image_dir ./test_images \
    --output results.txt
```

### Expected Training Time

- **2x RTX 4090**: ~2-4 hours for 10 epochs on 2-5K line images
- **Single RTX 4090**: ~4-8 hours

### Memory Usage

- **Qwen3-VL-2B**: ~12-16GB VRAM per GPU (with LoRA + gradient checkpointing)
- **Qwen3-VL-8B**: ~20-24GB VRAM per GPU

## 2. Commercial API Integration

### Purpose

Use cloud-based vision models (OpenAI, Gemini, Claude) for transcription without local GPU requirements.

### Supported Providers

1. **OpenAI GPT-4o**
   - Best general-purpose accuracy
   - Models: `gpt-4o`, `gpt-4o-mini`, `gpt-4-turbo`

2. **Google Gemini**
   - Fast and cost-effective
   - Models: `gemini-2.0-flash`, `gemini-1.5-pro-002`

3. **Anthropic Claude**
   - Best for text correction
   - Models: `claude-3-5-sonnet-20241022`, `claude-3-5-haiku-20241022`, `claude-3-opus-20240229`

### Files Created

- [inference_commercial_api.py](c:\Users\Achim\Documents\TrOCR\dhlab-slavistik\inference_commercial_api.py) - API inference wrappers

### GUI Integration

The GUI now has a **"Commercial APIs"** tab with:

- **Provider selection**: Choose OpenAI, Gemini, or Claude
- **API key input**: Secure password field with show/hide toggle
- **Model selection**: Provider-specific model dropdown
- **Custom prompt**: Customize transcription instructions
- **Advanced settings**: Temperature, max tokens

### Installation

Install required API libraries:

```bash
# OpenAI
pip install openai

# Gemini
pip install google-generativeai

# Claude
pip install anthropic

# Or install all at once
pip install openai google-generativeai anthropic
```

### Command-Line Usage

```bash
# OpenAI
python inference_commercial_api.py openai sk-... image.jpg

# Gemini
python inference_commercial_api.py gemini AIza... image.jpg

# Claude
python inference_commercial_api.py claude sk-ant-... image.jpg
```

### GUI Usage

1. Open GUI: `python transcription_gui_qt.py`
2. Load image (drag & drop or File > Open)
3. Switch to **"Commercial APIs"** tab
4. Select provider (OpenAI / Gemini / Claude)
5. Enter API key
6. Choose model
7. (Optional) Customize prompt
8. Click **"Transcribe with [Provider]"**

### API Key Security

- API keys are stored in memory only (not saved to disk by default)
- Use the 👁 button to toggle visibility
- Clear the key field when switching providers (optional)

### Pricing (Approximate, as of 2025)

- **OpenAI GPT-4o**: $2.50-5.00 per 1K images (depending on size)
- **Gemini 2.0 Flash**: $0.10-0.30 per 1K images
- **Claude 3.5 Sonnet**: $3.00-8.00 per 1K images

**Note**: Prices vary based on image size, token usage, and provider pricing changes. Check provider websites for current rates.

### Comparison: Local vs. Commercial

| Feature | Local (TrOCR/PyLaia) | Cloud APIs |
|---------|---------------------|-----------|
| **Cost** | Free (GPU electricity) | Pay-per-use |
| **Speed** | Fast (local GPU) | Moderate (network latency) |
| **Accuracy** | Good (fine-tuned) | Excellent (SOTA models) |
| **Privacy** | 100% local | Data sent to cloud |
| **Internet** | Not required | Required |
| **Setup** | Complex (GPU, models) | Simple (API key) |

### Best Practices

1. **Start with Gemini Flash**: Fast and cheap for initial testing
2. **Use Claude for corrections**: Best at refining initial transcriptions
3. **Fine-tune local models**: For large-scale production (lower cost)
4. **Hybrid approach**: Use APIs for difficult pages, local models for bulk

## Integration with Existing Pipeline

Both features integrate seamlessly with the existing pipeline:

### Qwen3-VL

- Uses same data format as TrOCR (`transkribus_parser.py` output)
- Compatible with existing aspect ratio preservation
- Can be used for inference via GUI (already integrated) or command-line

### Commercial APIs

- **New tab in GUI**: No changes to existing TrOCR/PyLaia/Qwen3 tabs
- **Full-page processing**: Like Qwen3, processes entire page (no segmentation needed)
- **Statistics panel**: Shows timing, model used, parameters

## Troubleshooting

### Qwen3-VL Fine-tuning

**Out of Memory (OOM)**:
- Reduce `--batch_size` (4 → 2 → 1)
- Increase `--gradient_accumulation` to maintain effective batch size
- Use smaller base model (2B instead of 8B)

**Slow Training**:
- Enable multi-GPU (`--multi_gpu` or `torchrun`)
- Increase `--batch_size` if VRAM available
- Use faster base model (2B instead of 8B)

**Low Accuracy**:
- Train for more epochs (`--epochs 20`)
- Check data quality (inspect `line_images/`)
- Ensure aspect ratio preservation was used
- Try different learning rate (`--learning_rate 3e-5` or `1e-4`)

### Commercial APIs

**API Key Error**:
- Check key format (OpenAI: `sk-...`, Claude: `sk-ant-...`, Gemini: `AIza...`)
- Verify key is active on provider dashboard
- Check billing/credits

**Rate Limit Exceeded**:
- Wait and retry (rate limits reset after time)
- Upgrade API tier on provider website
- Use batch processing with delays

**Network Error**:
- Check internet connection
- Verify firewall allows HTTPS to API endpoints
- Try different provider

**Poor Accuracy**:
- Customize prompt for your specific manuscript type
- Try different model (e.g., GPT-4o → Claude 3.5 Sonnet)
- Ensure image quality is good (use high DPI scans)

## Next Steps

1. **Test Qwen3 fine-tuning**: Start with small dataset (500-1K lines) for 5 epochs
2. **Try commercial APIs**: Test all three providers on sample pages
3. **Compare results**: Evaluate accuracy, speed, cost for your use case
4. **Scale up**: Choose best approach for your production workflow

## Additional Resources

- **Qwen3-VL Paper**: [Qwen Technical Report](https://arxiv.org/abs/2309.16609)
- **LoRA Paper**: [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)
- **Transcription Pearl**: [GitHub Repository](https://github.com/mhumphries2323/Transcription_Pearl)

## Credits

- **Qwen3-VL**: Alibaba DAMO Academy
- **LoRA**: Microsoft Research
- **Commercial API integration**: Inspired by Transcription Pearl
