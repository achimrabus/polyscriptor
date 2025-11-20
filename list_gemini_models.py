#!/usr/bin/env python3
"""List all available Gemini models with vision capabilities."""

import os
import google.generativeai as genai

# Get API key from environment
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("ERROR: GOOGLE_API_KEY not set in environment")
    print("Run: export GOOGLE_API_KEY='your-key-here'")
    exit(1)

genai.configure(api_key=api_key)

print("=" * 70)
print("Available Gemini Models with Vision Capabilities")
print("=" * 70)

vision_models = []
all_models = []

for model in genai.list_models():
    model_id = model.name.replace("models/", "")
    all_models.append(model_id)
    
    # Check if it supports generateContent (vision capability)
    if 'generateContent' in model.supported_generation_methods:
        vision_models.append(model_id)
        
        # Show details
        print(f"\n✓ {model_id}")
        print(f"   Display Name: {model.display_name}")
        print(f"   Methods: {', '.join(model.supported_generation_methods)}")
        
        # Show input/output limits if available
        if hasattr(model, 'input_token_limit'):
            print(f"   Input Limit: {model.input_token_limit:,} tokens")
        if hasattr(model, 'output_token_limit'):
            print(f"   Output Limit: {model.output_token_limit:,} tokens")

print("\n" + "=" * 70)
print(f"Summary: {len(vision_models)} vision-capable models found")
print("=" * 70)

# Show recommended models
print("\nRecommended for manuscript transcription:")
print("  • gemini-2.0-flash-exp (fast, excellent quality)")
print("  • gemini-1.5-pro-002 (slower, highest quality)")
print("  • gemini-1.5-flash-002 (good balance)")

print("\n" + "=" * 70)
print("All models (including non-vision):")
print("=" * 70)
for model_id in sorted(all_models):
    marker = "✓" if model_id in vision_models else "✗"
    print(f"  {marker} {model_id}")
