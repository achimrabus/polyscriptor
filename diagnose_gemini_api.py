#!/usr/bin/env python3
"""
Gemini API Diagnostic Tool

Tests your Gemini API setup to isolate blocking issues:
- Authentication and API key validity
- Available models and their capabilities
- Text-only generation (baseline)
- Vision generation with safe test image
- Vision generation with manuscript image across multiple models
- Safety settings impact

Usage:
    python diagnose_gemini_api.py [path/to/test/image.png]

If no image provided, will only test text generation.
"""

import os
import sys
from pathlib import Path
from PIL import Image
import json

try:
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
except ImportError:
    print("ERROR: google-generativeai not installed")
    print("Install with: pip install google-generativeai")
    sys.exit(1)


def load_api_key():
    """Load API key from environment or .env file."""
    # Try environment first
    api_key = os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        # Try .env file
        env_file = Path(__file__).parent / ".env"
        if env_file.exists():
            with open(env_file) as f:
                for line in f:
                    if line.startswith("GOOGLE_API_KEY="):
                        api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                        break
    
    if not api_key:
        print("ERROR: GOOGLE_API_KEY not found in environment or .env file")
        sys.exit(1)
    
    return api_key


def test_authentication(api_key):
    """Test 1: Verify API key authentication."""
    print("\n" + "="*70)
    print("TEST 1: API Key Authentication")
    print("="*70)
    
    try:
        genai.configure(api_key=api_key)
        print("✓ API key configured successfully")
        print(f"  Key prefix: {api_key[:10]}...")
        return True
    except Exception as e:
        print(f"✗ Authentication failed: {e}")
        return False


def test_list_models(api_key):
    """Test 2: List available models and capabilities."""
    print("\n" + "="*70)
    print("TEST 2: Available Models")
    print("="*70)
    
    try:
        genai.configure(api_key=api_key)
        models = list(genai.list_models())
        
        if not models:
            print("✗ No models available (API key may lack permissions)")
            return False
        
        print(f"✓ Found {len(models)} models")
        
        # Filter vision-capable models
        vision_models = []
        for model in models:
            if 'generateContent' in model.supported_generation_methods:
                model_name = model.name.replace("models/", "")
                supports_vision = hasattr(model, 'supported_generation_methods')
                vision_models.append(model_name)
                
                # Show key models
                if any(x in model_name for x in ['gemini-2.0', 'gemini-1.5', 'flash', 'pro']):
                    methods = ', '.join(model.supported_generation_methods)
                    print(f"  • {model_name}")
                    print(f"    Methods: {methods}")
        
        print(f"\n✓ {len(vision_models)} vision-capable models found")
        return vision_models
    
    except Exception as e:
        print(f"✗ Failed to list models: {e}")
        return False


def test_text_generation(api_key):
    """Test 3: Simple text-only generation (baseline)."""
    print("\n" + "="*70)
    print("TEST 3: Text-Only Generation (Baseline)")
    print("="*70)
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        prompt = "Say 'Hello, world!' in exactly those words."
        print(f"Prompt: {prompt}")
        
        response = model.generate_content(prompt)
        
        if response.parts:
            print(f"✓ Generation successful")
            print(f"  Response: {response.text[:100]}")
            return True
        else:
            print(f"✗ No response parts (blocked)")
            if hasattr(response, 'prompt_feedback'):
                print(f"  Feedback: {response.prompt_feedback}")
            return False
    
    except Exception as e:
        print(f"✗ Text generation failed: {e}")
        return False


def create_test_image():
    """Create a simple safe test image (white text on black)."""
    from PIL import Image, ImageDraw, ImageFont
    
    img = Image.new('RGB', (400, 100), color='white')
    draw = ImageDraw.Draw(img)
    
    # Simple text without font (default)
    text = "Test 1234"
    draw.text((20, 30), text, fill='black')
    
    return img


def test_vision_safe_image(api_key):
    """Test 4: Vision generation with simple safe image."""
    print("\n" + "="*70)
    print("TEST 4: Vision with Safe Test Image")
    print("="*70)
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        # Create simple test image
        test_img = create_test_image()
        print("Created test image: 400x100 white background with 'Test 1234'")
        
        prompt = "What text do you see in this image?"
        print(f"Prompt: {prompt}")
        
        response = model.generate_content([prompt, test_img])
        
        if response.parts:
            print(f"✓ Vision generation successful")
            print(f"  Response: {response.text[:200]}")
            return True
        else:
            print(f"✗ No response parts (blocked)")
            if hasattr(response, 'prompt_feedback'):
                print(f"  Feedback: {response.prompt_feedback}")
            return False
    
    except Exception as e:
        print(f"✗ Vision generation failed: {e}")
        return False


def test_vision_manuscript(api_key, image_path, models_to_test=None):
    """Test 5: Vision with actual manuscript across multiple models."""
    print("\n" + "="*70)
    print("TEST 5: Vision with Manuscript Image")
    print("="*70)
    
    if not image_path or not Path(image_path).exists():
        print("⊘ Skipped: No image provided")
        return None
    
    try:
        img = Image.open(image_path).convert('RGB')
        print(f"Loaded image: {image_path}")
        print(f"  Size: {img.width}x{img.height}")
    except Exception as e:
        print(f"✗ Failed to load image: {e}")
        return False
    
    genai.configure(api_key=api_key)
    
    # Default models to test
    if not models_to_test:
        models_to_test = [
            'gemini-2.0-flash-exp',
            'gemini-1.5-flash-002',
            'gemini-1.5-flash-8b',
            'gemini-1.5-pro-002',
        ]
    
    prompt = "Transcribe all handwritten text in this image. Output only the text."
    
    results = {}
    
    for model_name in models_to_test:
        print(f"\nTesting {model_name}...")
        
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content([prompt, img])
            
            if response.parts:
                text = response.text[:100] + ("..." if len(response.text) > 100 else "")
                print(f"  ✓ Success: {text}")
                results[model_name] = {"status": "success", "text": response.text}
            else:
                print(f"  ✗ Blocked")
                feedback = getattr(response, 'prompt_feedback', None)
                if feedback:
                    print(f"    Feedback: {feedback}")
                    results[model_name] = {"status": "blocked", "feedback": str(feedback)}
                else:
                    print(f"    No prompt_feedback available")
                    results[model_name] = {"status": "blocked", "feedback": "none"}
        
        except Exception as e:
            print(f"  ✗ Error: {e}")
            results[model_name] = {"status": "error", "error": str(e)}
    
    return results


def test_safety_settings(api_key, image_path):
    """Test 6: Impact of relaxed safety settings."""
    print("\n" + "="*70)
    print("TEST 6: Relaxed Safety Settings")
    print("="*70)
    
    if not image_path or not Path(image_path).exists():
        print("⊘ Skipped: No image provided")
        return None
    
    try:
        img = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"✗ Failed to load image: {e}")
        return False
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.0-flash-exp')
    
    prompt = "Transcribe all handwritten text in this image. Output only the text."
    
    # Test with default safety
    print("\n6a. Default safety settings:")
    try:
        response = model.generate_content([prompt, img])
        if response.parts:
            print(f"  ✓ Success with default safety")
            default_result = "success"
        else:
            print(f"  ✗ Blocked with default safety")
            default_result = "blocked"
    except Exception as e:
        print(f"  ✗ Error: {e}")
        default_result = "error"
    
    # Test with relaxed safety
    print("\n6b. Relaxed safety settings (BLOCK_ONLY_HIGH):")
    try:
        # Build safety settings dynamically
        relaxed_safety = []
        for category_name in dir(HarmCategory):
            if category_name.startswith('HARM_CATEGORY_'):
                category = getattr(HarmCategory, category_name)
                relaxed_safety.append({
                    "category": category,
                    "threshold": HarmBlockThreshold.BLOCK_ONLY_HIGH
                })
        
        print(f"  Using {len(relaxed_safety)} safety categories")
        
        response = model.generate_content(
            [prompt, img],
            safety_settings=relaxed_safety
        )
        
        if response.parts:
            print(f"  ✓ Success with relaxed safety")
            relaxed_result = "success"
        else:
            print(f"  ✗ Still blocked with relaxed safety")
            feedback = getattr(response, 'prompt_feedback', None)
            if feedback:
                print(f"    Feedback: {feedback}")
            relaxed_result = "blocked"
    
    except Exception as e:
        print(f"  ✗ Error: {e}")
        relaxed_result = "error"
    
    return {"default": default_result, "relaxed": relaxed_result}


def main():
    """Run all diagnostic tests."""
    print("\n" + "="*70)
    print("GEMINI API DIAGNOSTIC TOOL")
    print("="*70)
    
    # Get image path from args
    image_path = sys.argv[1] if len(sys.argv) > 1 else None
    
    if image_path:
        print(f"Test image: {image_path}")
    else:
        print("No test image provided (vision tests will be limited)")
    
    # Load API key
    api_key = load_api_key()
    
    # Run tests
    results = {
        "auth": test_authentication(api_key),
        "models": test_list_models(api_key),
        "text": test_text_generation(api_key),
        "vision_safe": test_vision_safe_image(api_key),
    }
    
    if image_path:
        results["vision_manuscript"] = test_vision_manuscript(api_key, image_path)
        results["safety_settings"] = test_safety_settings(api_key, image_path)
    
    # Summary
    print("\n" + "="*70)
    print("DIAGNOSTIC SUMMARY")
    print("="*70)
    
    print(f"\n1. Authentication: {'✓ PASS' if results['auth'] else '✗ FAIL'}")
    print(f"2. Model Access: {'✓ PASS' if results['models'] else '✗ FAIL'}")
    print(f"3. Text Generation: {'✓ PASS' if results['text'] else '✗ FAIL'}")
    print(f"4. Vision (Safe): {'✓ PASS' if results['vision_safe'] else '✗ FAIL'}")
    
    if image_path:
        if results.get('vision_manuscript'):
            success_count = sum(1 for r in results['vision_manuscript'].values() if r.get('status') == 'success')
            total_count = len(results['vision_manuscript'])
            print(f"5. Vision (Manuscript): {success_count}/{total_count} models succeeded")
            
            # Show which models worked
            for model_name, result in results['vision_manuscript'].items():
                status = result.get('status', 'unknown')
                icon = "✓" if status == "success" else "✗"
                print(f"   {icon} {model_name}: {status}")
        
        if results.get('safety_settings'):
            safety = results['safety_settings']
            print(f"6. Safety Settings:")
            print(f"   Default: {safety.get('default', 'unknown')}")
            print(f"   Relaxed: {safety.get('relaxed', 'unknown')}")
    
    # Diagnosis
    print("\n" + "="*70)
    print("DIAGNOSIS")
    print("="*70)
    
    if not results['auth']:
        print("❌ API key authentication failed")
        print("   → Check your GOOGLE_API_KEY in .env or environment")
    elif not results['models']:
        print("❌ No models available")
        print("   → Your API key may lack necessary permissions")
        print("   → Try regenerating the key in Google AI Studio")
    elif not results['text']:
        print("❌ Basic text generation failed")
        print("   → API quota may be exhausted or account has issues")
    elif not results['vision_safe']:
        print("❌ Vision generation fails even with safe content")
        print("   → Vision capabilities may not be enabled for your account")
        print("   → Check Google AI Studio settings")
    elif image_path and results.get('vision_manuscript'):
        manuscript_results = results['vision_manuscript']
        success_models = [m for m, r in manuscript_results.items() if r.get('status') == 'success']
        blocked_models = [m for m, r in manuscript_results.items() if r.get('status') == 'blocked']
        
        if success_models:
            print(f"✓ Manuscript transcription works with: {', '.join(success_models)}")
            if blocked_models:
                print(f"⚠ But blocked by: {', '.join(blocked_models)}")
                print("   → This is MODEL-SPECIFIC behavior")
                print("   → Use a working model in your application")
        else:
            print("❌ All models blocked manuscript transcription")
            print("   → This could be:")
            print("     1. Content safety filter (manuscript has problematic content)")
            print("     2. Preview model restrictions (try stable models)")
            print("     3. Account-level vision limitations")
            
            if results.get('safety_settings'):
                safety = results['safety_settings']
                if safety.get('relaxed') == 'success':
                    print("   ✓ Relaxed safety settings FIXED the issue")
                    print("     → Enable safety_relax in your application")
                elif safety.get('default') == safety.get('relaxed') == 'blocked':
                    print("   ✗ Even relaxed safety didn't help")
                    print("     → Likely model gating or account restriction")
    else:
        print("✓ All basic tests passed")
        print("  Run with image argument to test manuscript transcription:")
        print(f"  python {Path(__file__).name} path/to/manuscript/line.png")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
