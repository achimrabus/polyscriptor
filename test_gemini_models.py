"""
Test script to list available Gemini models with your API key.

Usage:
    python test_gemini_models.py YOUR_API_KEY
"""

import sys

if len(sys.argv) < 2:
    print("Usage: python test_gemini_models.py YOUR_API_KEY")
    sys.exit(1)

api_key = sys.argv[1]

try:
    import google.generativeai as genai

    print("Configuring Gemini API...")
    genai.configure(api_key=api_key)

    print("\nListing all available models...\n")
    models = genai.list_models()

    gemini_models = []
    for m in models:
        if 'gemini' in m.name.lower():
            model_id = m.name.replace('models/', '')
            supports_vision = 'generateContent' in m.supported_generation_methods

            gemini_models.append({
                'id': model_id,
                'name': m.display_name if hasattr(m, 'display_name') else model_id,
                'vision': supports_vision,
                'methods': m.supported_generation_methods
            })

    if not gemini_models:
        print("❌ No Gemini models found!")
        print("This might mean:")
        print("  - API key is invalid")
        print("  - API not enabled in Google Cloud Console")
        print("  - Billing not set up")
    else:
        print(f"✅ Found {len(gemini_models)} Gemini models:\n")

        # Group by version
        by_version = {}
        for model in gemini_models:
            if '2.5' in model['id']:
                version = '2.5'
            elif '2.0' in model['id']:
                version = '2.0'
            elif '1.5' in model['id']:
                version = '1.5'
            else:
                version = 'other'

            if version not in by_version:
                by_version[version] = []
            by_version[version].append(model)

        # Print by version
        for version in sorted(by_version.keys(), reverse=True):
            print(f"  Gemini {version} models:")
            for model in by_version[version]:
                vision_mark = "👁 " if model['vision'] else "   "
                print(f"    {vision_mark}{model['id']}")
            print()

        # Print vision-capable models specifically
        vision_models = [m for m in gemini_models if m['vision']]
        print(f"\n📷 Vision-capable models ({len(vision_models)}):")
        for model in vision_models:
            print(f"   • {model['id']}")

        print("\n" + "="*60)
        print("These models will appear in the GUI dropdown after validation.")
        print("="*60)

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
