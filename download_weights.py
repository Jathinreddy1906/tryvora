# download_weights.py
# Run this ONCE before starting the app
# python download_weights.py

import os
import sys

print("=" * 52)
print("   🔽 TRYVORA — MODEL WEIGHT SETUP")
print("=" * 52)

# Step 1: Check PyTorch
print("\n[1/4] Checking PyTorch...")
try:
    import torch
    print(f"  ✅ PyTorch {torch.__version__} — OK")
except ImportError:
    print("  ❌ PyTorch not found!")
    print("  Run: pip install torch --index-url "
          "https://download.pytorch.org/whl/cpu")
    sys.exit(1)

# Step 2: Check folder structure
print("\n[2/4] Checking folder structure...")
folders = [
    'models',
    'models/viton',
    'models/viton/weights',
    'static/outputs',
    'static/uploads',
]
for folder in folders:
    os.makedirs(folder, exist_ok=True)
    print(f"  ✅ {folder}/")

# Step 3: Create __init__.py files
print("\n[3/4] Setting up Python packages...")
for init_file in ['models/__init__.py',
                  'models/viton/__init__.py']:
    if not os.path.exists(init_file):
        with open(init_file, 'w') as f:
            f.write('# Auto-generated\n')
    print(f"  ✅ {init_file}")

# Step 4: Initialize weights
print("\n[4/4] Initializing model weights...")
try:
    from viton_model import download_weights
    success = download_weights()
    if success:
        print("  ✅ Weights ready!")
    else:
        print("  ❌ Weight initialization failed")
        sys.exit(1)
except Exception as e:
    print(f"  ❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 52)
print("  ✅ ALL DONE!")
print("  Run the app with: python app.py")
print("=" * 52)