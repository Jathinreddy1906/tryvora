# diagnose.py — Finds exactly what's going wrong
# Run: python diagnose.py

import os
import sys
import cv2
import numpy as np
from PIL import Image

print("\n" + "="*55)
print("   🔍 TRYVORA DIAGNOSTIC TOOL")
print("="*55)

# ── CHECK 1: Folder Structure ──────────────────────────────
print("\n📁 CHECK 1: Folder Structure")
folders = [
    'static',
    'static/uploads',
    'static/outputs',
    'templates'
]
for folder in folders:
    exists = os.path.exists(folder)
    print(f"   {'✅' if exists else '❌'} {folder}/")
    if not exists:
        os.makedirs(folder, exist_ok=True)
        print(f"      → Created missing folder!")

# ── CHECK 2: Uploaded Images ───────────────────────────────
print("\n🖼️  CHECK 2: Uploaded Images")
person_path   = 'static/uploads/person.jpg'
clothing_path = 'static/uploads/clothing.png'

person_ok   = os.path.exists(person_path)
clothing_ok = os.path.exists(clothing_path)

print(f"   {'✅' if person_ok else '❌'} person.jpg  → {person_path}")
print(f"   {'✅' if clothing_ok else '❌'} clothing.png → {clothing_path}")

if not person_ok or not clothing_ok:
    print("\n   ⚠️  MISSING IMAGES — This is why result wasn't generated!")
    print("   👉 Fix: Go to http://localhost:5000/upload")
    print("          Upload BOTH images, then run: python diagnose.py again")

    # Let's also check what IS in uploads
    print("\n   📂 Files currently in static/uploads/:")
    if os.path.exists('static/uploads'):
        files = os.listdir('static/uploads')
        if files:
            for f in files:
                size = os.path.getsize(f'static/uploads/{f}')
                print(f"      - {f}  ({size} bytes)")
        else:
            print("      (empty — no files uploaded yet!)")
    sys.exit()

# ── CHECK 3: Image Readability ─────────────────────────────
print("\n📖 CHECK 3: Can OpenCV Read the Images?")

person = cv2.imread(person_path)
clothing = cv2.imread(clothing_path, cv2.IMREAD_UNCHANGED)

if person is None:
    print("   ❌ person.jpg — OpenCV CANNOT read this file!")
    print("      Possible fix: Re-upload the image. Use JPG/PNG format only.")
else:
    size = os.path.getsize(person_path)
    print(f"   ✅ person.jpg — Shape: {person.shape}, Size: {size} bytes")

if clothing is None:
    print("   ❌ clothing.png — OpenCV CANNOT read this file!")
    print("      Possible fix: Re-upload the image. Use JPG/PNG format only.")
else:
    size = os.path.getsize(clothing_path)
    print(f"   ✅ clothing.png — Shape: {clothing.shape}, Size: {size} bytes")

if person is None or clothing is None:
    print("\n   ⚠️  Unreadable images — re-upload and try again.")
    sys.exit()

# ── CHECK 4: Alpha Channel ─────────────────────────────────
print("\n🔍 CHECK 4: Clothing Alpha Channel")
channels = clothing.shape[2] if len(clothing.shape) == 3 else 1
print(f"   Clothing channels: {channels}")

if channels == 4:
    alpha = clothing[:, :, 3]
    print(f"   ✅ Has alpha channel")
    print(f"   Alpha min: {alpha.min()}, max: {alpha.max()}")
    if alpha.max() == 0:
        print("   ⚠️  Alpha is ALL ZERO — clothing is fully transparent!")
        print("      Fix: Use a different PNG, or we'll auto-fix this below.")
elif channels == 3:
    print("   ℹ️  No alpha channel (3 channels only)")
    print("   → We will auto-create one by removing white background")
else:
    print(f"   ⚠️  Unexpected channel count: {channels}")

# ── CHECK 5: File Permissions ──────────────────────────────
print("\n🔐 CHECK 5: Can We Write to outputs/?")
test_file = 'static/outputs/test_write.txt'
try:
    with open(test_file, 'w') as f:
        f.write('test')
    os.remove(test_file)
    print("   ✅ Write permission OK")
except Exception as e:
    print(f"   ❌ Cannot write to outputs/: {e}")

# ── CHECK 6: Try a Manual Save ─────────────────────────────
print("\n💾 CHECK 6: Manual OpenCV Save Test")
test_img = np.zeros((100, 100, 3), dtype=np.uint8)
test_img[:] = [100, 180, 255]
save_path = 'static/outputs/diagnostic_test.jpg'
success = cv2.imwrite(save_path, test_img)
if success:
    print(f"   ✅ OpenCV can save images → {save_path}")
else:
    print("   ❌ OpenCV FAILED to save! Check outputs/ folder permissions.")

# ── CHECK 7: Run the Full Fix Pipeline ─────────────────────
print("\n🚀 CHECK 7: Running Fixed Pipeline...")

try:
    result = person.copy()

    # Ensure clothing has alpha
    if channels == 3:
        clothing_bgra = cv2.cvtColor(clothing, cv2.COLOR_BGR2BGRA)
        gray = cv2.cvtColor(clothing, cv2.COLOR_BGR2GRAY)
        _, bg_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
        clothing_bgra[:, :, 3] = cv2.bitwise_not(bg_mask)
        clothing = clothing_bgra
        print("   ✅ Auto-created alpha channel from white background")
    elif channels == 4:
        # Fix zero-alpha problem
        alpha = clothing[:, :, 3]
        if alpha.max() == 0:
            print("   ⚠️  Fixing zero-alpha clothing...")
            clothing[:, :, 3] = 255  # Make fully opaque
            print("   ✅ Alpha fixed to fully opaque")

    # Resize clothing
    ph, pw = person.shape[:2]
    target_w = int(pw * 0.70)
    target_h = int(ph * 0.45)
    clothing = cv2.resize(clothing, (target_w, target_h))
    print(f"   ✅ Clothing resized to {target_w}x{target_h}")

    # Position
    ch, cw = clothing.shape[:2]
    x = (pw - cw) // 2
    y = int(ph * 0.20)
    print(f"   ✅ Placement: x={x}, y={y}")

    # Overlay with alpha blending
    y_end = min(y + ch, ph)
    x_end = min(x + cw, pw)
    ch_clip = y_end - y
    cw_clip = x_end - x

    alpha_ch = clothing[:ch_clip, :cw_clip, 3].astype(float) / 255.0
    for i in range(3):
        cloth_px  = clothing[:ch_clip, :cw_clip, i].astype(float)
        person_px = result[y:y_end, x:x_end, i].astype(float)
        result[y:y_end, x:x_end, i] = (
            cloth_px * alpha_ch + person_px * (1 - alpha_ch)
        ).astype(np.uint8)

    print("   ✅ Alpha blending done")

    # Watermark
    h, w = result.shape[:2]
    overlay = result.copy()
    cv2.rectangle(overlay, (0, h-35), (w, h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.6, result, 0.4, 0, result)
    cv2.putText(result, 'Tryvora - Virtual Try-On',
                (10, h-12), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255,255,255), 1, cv2.LINE_AA)

    # Save
    out_path = 'static/outputs/tryon_result.jpg'
    saved = cv2.imwrite(out_path, result)
    if saved:
        size = os.path.getsize(out_path)
        print(f"   ✅ SAVED! → {out_path} ({size} bytes)")
    else:
        print("   ❌ cv2.imwrite returned False — save failed!")

except Exception as e:
    print(f"   ❌ Pipeline error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

# ── SUMMARY ───────────────────────────────────────────────
print("\n" + "="*55)
print("   📋 DIAGNOSTIC SUMMARY")
print("="*55)
print(f"   Person image:    {'✅ OK' if person_ok else '❌ MISSING'}")
print(f"   Clothing image:  {'✅ OK' if clothing_ok else '❌ MISSING'}")
print(f"   OpenCV read:     {'✅ OK' if person is not None else '❌ FAILED'}")
output_exists = os.path.exists('static/outputs/tryon_result.jpg')
print(f"   Result saved:    {'✅ YES — Check static/outputs/tryon_result.jpg!' if output_exists else '❌ NOT SAVED'}")
print("="*55 + "\n")