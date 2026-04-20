# test_opencv.py — OpenCV Learning Playground
# Run this file to learn how OpenCV works!
# Command: python test_opencv.py

import cv2          # OpenCV - image processing
import numpy as np  # NumPy - number arrays
from PIL import Image  # Pillow - extra image tools
import os

print("=" * 50)
print("   OpenCV Version:", cv2.__version__)
print("   NumPy Version:", np.__version__)
print("=" * 50)

# ─────────────────────────────────────────────────────
# LESSON 1: Read and Inspect an Image
# ─────────────────────────────────────────────────────
print("\n📖 LESSON 1: Reading an Image")

# Read person image
person_path = 'static/uploads/person.jpg'
clothing_path = 'static/uploads/clothing.png'

# Check if files exist first
if not os.path.exists(person_path):
    print("⚠️  No person.jpg found! Please upload images via the web app first.")
    print("   Go to http://localhost:5000/upload and upload both images.")
    print("   Then run this file again.")
    exit()

# cv2.imread() → reads image as a NumPy array
person_img = cv2.imread(person_path)

print(f"✅ Person image loaded!")
print(f"   Shape: {person_img.shape}")         # (height, width, channels)
print(f"   Height: {person_img.shape[0]} px")
print(f"   Width:  {person_img.shape[1]} px")
print(f"   Channels: {person_img.shape[2]}")   # 3 = BGR (Blue, Green, Red)
print(f"   Data type: {person_img.dtype}")      # uint8 = numbers 0-255
print(f"   Total pixels: {person_img.shape[0] * person_img.shape[1]}")

# Look at one pixel's value (top-left corner)
pixel = person_img[0, 0]  # [row, column]
print(f"\n   Pixel at (0,0): Blue={pixel[0]}, Green={pixel[1]}, Red={pixel[2]}")
print("   (OpenCV uses BGR order, not RGB!)")

# ─────────────────────────────────────────────────────
# LESSON 2: Color Space Conversion
# ─────────────────────────────────────────────────────
print("\n🎨 LESSON 2: Color Conversion")

# Convert BGR → RGB (for display compatibility)
person_rgb = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
print("✅ Converted BGR → RGB")

# Convert to Grayscale (1 channel instead of 3)
person_gray = cv2.cvtColor(person_img, cv2.COLOR_BGR2GRAY)
print(f"✅ Grayscale shape: {person_gray.shape}")  # Only (height, width) - no channels!

# Convert to HSV (Hue, Saturation, Value) - useful for color detection
person_hsv = cv2.cvtColor(person_img, cv2.COLOR_BGR2HSV)
print(f"✅ HSV shape: {person_hsv.shape}")

# ─────────────────────────────────────────────────────
# LESSON 3: Image Resizing
# ─────────────────────────────────────────────────────
print("\n📐 LESSON 3: Resizing Images")

# Resize to specific dimensions
small = cv2.resize(person_img, (200, 300))         # (width, height)
large = cv2.resize(person_img, (800, 1200))
print(f"✅ Original: {person_img.shape[:2]}")
print(f"✅ Small:    {small.shape[:2]}")
print(f"✅ Large:    {large.shape[:2]}")

# Resize by percentage (50% of original)
scale = 0.5
width  = int(person_img.shape[1] * scale)
height = int(person_img.shape[0] * scale)
half   = cv2.resize(person_img, (width, height))
print(f"✅ Half size: {half.shape[:2]}")

# ─────────────────────────────────────────────────────
# LESSON 4: Drawing on Images
# ─────────────────────────────────────────────────────
print("\n✏️  LESSON 4: Drawing on Images")

# Make a copy so we don't mess up the original
canvas = person_img.copy()

# Draw a rectangle: cv2.rectangle(image, top-left, bottom-right, color, thickness)
# Color is in BGR format!
cv2.rectangle(canvas, (10, 10), (150, 50), (0, 255, 0), 2)  # Green rectangle
print("✅ Drew a green rectangle")

# Draw a circle: cv2.circle(image, center, radius, color, thickness)
cv2.circle(canvas, (200, 100), 40, (255, 0, 0), 3)  # Blue circle
print("✅ Drew a blue circle")

# Write text: cv2.putText(image, text, position, font, size, color, thickness)
cv2.putText(canvas, 'Tryvora Test', (10, 80),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)  # Red text
print("✅ Added red text")

# Save this test image
cv2.imwrite('static/outputs/test_drawing.jpg', canvas)
print("✅ Saved test_drawing.jpg to static/outputs/")

# ─────────────────────────────────────────────────────
# LESSON 5: Working with PNG + Alpha Channel (Transparency)
# ─────────────────────────────────────────────────────
print("\n🔍 LESSON 5: PNG Transparency (Alpha Channel)")

if os.path.exists(clothing_path):
    # cv2.IMREAD_UNCHANGED → reads the 4th channel (alpha/transparency)
    cloth_img = cv2.imread(clothing_path, cv2.IMREAD_UNCHANGED)

    print(f"✅ Clothing image loaded!")
    print(f"   Shape: {cloth_img.shape}")

    if len(cloth_img.shape) == 3 and cloth_img.shape[2] == 4:
        print("   ✅ Has 4 channels: Blue, Green, Red, Alpha")
        print("   Alpha channel = transparency (0=invisible, 255=fully visible)")

        # Split into channels
        b, g, r, alpha = cv2.split(cloth_img)
        print(f"   Alpha min value: {alpha.min()} (0 = fully transparent)")
        print(f"   Alpha max value: {alpha.max()} (255 = fully opaque)")
    else:
        print("   ℹ️  Image has 3 channels (no transparency)")
        print("   Converting to 4 channels (adding alpha)...")
        # Add an alpha channel (fully opaque)
        cloth_img = cv2.cvtColor(cloth_img, cv2.COLOR_BGR2BGRA)
        print(f"   New shape: {cloth_img.shape}")
else:
    print("⚠️  No clothing.png found. Upload first at http://localhost:5000/upload")

# ─────────────────────────────────────────────────────
# LESSON 6: Image Blending
# ─────────────────────────────────────────────────────
print("\n🎭 LESSON 6: Image Blending")

# Create two colored squares to demonstrate blending
img_a = np.zeros((200, 200, 3), dtype=np.uint8)  # Black image
img_b = np.zeros((200, 200, 3), dtype=np.uint8)  # Black image

img_a[:] = [255, 100, 0]   # Blue-ish (BGR)
img_b[:] = [0, 255, 100]   # Green-ish (BGR)

# Blend: result = img_a * alpha + img_b * beta + gamma
# alpha + beta should equal 1.0 for natural blend
blended = cv2.addWeighted(img_a, 0.6, img_b, 0.4, 0)
cv2.imwrite('static/outputs/test_blend.jpg', blended)
print("✅ Blended two color images → saved as test_blend.jpg")

# ─────────────────────────────────────────────────────
# LESSON 7: Region of Interest (ROI) — KEY for try-on!
# ─────────────────────────────────────────────────────
print("\n🎯 LESSON 7: Region of Interest (ROI) — IMPORTANT!")
print("   ROI = selecting a specific area of an image")
print("   This is HOW we place clothing onto the person!")

person_copy = person_img.copy()

# Define a region: rows 100 to 300, columns 50 to 250
# Think of it like cropping a section
roi = person_copy[100:300, 50:250]
print(f"✅ Extracted ROI shape: {roi.shape}")

# Fill that region with a color (simulating clothing placement)
person_copy[100:300, 50:250] = [50, 50, 200]  # Fill with red (BGR)
cv2.imwrite('static/outputs/test_roi.jpg', person_copy)
print("✅ Saved ROI test → test_roi.jpg")

print("\n" + "=" * 50)
print("✅ ALL LESSONS COMPLETE!")
print("   Check static/outputs/ for test images:")
print("   - test_drawing.jpg  (rectangles, circles, text)")
print("   - test_blend.jpg    (color blending)")
print("   - test_roi.jpg      (region of interest)")
print("=" * 50)