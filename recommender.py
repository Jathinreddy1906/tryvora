# recommender.py — Color-Based Outfit Recommendation Engine
# How it works:
#   1. Read the clothing image
#   2. Find the dominant color (most common color in the image)
#   3. Figure out what COLOR FAMILY it belongs to
#      (red, blue, green, yellow, purple, neutral...)
#   4. Return matching outfit suggestions for that color

import cv2
import numpy as np
from PIL import Image
import os

# ─────────────────────────────────────────────────────────
# COLOR KNOWLEDGE BASE
# Each color family has:
#   - hue_range: (min, max) in HSV (0-180 in OpenCV)
#   - name: display name
#   - emoji: visual icon
#   - matches: list of outfit recommendations
#   - avoid: colors that clash
# ─────────────────────────────────────────────────────────

COLOR_KNOWLEDGE = {
    'red': {
        'hue_range': [(0, 10), (165, 180)],
        'name': 'Red',
        'emoji': '🔴',
        'hex': '#ef4444',
        'matches': [
            {
                'item': 'Dark Navy Jeans',
                'reason': 'Navy blue creates a classic contrast with red',
                'color': '#1e3a5f',
                'emoji': '👖'
            },
            {
                'item': 'Black Trousers',
                'reason': 'Black is neutral and makes red pop',
                'color': '#1a1a1a',
                'emoji': '🖤'
            },
            {
                'item': 'White Sneakers',
                'reason': 'White keeps the look fresh and clean',
                'color': '#f9fafb',
                'emoji': '👟'
            },
            {
                'item': 'Beige Chinos',
                'reason': 'Earthy tones complement red beautifully',
                'color': '#d4b896',
                'emoji': '🟤'
            },
        ],
        'palette': ['#ef4444', '#1e3a5f', '#1a1a1a', '#f9fafb', '#d4b896'],
        'style_tip': 'Red is bold — keep bottoms neutral (black/navy/beige)'
    },

    'blue': {
        'hue_range': [(100, 130)],
        'name': 'Blue',
        'emoji': '🔵',
        'hex': '#3b82f6',
        'matches': [
            {
                'item': 'Light Grey Pants',
                'reason': 'Grey and blue are a timeless combination',
                'color': '#9ca3af',
                'emoji': '👔'
            },
            {
                'item': 'White Chinos',
                'reason': 'White brightens and freshens the blue',
                'color': '#f9fafb',
                'emoji': '🤍'
            },
            {
                'item': 'Brown Leather Belt',
                'reason': 'Brown leather adds warmth to cool blues',
                'color': '#92400e',
                'emoji': '🟫'
            },
            {
                'item': 'Khaki Shorts',
                'reason': 'Earthy khaki pairs great with blue tops',
                'color': '#c3a06a',
                'emoji': '🌿'
            },
        ],
        'palette': ['#3b82f6', '#9ca3af', '#f9fafb', '#92400e', '#c3a06a'],
        'style_tip': 'Blue is versatile — works with almost everything!'
    },

    'green': {
        'hue_range': [(36, 85)],
        'name': 'Green',
        'emoji': '🟢',
        'hex': '#22c55e',
        'matches': [
            {
                'item': 'Cream / Ivory Pants',
                'reason': 'Light neutrals let green be the star',
                'color': '#fef3c7',
                'emoji': '🟡'
            },
            {
                'item': 'Dark Brown Trousers',
                'reason': 'Brown is earthy and natural with green',
                'color': '#78350f',
                'emoji': '🤎'
            },
            {
                'item': 'Tan Suede Shoes',
                'reason': 'Tan complements green tones perfectly',
                'color': '#b45309',
                'emoji': '👞'
            },
            {
                'item': 'White T-shirt (layering)',
                'reason': 'White creates a clean base under green',
                'color': '#ffffff',
                'emoji': '⚪'
            },
        ],
        'palette': ['#22c55e', '#fef3c7', '#78350f', '#b45309', '#ffffff'],
        'style_tip': 'Green tones work best with natural, earthy colors'
    },

    'yellow': {
        'hue_range': [(22, 35)],
        'name': 'Yellow',
        'emoji': '🟡',
        'hex': '#eab308',
        'matches': [
            {
                'item': 'Navy Blue Jeans',
                'reason': 'Navy is the perfect contrast for yellow',
                'color': '#1e3a5f',
                'emoji': '👖'
            },
            {
                'item': 'White Shorts',
                'reason': 'White creates a sunny, fresh summer look',
                'color': '#f9fafb',
                'emoji': '🤍'
            },
            {
                'item': 'Olive Green Jacket',
                'reason': 'Olive and yellow are complementary earth tones',
                'color': '#4d5d2a',
                'emoji': '🫒'
            },
            {
                'item': 'Light Denim',
                'reason': 'Casual denim pairs effortlessly with yellow',
                'color': '#7ea7c8',
                'emoji': '👖'
            },
        ],
        'palette': ['#eab308', '#1e3a5f', '#f9fafb', '#4d5d2a', '#7ea7c8'],
        'style_tip': 'Yellow is vibrant — pair with navy or white to balance'
    },

    'purple': {
        'hue_range': [(130, 165)],
        'name': 'Purple',
        'emoji': '🟣',
        'hex': '#a855f7',
        'matches': [
            {
                'item': 'Charcoal Grey Pants',
                'reason': 'Charcoal provides a sophisticated base',
                'color': '#374151',
                'emoji': '🖤'
            },
            {
                'item': 'Light Grey Jeans',
                'reason': 'Light grey softens the richness of purple',
                'color': '#d1d5db',
                'emoji': '👖'
            },
            {
                'item': 'White Button-Up',
                'reason': 'White keeps purple from overwhelming the look',
                'color': '#ffffff',
                'emoji': '🤍'
            },
            {
                'item': 'Black Boots',
                'reason': 'Black grounds the elegance of purple',
                'color': '#111827',
                'emoji': '🥾'
            },
        ],
        'palette': ['#a855f7', '#374151', '#d1d5db', '#ffffff', '#111827'],
        'style_tip': 'Purple is elegant — grey and white are your best friends'
    },

    'orange': {
        'hue_range': [(11, 22)],
        'name': 'Orange',
        'emoji': '🟠',
        'hex': '#f97316',
        'matches': [
            {
                'item': 'Dark Denim Jeans',
                'reason': 'Deep blue denim is the ultimate orange partner',
                'color': '#1e3a5f',
                'emoji': '👖'
            },
            {
                'item': 'Brown Chinos',
                'reason': 'Earthy brown harmonizes with warm orange',
                'color': '#92400e',
                'emoji': '🤎'
            },
            {
                'item': 'White Tee (layering)',
                'reason': 'White keeps the warm tones balanced',
                'color': '#f9fafb',
                'emoji': '⚪'
            },
            {
                'item': 'Olive Trousers',
                'reason': 'Olive and orange are natural earth tones',
                'color': '#4d5d2a',
                'emoji': '🫒'
            },
        ],
        'palette': ['#f97316', '#1e3a5f', '#92400e', '#f9fafb', '#4d5d2a'],
        'style_tip': 'Orange is warm and energetic — earth tones keep it grounded'
    },

    'neutral': {
        'hue_range': [],   # handled separately via saturation check
        'name': 'Neutral (White/Grey/Black)',
        'emoji': '⚪',
        'hex': '#9ca3af',
        'matches': [
            {
                'item': 'Any Color Bottom',
                'reason': 'Neutrals go with literally everything!',
                'color': '#6d28d9',
                'emoji': '🌈'
            },
            {
                'item': 'Bold Colored Jeans',
                'reason': 'Add a pop of color with patterned bottoms',
                'color': '#2563eb',
                'emoji': '👖'
            },
            {
                'item': 'Statement Accessories',
                'reason': 'Neutral top = perfect canvas for accessories',
                'color': '#d97706',
                'emoji': '💛'
            },
            {
                'item': 'Patterned Scarf / Cap',
                'reason': 'Accessories add personality to neutral outfits',
                'color': '#dc2626',
                'emoji': '🧣'
            },
        ],
        'palette': ['#9ca3af', '#1f2937', '#f3f4f6', '#2563eb', '#d97706'],
        'style_tip': 'Neutral clothing is the most versatile — wear anything!'
    }
}


# ─────────────────────────────────────────────────────────
# FUNCTION 1: Extract dominant color from clothing image
# ─────────────────────────────────────────────────────────
def get_dominant_color(image_path, n_colors=1):
    """
    Finds the most common (dominant) color in the clothing image.

    How it works:
    - Load image → remove background pixels → 
      find most frequent color → return as HSV + RGB
    """
    # Load image
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None, None

    # Handle different channel counts
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        # Has alpha — only use non-transparent pixels
        alpha = img[:, :, 3]
        bgr   = img[:, :, :3]
        # Keep only pixels where alpha > 50 (not transparent)
        mask  = alpha > 50
        pixels = bgr[mask]
    else:
        pixels = img.reshape(-1, 3)

    if 'pixels' not in dir() or len(pixels) == 0:
        pixels = img[:, :, :3].reshape(-1, 3)

    # Remove very dark (shadow) and very bright (highlight) pixels
    # We want the actual clothing color
    brightness = pixels.mean(axis=1)
    pixels = pixels[(brightness > 30) & (brightness < 240)]

    if len(pixels) == 0:
        return None, None

    # Convert to float32 for K-Means
    pixels_f = pixels.astype(np.float32)

    # Use K-Means clustering to find dominant color
    # K=3: find top 3 color clusters, take the biggest
    k = min(3, len(pixels_f))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                100, 0.2)

    try:
        _, labels, centers = cv2.kmeans(
            pixels_f, k, None, criteria, 5,
            cv2.KMEANS_RANDOM_CENTERS
        )
    except Exception:
        # Fallback: just average the color
        centers = [pixels_f.mean(axis=0)]
        labels  = np.zeros(len(pixels_f), dtype=np.int32)

    # Find which cluster has the most pixels
    counts = np.bincount(labels.flatten())
    dominant_bgr = centers[np.argmax(counts)].astype(np.uint8)

    # Convert BGR → RGB for display
    dominant_rgb = dominant_bgr[::-1]

    # Convert BGR → HSV for color family detection
    bgr_pixel   = dominant_bgr.reshape(1, 1, 3)
    hsv_pixel   = cv2.cvtColor(bgr_pixel, cv2.COLOR_BGR2HSV)
    dominant_hsv = hsv_pixel[0][0]

    return dominant_rgb.tolist(), dominant_hsv.tolist()


# ─────────────────────────────────────────────────────────
# FUNCTION 2: Identify color family from HSV values
# ─────────────────────────────────────────────────────────
def identify_color_family(hsv_color):
    """
    Given HSV values, returns which color family it belongs to.
    HSV in OpenCV: H=0-180, S=0-255, V=0-255

    Logic:
    - Low saturation (S < 40) = neutral (white/grey/black)
    - Otherwise check Hue range for color family
    """
    if hsv_color is None:
        return 'neutral'

    h, s, v = hsv_color

    # Low saturation = neutral color
    if s < 45:
        return 'neutral'

    # Very dark = probably black (neutral)
    if v < 40:
        return 'neutral'

    # Check each color family's hue range
    for family, data in COLOR_KNOWLEDGE.items():
        if family == 'neutral':
            continue
        for (low, high) in data['hue_range']:
            if low <= h <= high:
                return family

    # Default fallback
    return 'neutral'


# ─────────────────────────────────────────────────────────
# FUNCTION 3: Generate recommendations
# ─────────────────────────────────────────────────────────
def get_recommendations(clothing_image_path):
    """
    Main function called by app.py.
    Returns a dict with:
    - color_name: "Blue"
    - color_hex: "#3b82f6"
    - dominant_rgb: [59, 130, 246]
    - style_tip: "Blue is versatile..."
    - palette: list of hex colors
    - matches: list of outfit suggestions
    """
    # Step 1: Get dominant color
    dominant_rgb, dominant_hsv = get_dominant_color(clothing_image_path)

    if dominant_rgb is None:
        # Fallback if detection fails
        return {
            'color_name'   : 'Unknown',
            'color_emoji'  : '❓',
            'color_hex'    : '#9ca3af',
            'dominant_rgb' : [156, 163, 175],
            'style_tip'    : 'Upload a clearer clothing image for better recommendations',
            'palette'      : ['#9ca3af'],
            'matches'      : [],
            'family'       : 'neutral'
        }

    # Step 2: Identify color family
    family = identify_color_family(dominant_hsv)

    # Step 3: Get recommendations for that family
    data = COLOR_KNOWLEDGE.get(family, COLOR_KNOWLEDGE['neutral'])

    # Convert RGB to hex for display
    r, g, b = dominant_rgb
    detected_hex = f'#{r:02x}{g:02x}{b:02x}'

    return {
        'color_name'   : data['name'],
        'color_emoji'  : data['emoji'],
        'color_hex'    : detected_hex,       # actual detected color
        'family_hex'   : data['hex'],         # reference color
        'dominant_rgb' : dominant_rgb,
        'style_tip'    : data['style_tip'],
        'palette'      : data['palette'],
        'matches'      : data['matches'],
        'family'       : family
    }


# ── Quick test ─────────────────────────────────────────────
if __name__ == '__main__':
    path = 'static/uploads/clothing.png'
    if os.path.exists(path):
        result = get_recommendations(path)
        print(f"\n🎨 Detected Color: {result['color_emoji']} {result['color_name']}")
        print(f"   Hex: {result['color_hex']}")
        print(f"   Family: {result['family']}")
        print(f"   Tip: {result['style_tip']}")
        print(f"\n👗 Matches:")
        for m in result['matches']:
            print(f"   {m['emoji']} {m['item']} — {m['reason']}")
    else:
        print("Upload clothing first at http://localhost:5000/upload")