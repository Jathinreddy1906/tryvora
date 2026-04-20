# segmentation.py
# ─────────────────────────────────────────────────────────
# STEP 1: Human Segmentation — Remove Original Shirt
#
# What this file does:
#   1. Separate person from background (MediaPipe)
#   2. Find the upper-body / shirt zone (pose landmarks)
#   3. Build a precise shirt mask
#   4. Remove shirt pixels (inpainting fills the gap)
#   5. Return clean "shirt-removed" image + mask
#
# Beginner note:
#   A "mask" is a black-and-white image the same size as
#   the photo. White pixels = "this area", black = "not this area"
# ─────────────────────────────────────────────────────────

import cv2
import numpy as np
import os

# ── Safe MediaPipe import ─────────────────────────────────
MP_SEG_OK   = False
MP_POSE_OK  = False

try:
    import mediapipe as mp

    # Selfie Segmentation — separates person from background
    try:
        _mp_selfie = mp.solutions.selfie_segmentation
        MP_SEG_OK  = True
        print("[✓] MediaPipe Selfie Segmentation ready")
    except Exception as e:
        print(f"[!] Selfie seg unavailable: {e}")

    # Pose — gives us shoulder/hip/neck landmarks
    try:
        _mp_pose  = mp.solutions.pose
        MP_POSE_OK = True
        print("[✓] MediaPipe Pose ready")
    except Exception as e:
        print(f"[!] Pose unavailable: {e}")

except ImportError:
    print("[!] MediaPipe not installed")


# ═════════════════════════════════════════════════════════
# PART 1 — FULL BODY SEGMENTATION MASK
# Separates person pixels from background pixels
# ═════════════════════════════════════════════════════════

def get_person_mask(img_bgr):
    """
    Returns a mask where:
      255 = person pixel
      0   = background pixel

    Uses MediaPipe Selfie Segmentation.
    Falls back to GrabCut if MediaPipe unavailable.
    """
    h, w = img_bgr.shape[:2]

    # ── Method 1: MediaPipe Selfie Segmentation ──────────
    if MP_SEG_OK:
        try:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            with _mp_selfie.SelfieSegmentation(
                    model_selection=1) as seg:
                results = seg.process(img_rgb)

            if results.segmentation_mask is not None:
                # segmentation_mask values: 0.0 (bg) → 1.0 (person)
                raw  = results.segmentation_mask  # float array
                mask = (raw > 0.55).astype(np.uint8) * 255

                # Clean up with morphology
                k    = np.ones((7, 7), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                        np.ones((3,3), np.uint8))

                print("[✓] Person mask via MediaPipe Selfie Seg")
                return mask

        except Exception as e:
            print(f"[!] Selfie seg failed: {e} — trying GrabCut")

    # ── Method 2: GrabCut fallback ───────────────────────
    print("[~] Using GrabCut for person mask...")
    return _grabcut_person_mask(img_bgr)


def _grabcut_person_mask(img_bgr):
    """
    OpenCV GrabCut — segments person from background.
    Assumes person is roughly centered in the image.
    """
    h, w  = img_bgr.shape[:2]
    mask  = np.zeros((h, w), np.uint8)

    # Rectangle covering center 70% of image (where person usually is)
    margin_x = int(w * 0.08)
    margin_y = int(h * 0.02)
    rect     = (margin_x, margin_y,
                w - 2*margin_x, h - 2*margin_y)

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    cv2.grabCut(img_bgr, mask, rect,
                bgd_model, fgd_model, 5,
                cv2.GC_INIT_WITH_RECT)

    # GrabCut gives 4 values:
    #   0=bg, 1=fg, 2=probable_bg, 3=probable_fg
    # We want definite + probable foreground
    person_mask = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
        255, 0
    ).astype(np.uint8)

    # Smooth edges
    person_mask = cv2.GaussianBlur(person_mask, (9,9), 0)
    _, person_mask = cv2.threshold(
        person_mask, 127, 255, cv2.THRESH_BINARY
    )

    print("[✓] Person mask via GrabCut")
    return person_mask


# ═════════════════════════════════════════════════════════
# PART 2 — UPPER BODY LANDMARKS
# Finds exact pixel positions of shoulders, hips, neck
# ═════════════════════════════════════════════════════════

def get_upper_body_landmarks(img_bgr):
    """
    Returns key upper-body landmark positions (in pixels):
      left_shoulder, right_shoulder, left_hip, right_hip,
      nose (head top reference)

    Returns None if detection fails.
    """
    if not MP_POSE_OK:
        return None

    h, w    = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    try:
        with _mp_pose.Pose(
                static_image_mode=True,
                model_complexity=1,
                min_detection_confidence=0.4) as pose:
            results = pose.process(img_rgb)

        if not results.pose_landmarks:
            print("[!] No pose landmarks found")
            return None

        lm = results.pose_landmarks.landmark

        def px(lmk):
            # Clamp to image boundaries
            x = max(0, min(int(lmk.x * w), w-1))
            y = max(0, min(int(lmk.y * h), h-1))
            return (x, y)

        landmarks = {
            'left_shoulder' : px(lm[_mp_pose.PoseLandmark.LEFT_SHOULDER]),
            'right_shoulder': px(lm[_mp_pose.PoseLandmark.RIGHT_SHOULDER]),
            'left_hip'      : px(lm[_mp_pose.PoseLandmark.LEFT_HIP]),
            'right_hip'     : px(lm[_mp_pose.PoseLandmark.RIGHT_HIP]),
            'left_elbow'    : px(lm[_mp_pose.PoseLandmark.LEFT_ELBOW]),
            'right_elbow'   : px(lm[_mp_pose.PoseLandmark.RIGHT_ELBOW]),
            'nose'          : px(lm[_mp_pose.PoseLandmark.NOSE]),
            'left_wrist'    : px(lm[_mp_pose.PoseLandmark.LEFT_WRIST]),
            'right_wrist'   : px(lm[_mp_pose.PoseLandmark.RIGHT_WRIST]),
        }

        print(f"[✓] Landmarks: "
              f"L-shoulder={landmarks['left_shoulder']}  "
              f"R-shoulder={landmarks['right_shoulder']}  "
              f"L-hip={landmarks['left_hip']}")
        return landmarks

    except Exception as e:
        print(f"[!] Landmark detection error: {e}")
        return None


# ═════════════════════════════════════════════════════════
# PART 3 — SHIRT / UPPER-BODY MASK
# Builds a precise mask covering only the shirt area
# ═════════════════════════════════════════════════════════

def build_shirt_mask(img_bgr, person_mask, landmarks=None):
    """
    Builds a mask that covers ONLY the shirt/torso region.

    Strategy:
      A) If landmarks available:
           Use shoulder → hip polygon to define shirt zone
           Then intersect with person_mask (stay inside body)

      B) If no landmarks:
           Use upper 55% of person_mask as shirt zone

    Returns uint8 mask: 255=shirt, 0=not shirt
    """
    h, w = img_bgr.shape[:2]

    if landmarks is not None:
        mask = _landmark_shirt_mask(h, w, landmarks, person_mask)
    else:
        mask = _fallback_shirt_mask(h, w, person_mask)

    # Final cleanup — smooth edges
    k    = np.ones((9, 9), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.GaussianBlur(mask, (15, 15), 0)
    _, mask = cv2.threshold(mask, 80, 255, cv2.THRESH_BINARY)

    return mask


def _landmark_shirt_mask(h, w, lm, person_mask):
    """
    Creates shirt mask using body landmark polygon.

    Shirt polygon points:
      → Start above left shoulder (neck-left)
      → Across to above right shoulder (neck-right)
      → Down to right hip
      → Across to left hip
      → Back up to left shoulder

    We add generous padding so we capture the whole shirt.
    """
    ls = lm['left_shoulder']
    rs = lm['right_shoulder']
    lh = lm['left_hip']
    rh = lm['right_hip']

    # Padding: extend outward from landmarks
    shoulder_pad_x = int((rs[0] - ls[0]) * 0.18)  # 18% extra width
    shoulder_pad_y = int(h * 0.04)                  # slight upward push

    # Neck points (between shoulder and head)
    neck_l = (ls[0] - shoulder_pad_x,
              ls[1] - shoulder_pad_y)
    neck_r = (rs[0] + shoulder_pad_x,
              rs[1] - shoulder_pad_y)

    # Hip points (slight downward extension for shirt hem)
    hip_pad_y = int(h * 0.04)
    hip_l     = (lh[0] - shoulder_pad_x, lh[1] + hip_pad_y)
    hip_r     = (rh[0] + shoulder_pad_x, rh[1] + hip_pad_y)

    # Sleeve extension — down to elbow area
    le = lm.get('left_elbow',  (ls[0] - shoulder_pad_x*2, ls[1]+50))
    re = lm.get('right_elbow', (rs[0] + shoulder_pad_x*2, rs[1]+50))

    sleeve_l = (min(ls[0], le[0]) - shoulder_pad_x,
                max(ls[1], le[1]))
    sleeve_r = (max(rs[0], re[0]) + shoulder_pad_x,
                max(rs[1], re[1]))

    # Full shirt polygon (convex hull of all points)
    points = np.array([
        neck_l, neck_r,
        sleeve_r, hip_r,
        hip_l,   sleeve_l,
    ], dtype=np.int32)

    # Clamp to image size
    points[:, 0] = np.clip(points[:, 0], 0, w-1)
    points[:, 1] = np.clip(points[:, 1], 0, h-1)

    # Draw filled polygon
    shirt_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(shirt_mask, [points], 255)

    # Only keep pixels that are ALSO part of the person
    shirt_mask = cv2.bitwise_and(shirt_mask, person_mask)

    return shirt_mask


def _fallback_shirt_mask(h, w, person_mask):
    """
    When no landmarks: take upper 58% of person mask as shirt.
    """
    shirt_zone = np.zeros((h, w), dtype=np.uint8)
    cutoff     = int(h * 0.58)
    start      = int(h * 0.12)   # skip head area
    shirt_zone[start:cutoff, :] = 255

    return cv2.bitwise_and(shirt_zone, person_mask)


# ═════════════════════════════════════════════════════════
# PART 4 — SHIRT REMOVAL (INPAINTING)
# Fills the removed shirt area with background estimation
# ═════════════════════════════════════════════════════════

def remove_shirt(img_bgr, shirt_mask):
    """
    Removes the shirt from the image by:
      1. Sampling background color from image edges
      2. Creating a smooth background fill
      3. Using OpenCV inpainting for realistic fill
      4. Blending inpainted result with original

    Returns:
      cleaned_img : image with shirt area filled
      shirt_mask  : the mask used (for use in overlay step)
    """
    h, w = img_bgr.shape[:2]

    # ── Step A: Sample background color ──────────────────
    # Sample from image corners and edges (likely background)
    bg_color = _sample_background_color(img_bgr, shirt_mask)
    print(f"[✓] Background color sampled: BGR={bg_color}")

    # ── Step B: Create background fill canvas ────────────
    bg_canvas = np.full_like(img_bgr, bg_color)

    # Add slight gradient to make it look natural
    for c in range(3):
        # Vertical gradient matching top/bottom of image
        top_col = float(img_bgr[0, w//2, c])
        bot_col = float(img_bgr[min(h-1, int(h*0.8)), w//2, c])
        for row in range(h):
            t = row / h
            bg_canvas[row, :, c] = np.clip(
                top_col * (1-t) + bot_col * t, 0, 255
            ).astype(np.uint8)

    # ── Step C: OpenCV Inpainting ─────────────────────────
    # Inpainting = smart fill that looks at surrounding pixels
    # We dilate the mask slightly so edges are clean
    dilate_k       = np.ones((5, 5), np.uint8)
    inpaint_mask   = cv2.dilate(shirt_mask, dilate_k, iterations=2)

    # INPAINT_TELEA = fast, high quality
    inpainted = cv2.inpaint(
        img_bgr, inpaint_mask,
        inpaintRadius=12,
        flags=cv2.INPAINT_TELEA
    )
    print("[✓] Inpainting complete")

    # ── Step D: Blend inpainted + bg_canvas ───────────────
    # In the shirt zone: 70% inpainted + 30% flat bg
    # This handles cases where inpainting bleeds shirt color
    alpha_blend  = 0.75
    shirt_float  = shirt_mask.astype(np.float32) / 255.0
    shirt_3ch    = np.stack([shirt_float]*3, axis=2)

    blended = (
        inpainted.astype(np.float32)  * alpha_blend * shirt_3ch +
        bg_canvas.astype(np.float32)  * (1-alpha_blend) * shirt_3ch +
        img_bgr.astype(np.float32)    * (1 - shirt_3ch)
    ).astype(np.uint8)

    # ── Step E: Smooth the transition edge ────────────────
    edge_mask   = _create_edge_blend_mask(shirt_mask, radius=30)
    edge_3ch    = np.stack([edge_mask]*3, axis=2)

    final = (
        blended.astype(np.float32)  * (1 - edge_3ch) +
        img_bgr.astype(np.float32)  * edge_3ch
    ).astype(np.uint8)

    return final, shirt_mask


def _sample_background_color(img_bgr, shirt_mask):
    """
    Sample background color from areas NOT covered by person/shirt.
    Uses edges and corners of image.
    """
    h, w = img_bgr.shape[:2]

    # Sample regions: top strip, left strip, right strip
    sample_regions = [
        img_bgr[0:int(h*0.05), :],          # top 5%
        img_bgr[:, 0:int(w*0.05)],           # left 5%
        img_bgr[:, int(w*0.95):],            # right 5%
    ]

    all_pixels = []
    for region in sample_regions:
        if region.size > 0:
            pixels = region.reshape(-1, 3)
            all_pixels.append(pixels)

    if all_pixels:
        combined = np.vstack(all_pixels)
        # Remove very dark and very bright outliers
        brightness = combined.mean(axis=1)
        filtered   = combined[(brightness > 20) & (brightness < 240)]
        if len(filtered) > 0:
            return tuple(map(int, np.median(filtered, axis=0)))

    # Fallback: light gray
    return (200, 200, 200)


def _create_edge_blend_mask(mask, radius=25):
    """
    Creates a soft transition mask at the edges of the shirt mask.
    This prevents a visible "seam" where shirt was removed.

    Returns float mask: 0.0 (inside shirt) → 1.0 (outside shirt)
    """
    # Distance transform: distance from shirt mask edge
    dist = cv2.distanceTransform(
        cv2.bitwise_not(mask), cv2.DIST_L2, 5
    )

    # Normalize: 0 at edge, 1 beyond radius
    edge_blend = np.clip(dist / radius, 0, 1).astype(np.float32)

    return edge_blend


# ═════════════════════════════════════════════════════════
# PART 5 — VISUALIZE / DEBUG (optional)
# ═════════════════════════════════════════════════════════

def save_debug_masks(img_bgr, person_mask, shirt_mask,
                     cleaned_img, output_dir='static/outputs'):
    """
    Saves debug images so you can see what each step produced.
    Useful for understanding what the code is doing!
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Person mask (green overlay)
    person_viz = img_bgr.copy()
    overlay    = np.zeros_like(img_bgr)
    overlay[:,:,1] = person_mask  # green channel
    person_viz = cv2.addWeighted(person_viz, 0.7, overlay, 0.3, 0)
    cv2.imwrite(f'{output_dir}/debug_person_mask.jpg', person_viz)

    # 2. Shirt mask (red overlay)
    shirt_viz    = img_bgr.copy()
    shirt_overlay = np.zeros_like(img_bgr)
    shirt_overlay[:,:,2] = shirt_mask  # red channel
    shirt_viz = cv2.addWeighted(shirt_viz, 0.6, shirt_overlay, 0.4, 0)
    cv2.imwrite(f'{output_dir}/debug_shirt_mask.jpg', shirt_viz)

    # 3. Cleaned image (shirt removed)
    cv2.imwrite(f'{output_dir}/debug_shirt_removed.jpg', cleaned_img)

    # 4. Side-by-side comparison
    h = max(img_bgr.shape[0], cleaned_img.shape[0])
    compare = np.hstack([
        cv2.resize(img_bgr, (img_bgr.shape[1], h)),
        cv2.resize(cleaned_img, (cleaned_img.shape[1], h))
    ])
    # Add labels
    cv2.putText(compare, 'ORIGINAL',
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0,255,0), 2)
    cv2.putText(compare, 'SHIRT REMOVED',
                (img_bgr.shape[1]+10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0,255,255), 2)
    cv2.imwrite(f'{output_dir}/debug_comparison.jpg', compare)

    print(f"[✓] Debug images saved to {output_dir}/")
    print( "    debug_person_mask.jpg  ← green = detected person")
    print( "    debug_shirt_mask.jpg   ← red   = shirt zone")
    print( "    debug_shirt_removed.jpg← shirt removed result")
    print( "    debug_comparison.jpg   ← before vs after")


# ═════════════════════════════════════════════════════════
# PART 6 — MAIN FUNCTION (called by image_processor.py)
# ═════════════════════════════════════════════════════════

def segment_and_remove_shirt(person_path, save_debug=True):
    """
    Complete shirt removal pipeline.

    Input:  person_path — path to uploaded person image
    Output: (cleaned_img, shirt_mask, landmarks)
            cleaned_img  = person image with shirt removed
            shirt_mask   = binary mask of removed area
            landmarks    = pose landmarks (or None)

    Usage in image_processor.py:
        from segmentation import segment_and_remove_shirt
        cleaned, mask, landmarks = segment_and_remove_shirt(person_path)
    """
    print("\n── SEGMENTATION PIPELINE ──────────────────────")

    # Load image
    img = cv2.imread(person_path)
    if img is None:
        raise ValueError(f"Cannot read image: {person_path}")

    h, w = img.shape[:2]
    print(f"[✓] Image loaded: {w}×{h}")

    # Step 1: Full person mask
    print("[~] Step 1: Building person mask...")
    person_mask = get_person_mask(img)

    # Step 2: Get pose landmarks
    print("[~] Step 2: Getting pose landmarks...")
    landmarks = get_upper_body_landmarks(img)

    # Step 3: Build shirt mask
    print("[~] Step 3: Building shirt mask...")
    shirt_mask = build_shirt_mask(img, person_mask, landmarks)

    # Step 4: Remove shirt (inpainting)
    print("[~] Step 4: Removing shirt (inpainting)...")
    cleaned_img, shirt_mask = remove_shirt(img, shirt_mask)

    # Step 5: Save debug images
    if save_debug:
        save_debug_masks(img, person_mask, shirt_mask, cleaned_img)

    print("[✓] Segmentation pipeline complete!")
    print("────────────────────────────────────────────────\n")

    return cleaned_img, shirt_mask, landmarks


# ── Quick test ─────────────────────────────────────────────
if __name__ == '__main__':
    path = 'static/uploads/person.jpg'
    if not os.path.exists(path):
        print("❌ No person.jpg found!")
        print("   Go to http://localhost:5000/upload first")
    else:
        print("🔬 Testing segmentation pipeline...")
        cleaned, mask, lm = segment_and_remove_shirt(path, save_debug=True)
        print("\n✅ Test complete! Check static/outputs/ for debug images:")
        print("   debug_person_mask.jpg   ← person detected")
        print("   debug_shirt_mask.jpg    ← shirt region")
        print("   debug_shirt_removed.jpg ← shirt removed")
        print("   debug_comparison.jpg    ← before vs after")