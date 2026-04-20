# input_preparator.py
# ─────────────────────────────────────────────────────────
# STEP 3: Input Preparation for VITON Model
#
# This file converts raw uploaded images into the
# exact format the VITON model needs.
#
# Beginner note:
#   Think of this like a "translator" —
#   your camera takes photos in one format,
#   but the AI speaks a different language.
#   This file translates between them.
# ─────────────────────────────────────────────────────────

import cv2
import numpy as np
import os

# VITON standard dimensions (do NOT change these!)
VITON_W = 192   # width
VITON_H = 256   # height

# Safe MediaPipe import
MP_POSE_OK = False
try:
    import mediapipe as mp
    try:
        _mp_pose   = mp.solutions.pose
        MP_POSE_OK = True
    except Exception:
        pass
except ImportError:
    pass


# ═════════════════════════════════════════════════════════
# PART 1 — PERSON IMAGE PREPARATION
# ═════════════════════════════════════════════════════════

def prepare_person_image(person_img_bgr):
    """
    Prepares the person image for VITON input.

    Steps:
      1. Smart crop to portrait aspect ratio (3:4)
      2. Resize to 192×256
      3. Normalize pixel values
      4. Return both processed image AND scale info
         (we need scale info to map results back later)

    Input:  BGR numpy array (any size)
    Output: dict with prepared arrays
    """
    orig_h, orig_w = person_img_bgr.shape[:2]

    # ── Step 1: Smart crop to 3:4 ratio ──────────────────
    # VITON expects portrait format (3 wide : 4 tall)
    cropped, crop_info = smart_portrait_crop(person_img_bgr)

    # ── Step 2: Resize to exact VITON dimensions ──────────
    resized = cv2.resize(
        cropped,
        (VITON_W, VITON_H),
        interpolation=cv2.INTER_LANCZOS4
    )

    # ── Step 3: Color correction ──────────────────────────
    # Normalize brightness so model works consistently
    corrected = normalize_image_brightness(resized)

    # ── Step 4: Create RGB version (VITON uses RGB) ───────
    rgb = cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB)

    print(f"[✓] Person prepared: "
          f"{orig_w}×{orig_h} → {VITON_W}×{VITON_H}")

    return {
        'bgr'       : corrected,        # BGR for OpenCV ops
        'rgb'       : rgb,              # RGB for model
        'orig_h'    : orig_h,
        'orig_w'    : orig_w,
        'crop_info' : crop_info,        # for mapping back
    }


def smart_portrait_crop(img_bgr):
    """
    Crops image to 3:4 portrait ratio intelligently.

    Strategy:
      - If image is already portrait → minimal crop
      - If image is landscape → crop sides equally
      - Always keep the CENTER of the image
        (where the person usually is)

    Returns: (cropped_image, crop_info_dict)
    """
    h, w = img_bgr.shape[:2]
    target_ratio = 3.0 / 4.0   # width/height

    current_ratio = w / h

    if current_ratio > target_ratio:
        # Image is too wide → crop left and right
        new_w    = int(h * target_ratio)
        start_x  = (w - new_w) // 2
        cropped  = img_bgr[:, start_x:start_x + new_w]
        crop_info = {
            'type': 'width_crop',
            'start_x': start_x,
            'start_y': 0,
            'crop_w': new_w,
            'crop_h': h
        }
    elif current_ratio < target_ratio:
        # Image is too tall → crop top and bottom
        new_h    = int(w / target_ratio)
        # Keep slightly more from top (head) than bottom (feet)
        start_y  = max(0, int((h - new_h) * 0.3))
        cropped  = img_bgr[start_y:start_y + new_h, :]
        crop_info = {
            'type': 'height_crop',
            'start_x': 0,
            'start_y': start_y,
            'crop_w': w,
            'crop_h': new_h
        }
    else:
        # Already correct ratio
        cropped   = img_bgr.copy()
        crop_info = {
            'type': 'none',
            'start_x': 0,
            'start_y': 0,
            'crop_w': w,
            'crop_h': h
        }

    return cropped, crop_info


def normalize_image_brightness(img_bgr, target_mean=127.0):
    """
    Normalizes image brightness so it's consistent.
    Too dark or too bright images confuse the model.

    target_mean = 127 means we aim for middle brightness.
    """
    current_mean = img_bgr.mean()
    diff = target_mean - current_mean

    # Only adjust if significantly off (more than 30 units)
    if abs(diff) > 30:
        adjusted = np.clip(
            img_bgr.astype(np.float32) + diff * 0.4,
            0, 255
        ).astype(np.uint8)
        return adjusted

    return img_bgr


# ═════════════════════════════════════════════════════════
# PART 2 — CLOTHING IMAGE PREPARATION
# ═════════════════════════════════════════════════════════

def prepare_clothing_image(cloth_img):
    """
    Prepares clothing image for VITON input.

    Clothing images need special treatment:
      1. Remove background → white background standard
      2. Center the garment in the frame
      3. Resize to VITON dimensions
      4. Create clothing mask (white=cloth, black=background)

    Input:  BGR or BGRA numpy array
    Output: dict with prepared arrays + mask
    """
    # ── Step 1: Convert to BGRA (ensure alpha channel) ───
    if len(cloth_img.shape) == 2:
        # Grayscale → BGR
        cloth_img = cv2.cvtColor(cloth_img, cv2.COLOR_GRAY2BGR)

    if cloth_img.shape[2] == 3:
        # No alpha → remove white background
        cloth_bgra = _remove_cloth_background(cloth_img)
    else:
        cloth_bgra = cloth_img.copy()

    # ── Step 2: Center garment on white background ────────
    cloth_centered = _center_garment(cloth_bgra)

    # ── Step 3: Resize to VITON dimensions ────────────────
    cloth_resized = cv2.resize(
        cloth_centered,
        (VITON_W, VITON_H),
        interpolation=cv2.INTER_LANCZOS4
    )

    # ── Step 4: Extract cloth mask ────────────────────────
    # Mask: where the cloth is (not background)
    cloth_mask = _extract_cloth_mask(cloth_resized)

    # ── Step 5: Create white-background version ───────────
    # Place cloth on white background for model input
    cloth_on_white = _composite_on_white(cloth_resized)

    # RGB version
    rgb = cv2.cvtColor(cloth_on_white, cv2.COLOR_BGR2RGB)

    print(f"[✓] Clothing prepared: {VITON_W}×{VITON_H}")
    print(f"    Cloth mask coverage: "
          f"{(cloth_mask > 0).mean()*100:.1f}%")

    return {
        'bgr'        : cloth_on_white,
        'rgb'        : rgb,
        'mask'       : cloth_mask,      # binary mask
        'bgra'       : cloth_resized,   # with alpha
    }


def _remove_cloth_background(bgr_img):
    """
    Remove white/light background from clothing.
    Returns BGRA image.
    """
    bgra = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2BGRA)
    hsv  = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)

    # White background detection
    bg_mask = cv2.inRange(
        hsv,
        np.array([0,   0,  170], np.uint8),
        np.array([180, 60, 255], np.uint8)
    )

    # Clean up
    k      = np.ones((5,5), np.uint8)
    bg_mask = cv2.morphologyEx(bg_mask, cv2.MORPH_CLOSE, k)
    bg_mask = cv2.morphologyEx(bg_mask, cv2.MORPH_OPEN,  k)

    # Cloth pixels
    cloth_alpha = cv2.bitwise_not(bg_mask)

    # Smooth edges
    cloth_alpha = cv2.GaussianBlur(cloth_alpha, (11, 11), 0)
    bgra[:, :, 3] = cloth_alpha

    return bgra


def _center_garment(cloth_bgra):
    """
    Centers the garment within the frame.
    Adds padding if needed so garment is centered.
    """
    if cloth_bgra.shape[2] < 4:
        return cloth_bgra

    alpha = cloth_bgra[:, :, 3]
    h, w  = alpha.shape

    # Find bounding box of non-transparent pixels
    rows = np.any(alpha > 10, axis=1)
    cols = np.any(alpha > 10, axis=0)

    if not rows.any() or not cols.any():
        return cloth_bgra

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # Crop to content
    content = cloth_bgra[rmin:rmax+1, cmin:cmax+1]
    ch, cw  = content.shape[:2]

    # Create white background canvas
    # Use 3:4 ratio matching VITON
    target_h = int(max(ch, cw * 4/3) * 1.1)
    target_w = int(target_h * 3/4)

    canvas = np.ones((target_h, target_w, 4),
                     dtype=np.uint8) * 255
    canvas[:, :, 3] = 255  # fully opaque white

    # Center content in canvas
    pad_y = (target_h - ch) // 2
    pad_x = (target_w - cw) // 2

    # Clamp
    pad_y = max(0, min(pad_y, target_h - ch))
    pad_x = max(0, min(pad_x, target_w - cw))
    ch    = min(ch, target_h - pad_y)
    cw    = min(cw, target_w - pad_x)

    canvas[pad_y:pad_y+ch, pad_x:pad_x+cw] = content[:ch, :cw]
    return canvas


def _extract_cloth_mask(cloth_bgra):
    """
    Extracts binary mask from cloth alpha channel.
    Returns uint8 mask: 255=cloth, 0=background
    """
    if cloth_bgra.shape[2] == 4:
        alpha = cloth_bgra[:, :, 3]
        _, mask = cv2.threshold(alpha, 10, 255, cv2.THRESH_BINARY)
    else:
        # No alpha → detect non-white pixels
        bgr  = cloth_bgra[:, :, :3]
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(
            gray, 240, 255, cv2.THRESH_BINARY_INV
        )

    # Clean up
    k    = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask


def _composite_on_white(cloth_bgra):
    """
    Composites BGRA cloth onto white background → BGR.
    """
    h, w = cloth_bgra.shape[:2]
    white = np.ones((h, w, 3), dtype=np.uint8) * 255

    if cloth_bgra.shape[2] == 4:
        alpha = cloth_bgra[:, :, 3:4].astype(np.float32) / 255
        cloth = cloth_bgra[:, :, :3].astype(np.float32)
        result = (cloth * alpha +
                  white.astype(np.float32) * (1-alpha))
        return result.astype(np.uint8)
    else:
        return cloth_bgra[:, :, :3].copy()


# ═════════════════════════════════════════════════════════
# PART 3 — POSE HEATMAP GENERATION
# Creates a visual map showing where body joints are
# ═════════════════════════════════════════════════════════

def generate_pose_heatmap(person_img_bgr):
    """
    Generates a pose heatmap for VITON.

    What is a pose heatmap?
      Instead of giving the model (x,y) coordinates,
      we draw a Gaussian "blob" at each joint location.
      18 joints = 18 channels of blobs.
      Model reads this like a body blueprint.

    Returns: heatmap array [VITON_H, VITON_W, 18]
    """
    h, w   = person_img_bgr.shape[:2]
    n_joints = 18

    # Initialize heatmap (all zeros = no joints)
    heatmap = np.zeros((VITON_H, VITON_W, n_joints),
                       dtype=np.float32)

    # ── Get landmarks ─────────────────────────────────────
    landmarks = _get_pose_landmarks(person_img_bgr)

    if landmarks is None:
        print("[!] No pose for heatmap → using estimated joints")
        landmarks = _estimate_landmarks_from_proportions(h, w)

    # ── Draw Gaussian blob at each joint ─────────────────
    # Joint index mapping (COCO 18-joint format)
    joint_map = {
        0 : 'nose',
        1 : 'left_eye',
        2 : 'right_eye',
        5 : 'left_shoulder',
        6 : 'right_shoulder',
        7 : 'left_elbow',
        8 : 'right_elbow',
        9 : 'left_wrist',
        10: 'right_wrist',
        11: 'left_hip',
        12: 'right_hip',
        13: 'left_knee',
        14: 'right_knee',
    }

    for joint_idx, joint_name in joint_map.items():
        if joint_name in landmarks:
            px, py = landmarks[joint_name]

            # Scale to VITON dimensions
            px_v = int(px * VITON_W / w)
            py_v = int(py * VITON_H / h)

            # Draw Gaussian blob centered at joint
            heatmap[:, :, joint_idx] = _gaussian_blob(
                VITON_H, VITON_W,
                center_y=py_v,
                center_x=px_v,
                sigma=8
            )

    print(f"[✓] Pose heatmap: {n_joints} channels, "
          f"{VITON_W}×{VITON_H}")
    return heatmap


def _gaussian_blob(h, w, center_y, center_x, sigma=8):
    """
    Creates a 2D Gaussian blob (circular gradient).
    Peaks at (center_y, center_x), falls off smoothly.

    This is how each joint is represented visually.
    """
    y_grid, x_grid = np.ogrid[:h, :w]
    dist_sq = ((y_grid - center_y)**2 +
               (x_grid - center_x)**2)
    blob = np.exp(-dist_sq / (2 * sigma**2))
    return blob.astype(np.float32)


def _get_pose_landmarks(img_bgr):
    """Get pose landmarks using MediaPipe."""
    if not MP_POSE_OK:
        return None

    h, w = img_bgr.shape[:2]
    rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    try:
        with _mp_pose.Pose(
                static_image_mode=True,
                model_complexity=1,
                min_detection_confidence=0.35) as pose:
            results = pose.process(rgb)

        if not results.pose_landmarks:
            return None

        lm = results.pose_landmarks.landmark

        def px(lmk):
            return (
                max(0, min(int(lmk.x * w), w-1)),
                max(0, min(int(lmk.y * h), h-1))
            )

        p = _mp_pose.PoseLandmark
        return {
            'nose'          : px(lm[p.NOSE]),
            'left_eye'      : px(lm[p.LEFT_EYE]),
            'right_eye'     : px(lm[p.RIGHT_EYE]),
            'left_shoulder' : px(lm[p.LEFT_SHOULDER]),
            'right_shoulder': px(lm[p.RIGHT_SHOULDER]),
            'left_elbow'    : px(lm[p.LEFT_ELBOW]),
            'right_elbow'   : px(lm[p.RIGHT_ELBOW]),
            'left_wrist'    : px(lm[p.LEFT_WRIST]),
            'right_wrist'   : px(lm[p.RIGHT_WRIST]),
            'left_hip'      : px(lm[p.LEFT_HIP]),
            'right_hip'     : px(lm[p.RIGHT_HIP]),
            'left_knee'     : px(lm[p.LEFT_KNEE]),
            'right_knee'    : px(lm[p.RIGHT_KNEE]),
        }
    except Exception as e:
        print(f"[!] Landmark error: {e}")
        return None


def _estimate_landmarks_from_proportions(h, w):
    """
    Estimates body joint positions using standard
    human body proportions when MediaPipe fails.

    Based on average human proportions:
      Head = top 15% of body
      Shoulders = ~20% from top
      Waist = ~45% from top
      Hips = ~52% from top
    """
    return {
        'nose'          : (w//2,           int(h*0.08)),
        'left_eye'      : (int(w*0.46),    int(h*0.07)),
        'right_eye'     : (int(w*0.54),    int(h*0.07)),
        'left_shoulder' : (int(w*0.35),    int(h*0.22)),
        'right_shoulder': (int(w*0.65),    int(h*0.22)),
        'left_elbow'    : (int(w*0.25),    int(h*0.38)),
        'right_elbow'   : (int(w*0.75),    int(h*0.38)),
        'left_wrist'    : (int(w*0.18),    int(h*0.54)),
        'right_wrist'   : (int(w*0.82),    int(h*0.54)),
        'left_hip'      : (int(w*0.38),    int(h*0.52)),
        'right_hip'     : (int(w*0.62),    int(h*0.52)),
        'left_knee'     : (int(w*0.36),    int(h*0.72)),
        'right_knee'    : (int(w*0.64),    int(h*0.72)),
    }


# ═════════════════════════════════════════════════════════
# PART 4 — SEGMENTATION MAP
# Labels each pixel: background/hair/face/shirt/pants/skin
# ═════════════════════════════════════════════════════════

def generate_segmentation_map(person_img_bgr,
                               shirt_mask=None,
                               landmarks=None):
    """
    Creates a semantic segmentation map.

    Each pixel gets a LABEL number:
      0 = background
      1 = hat/hair
      2 = face/head skin
      3 = upper clothes (shirt) ← most important!
      4 = pants/skirt
      5 = left arm skin
      6 = right arm skin
      7 = left leg
      8 = right leg
      9 = left shoe
      10 = right shoe
      11 = body/torso (neck area)

    This tells the model: "HERE is where the shirt goes"
    The model uses this as guidance for cloth placement.

    Returns: segmap [VITON_H, VITON_W] with label values
    """
    h, w   = person_img_bgr.shape[:2]
    segmap = np.zeros((VITON_H, VITON_W), dtype=np.uint8)

    # ── Get landmarks if not provided ─────────────────────
    if landmarks is None:
        landmarks = _get_pose_landmarks(person_img_bgr)
    if landmarks is None:
        landmarks = _estimate_landmarks_from_proportions(h, w)

    # Scale landmarks to VITON dimensions
    lm_v = {}
    for name, (px, py) in landmarks.items():
        lm_v[name] = (
            int(px * VITON_W / w),
            int(py * VITON_H / h)
        )

    # ── Build segmentation zones ──────────────────────────
    # We build from bottom to top so upper layers override

    # 1. Full background
    segmap[:] = 0

    # 2. Lower body zones (estimated)
    _draw_lower_body(segmap, lm_v)

    # 3. Arms
    _draw_arms(segmap, lm_v)

    # 4. Upper clothes / shirt zone (label=3)
    if shirt_mask is not None:
        # Resize provided shirt mask to VITON size
        shirt_v = cv2.resize(
            shirt_mask, (VITON_W, VITON_H),
            interpolation=cv2.INTER_NEAREST
        )
        segmap[shirt_v > 127] = 3
    else:
        _draw_shirt_zone(segmap, lm_v)

    # 5. Face/neck (always on top)
    _draw_face(segmap, lm_v)

    # 6. Hair (very top)
    _draw_hair(segmap, lm_v)

    print(f"[✓] Segmap generated: {VITON_W}×{VITON_H}, "
          f"labels {np.unique(segmap).tolist()}")
    return segmap


def _draw_shirt_zone(seg, lm):
    """Draw shirt region (label 3) using landmarks."""
    if 'left_shoulder' not in lm or 'right_shoulder' not in lm:
        return

    ls = lm['left_shoulder']
    rs = lm['right_shoulder']
    lh = lm.get('left_hip',  (ls[0], ls[1]+80))
    rh = lm.get('right_hip', (rs[0], rs[1]+80))

    pad_x = int((rs[0] - ls[0]) * 0.18)
    pts   = np.array([
        (ls[0]-pad_x, ls[1]-10),
        (rs[0]+pad_x, rs[1]-10),
        (rh[0]+pad_x, rh[1]+10),
        (lh[0]-pad_x, lh[1]+10),
    ], dtype=np.int32)

    pts[:, 0] = np.clip(pts[:, 0], 0, VITON_W-1)
    pts[:, 1] = np.clip(pts[:, 1], 0, VITON_H-1)
    cv2.fillPoly(seg, [pts], 3)


def _draw_face(seg, lm):
    """Draw face region (label 2)."""
    nose = lm.get('nose', (VITON_W//2, int(VITON_H*0.12)))
    le   = lm.get('left_eye',  (nose[0]-12, nose[1]-8))
    re   = lm.get('right_eye', (nose[0]+12, nose[1]-8))
    ls   = lm.get('left_shoulder',  (nose[0]-30, nose[1]+40))
    rs   = lm.get('right_shoulder', (nose[0]+30, nose[1]+40))

    face_w = abs(re[0] - le[0]) + 20
    face_h = int(face_w * 1.3)
    cx, cy = nose[0], nose[1] - 5

    pts = np.array([
        (cx - face_w//2, cy - face_h//2),
        (cx + face_w//2, cy - face_h//2),
        (cx + face_w//2, cy + face_h//2),
        (cx - face_w//2, cy + face_h//2),
    ], dtype=np.int32)
    pts[:, 0] = np.clip(pts[:, 0], 0, VITON_W-1)
    pts[:, 1] = np.clip(pts[:, 1], 0, VITON_H-1)
    cv2.fillPoly(seg, [pts], 2)

    # Neck
    neck_top = cy + face_h//2
    neck_bot = lm.get('left_shoulder', (0, neck_top+20))[1]
    nx1 = max(0, cx - 10)
    nx2 = min(VITON_W-1, cx + 10)
    if neck_bot > neck_top:
        seg[neck_top:neck_bot, nx1:nx2] = 11


def _draw_hair(seg, lm):
    """Draw hair region (label 1) above face."""
    nose  = lm.get('nose', (VITON_W//2, int(VITON_H*0.12)))
    le    = lm.get('left_eye',  (nose[0]-12, nose[1]-8))
    re    = lm.get('right_eye', (nose[0]+12, nose[1]-8))
    face_w = abs(re[0]-le[0]) + 22
    cy     = nose[1] - 5
    top    = max(0, cy - int(face_w*0.9))
    bot    = cy - int(face_w*0.1)
    cx     = nose[0]
    x1     = max(0, cx - face_w//2 - 5)
    x2     = min(VITON_W-1, cx + face_w//2 + 5)
    if bot > top:
        seg[top:bot, x1:x2] = 1


def _draw_arms(seg, lm):
    """Draw arm skin regions (labels 5, 6)."""
    pairs = [
        ('left_shoulder',  'left_elbow',  'left_wrist',  5),
        ('right_shoulder', 'right_elbow', 'right_wrist', 6),
    ]
    for sh_key, el_key, wr_key, label in pairs:
        if sh_key not in lm:
            continue
        sh = lm[sh_key]
        el = lm.get(el_key, (sh[0], sh[1]+30))
        wr = lm.get(wr_key, (el[0], el[1]+30))
        r  = 9
        for pt in [el, wr]:
            cv2.circle(seg, pt, r, label, -1)
        pts = np.array([
            (el[0]-r, el[1]),
            (el[0]+r, el[1]),
            (wr[0]+r, wr[1]),
            (wr[0]-r, wr[1]),
        ], dtype=np.int32)
        pts[:,0] = np.clip(pts[:,0], 0, VITON_W-1)
        pts[:,1] = np.clip(pts[:,1], 0, VITON_H-1)
        cv2.fillPoly(seg, [pts], label)


def _draw_lower_body(seg, lm):
    """Draw legs and pants (labels 4,7,8,9,10)."""
    lh = lm.get('left_hip',  (int(VITON_W*0.38), int(VITON_H*0.52)))
    rh = lm.get('right_hip', (int(VITON_W*0.62), int(VITON_H*0.52)))
    lk = lm.get('left_knee', (lh[0],             int(VITON_H*0.72)))
    rk = lm.get('right_knee',(rh[0],             int(VITON_H*0.72)))

    # Pants zone (between hips and knees)
    pts = np.array([
        (lh[0]-14, lh[1]),
        (rh[0]+14, rh[1]),
        (rk[0]+12, rk[1]),
        (lk[0]-12, lk[1]),
    ], dtype=np.int32)
    pts[:,0] = np.clip(pts[:,0], 0, VITON_W-1)
    pts[:,1] = np.clip(pts[:,1], 0, VITON_H-1)
    cv2.fillPoly(seg, [pts], 4)

    # Lower legs
    bot = min(VITON_H-1, rk[1]+60)
    cv2.rectangle(seg,
                  (lk[0]-10, lk[1]),
                  (lk[0]+10, bot), 7, -1)
    cv2.rectangle(seg,
                  (rk[0]-10, rk[1]),
                  (rk[0]+10, bot), 8, -1)


# ═════════════════════════════════════════════════════════
# PART 5 — AGNOSTIC PERSON IMAGE
# Person with shirt area blanked out
# ═════════════════════════════════════════════════════════

def create_agnostic_image(person_img_bgr,
                           shirt_mask,
                           segmap=None):
    """
    Creates the "agnostic" person image.

    "Agnostic" means we REMOVE all clothing info
    from the person image so the model doesn't get
    confused by the original shirt.

    We replace the shirt area with:
      - Skin color estimation (for seamless look)
      - Slight blur at boundaries
      - Preserved: face, hair, arms, legs, background

    Returns: BGR image same size as input
    """
    h, w    = person_img_bgr.shape[:2]
    result  = person_img_bgr.copy()

    # Resize mask to match image
    if shirt_mask.shape[:2] != (h, w):
        shirt_r = cv2.resize(
            shirt_mask, (w, h),
            interpolation=cv2.INTER_LINEAR
        )
    else:
        shirt_r = shirt_mask.copy()

    _, shirt_binary = cv2.threshold(
        shirt_r, 127, 255, cv2.THRESH_BINARY
    )

    # ── Estimate skin/body color ──────────────────────────
    body_color = _estimate_body_color(person_img_bgr,
                                       shirt_binary)

    # ── Create smooth fill for shirt area ─────────────────
    # Use inpainting for natural look
    dilated_mask = cv2.dilate(
        shirt_binary, np.ones((5,5), np.uint8), iterations=2
    )
    inpainted = cv2.inpaint(
        person_img_bgr, dilated_mask,
        inpaintRadius=10,
        flags=cv2.INPAINT_TELEA
    )

    # ── Blend: inpainted in shirt zone ────────────────────
    mask_f   = shirt_binary.astype(np.float32) / 255.0
    mask_f   = cv2.GaussianBlur(mask_f, (31, 31), 0)
    mask_3ch = np.stack([mask_f]*3, axis=2)

    result = (
        inpainted.astype(np.float32) * mask_3ch +
        person_img_bgr.astype(np.float32) * (1-mask_3ch)
    ).astype(np.uint8)

    # Resize to VITON dims
    result_v = cv2.resize(
        result, (VITON_W, VITON_H),
        interpolation=cv2.INTER_LANCZOS4
    )

    print(f"[✓] Agnostic image created: {VITON_W}×{VITON_H}")
    return result_v


def _estimate_body_color(img_bgr, shirt_mask):
    """
    Estimates body/skin color by sampling areas
    near but outside the shirt mask.
    """
    h, w = img_bgr.shape[:2]

    # Dilate shirt mask to get nearby region
    dilated  = cv2.dilate(
        shirt_mask, np.ones((20,20), np.uint8)
    )
    boundary = cv2.bitwise_and(
        dilated, cv2.bitwise_not(shirt_mask)
    )

    # Sample pixels in boundary region
    pixels = img_bgr[boundary > 0]
    if len(pixels) > 10:
        return tuple(map(int, np.median(pixels, axis=0)))

    return (200, 180, 160)  # default skin-ish


# ═════════════════════════════════════════════════════════
# PART 6 — VISUALIZE PREPARED INPUTS (Debug)
# ═════════════════════════════════════════════════════════

def visualize_inputs(person_prep, cloth_prep,
                     pose_heatmap, segmap,
                     agnostic_img,
                     output_dir='static/outputs'):
    """
    Saves all prepared inputs as images so you can
    see exactly what the model receives.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Person prepared
    cv2.imwrite(
        f'{output_dir}/prep_person.jpg',
        person_prep['bgr']
    )

    # 2. Cloth prepared
    cv2.imwrite(
        f'{output_dir}/prep_cloth.jpg',
        cloth_prep['bgr']
    )

    # 3. Cloth mask
    cv2.imwrite(
        f'{output_dir}/prep_cloth_mask.jpg',
        cloth_prep['mask']
    )

    # 4. Pose heatmap (sum all channels for visualization)
    heatmap_viz = pose_heatmap.sum(axis=2)
    heatmap_viz = np.clip(heatmap_viz * 255, 0, 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_viz, cv2.COLORMAP_JET)
    cv2.imwrite(
        f'{output_dir}/prep_pose_heatmap.jpg',
        heatmap_color
    )

    # 5. Segmentation map (colorized)
    seg_color = _colorize_segmap(segmap)
    cv2.imwrite(
        f'{output_dir}/prep_segmap.jpg',
        seg_color
    )

    # 6. Agnostic image
    cv2.imwrite(
        f'{output_dir}/prep_agnostic.jpg',
        agnostic_img
    )

    # 7. Side-by-side summary
    h = VITON_H
    summary = np.hstack([
        cv2.resize(person_prep['bgr'],   (VITON_W, h)),
        cv2.resize(agnostic_img,         (VITON_W, h)),
        cv2.resize(cloth_prep['bgr'],    (VITON_W, h)),
        cv2.resize(seg_color,            (VITON_W, h)),
    ])

    labels = ['Person', 'Agnostic', 'Cloth', 'SegMap']
    for i, label in enumerate(labels):
        cv2.putText(summary, label,
                    (i*VITON_W + 5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0,255,0), 1)

    cv2.imwrite(f'{output_dir}/prep_summary.jpg', summary)

    print(f"\n[✓] Preparation debug images saved:")
    print(f"    prep_person.jpg      ← resized person")
    print(f"    prep_cloth.jpg       ← resized clothing")
    print(f"    prep_cloth_mask.jpg  ← cloth binary mask")
    print(f"    prep_pose_heatmap.jpg← joint locations")
    print(f"    prep_segmap.jpg      ← body part labels")
    print(f"    prep_agnostic.jpg    ← shirt-removed person")
    print(f"    prep_summary.jpg     ← all side by side")


def _colorize_segmap(segmap):
    """Assigns distinct colors to each segmap label."""
    colors = {
        0 : (20,  20,  20),    # background
        1 : (60,  20,  100),   # hair
        2 : (220, 180, 140),   # face
        3 : (60,  100, 200),   # upper clothes (blue)
        4 : (40,  80,  40),    # pants
        5 : (200, 150, 120),   # left arm
        6 : (180, 130, 100),   # right arm
        7 : (40,  60,  100),   # left leg
        8 : (30,  50,  80),    # right leg
        9 : (20,  20,  40),    # left shoe
        10: (15,  15,  35),    # right shoe
        11: (190, 160, 130),   # neck/torso
    }
    h, w = segmap.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)
    for label, color in colors.items():
        color_img[segmap == label] = color
    return color_img


# ═════════════════════════════════════════════════════════
# PART 7 — MAIN FUNCTION
# ═════════════════════════════════════════════════════════

def prepare_all_inputs(person_path, cloth_path,
                        shirt_mask=None,
                        landmarks=None,
                        save_debug=True):
    """
    Master function — prepares ALL inputs for VITON model.

    Returns a dict with everything the model needs:
    {
      'person'    : prepared person image (BGR)
      'cloth'     : prepared cloth image (BGR)
      'cloth_mask': cloth binary mask
      'pose_hmap' : pose heatmap [H,W,18]
      'segmap'    : segmentation map [H,W]
      'agnostic'  : person with shirt removed
      'orig_h'    : original person height
      'orig_w'    : original person width
    }
    """
    print("\n── INPUT PREPARATION PIPELINE ─────────────────")

    # Load images
    person_bgr = cv2.imread(person_path)
    cloth_bgr  = cv2.imread(cloth_path, cv2.IMREAD_UNCHANGED)

    if person_bgr is None:
        raise ValueError(f"Cannot read: {person_path}")
    if cloth_bgr is None:
        raise ValueError(f"Cannot read: {cloth_path}")

    orig_h, orig_w = person_bgr.shape[:2]

    # Step 1: Prepare person image
    print("[1/5] Preparing person image...")
    person_prep = prepare_person_image(person_bgr)

    # Step 2: Prepare clothing image
    print("[2/5] Preparing clothing image...")
    cloth_prep = prepare_clothing_image(cloth_bgr)

    # Step 3: Generate pose heatmap
    print("[3/5] Generating pose heatmap...")
    pose_hmap = generate_pose_heatmap(person_bgr)

    # Step 4: Generate segmentation map
    print("[4/5] Generating segmentation map...")
    segmap = generate_segmentation_map(
        person_bgr,
        shirt_mask=shirt_mask,
        landmarks=landmarks
    )

    # Step 5: Create agnostic image
    print("[5/5] Creating agnostic image...")
    if shirt_mask is None:
        # Create a basic shirt mask from segmap
        shirt_mask_from_seg = (
            (segmap == 3).astype(np.uint8) * 255
        )
        shirt_mask_from_seg = cv2.resize(
            shirt_mask_from_seg,
            (orig_w, orig_h),
            interpolation=cv2.INTER_NEAREST
        )
        use_mask = shirt_mask_from_seg
    else:
        use_mask = shirt_mask

    agnostic = create_agnostic_image(
        person_bgr, use_mask, segmap
    )

    inputs = {
        'person'    : person_prep['bgr'],
        'cloth'     : cloth_prep['bgr'],
        'cloth_mask': cloth_prep['mask'],
        'pose_hmap' : pose_hmap,
        'segmap'    : segmap,
        'agnostic'  : agnostic,
        'orig_h'    : orig_h,
        'orig_w'    : orig_w,
    }

    # Save debug images
    if save_debug:
        visualize_inputs(
            person_prep, cloth_prep,
            pose_hmap, segmap, agnostic
        )

    print("[✓] All inputs prepared!")
    print("────────────────────────────────────────────────\n")
    return inputs


# ── Quick test ─────────────────────────────────────────────
if __name__ == '__main__':
    person = 'static/uploads/person.jpg'
    cloth  = 'static/uploads/clothing.png'

    if not os.path.exists(person):
        print("❌ Upload images first at http://localhost:5000/upload")
    else:
        print("🔬 Testing input preparation...")
        inputs = prepare_all_inputs(
            person, cloth, save_debug=True
        )
        print("\n✅ Done! Check static/outputs/ for:")
        print("   prep_summary.jpg ← all inputs side by side")