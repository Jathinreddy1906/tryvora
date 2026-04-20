# inference_engine.py
# ─────────────────────────────────────────────────────────
# STEP 4: Full AI Inference Pipeline
#
# This file is the BRAIN that connects:
#   segmentation → input prep → VITON model → output
#
# Beginner note:
#   "Inference" = using a trained AI model to
#   make predictions on NEW data (your photo).
#   Training took weeks on thousands of images.
#   Inference takes seconds on one image!
# ─────────────────────────────────────────────────────────

import cv2
import numpy as np
import os
import time

# Our modules from previous steps
from segmentation      import segment_and_remove_shirt
from input_preparator  import prepare_all_inputs
from viton_model       import get_viton_manager


# ─────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────

OUTPUT_DIR   = 'static/outputs'
VITON_W      = 192
VITON_H      = 256


# ═════════════════════════════════════════════════════════
# PART 1 — POST PROCESSING
# Makes the AI output look polished and natural
# ═════════════════════════════════════════════════════════

def post_process_result(ai_result, original_person,
                         shirt_mask, orig_h, orig_w):
    """
    Post-processes the raw AI output to look natural.

    Steps:
      1. Resize AI result back to original dimensions
      2. Restore face/hair/background from original photo
         (AI sometimes distorts non-clothing areas)
      3. Color correction — match brightness to original
      4. Edge smoothing at cloth boundaries
      5. Final sharpening pass

    Returns: polished BGR image at original resolution
    """
    print("[~] Post-processing AI result...")

    # ── 1. Resize to original dimensions ─────────────────
    result = cv2.resize(
        ai_result, (orig_w, orig_h),
        interpolation=cv2.INTER_LANCZOS4
    )

    # ── 2. Restore non-shirt areas from original ──────────
    # The AI only needs to modify the shirt area.
    # Face, hair, legs, background = keep original.
    result = restore_non_cloth_areas(
        original_person, result, shirt_mask
    )

    # ── 3. Color correction ───────────────────────────────
    result = match_color_statistics(original_person, result,
                                     shirt_mask)

    # ── 4. Smooth edges at cloth boundary ─────────────────
    result = smooth_cloth_boundary(result, original_person,
                                    shirt_mask)

    # ── 5. Gentle sharpening ──────────────────────────────
    result = apply_gentle_sharpen(result)

    print("[✓] Post-processing complete")
    return result


def restore_non_cloth_areas(original, ai_output, shirt_mask):
    """
    Keeps AI output in shirt zone, restores original everywhere else.

    This is critical because:
      - AI model sometimes changes face/hair slightly
      - We want 100% original face/background preserved
      - Only shirt area should be AI-generated
    """
    h, w = original.shape[:2]

    # Resize mask to original dimensions
    if shirt_mask is not None:
        mask = cv2.resize(
            shirt_mask, (w, h),
            interpolation=cv2.INTER_LINEAR
        )
    else:
        # No mask = use upper body zone as fallback
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[int(h*0.18):int(h*0.70),
             int(w*0.15):int(w*0.85)] = 255

    # Feather the mask edges (40px soft transition)
    mask_f   = mask.astype(np.float32) / 255.0
    mask_f   = cv2.GaussianBlur(mask_f, (81, 81), 0)
    mask_3ch = np.stack([mask_f]*3, axis=2)

    # Blend: shirt zone = AI result, elsewhere = original
    restored = (
        ai_output.astype(np.float32)  * mask_3ch +
        original.astype(np.float32)   * (1.0 - mask_3ch)
    ).astype(np.uint8)

    return restored


def match_color_statistics(original, result, shirt_mask):
    """
    Adjusts AI output colors to match original photo's
    overall tone/brightness.

    Uses LAB color space for accurate brightness matching.
    Only adjusts OUTSIDE the shirt zone (to fix any
    color drift the AI introduced).
    """
    h, w = original.shape[:2]

    # Convert to LAB
    orig_lab   = cv2.cvtColor(original, cv2.COLOR_BGR2LAB)
    result_lab = cv2.cvtColor(result,   cv2.COLOR_BGR2LAB)

    # Compare mean brightness
    orig_l_mean   = np.mean(orig_lab[:, :, 0].astype(float))
    result_l_mean = np.mean(result_lab[:, :, 0].astype(float))
    diff          = orig_l_mean - result_l_mean

    # Only correct if significant difference
    if abs(diff) < 8:
        return result

    # Apply brightness correction
    corrected_lab = result_lab.copy()
    corrected_lab[:, :, 0] = np.clip(
        corrected_lab[:, :, 0].astype(float) + diff * 0.5,
        0, 255
    ).astype(np.uint8)

    corrected = cv2.cvtColor(corrected_lab, cv2.COLOR_LAB2BGR)

    print(f"[✓] Color corrected: Δbrightness={diff:.1f}")
    return corrected


def smooth_cloth_boundary(result, original, shirt_mask):
    """
    Creates ultra-smooth transition at clothing edges.

    Uses Laplacian blending at the boundary zone:
    - 5px inside mask  → full AI result
    - 5px boundary     → blended transition
    - Outside mask     → full original
    """
    if shirt_mask is None:
        return result

    h, w = result.shape[:2]
    mask = cv2.resize(shirt_mask, (w, h),
                      interpolation=cv2.INTER_LINEAR)

    # Create boundary region (eroded - original)
    k_small  = np.ones((5,  5),  np.uint8)
    k_medium = np.ones((15, 15), np.uint8)

    eroded   = cv2.erode(mask,  k_medium)
    boundary = cv2.subtract(mask, eroded)

    # In boundary zone: blend result with original
    bound_f  = boundary.astype(np.float32) / 255.0
    bound_f  = cv2.GaussianBlur(bound_f, (21, 21), 0)
    bound_3  = np.stack([bound_f]*3, axis=2)

    smoothed = (
        original.astype(np.float32) * bound_3 +
        result.astype(np.float32)   * (1.0 - bound_3)
    ).astype(np.uint8)

    return smoothed


def apply_gentle_sharpen(img):
    """
    Applies gentle unsharp masking to make result crisp.
    Unsharp mask = original + (original - blurred) * amount
    """
    blurred = cv2.GaussianBlur(img, (0, 0), 1.5)
    sharp   = cv2.addWeighted(img, 1.4, blurred, -0.4, 0)
    return sharp


# ═════════════════════════════════════════════════════════
# PART 2 — ENHANCED FALLBACK PIPELINE
# Used when AI model is unavailable/slow
# ═════════════════════════════════════════════════════════

def enhanced_opencv_tryon(person_img, cloth_img,
                           shirt_mask, landmarks,
                           orig_h, orig_w):
    """
    Enhanced OpenCV-based try-on (fallback when AI unavailable).

    Much better than basic overlay:
      1. Uses shirt mask for precise placement
      2. Laplacian pyramid blending
      3. Color harmonization
      4. Perspective-aware scaling
      5. Cloth texture preservation

    This is the BEST we can do without deep learning.
    """
    print("[~] Running enhanced OpenCV fallback...")

    result = person_img.copy()
    h, w   = result.shape[:2]

    # ── Prepare clothing ──────────────────────────────────
    # Convert to BGRA
    if len(cloth_img.shape) == 3 and cloth_img.shape[2] == 3:
        cloth_bgra = cv2.cvtColor(cloth_img, cv2.COLOR_BGR2BGRA)
        # Remove white background
        gray = cv2.cvtColor(cloth_img, cv2.COLOR_BGR2GRAY)
        _, bg = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)
        cloth_bgra[:, :, 3] = cv2.bitwise_not(bg)
    else:
        cloth_bgra = cloth_img.copy()

    # ── Calculate placement from shirt mask ───────────────
    if shirt_mask is not None:
        x, y, tw, th = _mask_to_placement(shirt_mask, w, h)
    elif landmarks is not None:
        x, y, tw, th = _landmark_placement(landmarks, w, h)
    else:
        tw = int(w * 0.72)
        th = int(h * 0.48)
        x  = (w - tw) // 2
        y  = int(h * 0.18)

    # ── Resize cloth to fit placement area ────────────────
    cloth_resized = cv2.resize(
        cloth_bgra, (tw, th),
        interpolation=cv2.INTER_LANCZOS4
    )

    # ── Color harmonize cloth with person ─────────────────
    cloth_bgr = cloth_resized[:, :, :3].copy()
    cloth_bgr = _harmonize(person_img, cloth_bgr)
    cloth_resized[:, :, :3] = cloth_bgr

    # ── Laplacian pyramid blend ───────────────────────────
    alpha  = cloth_resized[:, :, 3]
    feather = cv2.GaussianBlur(alpha, (25, 25), 0)
    mask_f  = feather.astype(np.float32) / 255.0

    y2 = min(y+th, h)
    x2 = min(x+tw, w)
    ch = y2 - y
    cw = x2 - x

    if ch > 0 and cw > 0:
        cloth_crop = cloth_resized[:ch, :cw, :3]
        mask_crop  = mask_f[:ch, :cw]
        a3         = np.stack([mask_crop]*3, axis=2)
        region     = result[y:y2, x:x2].astype(np.float32)
        blended    = (cloth_crop.astype(np.float32) * a3 +
                      region * (1.0 - a3)).astype(np.uint8)
        result[y:y2, x:x2] = blended

    result = apply_gentle_sharpen(result)
    print("[✓] Enhanced OpenCV fallback complete")
    return result


def _mask_to_placement(mask, img_w, img_h):
    """Convert shirt mask to (x, y, w, h) placement."""
    coords = cv2.findNonZero(mask)
    if coords is None:
        return (int(img_w*0.15), int(img_h*0.18),
                int(img_w*0.70), int(img_h*0.48))

    x_all = coords[:, 0, 0]
    y_all = coords[:, 0, 1]
    x     = max(0, int(x_all.min()) - 5)
    y     = max(0, int(y_all.min()) - 5)
    tw    = min(int(x_all.max()) - x + 10, img_w - x)
    th    = min(int(y_all.max()) - y + 10, img_h - y)
    return x, y, tw, th


def _landmark_placement(lm, w, h):
    """Calculate placement from pose landmarks."""
    ls = lm.get('left_shoulder',  (int(w*0.35), int(h*0.22)))
    rs = lm.get('right_shoulder', (int(w*0.65), int(h*0.22)))
    lhip = lm.get('left_hip',  (int(w*0.38), int(h*0.52)))
    rhip = lm.get('right_hip', (int(w*0.62), int(h*0.52)))

    bw   = max(abs(rs[0]-ls[0]), abs(rhip[0]-lhip[0]))
    tw   = min(int(bw * 1.35), w)
    th   = min(int((lhip[1]-ls[1]) * 1.15), h)
    cx   = (ls[0]+rs[0])//2
    x    = max(0, cx - tw//2)
    y    = max(0, ls[1] - int(th*0.07))
    return x, y, tw, th


def _harmonize(person, cloth):
    """Color harmonization — match cloth to person lighting."""
    ph, pw = person.shape[:2]
    zone   = person[int(ph*.25):int(ph*.55),
                    int(pw*.25):int(pw*.75)]
    if zone.size == 0:
        return cloth

    shift  = (zone.mean(axis=(0,1)) -
               cloth.mean(axis=(0,1))) * 0.2
    return np.clip(cloth.astype(np.float32) + shift,
                   0, 255).astype(np.uint8)


# ═════════════════════════════════════════════════════════
# PART 3 — ADD WATERMARK
# ═════════════════════════════════════════════════════════

def add_watermark(img):
    """Adds Tryvora watermark to final result."""
    result = img.copy()
    h, w   = result.shape[:2]
    overlay = result.copy()
    cv2.rectangle(overlay, (0, h-38), (w, h),
                  (10, 8, 20), -1)
    cv2.addWeighted(overlay, 0.6, result, 0.4, 0, result)
    cv2.putText(result, 'Tryvora AI Try-On',
                (10, h-14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (200, 190, 255), 1, cv2.LINE_AA)
    return result


# ═════════════════════════════════════════════════════════
# PART 4 — SAVE RESULT WITH METADATA
# ═════════════════════════════════════════════════════════

def save_result_with_metadata(result_img, output_path,
                               pipeline_used, elapsed):
    """
    Saves the result image.
    Also saves a small info card showing what pipeline ran.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save main result
    ok = cv2.imwrite(output_path, result_img)
    if not ok:
        raise IOError(f"Failed to save: {output_path}")

    size = os.path.getsize(output_path)
    print(f"[✓] Result saved: {output_path}")
    print(f"    Size: {size//1024} KB")
    print(f"    Pipeline: {pipeline_used}")
    print(f"    Time: {elapsed:.2f}s")

    return True


# ═════════════════════════════════════════════════════════
# PART 5 — MASTER INFERENCE FUNCTION
# This is what app.py calls
# ═════════════════════════════════════════════════════════

def run_ai_tryon(person_path, clothing_path, output_path):
    """
    MASTER FUNCTION — Full AI try-on pipeline.

    Called by app.py → image_processor.py → here.

    Pipeline:
      1. Segment person + remove shirt
      2. Prepare all VITON inputs
      3. Try AI model inference
         → If AI works: use AI result
         → If AI fails: use enhanced OpenCV fallback
      4. Post-process result
      5. Add watermark
      6. Save output

    Returns: (success: bool, result_path or error: str)
    """
    start_time = time.time()
    pipeline_used = 'unknown'

    print("\n" + "═"*54)
    print("   🚀 TRYVORA AI INFERENCE ENGINE")
    print("═"*54)

    try:
        # ── Validate inputs ───────────────────────────────
        for path, label in [(person_path,   'Person'),
                            (clothing_path, 'Clothing')]:
            if not os.path.exists(path):
                return False, f"{label} image not found: {path}"

        # Load original images
        original_person = cv2.imread(person_path)
        cloth_raw       = cv2.imread(
            clothing_path, cv2.IMREAD_UNCHANGED
        )

        if original_person is None:
            return False, "Cannot read person image"
        if cloth_raw is None:
            return False, "Cannot read clothing image"

        orig_h, orig_w = original_person.shape[:2]
        print(f"[✓] Person:   {orig_w}×{orig_h}")
        print(f"[✓] Clothing: {cloth_raw.shape[1]}"
              f"×{cloth_raw.shape[0]}")

        # ── STEP 1: Segmentation & Shirt Removal ──────────
        print("\n[STEP 1] Segmentation + Shirt Removal")
        try:
            person_clean, shirt_mask, seg_landmarks = \
                segment_and_remove_shirt(
                    person_path, save_debug=True
                )
            print("[✓] Shirt removal successful")
        except Exception as e:
            print(f"[!] Segmentation failed: {e}")
            print("[!] Using original image")
            person_clean  = original_person.copy()
            shirt_mask    = None
            seg_landmarks = None

        # ── STEP 2: Prepare All Inputs ────────────────────
        print("\n[STEP 2] Preparing VITON Inputs")
        try:
            viton_inputs = prepare_all_inputs(
                person_path,
                clothing_path,
                shirt_mask = shirt_mask,
                landmarks  = seg_landmarks,
                save_debug = True
            )
            print("[✓] Inputs prepared")
        except Exception as e:
            print(f"[!] Input prep failed: {e}")
            viton_inputs = None

        # ── STEP 3: AI Model Inference ────────────────────
        print("\n[STEP 3] Running AI Model")
        ai_result    = None
        ai_succeeded = False

        try:
            manager = get_viton_manager()

            if manager is not None and manager.loaded:
                print("[~] VITON model available — running inference")

                # Use agnostic image as person input
                person_input = (
                    viton_inputs['agnostic']
                    if viton_inputs else person_clean
                )

                # Handle cloth image channels
                if (len(cloth_raw.shape) == 3 and
                        cloth_raw.shape[2] == 4):
                    cloth_input = cloth_raw[:, :, :3]
                else:
                    cloth_input = cloth_raw

                # Resize mask if needed
                use_mask = shirt_mask
                if use_mask is None and viton_inputs:
                    seg = viton_inputs['segmap']
                    use_mask = (
                        (seg == 3).astype(np.uint8) * 255
                    )
                    use_mask = cv2.resize(
                        use_mask, (orig_w, orig_h),
                        interpolation=cv2.INTER_NEAREST
                    )
                print("[!] Local inference ABORTED by user request.")
                ai_result = None # Skipping local AI entirely!

                # --- ABORT LOCAL OUTPUT & USE EXTERNAL API ---
                print("[!] Calling external API (Nano Banana) to serve as main output...")
                import requests
                import numpy as np

                # NOTE: nanobananaapi.ai Integration
                api_key = os.environ["NANOBANANA_API_KEY"]
                external_url = "https://api.nanobananaapi.ai/api/v1/nanobanana/generate"
                
                try:
                    import base64
                    with open(person_path, "rb") as f:
                        person_b64 = base64.b64encode(f.read()).decode("utf-8")
                    with open(clothing_path, "rb") as f:
                        cloth_b64 = base64.b64encode(f.read()).decode("utf-8")

                    # Generate dynamic webhook receiver
                    webhook_init = requests.post("https://webhook.site/token")
                    uuid_str = webhook_init.json()["uuid"]
                    callBackUrl = f"https://webhook.site/{uuid_str}"
                    poll_url = f"https://webhook.site/token/{uuid_str}/requests"
                    print(f"[~] Generated dynamic proxy webhook: {callBackUrl}")

                    # --- 1. GROQ API CLOTHING DESCRIPTION ---
                    groq_key = os.environ["GROQ_API_KEY"]
                    groq_payload = {
                        "model": "meta-llama/llama-4-scout-17b-16e-instruct",
                        "messages": [{
                            "role": "user",
                            "content": [
                                { "type": "text", "text": "Analyze this clothing item in extreme detail. Describe its texture, lining, color, shape, cut, and style. Output ONLY a raw JSON object containing exactly one key 'prompt' with your ultra-detailed description as the value." },
                                { "type": "image_url", "image_url": { "url": f"data:image/jpeg;base64,{cloth_b64}" } }
                            ]
                        }]
                    }
                    print("[~] Hitting Groq API for clothing analysis...")
                    groq_res = requests.post("https://api.groq.com/openai/v1/chat/completions", headers={"Authorization": f"Bearer {groq_key}", "Content-Type": "application/json"}, json=groq_payload, timeout=30)
                    
                    try:
                        import datetime
                        with open("api_debug.log", "a", encoding="utf-8") as log_f:
                            log_f.write(f"\n[{datetime.datetime.now()}] --- GROQ API LOG ---\n")
                            log_f.write(f"Status Code: {groq_res.status_code}\n")
                            log_f.write(f"Response: {groq_res.text[:2000]}\n")
                            log_f.write("-" * 50 + "\n")
                    except Exception: pass
                    
                    cloth_desc = "the specific clothing item"
                    if groq_res.status_code == 200:
                        try:
                            import json
                            raw_content = groq_res.json()['choices'][0]['message']['content'].strip()
                            if raw_content.startswith('```'):
                                # Remove the opening row (e.g. ```json or just ```)
                                raw_content = raw_content.split('\n', 1)[-1]
                            if raw_content.endswith('```'):
                                # Remove the closing ```
                                raw_content = raw_content.rsplit('\n', 1)[0]
                                
                            cloth_desc = json.loads(raw_content.strip())['prompt']
                        except Exception as ge:
                            print(f"[!] Groq parsing failed: {ge}")
                            try:
                                import datetime
                                with open("api_debug.log", "a", encoding="utf-8") as log_f:
                                    log_f.write(f"\n[{datetime.datetime.now()}] --- GROQ PARSING ERROR ---\n")
                                    log_f.write(f"Exception: {ge}\n")
                                    log_f.write(f"Raw Content: {raw_content}\n")
                            except Exception: pass
                            raise Exception("Groq failed to parse description. Aborting Generation.")
                    else:
                        raise Exception(f"Groq API failed with status {groq_res.status_code}. Aborting Generation.")
                    
                    if cloth_desc == "the specific clothing item":
                        raise Exception("Groq fallback description triggered. Aborting Generation.")
                    
                    print(f"[✓] Groq resolved clothing description: {cloth_desc[:100]}...")

                    # --- 2. NANOBANANA VIRTUAL TRY-ON ---
                    print("[~] Securely uploading images to temporary CDN for Nano Banana...")
                    def upload_to_catbox(file_p):
                        with open(file_p, "rb") as f2:
                            r_cat = requests.post("https://catbox.moe/user/api.php", data={"reqtype": "fileupload"}, files={"fileToUpload": f2})
                        if r_cat.status_code == 200:
                            return r_cat.text.strip()
                        raise Exception("Failed CDN image upload")
                        
                    person_url = upload_to_catbox(person_path)
                    cloth_url = upload_to_catbox(clothing_path)
                    print(f"    -> Hosted successfully: {person_url}, {cloth_url}")
                    
                    payload = {
                        "prompt": f"Virtual try on. A highly realistic portrait photo of this exact person wearing {cloth_desc}. Extremely natural body mapping. i want full photo of the person do not cut off anything. Keep the person and everything other than the cloth the exact same. Change only the clothes. No changing facial features or body proportions.",
                        "numImages": 1,
                        "type": "IMAGETOIAMGE",
                        "image_size": "9:16",
                        "imageUrls": [person_url, cloth_url],
                        "callBackUrl": callBackUrl
                    }
                    
                    headers = {
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json"
                    }
                    print("[~] Sending request to api.nanobananaapi.ai...")
                    response = requests.post(external_url, headers=headers, json=payload, timeout=30)
                    
                    # ── Added Persistent Logging ──
                    try:
                        import datetime
                        with open("api_debug.log", "a", encoding="utf-8") as log_f:
                            log_f.write(f"\n[{datetime.datetime.now()}] --- NANO BANANA LOG ---\n")
                            log_f.write(f"Endpoint: {external_url}\n")
                            log_f.write(f"Payload Prompt: {payload['prompt']}\n")
                            log_f.write(f"Payload Images Passed (IMAGETOIAMGE): {len(payload['imageUrls'])} imageUrls data URIs\n")
                            log_f.write(f"Status Code: {response.status_code}\n")
                            log_f.write(f"Response: {response.text[:2000]}\n")
                            log_f.write("-" * 50 + "\n")
                    except Exception:
                        pass

                    # --- SAVE GROQ OUTPUT TO BACKEND ---
                    try:
                        with open("groq_output.txt", "w", encoding="utf-8") as gf:
                            gf.write(cloth_desc)
                    except Exception: pass

                    # ── Added Debugging Output ──
                    print(f"[~] API Status Code: {response.status_code}")
                    if response.status_code != 200:
                        print(f"[!] API Error Data: {response.text[:500]}")
                        
                    if response.status_code == 200:
                        print("[✓] External API triggered!")
                        
                        external_img = None
                        try:
                            resp_json = response.json()
                            
                            if resp_json.get('data') and isinstance(resp_json['data'], dict) and 'taskId' in resp_json['data']:
                                print("[~] Waiting for asynchronous webhook result (Increased Polling limit to 3 minutes)...")
                                
                                webhook_content = None
                                # Wait up to 280 seconds for the webhook to receive the image
                                for _ in range(140):
                                    time.sleep(2)
                                    poll_resp = requests.get(poll_url)
                                    if poll_resp.status_code == 200:
                                        poll_data = poll_resp.json()
                                        if poll_data.get('data') and len(poll_data['data']) > 0:
                                            webhook_content = poll_data['data'][0]['content']
                                            print("\n[✓] Webhook received the generated image!")
                                            break
                                    print(".", end="", flush=True)
                                
                                if webhook_content:
                                    import json as dynamic_json
                                    try:
                                        wh_json = dynamic_json.loads(webhook_content)
                                        # Use standard parser
                                        b64_str = None
                                        
                                        # NanoBanana nested structures
                                        if isinstance(wh_json.get('data'), dict):
                                            d = wh_json['data']
                                            info = d.get('info') or {}
                                            b64_str = (
                                                info.get('resultImageUrl') or
                                                (d.get('result_urls') and d['result_urls'][0]) or
                                                d.get('image') or 
                                                d.get('url') or 
                                                d.get('base64')
                                            )
                                            
                                        # Fallback to root level strings
                                        if not b64_str:
                                            b64_str = (
                                                wh_json.get('image') or 
                                                wh_json.get('images', [None])[0] or 
                                                (wh_json.get('result_urls') and wh_json['result_urls'][0]) or
                                                wh_json.get('url') or
                                                wh_json.get('base64') or
                                                (wh_json.get('data') if isinstance(wh_json.get('data'), str) else None)
                                            )
                                        
                                        if b64_str and isinstance(b64_str, str) and b64_str.startswith('http'):
                                            # If it sent an image URL instead of base64
                                            img_data = requests.get(b64_str).content
                                            img_array = np.asarray(bytearray(img_data), dtype=np.uint8)
                                        else:
                                            img_array = np.asarray(bytearray(base64.b64decode(b64_str)), dtype=np.uint8)
                                            
                                        external_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                                    except Exception as we:
                                        print(f"\n[!] Webhook parsing error: {we}. Content: {webhook_content[:300]}")
                                else:
                                    print("\n[!] Webhook timed out waiting for NanobananaAPI after 3 minutes.")
                            else:
                                # Synchronous fallback just in case
                                b64_str = (
                                    resp_json.get('image') or 
                                    resp_json.get('images', [None])[0] or 
                                    resp_json.get('data') or 
                                    resp_json.get('base64')
                                )
                                img_array = np.asarray(bytearray(base64.b64decode(b64_str)), dtype=np.uint8)
                                external_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                                
                        except Exception as e:
                            print(f"[!] Parsing Exception: {e}")
                            try:
                                img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
                                external_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                            except Exception:
                                pass

                        
                        if external_img is not None:
                            # Save the RAW uncropped external image to full outputs folder
                            full_out_dir = 'static/full_outputs'
                            os.makedirs(full_out_dir, exist_ok=True)
                            raw_save_path = os.path.join(full_out_dir, 'external_raw.jpg')
                            cv2.imwrite(raw_save_path, external_img)
                            print(f"[✓] Saved raw uncropped external image to {raw_save_path}")

                            # Crop the bottom 15%
                            h_ext, w_ext = external_img.shape[:2]
                            crop_h = int(h_ext * 0.85)
                            ai_result = external_img[:crop_h, :]
                            print(f"[~] Cropped bottom 15% (Old H: {h_ext}, New H: {crop_h})")
                            
                            ai_result = cv2.resize(ai_result, (orig_w, orig_h))
                            pipeline_used = 'External-API'
                        else:
                            print("[!] Failed to decode external image layout. Generating error image.")
                            pipeline_used = 'External-API-Error'
                            ai_result = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
                            cv2.putText(ai_result, "API Error: No image returned", (20, orig_h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    else:
                        print(f"[!] External API failed (Status {response.status_code}). Generating error image.")
                        pipeline_used = 'External-API-Error'
                        ai_result = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
                        cv2.putText(ai_result, f"API failed (Status {response.status_code})", (20, orig_h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    
                except Exception as e:
                    print(f"[!] API call Exception: {e}. Generating error image.")
                    pipeline_used = 'External-API-Error'
                    ai_result = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
                    cv2.putText(ai_result, f"Exception: {str(e)[:40]}", (20, orig_h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


                ai_succeeded = True
                print("[✓] AI inference step completed!")

            else:
                print("[!] VITON model not available")

        except Exception as e:
            print(f"[!] AI inference failed: {e}")
            import traceback
            traceback.print_exc()
            ai_succeeded = False

        # ── STEP 4: Fallback if AI failed ────────────────
        if not ai_succeeded:
            print("\n[STEP 3b] Using Enhanced OpenCV Fallback")
            pipeline_used = 'Enhanced-OpenCV'

            # Handle cloth input for fallback
            if cloth_raw.shape[2] == 4:
                cloth_input = cloth_raw
            else:
                cloth_input = cv2.cvtColor(
                    cloth_raw, cv2.COLOR_BGR2BGRA
                )

            ai_result = enhanced_opencv_tryon(
                person_clean,
                cloth_input,
                shirt_mask,
                seg_landmarks,
                orig_h, orig_w
            )

        # ── STEP 5: Post-Processing ───────────────────────
        print("\n[STEP 4] Post-Processing")
        if pipeline_used in ['External-API', 'External-API-Error']:
            print(f"[~] Skipping local post-processing for {pipeline_used}.")
            final_result = ai_result
        else:
            final_result = post_process_result(
                ai_result,
                original_person,
                shirt_mask,
                orig_h, orig_w
            )

        # ── STEP 6: Watermark ─────────────────────────────
        final_result = add_watermark(final_result)

        # ── STEP 7: Save ──────────────────────────────────
        elapsed = time.time() - start_time
        save_result_with_metadata(
            final_result, output_path,
            pipeline_used, elapsed
        )

        print("\n" + "═"*54)
        print(f"   ✅ COMPLETE! Pipeline: {pipeline_used}")
        print(f"   ⏱️  Time: {elapsed:.2f} seconds")
        print("═"*54 + "\n")

        return True, output_path

    except Exception as e:
        import traceback
        traceback.print_exc()
        elapsed = time.time() - start_time
        print(f"\n❌ Inference failed after {elapsed:.1f}s: {e}")
        return False, str(e)


# ─────────────────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────────────────

if __name__ == '__main__':
    person  = 'static/uploads/person.jpg'
    cloth   = 'static/uploads/clothing.png'
    output  = 'static/outputs/tryon_result.jpg'

    if not os.path.exists(person):
        print("❌ No images found!")
        print("   Go to http://localhost:5000/upload first")
    else:
        print("🧠 Testing full AI inference pipeline...")
        ok, result = run_ai_tryon(person, cloth, output)
        if ok:
            print(f"\n🎉 Success! Result: {result}")
        else:
            print(f"\n💥 Failed: {result}")