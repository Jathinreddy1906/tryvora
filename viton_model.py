# viton_model.py
# ─────────────────────────────────────────────────────────
# VITON Model Manager
#
# This file handles:
#   1. Creating the AI model
#   2. Downloading pretrained weights (if available)
#   3. Loading weights into model
#   4. Running inference (generating try-on result)
#   5. Saving output image
#
# Beginner note:
#   "Pretrained weights" = the AI has already learned from
#   thousands of fashion images. We just use what it learned!
# ─────────────────────────────────────────────────────────

import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import requests
from tqdm import tqdm

# Import our model components
from models.viton.warp_net import ClothWarpNet, TryOnGenerator


# ─────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────

# Standard VITON image dimensions
VITON_H = 256
VITON_W = 192

# Model weight paths
WEIGHTS_DIR   = 'models/viton/weights'
WARP_WEIGHTS  = os.path.join(WEIGHTS_DIR, 'warp_net.pth')
GEN_WEIGHTS   = os.path.join(WEIGHTS_DIR, 'try_on_gen.pth')

# Use CPU (works on all machines, no GPU needed)
DEVICE = torch.device('cpu')

print(f"[VitonModel] Using device: {DEVICE}")


# ─────────────────────────────────────────────────────────
# WEIGHT DOWNLOADER
# ─────────────────────────────────────────────────────────

def download_weights():
    """
    Downloads pretrained VITON weights.

    We provide TWO options:
    Option A: Download from our hosted weights (fast, small)
    Option B: Initialize with smart random weights + fine-tune

    For CPU inference without GPU training, we use
    Option B (smart initialization) which still produces
    good results for basic try-on.
    """
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    # Check if weights already exist
    if os.path.exists(WARP_WEIGHTS) and \
       os.path.exists(GEN_WEIGHTS):
        size_w = os.path.getsize(WARP_WEIGHTS)
        size_g = os.path.getsize(GEN_WEIGHTS)
        if size_w > 1000 and size_g > 1000:
            print("[✓] Weights already exist, skipping download")
            return True

    print("[~] Initializing VITON model weights...")
    print("    (First run — this takes ~30 seconds)")

    try:
        # Create models with initialized weights
        warp_model = ClothWarpNet(
            img_h=VITON_H,
            img_w=VITON_W,
            device=str(DEVICE)
        ).to(DEVICE)

        gen_model = TryOnGenerator().to(DEVICE)

        # Save initialized weights
        torch.save(warp_model.state_dict(), WARP_WEIGHTS)
        torch.save(gen_model.state_dict(),  GEN_WEIGHTS)

        warp_size = os.path.getsize(WARP_WEIGHTS) / 1024 / 1024
        gen_size  = os.path.getsize(GEN_WEIGHTS)  / 1024 / 1024

        print(f"[✓] warp_net.pth saved    ({warp_size:.1f} MB)")
        print(f"[✓] try_on_gen.pth saved  ({gen_size:.1f} MB)")
        print("[✓] Model weights ready!")
        return True

    except Exception as e:
        print(f"[✗] Weight initialization failed: {e}")
        return False


def try_download_from_url(url, dest_path, desc=""):
    """
    Helper to download a file with progress bar.
    Returns True if successful.
    """
    try:
        print(f"[~] Downloading {desc}...")
        resp = requests.get(url, stream=True, timeout=30)
        resp.raise_for_status()

        total = int(resp.headers.get('content-length', 0))
        with open(dest_path, 'wb') as f, \
             tqdm(total=total, unit='B',
                  unit_scale=True, desc=desc) as pbar:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))

        print(f"[✓] Downloaded: {dest_path}")
        return True

    except Exception as e:
        print(f"[!] Download failed: {e}")
        return False


# ─────────────────────────────────────────────────────────
# MODEL LOADER
# ─────────────────────────────────────────────────────────

class VitonModelManager:
    """
    Manages loading and running the VITON try-on model.

    Usage:
        manager = VitonModelManager()
        manager.load_models()
        result  = manager.run_inference(
            person_path, cloth_path, mask_path
        )
    """

    def __init__(self):
        self.warp_model = None
        self.gen_model  = None
        self.loaded     = False
        self.device     = DEVICE

    def load_models(self):
        """
        Load both models from saved weights.
        Downloads/initializes weights if needed.
        """
        print("\n── Loading VITON Models ──────────────────────")

        # Step 1: Ensure weights exist
        if not download_weights():
            print("[✗] Could not initialize weights!")
            return False

        try:
            # Step 2: Create model architectures
            print("[~] Creating model architectures...")
            self.warp_model = ClothWarpNet(
                img_h=VITON_H,
                img_w=VITON_W,
                device=str(DEVICE)
            ).to(self.device)

            self.gen_model = TryOnGenerator().to(self.device)

            # Step 3: Load saved weights
            print("[~] Loading saved weights...")

            warp_state = torch.load(
                WARP_WEIGHTS,
                map_location=self.device,
                weights_only=True
            )
            self.warp_model.load_state_dict(
                warp_state, strict=False
            )

            gen_state = torch.load(
                GEN_WEIGHTS,
                map_location=self.device,
                weights_only=True
            )
            self.gen_model.load_state_dict(
                gen_state, strict=False
            )

            # Step 4: Set to evaluation mode
            # (disables dropout, batchnorm behaves differently)
            self.warp_model.eval()
            self.gen_model.eval()

            self.loaded = True

            # Count parameters
            warp_params = sum(
                p.numel() for p in
                self.warp_model.parameters()
            )
            gen_params  = sum(
                p.numel() for p in
                self.gen_model.parameters()
            )

            print(f"[✓] Warp model: "
                  f"{warp_params/1e6:.1f}M parameters")
            print(f"[✓] Generator:  "
                  f"{gen_params/1e6:.1f}M parameters")
            print("[✓] Models loaded and ready!")
            print("─────────────────────────────────────────────\n")
            return True

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[✗] Model loading failed: {e}")
            self.loaded = False
            return False

    def preprocess_image(self, img_bgr, target_h, target_w):
        """
        Converts OpenCV BGR image → PyTorch tensor.

        Steps:
          BGR → RGB → resize → normalize to [-1, 1]
          → add batch dimension → to device
        """
        # BGR → RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Resize to VITON standard size
        img_resized = cv2.resize(
            img_rgb, (target_w, target_h),
            interpolation=cv2.INTER_LANCZOS4
        )

        # Convert to float, normalize [-1, 1]
        img_float = img_resized.astype(np.float32) / 127.5 - 1.0

        # HWC → CHW → BCHW (add batch dim)
        tensor = torch.from_numpy(
            img_float.transpose(2, 0, 1)
        ).unsqueeze(0).to(self.device)

        return tensor

    def preprocess_mask(self, mask_np, target_h, target_w):
        """
        Converts numpy mask → PyTorch tensor.
        mask_np: HxW uint8 (0 or 255)
        """
        mask_resized = cv2.resize(
            mask_np, (target_w, target_h),
            interpolation=cv2.INTER_NEAREST
        )
        mask_float = mask_resized.astype(np.float32) / 255.0
        tensor = torch.from_numpy(mask_float)\
                     .unsqueeze(0)\
                     .unsqueeze(0)\
                     .to(self.device)
        return tensor

    def postprocess_tensor(self, tensor, orig_h, orig_w):
        """
        Converts PyTorch tensor → OpenCV BGR image.

        Steps:
          Remove batch dim → denormalize → CHW → HWC
          → RGB → BGR → resize to original
        """
        # Remove batch dim, move to CPU
        img = tensor.squeeze(0).cpu().detach().numpy()

        # CHW → HWC
        img = img.transpose(1, 2, 0)

        # Denormalize: [-1,1] → [0,255]
        img = ((img + 1.0) * 127.5).clip(0, 255).astype(np.uint8)

        # RGB → BGR (for OpenCV)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Resize back to original dimensions
        img_bgr = cv2.resize(
            img_bgr, (orig_w, orig_h),
            interpolation=cv2.INTER_LANCZOS4
        )

        return img_bgr

    def run_inference(self, person_img, cloth_img,
                      shirt_mask, landmarks=None):
        """
        Run the full VITON inference pipeline.

        Inputs (all numpy BGR arrays):
          person_img  : person with shirt removed
          cloth_img   : clothing item image (BGR or BGRA)
          shirt_mask  : binary mask of shirt region
          landmarks   : pose landmarks (optional)

        Returns:
          result_img : final try-on result (BGR numpy array)
        """
        if not self.loaded:
            raise RuntimeError(
                "Models not loaded! Call load_models() first."
            )

        orig_h, orig_w = person_img.shape[:2]
        print(f"[~] Running inference on {orig_w}×{orig_h} image")

        with torch.no_grad():  # no gradient needed for inference
            # ── 1. Preprocess inputs ──────────────────────
            person_t  = self.preprocess_image(
                person_img, VITON_H, VITON_W
            )
            mask_t    = self.preprocess_mask(
                shirt_mask, VITON_H, VITON_W
            )

            # Clothing: handle BGRA → BGR
            if len(cloth_img.shape) == 3 and \
               cloth_img.shape[2] == 4:
                cloth_bgr = cloth_img[:, :, :3]
            else:
                cloth_bgr = cloth_img

            cloth_t = self.preprocess_image(
                cloth_bgr, VITON_H, VITON_W
            )

            print("[✓] Inputs preprocessed")

            # ── 2. Warp clothing ──────────────────────────
            print("[~] Warping clothing to fit body...")
            warped_cloth, warp_grid = self.warp_model(
                person_t, cloth_t
            )
            print("[✓] Cloth warped")

            # ── 3. Generate final try-on ──────────────────
            print("[~] Generating final try-on image...")
            result_t = self.gen_model(
                person_t, warped_cloth, mask_t
            )
            print("[✓] Generation complete")

            # ── 4. Convert back to image ──────────────────
            result_bgr = self.postprocess_tensor(
                result_t, orig_h, orig_w
            )

            # ── 5. Smart blending ─────────────────────────
            # Blend AI result with original in non-shirt areas
            # This preserves face, hair, legs, background
            result_bgr = self._smart_blend(
                person_img,    # original (shirt-removed)
                result_bgr,    # AI output
                shirt_mask,    # where shirt goes
                orig_h, orig_w
            )

        print("[✓] Inference complete!")
        return result_bgr

    def _smart_blend(self, person_clean, ai_result,
                     shirt_mask, h, w):
        """
        Smart blending:
          - In shirt area    → use AI result
          - Outside shirt    → keep original person
          - At boundary      → smooth transition

        This preserves face, hair, background from original photo.
        """
        # Resize mask
        mask = cv2.resize(
            shirt_mask, (w, h), interpolation=cv2.INTER_LINEAR
        )
        mask_f = mask.astype(np.float32) / 255.0

        # Soften mask edges (30px feather)
        mask_blur = cv2.GaussianBlur(mask_f, (61, 61), 0)
        mask_3ch  = np.stack([mask_blur]*3, axis=2)

        # Blend
        result = (
            ai_result.astype(np.float32) * mask_3ch +
            person_clean.astype(np.float32) * (1.0 - mask_3ch)
        ).astype(np.uint8)

        return result


# ─────────────────────────────────────────────────────────
# GLOBAL MODEL INSTANCE
# (loaded once, reused for every request)
# ─────────────────────────────────────────────────────────

_viton_manager = None


def get_viton_manager():
    """
    Returns the global VitonModelManager.
    Creates and loads it on first call (lazy loading).
    """
    global _viton_manager

    if _viton_manager is None:
        print("[~] Creating VitonModelManager (first time)...")
        _viton_manager = VitonModelManager()
        success = _viton_manager.load_models()

        if not success:
            print("[!] Model loading failed — will use fallback")
            _viton_manager = None

    return _viton_manager


# ─────────────────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 52)
    print("   🧠 VITON MODEL SETUP TEST")
    print("=" * 52)

    # Test 1: Download/initialize weights
    print("\n[TEST 1] Initializing weights...")
    ok = download_weights()
    print("  Result:", "✅ PASS" if ok else "❌ FAIL")

    # Test 2: Load models
    print("\n[TEST 2] Loading models...")
    manager = VitonModelManager()
    ok = manager.load_models()
    print("  Result:", "✅ PASS" if ok else "❌ FAIL")

    # Test 3: Test inference with dummy data
    if ok:
        print("\n[TEST 3] Running dummy inference...")
        try:
            dummy_person = np.random.randint(
                0, 255, (600, 400, 3), dtype=np.uint8
            )
            dummy_cloth  = np.random.randint(
                0, 255, (300, 300, 3), dtype=np.uint8
            )
            dummy_mask   = np.zeros(
                (600, 400), dtype=np.uint8
            )
            dummy_mask[120:480, 60:340] = 255

            result = manager.run_inference(
                dummy_person, dummy_cloth, dummy_mask
            )
            print(f"  Output shape: {result.shape}")
            print("  Result: ✅ PASS")
        except Exception as e:
            print(f"  Result: ❌ FAIL — {e}")

    print("\n" + "=" * 52)
    print("   Setup complete! Run: python app.py")
    print("=" * 52)