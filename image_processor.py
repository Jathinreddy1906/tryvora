# image_processor.py — FINAL VERSION (Step 4)
# Clean connector between Flask and the AI pipeline

import cv2
import numpy as np
import os

# Connect to our AI inference engine
from inference_engine import run_ai_tryon


# ─────────────────────────────────────────────────────────
# MAIN PIPELINE FUNCTION
# Called by app.py for every try-on request
# ─────────────────────────────────────────────────────────

def run_tryon_pipeline(person_path, clothing_path,
                        output_path):
    """
    Entry point called by Flask (app.py).

    Simply delegates to the AI inference engine.
    Returns: (success: bool, result_path or error: str)
    """
    return run_ai_tryon(person_path, clothing_path,
                         output_path)


# ─────────────────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────────────────

if __name__ == '__main__':
    ok, msg = run_tryon_pipeline(
        'static/uploads/person.jpg',
        'static/uploads/clothing.png',
        'static/outputs/tryon_result.jpg'
    )
    print("✅ Done!" if ok else f"❌ Failed: {msg}")