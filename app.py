# app.py — TRYVORA FINAL VERSION (Day 5)
import json  # ADD this line at the top of app.py
import os
from flask import (Flask, render_template, request,
                   redirect, url_for, flash, jsonify)
from PIL import Image
from image_processor import run_tryon_pipeline 
# ADD this line near the top of app.py
from recommender import get_recommendations

app = Flask(__name__)
app.secret_key = 'tryvora_secret_key_2025'

# ── CONFIG ────────────────────────────────────────────────
UPLOAD_FOLDER   = 'static/uploads'
OUTPUT_FOLDER   = 'static/outputs'
ALLOWED_EXT     = {'png', 'jpg', 'jpeg', 'webp'}
MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10MB

app.config['UPLOAD_FOLDER']      = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_UPLOAD_SIZE

def allowed(filename):
    return ('.' in filename and
            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT)

def images_uploaded():
    """Check both required images exist"""
    return (os.path.exists(os.path.join(UPLOAD_FOLDER, 'person.jpg')) and
            os.path.exists(os.path.join(UPLOAD_FOLDER, 'clothing.png')))

# ── ROUTES ────────────────────────────────────────────────

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/upload')
def upload_page():
    return render_template('upload.html')


@app.route('/upload', methods=['POST'])
def handle_upload():
    pf = request.files.get('person_image')
    cf = request.files.get('clothing_image')

    # Validate both files present
    if not pf or not cf:
        flash('❌ Please select both images!', 'error')
        return redirect(url_for('upload_page'))

    if pf.filename == '' or cf.filename == '':
        flash('❌ No file selected!', 'error')
        return redirect(url_for('upload_page'))

    if not allowed(pf.filename) or not allowed(cf.filename):
        flash('❌ Only JPG, PNG, WEBP files are allowed!', 'error')
        return redirect(url_for('upload_page'))

    # Save paths
    person_path   = os.path.join(UPLOAD_FOLDER, 'person.jpg')
    clothing_path = os.path.join(UPLOAD_FOLDER, 'clothing.png')

    try:
        # Save raw uploads
        pf.save(person_path)
        cf.save(clothing_path)

        # Resize with Pillow
        p = Image.open(person_path).convert('RGB')
        p = p.resize((400, 600), Image.LANCZOS)
        p.save(person_path, quality=92)

        c = Image.open(clothing_path).convert('RGBA')
        c = c.resize((300, 300), Image.LANCZOS)
        c.save(clothing_path)

        flash('✅ Images uploaded successfully!', 'success')
        return redirect(url_for('preview_page'))

    except Exception as e:
        flash(f'❌ Upload error: {str(e)}', 'error')
        return redirect(url_for('upload_page'))


@app.route('/preview')
def preview_page():
    if not images_uploaded():
        flash('⚠️ Please upload both images first.', 'error')
        return redirect(url_for('upload_page'))

    return render_template('preview.html',
                           person_image='uploads/person.jpg',
                           clothing_image='uploads/clothing.png')


# REPLACE your existing /tryon route with this upgraded version

# REPLACE your existing /tryon route in app.py

import time  # add this import at top of app.py if not there

@app.route('/tryon', methods=['POST'])
def run_tryon():
    """
    Main try-on endpoint.
    Called by JavaScript fetch() in preview.html.
    Runs the full AI pipeline and returns JSON.
    """
    if not images_uploaded():
        return jsonify({
            'success': False,
            'error'  : 'Images missing. Please upload first.'
        })

    person_path   = os.path.join(UPLOAD_FOLDER, 'person.jpg')
    clothing_path = os.path.join(UPLOAD_FOLDER, 'clothing.png')
    output_path   = os.path.join(OUTPUT_FOLDER, 'tryon_result.jpg')

    try:
        from image_processor import run_tryon_pipeline

        start   = time.time()
        success, result = run_tryon_pipeline(
            person_path, clothing_path, output_path
        )
        elapsed = round(time.time() - start, 1)

        if success:
            # Cache-busting timestamp prevents browser
            # showing old cached image
            ts = int(time.time())
            return jsonify({
                'success'      : True,
                'result_url'   : f'/static/outputs/tryon_result.jpg?v={ts}',
                'elapsed'      : elapsed,
                'pipeline'     : 'AI'
            })
        else:
            return jsonify({
                'success': False,
                'error'  : str(result)
            })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error'  : f'Server error: {str(e)}'
        })


@app.route('/pipeline_status')
def pipeline_status():
    """
    Returns current status of AI model.
    Called by frontend to show loading message.
    """
    try:
        from viton_model import get_viton_manager
        manager = get_viton_manager()
        ai_ready = manager is not None and manager.loaded
    except Exception:
        ai_ready = False

    return jsonify({
        'ai_ready'  : ai_ready,
        'pipeline'  : 'VITON-AI' if ai_ready
                      else 'Enhanced-OpenCV',
        'device'    : 'CPU'
    })


@app.route('/result')
def result_page():
    result_path = os.path.join(OUTPUT_FOLDER, 'tryon_result.jpg')

    if not os.path.exists(result_path):
        flash('⚠️ No result yet. Run the try-on first.', 'error')
        return redirect(url_for('preview_page'))

    return render_template('result.html',
                           result_image='outputs/tryon_result.jpg',
                           person_image='uploads/person.jpg',
                           clothing_image='uploads/clothing.png') 

# ADD this new route to app.py

@app.route('/recommend')
def recommend():
    """
    Reads the uploaded clothing image,
    detects its color, and returns outfit recommendations.
    Called by JavaScript via fetch().
    """
    clothing_path = os.path.join(UPLOAD_FOLDER, 'clothing.png')

    if not os.path.exists(clothing_path):
        return jsonify({
            'success': False,
            'error'  : 'No clothing image found'
        })

    try:
        recs = get_recommendations(clothing_path)
        return jsonify({
            'success': True,
            'data'   : recs
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error'  : str(e)
        })


# 404 handler
@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404


# 413 handler (file too large)
@app.errorhandler(413)
def file_too_large(e):
    flash('❌ File too large! Maximum size is 10MB.', 'error')
    return redirect(url_for('upload_page')) 

# ADD this route to app.py

@app.route('/viewer')
def viewer3d():
    """
    3D Model Viewer page.
    Passes clothing color + result image URL to the template.
    """
    result_path   = os.path.join(OUTPUT_FOLDER, 'tryon_result.jpg')
    clothing_path = os.path.join(UPLOAD_FOLDER,  'clothing.png')

    result_exists = os.path.exists(result_path)

    # Get clothing color for the 3D model
    cloth_color = '#6366f1'  # default purple

    if os.path.exists(clothing_path):
        try:
            from recommender import get_recommendations
            recs        = get_recommendations(clothing_path)
            cloth_color = recs.get('color_hex', '#6366f1')
        except Exception:
            pass  # use default color if recommender fails

    return render_template(
        'viewer3d.html',
        cloth_color   = cloth_color,
        result_image  = 'outputs/tryon_result.jpg' if result_exists else None,
        person_image  = 'uploads/person.jpg'
                        if os.path.exists(
                            os.path.join(UPLOAD_FOLDER, 'person.jpg')
                        ) else None
    )
# ADD this route to app.py

@app.route('/debug_view')
def debug_view():
    """
    Shows all AI debug images in a gallery.
    Helps you understand what each step produced.
    """
    base = '/static/outputs/'

    # Final outputs
    output_images = [
        {
            'url'  : base + 'tryon_result.jpg',
            'title': 'Final Try-On Result',
            'desc' : 'Complete AI pipeline output'
        },
    ]

    # Segmentation debug images
    seg_images = [
        {
            'url'  : base + 'debug_comparison.jpg',
            'title': 'Before vs After (Shirt Removal)',
            'desc' : 'Original vs shirt-removed person'
        },
        {
            'url'  : base + 'debug_person_mask.jpg',
            'title': 'Person Mask',
            'desc' : 'Green = detected person pixels'
        },
        {
            'url'  : base + 'debug_shirt_mask.jpg',
            'title': 'Shirt Mask',
            'desc' : 'Red = shirt region to replace'
        },
        {
            'url'  : base + 'debug_shirt_removed.jpg',
            'title': 'Shirt Removed',
            'desc' : 'Inpainted shirt-removed person'
        },
    ]

    # Input preparation debug images
    prep_images = [
        {
            'url'  : base + 'prep_summary.jpg',
            'title': 'All Inputs Side-by-Side',
            'desc' : 'Person → Agnostic → Cloth → SegMap'
        },
        {
            'url'  : base + 'prep_person.jpg',
            'title': 'Prepared Person (192×256)',
            'desc' : 'Resized + normalized to VITON format'
        },
        {
            'url'  : base + 'prep_agnostic.jpg',
            'title': 'Agnostic Person',
            'desc' : 'Person with clothing info removed'
        },
        {
            'url'  : base + 'prep_cloth.jpg',
            'title': 'Prepared Clothing',
            'desc' : 'Centred on white background'
        },
        {
            'url'  : base + 'prep_cloth_mask.jpg',
            'title': 'Cloth Mask',
            'desc' : 'White = clothing pixels'
        },
        {
            'url'  : base + 'prep_pose_heatmap.jpg',
            'title': 'Pose Heatmap (18 joints)',
            'desc' : 'Colourful blobs = detected joints'
        },
        {
            'url'  : base + 'prep_segmap.jpg',
            'title': 'Segmentation Map',
            'desc' : 'Each colour = body part label'
        },
    ]

    return render_template(
        'debug_view.html',
        output_images = output_images,
        seg_images    = seg_images,
        prep_images   = prep_images,
    )

# ADD to app.py — Avatar route

@app.route('/avatar')
def avatar_page():
    """3D Avatar Try-On page using Ready Player Me"""
    result_image = None
    result_path  = os.path.join(OUTPUT_FOLDER,
                                'tryon_result.jpg')
    if os.path.exists(result_path):
        result_image = 'outputs/tryon_result.jpg'

    clothing_image = None
    clothing_path  = os.path.join(UPLOAD_FOLDER,
                                  'clothing.png')
    if os.path.exists(clothing_path):
        clothing_image = 'uploads/clothing.png'

    return render_template(
        'avatar.html',
        result_image  = result_image,
        clothing_image = clothing_image
    )


# ── START ─────────────────────────────────────────────────
if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    print("\n" + "="*45)
    print("   👗 TRYVORA is starting...")
    print("   🌐 Open: http://localhost:5000")
    print("="*45 + "\n")
    app.run(debug=True)