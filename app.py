import os
import secrets
import time
from flask import (Flask, render_template, request,
                   redirect, url_for, flash, jsonify, abort,
                   send_from_directory, session)
from PIL import Image
from image_processor import run_tryon_pipeline
from recommender import get_recommendations

app = Flask(__name__, instance_relative_config=True)
app.secret_key = os.environ.get('FLASK_SECRET_KEY')
if not app.secret_key:
    if os.environ.get('APP_ENV') == 'production':
        raise RuntimeError('FLASK_SECRET_KEY must be configured in production')
    app.secret_key = secrets.token_hex(32)

# ── CONFIG ────────────────────────────────────────────────
UPLOAD_FOLDER   = os.path.join(app.instance_path, 'uploads')
OUTPUT_FOLDER   = 'outputs'
ALLOWED_EXT     = {'png', 'jpg', 'jpeg', 'webp'}
MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10MB
MEDIA_FILES = {
    'uploads': {'person.jpg', 'clothing.png'},
    'outputs': {
        'tryon_result.jpg',
        'debug_comparison.jpg',
        'debug_person_mask.jpg',
        'debug_shirt_mask.jpg',
        'debug_shirt_removed.jpg',
        'prep_summary.jpg',
        'prep_person.jpg',
        'prep_agnostic.jpg',
        'prep_cloth.jpg',
        'prep_cloth_mask.jpg',
        'prep_pose_heatmap.jpg',
        'prep_segmap.jpg',
    },
}

app.config['MAX_CONTENT_LENGTH'] = MAX_UPLOAD_SIZE
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_COOKIE_SECURE'] = os.environ.get('APP_ENV') == 'production'
app.config['PERMANENT_SESSION_LIFETIME'] = 60 * 60 * 24


@app.before_request
def block_public_user_images():
    private_prefixes = (
        '/static/uploads/',
        '/static/outputs/',
        '/static/full_outputs/',
    )
    if request.path.startswith(private_prefixes):
        abort(404)


def allowed(filename):
    return ('.' in filename and
            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT)


def user_storage_dir(create=False):
    """Return the private storage directory associated with this browser."""
    upload_id = session.get('upload_id')
    if not upload_id and create:
        upload_id = secrets.token_hex(16)
        session['upload_id'] = upload_id
        session.permanent = True
    if not upload_id:
        return None
    return os.path.join(UPLOAD_FOLDER, upload_id)


def images_uploaded():
    """Check both required images exist"""
    storage_dir = user_storage_dir()
    return bool(storage_dir and
                os.path.exists(os.path.join(storage_dir, 'uploads',
                                             'person.jpg')) and
                os.path.exists(os.path.join(storage_dir, 'uploads',
                                             'clothing.png')))


def clear_user_outputs(storage_dir):
    output_dir = os.path.join(storage_dir, OUTPUT_FOLDER)
    for filename in MEDIA_FILES['outputs']:
        path = os.path.join(output_dir, filename)
        if os.path.isfile(path):
            os.remove(path)


@app.route('/media/<file_type>/<filename>')
def private_media(file_type, filename):
    if file_type not in MEDIA_FILES or filename not in MEDIA_FILES[file_type]:
        abort(404)
    storage_dir = user_storage_dir()
    if not storage_dir:
        abort(404)
    return send_from_directory(
        os.path.join(storage_dir, file_type), filename, max_age=0
    )

# ── ROUTES ────────────────────────────────────────────────

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/healthz')
def health_check():
    return jsonify({'status': 'ok'})


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

    try:
        storage_dir = user_storage_dir(create=True)
        upload_dir = os.path.join(storage_dir, 'uploads')
        os.makedirs(upload_dir, exist_ok=True)

        with Image.open(pf.stream) as image:
            image.load()
            person = image.convert('RGB').resize(
                (400, 600), Image.Resampling.LANCZOS
            )
            person.save(os.path.join(upload_dir, 'person.jpg'),
                        format='JPEG', quality=92)

        with Image.open(cf.stream) as image:
            image.load()
            clothing = image.convert('RGBA').resize(
                (300, 300), Image.Resampling.LANCZOS
            )
            clothing.save(os.path.join(upload_dir, 'clothing.png'),
                          format='PNG')

        clear_user_outputs(storage_dir)
        flash('✅ Images uploaded successfully!', 'success')
        return redirect(url_for('preview_page'))

    except Exception as e:
        app.logger.warning('Upload rejected: %s', e)
        flash('❌ Could not process those images. Please try valid JPG, PNG, or WEBP files.', 'error')
        return redirect(url_for('upload_page'))


@app.route('/preview')
def preview_page():
    if not images_uploaded():
        flash('⚠️ Please upload both images first.', 'error')
        return redirect(url_for('upload_page'))

    return render_template('preview.html',
                           person_image='person.jpg',
                           clothing_image='clothing.png')

@app.route('/tryon', methods=['POST'])
def run_tryon():
    """
    Main try-on endpoint.
    Called by JavaScript fetch() in preview.html.
    Runs the local OpenCV pipeline and returns JSON.
    """
    if not images_uploaded():
        return jsonify({
            'success': False,
            'error'  : 'Images missing. Please upload first.'
        }), 400

    storage_dir = user_storage_dir()
    upload_dir = os.path.join(storage_dir, 'uploads')
    output_dir = os.path.join(storage_dir, OUTPUT_FOLDER)
    os.makedirs(output_dir, exist_ok=True)
    person_path = os.path.join(upload_dir, 'person.jpg')
    clothing_path = os.path.join(upload_dir, 'clothing.png')
    output_path = os.path.join(output_dir, 'tryon_result.jpg')

    try:
        start   = time.time()
        success, _result = run_tryon_pipeline(
            person_path, clothing_path, output_path,
            debug_output_dir=output_dir
        )
        elapsed = round(time.time() - start, 1)

        if success:
            # Cache-busting timestamp prevents browser
            # showing old cached image
            ts = int(time.time())
            return jsonify({
                'success'      : True,
                'result_url'   : url_for(
                    'private_media', file_type='outputs',
                    filename='tryon_result.jpg', v=ts
                ),
                'elapsed'      : elapsed,
                'pipeline'     : 'Enhanced-OpenCV'
            })
        else:
            return jsonify({
                'success': False,
                'error'  : 'Unable to create the preview. Please try again.'
            }), 500

    except Exception:
        app.logger.exception('Try-on processing failed')
        return jsonify({
            'success': False,
            'error'  : 'Unable to create the preview. Please try again.'
        }), 500


@app.route('/pipeline_status')
def pipeline_status():
    return jsonify({
        'ai_ready'  : False,
        'pipeline'  : 'Enhanced-OpenCV',
        'device'    : 'CPU'
    })


@app.route('/result')
def result_page():
    storage_dir = user_storage_dir()
    result_path = (os.path.join(storage_dir, OUTPUT_FOLDER,
                                 'tryon_result.jpg')
                   if storage_dir else None)

    if not result_path or not os.path.exists(result_path):
        flash('⚠️ No result yet. Run the try-on first.', 'error')
        return redirect(url_for('preview_page'))

    return render_template('result.html',
                           result_image='tryon_result.jpg',
                           person_image='person.jpg',
                           clothing_image='clothing.png')

# ADD this new route to app.py

@app.route('/recommend')
def recommend():
    """
    Reads the uploaded clothing image,
    detects its color, and returns outfit recommendations.
    Called by JavaScript via fetch().
    """
    storage_dir = user_storage_dir()
    clothing_path = (os.path.join(storage_dir, 'uploads', 'clothing.png')
                     if storage_dir else None)

    if not clothing_path or not os.path.exists(clothing_path):
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
    except Exception:
        app.logger.exception('Outfit recommendation failed')
        return jsonify({
            'success': False,
            'error'  : 'Unable to create recommendations right now.'
        }), 500


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
    storage_dir = user_storage_dir()
    result_path = (os.path.join(storage_dir, OUTPUT_FOLDER,
                                'tryon_result.jpg')
                   if storage_dir else None)
    clothing_path = (os.path.join(storage_dir, 'uploads', 'clothing.png')
                     if storage_dir else None)

    result_exists = bool(result_path and os.path.exists(result_path))

    # Get clothing color for the 3D model
    cloth_color = '#6366f1'  # default purple

    if clothing_path and os.path.exists(clothing_path):
        try:
            from recommender import get_recommendations
            recs        = get_recommendations(clothing_path)
            cloth_color = recs.get('color_hex', '#6366f1')
        except Exception:
            pass  # use default color if recommender fails

    return render_template(
        'viewer3d.html',
        cloth_color   = cloth_color,
        result_image = 'tryon_result.jpg' if result_exists else None,
        clothing_image = 'clothing.png'
                         if clothing_path and os.path.exists(clothing_path)
                         else None,
        person_image = 'person.jpg'
                       if images_uploaded() else None
    )
# ADD this route to app.py

@app.route('/debug_view')
def debug_view():
    """
    Shows all AI debug images in a gallery.
    Helps you understand what each step produced.
    """
    def debug_url(filename):
        return url_for('private_media', file_type='outputs',
                       filename=filename)

    # Final outputs
    output_images = [
        {
            'url'  : debug_url('tryon_result.jpg'),
            'title': 'Final Try-On Result',
            'desc' : 'Complete local OpenCV pipeline output'
        },
    ]

    # Segmentation debug images
    seg_images = [
        {
            'url'  : debug_url('debug_comparison.jpg'),
            'title': 'Before vs After (Shirt Removal)',
            'desc' : 'Original vs shirt-removed person'
        },
        {
            'url'  : debug_url('debug_person_mask.jpg'),
            'title': 'Person Mask',
            'desc' : 'Green = detected person pixels'
        },
        {
            'url'  : debug_url('debug_shirt_mask.jpg'),
            'title': 'Shirt Mask',
            'desc' : 'Red = shirt region to replace'
        },
        {
            'url'  : debug_url('debug_shirt_removed.jpg'),
            'title': 'Shirt Removed',
            'desc' : 'Inpainted shirt-removed person'
        },
    ]

    # Input preparation debug images
    prep_images = [
        {
            'url'  : debug_url('prep_summary.jpg'),
            'title': 'All Inputs Side-by-Side',
            'desc' : 'Person → Agnostic → Cloth → SegMap'
        },
        {
            'url'  : debug_url('prep_person.jpg'),
            'title': 'Prepared Person (192×256)',
            'desc' : 'Resized + normalized to VITON format'
        },
        {
            'url'  : debug_url('prep_agnostic.jpg'),
            'title': 'Agnostic Person',
            'desc' : 'Person with clothing info removed'
        },
        {
            'url'  : debug_url('prep_cloth.jpg'),
            'title': 'Prepared Clothing',
            'desc' : 'Centred on white background'
        },
        {
            'url'  : debug_url('prep_cloth_mask.jpg'),
            'title': 'Cloth Mask',
            'desc' : 'White = clothing pixels'
        },
        {
            'url'  : debug_url('prep_pose_heatmap.jpg'),
            'title': 'Pose Heatmap (18 joints)',
            'desc' : 'Colourful blobs = detected joints'
        },
        {
            'url'  : debug_url('prep_segmap.jpg'),
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
    storage_dir = user_storage_dir()
    result_path = (os.path.join(storage_dir, OUTPUT_FOLDER,
                                'tryon_result.jpg')
                   if storage_dir else None)
    clothing_path = (os.path.join(storage_dir, 'uploads', 'clothing.png')
                     if storage_dir else None)
    result_image = ('tryon_result.jpg'
                    if result_path and os.path.exists(result_path) else None)
    clothing_image = ('clothing.png'
                      if clothing_path and os.path.exists(clothing_path)
                      else None)

    return render_template(
        'avatar.html',
        result_image  = result_image,
        clothing_image = clothing_image
    )


# ── START ─────────────────────────────────────────────────
if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    print("\n" + "="*45)
    print("   👗 TRYVORA is starting...")
    print("   🌐 Open: http://localhost:5000")
    print("="*45 + "\n")
    app.run(debug=os.environ.get('APP_ENV') != 'production')