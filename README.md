# Tryvora

Tryvora is a Flask virtual outfit preview app. The web app processes photos on
the server with its local OpenCV pipeline; it does not send uploaded photos to
external image-generation services.

## Run locally

Use Python 3.11, then install the web runtime dependencies:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

Open `http://localhost:5000`. The app stores each browser session's uploads and
outputs under `instance/uploads/`; these files are not served as public static
assets. The session cookie is required to retrieve them.

## Production

Configure `APP_ENV=production` and a long random `FLASK_SECRET_KEY`, then start
the app with:

```sh
gunicorn --workers 1 --bind 0.0.0.0:${PORT:-8000} app:app
```

The host must support Python 3.11 and the MediaPipe/OpenCV CPU dependencies.
Keep a single app instance unless uploads and sessions are moved to shared,
private storage. Local instance storage is temporary on hosts with ephemeral
filesystems.

`requirements-ml.txt` includes the optional PyTorch dependencies for the
separate VITON setup scripts; the web runtime does not load that model.

## Deploy to Render

The included `render.yaml` configures a free Render web service. Push this
branch to GitHub, then create a Blueprint in Render from the repository. The
free instance may sleep when idle and has limited CPU/memory, so it is intended
for a small demo; virtual try-on requests may be slow or exceed its limits.
Uploads are session-private but stored on the instance's ephemeral filesystem
and are lost when that filesystem is replaced.
