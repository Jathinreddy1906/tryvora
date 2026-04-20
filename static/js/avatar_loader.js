// static/js/avatar_loader.js
// ─────────────────────────────────────────────────────
// STEP 1: Avatar Loader
//
// Handles:
//   1. Ready Player Me iframe → get avatar GLB URL
//   2. Three.js scene setup (basic)
//   3. GLTF model loading
//   4. Loading state management
// ─────────────────────────────────────────────────────

// ── Three.js imports from CDN ─────────────────────────
// These are loaded via <script> tags in the HTML.
// We use global THREE object.

'use strict';

// ════════════════════════════════════════════════════════
// CONFIG
// ════════════════════════════════════════════════════════

const CONFIG = {
    // Ready Player Me subdomain
    // Replace 'demo' with your own RPM subdomain
    // (free at readyplayer.me)
    RPM_SUBDOMAIN : 'demo',

    // Default avatar if none created yet
    // (a free public RPM avatar for testing)
    DEFAULT_AVATAR_URL:
        'https://models.readyplayer.me/' +
        '6409c84a66d3b08f4d2d3c7b.glb?' +
        'morphTargets=ARKit&textureAtlas=1024',

    CAMERA_FOV   : 45,
    CAMERA_NEAR  : 0.1,
    CAMERA_FAR   : 100,
    CAMERA_START : { x: 0, y: 1.5, z: 3.5 },

    // Avatar display settings
    AVATAR_SCALE : 1.0,
    AVATAR_Y_POS : -1.0,    // move down so feet touch "ground"
};


// ════════════════════════════════════════════════════════
// GLOBAL STATE
// ════════════════════════════════════════════════════════

const State = {
    scene         : null,
    camera        : null,
    renderer      : null,
    avatar        : null,       // loaded GLTF object
    avatarMixer   : null,       // animation mixer
    avatarUrl     : null,       // current GLB URL
    isLoading     : false,
    rpmOpen       : false,
    clock         : null,
};


// ════════════════════════════════════════════════════════
// SCENE SETUP (basic — lighting added in Step 2)
// ════════════════════════════════════════════════════════

// REPLACE initScene() in avatar_loader.js

function initScene() {
    const canvas = document.getElementById('avatarCanvas');
    if (!canvas) {
        console.error('avatarCanvas not found!');
        return;
    }

    // ── Renderer ──────────────────────────────────────
    State.renderer = new THREE.WebGLRenderer({
        canvas,
        antialias       : true,
        alpha           : false,
        powerPreference : 'high-performance',
    });

    State.renderer.setPixelRatio(
        Math.min(window.devicePixelRatio, 2)
    );
    State.renderer.shadowMap.enabled = true;
    State.renderer.shadowMap.type    = THREE.PCFSoftShadowMap;
    State.renderer.outputEncoding    = THREE.sRGBEncoding;
    State.renderer.toneMapping       = THREE.ACESFilmicToneMapping;
    State.renderer.toneMappingExposure = 1.1;
    State.renderer.physicallyCorrectLights = true;

    // ── Scene ─────────────────────────────────────────
    State.scene           = new THREE.Scene();
    State.scene.background = new THREE.Color(0x080614);
    State.scene.fog        = new THREE.FogExp2(
        0x080614, 0.038
    );

    // ── Camera ────────────────────────────────────────
    State.camera = new THREE.PerspectiveCamera(
        42, 1, 0.1, 100
    );
    State.camera.position.set(0, 1.5, 5);
    State.camera.lookAt(0, 1.0, 0);

    // ── Clock ─────────────────────────────────────────
    State.clock = new THREE.Clock();

    // ── Full scene setup (NEW — calls scene_setup.js) ─
    setupFullScene(
        State.scene,
        State.camera,
        State.renderer
    );

    // ── Resize handler ─────────────────────────────────
    window.addEventListener('resize', onResize);
    onResize();

    console.log('[✓] Scene initialised with studio setup');
}


function onResize() {
    const col = document.querySelector('.canvas-col');
    if (!col || !State.renderer || !State.camera) return;

    const w = col.clientWidth;
    const h = col.clientHeight || window.innerHeight - 55;

    State.renderer.setSize(w, h);
    State.camera.aspect = w / h;
    State.camera.updateProjectionMatrix();
}


// ════════════════════════════════════════════════════════
// READY PLAYER ME — IFRAME INTEGRATION
// ════════════════════════════════════════════════════════

function openRPMCreator() {
    const overlay = document.getElementById('rpmOverlay');
    const iframe  = document.getElementById('rpmIframe');

    if (!overlay || !iframe) return;

    // Build RPM URL with parameters
    // - frameApi=1   → enables postMessage communication
    // - clearCache=1 → always show fresh creator
    const rpmUrl =
        `https://${CONFIG.RPM_SUBDOMAIN}.readyplayer.me/` +
        `avatar?frameApi&clearCache=1&` +
        `bodyType=fullbody&` +
        `quickStart=false`;

    iframe.src   = rpmUrl;
    State.rpmOpen = true;
    overlay.classList.add('open');

    showToast('🎨 Create your avatar!', 'info');
    updateStatus('rpm', 'loading');
    console.log('[~] RPM creator opened:', rpmUrl);
}


function closeRPMCreator() {
    const overlay = document.getElementById('rpmOverlay');
    const iframe  = document.getElementById('rpmIframe');

    if (!overlay) return;

    overlay.classList.remove('open');
    State.rpmOpen = false;

    // Clear iframe to stop it running
    if (iframe) iframe.src = 'about:blank';

    updateStatus('rpm', 'off');
}


// ── Listen for RPM postMessage ────────────────────────
// RPM sends the GLB URL via window.postMessage
// when user clicks "Done" / "Save"

function setupRPMListener() {
    window.addEventListener('message', (event) => {

        // Only handle RPM messages
        if (!event.origin.includes('readyplayer.me')) {
            return;
        }

        const data = event.data;
        console.log('[RPM] Message received:', data);

        // Parse the message
        if (typeof data === 'string') {
            // Older RPM API sends URL string directly
            if (data.includes('models.readyplayer.me') &&
                data.endsWith('.glb')) {
                handleAvatarUrl(data);
            }
        } else if (typeof data === 'object' && data !== null) {
            // Newer RPM API sends object
            if (data.eventName === 'v1.avatar.exported') {
                handleAvatarUrl(data.data?.url || data.url);
            } else if (data.source === 'readyplayerme') {
                if (data.eventName === 'v1.avatar.exported') {
                    handleAvatarUrl(data.data?.url);
                }
            }
        }
    });

    console.log('[✓] RPM postMessage listener ready');
}


function handleAvatarUrl(url) {
    if (!url) return;

    console.log('[✓] Avatar URL received:', url);

    // Append quality params if not already there
    if (!url.includes('morphTargets')) {
        url += (url.includes('?') ? '&' : '?') +
               'morphTargets=ARKit&textureAtlas=1024';
    }

    State.avatarUrl = url;

    // Close RPM iframe
    closeRPMCreator();

    // Update UI
    const urlInput = document.getElementById('avatarUrlInput');
    if (urlInput) urlInput.value = url;

    updateStatus('rpm', 'on');
    updateInfoRow('avatar-url', shortenUrl(url));

    showToast('✅ Avatar created! Loading model…', 'success');

    // Load the avatar into Three.js scene
    loadAvatarModel(url);
}


// ════════════════════════════════════════════════════════
// GLTF MODEL LOADER
// ════════════════════════════════════════════════════════

function loadAvatarModel(url) {
    if (!url) {
        showToast('❌ No avatar URL', 'error');
        return;
    }

    if (State.isLoading) {
        showToast('⏳ Already loading…', 'info');
        return;
    }

    State.isLoading = true;
    showCanvasLoader(true, 'Loading avatar model…');
    updateStatus('model', 'loading');

    console.log('[~] Loading avatar from:', url);

    // Remove old avatar from scene
    if (State.avatar) {
        State.scene.remove(State.avatar);
        State.avatar.traverse(child => {
            if (child.isMesh) {
                child.geometry.dispose();
                if (Array.isArray(child.material)) {
                    child.material.forEach(m => m.dispose());
                } else {
                    child.material.dispose();
                }
            }
        });
        State.avatar = null;
    }

    // ── Create GLTFLoader ──────────────────────────────
    // GLTFLoader is in the Three.js extras
    const loader = new THREE.GLTFLoader();

    // Add DRACO decoder for compressed models
    // (RPM models are sometimes DRACO compressed)
    if (THREE.DRACOLoader) {
        const draco = new THREE.DRACOLoader();
        draco.setDecoderPath(
            'https://www.gstatic.com/draco/versioned/' +
            'decoders/1.5.6/'
        );
        loader.setDRACOLoader(draco);
    }

    // ── Load the model ─────────────────────────────────
    loader.load(
        url,

        // ── onLoad ────────────────────────────────────
        function(gltf) {
            console.log('[✓] GLTF loaded:', gltf);
            onAvatarLoaded(gltf);
        },

        // ── onProgress ────────────────────────────────
        function(progress) {
            if (progress.total > 0) {
                const pct = Math.round(
                    (progress.loaded / progress.total) * 100
                );
                showCanvasLoader(
                    true,
                    `Loading avatar… ${pct}%`
                );
                console.log(`[~] Loading: ${pct}%`);
            }
        },

        // ── onError ───────────────────────────────────
        function(error) {
            console.error('[✗] GLTF load error:', error);
            State.isLoading = false;
            showCanvasLoader(false);
            updateStatus('model', 'off');

            // Show helpful error message
            if (error.message?.includes('CORS') ||
                error.message?.includes('fetch')) {
                showToast(
                    '❌ CORS error — try the manual URL method',
                    'error'
                );
            } else {
                showToast(
                    '❌ Failed to load model — check URL',
                    'error'
                );
            }
        }
    );
}


function onAvatarLoaded(gltf) {
    const model = gltf.scene;

    // ── Scale and position ─────────────────────────────
    model.scale.setScalar(CONFIG.AVATAR_SCALE);
    model.position.set(0, CONFIG.AVATAR_Y_POS, 0);

    // Enable shadows on all meshes
    model.traverse(child => {
        if (child.isMesh) {
            child.castShadow    = true;
            child.receiveShadow = true;

            // Improve material quality
            if (child.material) {
                child.material.envMapIntensity = 0.8;
            }
        }
    });

    // Add to scene
    State.scene.add(model);
    // ADD these two lines inside onAvatarLoaded(),
// right after:  State.scene.add(model);

    // Enhance materials for better look
    enhanceAvatarMaterials(model);

    // Auto-frame camera to avatar
    autoFrameAvatar(
        State.camera,
        model,
        State.controls || null
    );
    State.avatar = model;

    // ── Setup animations if available ─────────────────
    if (gltf.animations && gltf.animations.length > 0) {
        State.avatarMixer = new THREE.AnimationMixer(model);
        // Play first animation (usually idle)
        const idleAction = State.avatarMixer.clipAction(
            gltf.animations[0]
        );
        idleAction.play();
        console.log('[✓] Animation playing:',
                    gltf.animations[0].name);
    }

    // ── Update UI ──────────────────────────────────────
    State.isLoading = false;
    showCanvasLoader(false);
    updateStatus('model', 'on');

    // Count meshes
    let meshCount = 0;
    model.traverse(c => { if (c.isMesh) meshCount++; });

    updateInfoRow('mesh-count', meshCount + ' meshes');
    updateInfoRow('avatar-url',
        State.avatarUrl
            ? shortenUrl(State.avatarUrl)
            : 'Ready Player Me'
    );

    // Enable apply texture button
    const applyBtn = document.getElementById('applyTextureBtn');
    if (applyBtn) applyBtn.disabled = false;

    showToast('🎉 Avatar loaded!', 'success');
    console.log('[✓] Avatar added to scene, meshes:', meshCount);
}


// ════════════════════════════════════════════════════════
// ANIMATION LOOP
// ════════════════════════════════════════════════════════
// REPLACE startRenderLoop() in avatar_loader.js

function startRenderLoop() {
    function animate() {
        requestAnimationFrame(animate);

        const delta = State.clock.getDelta();

        // Update avatar animations
        if (State.avatarMixer) {
            State.avatarMixer.update(delta);
        }

        // Update scene objects (particles, ring pulse etc)
        updateSceneObjects(delta);

        // Update OrbitControls (added in Step 4)
        if (State.controls) {
            State.controls.update();
        }

        State.renderer.render(State.scene, State.camera);
    }

    animate();
    console.log('[✓] Render loop started');
}

// ════════════════════════════════════════════════════════
// LOAD DEFAULT / DEMO AVATAR
// ════════════════════════════════════════════════════════

function loadDefaultAvatar() {
    showToast('🔄 Loading demo avatar…', 'info');
    const url = CONFIG.DEFAULT_AVATAR_URL;
    State.avatarUrl = url;

    const urlInput = document.getElementById('avatarUrlInput');
    if (urlInput) urlInput.value = url;

    loadAvatarModel(url);
}


function loadFromUrlInput() {
    const input = document.getElementById('avatarUrlInput');
    if (!input || !input.value.trim()) {
        showToast('❌ Enter a GLB URL first', 'error');
        return;
    }

    let url = input.value.trim();
    if (!url.includes('.glb') &&
        !url.includes('readyplayer.me')) {
        showToast('❌ URL must be a .glb file', 'error');
        return;
    }

    State.avatarUrl = url;
    loadAvatarModel(url);
}


// ════════════════════════════════════════════════════════
// UI HELPERS
// ════════════════════════════════════════════════════════

function showCanvasLoader(show, text = 'Loading…') {
    const loader  = document.getElementById('canvasLoader');
    const loaderT = document.getElementById('loaderText');

    if (!loader) return;

    if (show) {
        loader.classList.remove('hidden');
        if (loaderT) loaderT.textContent = text;
    } else {
        loader.classList.add('hidden');
    }
}


function updateStatus(key, state) {
    // state: 'on' | 'off' | 'loading'
    const dot = document.getElementById(`status-${key}`);
    if (!dot) return;
    dot.className = `status-dot ${state}`;
}


function updateInfoRow(key, value) {
    const el = document.getElementById(`info-${key}`);
    if (el) el.textContent = value;
}


// Toast notification
let toastTimer = null;

function showToast(msg, type = 'info') {
    let toast = document.getElementById('globalToast');
    if (!toast) {
        toast = document.createElement('div');
        toast.id = 'globalToast';
        toast.className = 'toast';
        document.body.appendChild(toast);
    }

    toast.textContent = msg;
    toast.className   = `toast toast-${type} show`;

    if (toastTimer) clearTimeout(toastTimer);
    toastTimer = setTimeout(() => {
        toast.classList.remove('show');
    }, 3000);
}


function shortenUrl(url) {
    if (!url) return '—';
    // Extract model ID from RPM URL
    const match = url.match(/\/([a-f0-9]{24})\.glb/);
    if (match) return match[1].slice(0,8) + '…';
    // Generic shorten
    if (url.length > 28) return url.slice(0,25) + '…';
    return url;
}


// ════════════════════════════════════════════════════════
// INIT — called when page loads
// ════════════════════════════════════════════════════════

function initAvatarSystem() {
    console.log('══ Tryvora Avatar System Starting ══');

    // 1. Set up Three.js scene
    initScene();

    // 2. Start render loop (shows black scene for now)
    startRenderLoop();

    // 3. Set up RPM postMessage listener
    setupRPMListener();

    // 4. Load demo avatar immediately so page isn't blank
    loadDefaultAvatar();

    console.log('[✓] Avatar system ready');
}