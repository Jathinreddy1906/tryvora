// static/js/texture_applier.js
// ─────────────────────────────────────────────────────
// STEP 3: Clothing Texture Applier
//
// How it works:
//   1. Scan avatar for shirt/torso meshes
//   2. Save original materials (for undo)
//   3. Load clothing image as Three.js texture
//   4. Wrap texture around torso mesh UV coords
//   5. Provide controls: scale, blend, opacity
// ─────────────────────────────────────────────────────

'use strict';

// ════════════════════════════════════════════════════════
// CONFIG
// ════════════════════════════════════════════════════════

const TEX_CONFIG = {
    // Mesh name keywords that indicate shirt/torso
    // RPM meshes are named like: "Wolf3D_Outfit_Top"
    SHIRT_KEYWORDS: [
        'outfit_top', 'shirt', 'top', 'torso',
        'body',       'jacket', 'hoodie', 'sweater',
        'tshirt',     'blouse', 'coat',  'vest',
        'upper',      'chest',  'outfit'
    ],

    // Mesh name keywords to SKIP
    SKIP_KEYWORDS: [
        'eye', 'teeth', 'tongue', 'hair', 'beard',
        'glasses', 'hat', 'shoe', 'bottom', 'pants',
        'skirt', 'sock', 'foot', 'leg', 'eyebrow',
        'eyelash', 'nail'
    ],

    // Default texture settings
    DEFAULT_SCALE  : 1.0,
    DEFAULT_OPACITY: 1.0,
    DEFAULT_BLEND  : 'replace',
};


// ════════════════════════════════════════════════════════
// GLOBAL TEXTURE STATE
// ════════════════════════════════════════════════════════

const TexState = {
    // Map: mesh UUID → original material(s)
    originalMaterials : new Map(),

    // Map: mesh UUID → modified material(s)
    modifiedMaterials : new Map(),

    // Currently loaded texture
    currentTexture    : null,

    // Settings
    scale    : TEX_CONFIG.DEFAULT_SCALE,
    opacity  : TEX_CONFIG.DEFAULT_OPACITY,
    blendMode: TEX_CONFIG.DEFAULT_BLEND,

    // Which meshes are shirt meshes
    shirtMeshes : [],

    // Is texture applied?
    applied : false,
};


// ════════════════════════════════════════════════════════
// PART 1 — SCAN AVATAR FOR SHIRT MESHES
// ════════════════════════════════════════════════════════

function scanAvatarForShirtMeshes(avatar) {
    /**
     * Walks through all meshes in the avatar and
     * identifies which ones are shirt/torso meshes.
     *
     * RPM avatars name their meshes like:
     *   Wolf3D_Body
     *   Wolf3D_Outfit_Top
     *   Wolf3D_Outfit_Bottom
     *
     * We match against SHIRT_KEYWORDS list.
     */

    TexState.shirtMeshes = [];
    const allMeshes      = [];

    console.log('[~] Scanning avatar meshes...');

    avatar.traverse(child => {
        if (!child.isMesh) return;

        const name = child.name.toLowerCase();
        allMeshes.push(child.name);

        // Check skip list first
        const shouldSkip = TEX_CONFIG.SKIP_KEYWORDS
            .some(kw => name.includes(kw));

        if (shouldSkip) return;

        // Check shirt keywords
        const isShirt = TEX_CONFIG.SHIRT_KEYWORDS
            .some(kw => name.includes(kw));

        if (isShirt) {
            TexState.shirtMeshes.push(child);
            console.log(`  ✓ Shirt mesh: "${child.name}"`);
        }
    });

    // If nothing matched, try taking the largest mesh
    // (usually the body/outfit on simple avatars)
    if (TexState.shirtMeshes.length === 0) {
        console.log('[!] No shirt mesh found by name');
        console.log('    All meshes:', allMeshes);

        const largest = findLargestMesh(avatar);
        if (largest) {
            TexState.shirtMeshes = [largest];
            console.log(`[~] Using largest mesh: "${largest.name}"`);
        }
    }

    console.log(`[✓] Shirt meshes: ${TexState.shirtMeshes.length}`);
    return TexState.shirtMeshes;
}


function findLargestMesh(avatar) {
    /**
     * Finds the mesh with the most vertices.
     * Usually the body/torso mesh on simple avatars.
     */
    let largest     = null;
    let largestCount = 0;

    avatar.traverse(child => {
        if (!child.isMesh || !child.geometry) return;

        const name  = child.name.toLowerCase();
        // Skip obvious non-body meshes
        if (name.includes('eye')  ||
            name.includes('hair') ||
            name.includes('teeth')) return;

        const count = child.geometry.attributes
                           .position?.count || 0;
        if (count > largestCount) {
            largestCount = count;
            largest      = child;
        }
    });

    return largest;
}


// ════════════════════════════════════════════════════════
// PART 2 — SAVE ORIGINAL MATERIALS
// Needed for the "Undo" / "Remove Cloth" feature
// ════════════════════════════════════════════════════════

function saveOriginalMaterials(avatar) {
    /**
     * Before we modify anything, save the original
     * materials so we can restore them later.
     *
     * Materials can be single or array (multi-material).
     * We handle both cases.
     */

    TexState.originalMaterials.clear();

    avatar.traverse(child => {
        if (!child.isMesh) return;

        // Clone the material(s)
        if (Array.isArray(child.material)) {
            TexState.originalMaterials.set(
                child.uuid,
                child.material.map(m => m.clone())
            );
        } else if (child.material) {
            TexState.originalMaterials.set(
                child.uuid,
                child.material.clone()
            );
        }
    });

    console.log(
        `[✓] Saved ${TexState.originalMaterials.size}` +
        ` original materials`
    );
}


// ════════════════════════════════════════════════════════
// PART 3 — LOAD IMAGE AS TEXTURE
// Converts image URL/file → Three.js texture
// ════════════════════════════════════════════════════════

function loadImageAsTexture(imageSource) {
    /**
     * imageSource can be:
     *   - URL string (https://... or /static/...)
     *   - Base64 data URL (from FileReader)
     *   - Blob URL
     *
     * Returns a Promise that resolves with the texture.
     */

    return new Promise((resolve, reject) => {
        const loader = new THREE.TextureLoader();

        loader.load(
            imageSource,

            // onLoad
            texture => {
                // Texture settings for clothing
                texture.wrapS     = THREE.RepeatWrapping;
                texture.wrapT     = THREE.RepeatWrapping;
                texture.repeat.set(
                    TexState.scale,
                    TexState.scale
                );
                texture.flipY     = false; // GLB uses this
                texture.encoding  = THREE.sRGBEncoding;
                texture.anisotropy = 4;   // sharper at angles
                texture.needsUpdate = true;

                TexState.currentTexture = texture;
                console.log('[✓] Texture loaded:', imageSource
                    .slice(0, 50) + '...');
                resolve(texture);
            },

            // onProgress
            undefined,

            // onError
            error => {
                console.error('[✗] Texture load failed:', error);
                reject(error);
            }
        );
    });
}


// ════════════════════════════════════════════════════════
// PART 4 — CREATE CLOTH MATERIAL
// Builds a new material with the clothing texture
// ════════════════════════════════════════════════════════

function createClothMaterial(originalMaterial, texture) {
    /**
     * Creates a new material that looks like fabric
     * wearing the clothing texture.
     *
     * We base it on the original material so we keep
     * the same mesh normals, shadows, etc.
     * Only the color/map changes.
     */

    let newMat;

    // Clone original to preserve settings
    if (originalMaterial &&
        originalMaterial.isMeshStandardMaterial) {
        newMat = originalMaterial.clone();
    } else {
        newMat = new THREE.MeshStandardMaterial();
    }

    // ── Apply texture ──────────────────────────────────
    newMat.map     = texture;
    newMat.opacity = TexState.opacity;

    // ── Blend mode ────────────────────────────────────
    switch (TexState.blendMode) {

        case 'replace':
            // Pure texture, ignore original color
            newMat.color.set(0xffffff);
            newMat.map         = texture;
            newMat.roughness   = 0.75;
            newMat.metalness   = 0.0;
            break;

        case 'overlay':
            // Mix texture with original color
            newMat.map         = texture;
            newMat.color.set(0xdddddd);
            newMat.roughness   = 0.8;
            newMat.metalness   = 0.0;
            break;

        case 'multiply':
            // Darken original with texture
            newMat.color.set(0xaaaaaa);
            newMat.map         = texture;
            newMat.roughness   = 0.85;
            break;

        case 'screen':
            // Lighten original with texture
            newMat.color.set(0xffffff);
            newMat.map         = texture;
            newMat.roughness   = 0.6;
            break;
    }

    // ── Fabric look ────────────────────────────────────
    newMat.roughness      = 0.78;
    newMat.metalness      = 0.0;
    newMat.envMapIntensity = 0.3;

    // Transparent if opacity < 1
    if (TexState.opacity < 1.0) {
        newMat.transparent = true;
        newMat.opacity     = TexState.opacity;
    }

    newMat.needsUpdate = true;
    return newMat;
}


// ════════════════════════════════════════════════════════
// PART 5 — APPLY TEXTURE TO SHIRT MESHES
// The main function
// ════════════════════════════════════════════════════════

async function applyTextureToAvatar(
    avatar, imageSource, progressCallback
) {
    /**
     * Full texture application pipeline:
     *
     * 1. Scan for shirt meshes (if not done)
     * 2. Save originals (if not done)
     * 3. Load image as texture
     * 4. Apply to each shirt mesh
     * 5. Update UI
     *
     * progressCallback(0-100) for loading bar.
     */

    if (!avatar) {
        throw new Error('No avatar loaded!');
    }

    console.log('[~] Applying texture to avatar...');
    if (progressCallback) progressCallback(10);

    // Step 1: Scan for shirt meshes
    if (TexState.shirtMeshes.length === 0) {
        scanAvatarForShirtMeshes(avatar);
    }

    // Step 2: Save original materials
    if (TexState.originalMaterials.size === 0) {
        saveOriginalMaterials(avatar);
    }

    if (progressCallback) progressCallback(25);

    // Step 3: Load texture
    let texture;
    try {
        texture = await loadImageAsTexture(imageSource);
    } catch (e) {
        throw new Error(`Failed to load texture: ${e.message}`);
    }

    if (progressCallback) progressCallback(60);

    // Step 4: Apply to shirt meshes
    if (TexState.shirtMeshes.length === 0) {
        console.warn('[!] No shirt meshes — applying to all');
        // Fallback: apply to all non-face meshes
        applyToAllMeshes(avatar, texture);
    } else {
        applyToShirtMeshes(texture);
    }

    if (progressCallback) progressCallback(90);

    TexState.applied = true;

    if (progressCallback) progressCallback(100);

    console.log('[✓] Texture applied successfully!');
    return true;
}


function applyToShirtMeshes(texture) {
    /**
     * Applies texture to detected shirt meshes only.
     */

    TexState.shirtMeshes.forEach(mesh => {
        const origMat = TexState.originalMaterials
                            .get(mesh.uuid);

        if (Array.isArray(mesh.material)) {
            // Multi-material mesh
            mesh.material = mesh.material.map((m, i) => {
                const orig = Array.isArray(origMat)
                    ? origMat[i] : origMat;
                return createClothMaterial(orig, texture);
            });
        } else {
            // Single material
            mesh.material = createClothMaterial(
                origMat, texture
            );
        }

        mesh.material.needsUpdate = true;
        console.log(`  ✓ Applied to: "${mesh.name}"`);
    });
}


function applyToAllMeshes(avatar, texture) {
    /**
     * Fallback: applies to ALL non-face meshes.
     * Used when shirt mesh detection fails.
     */

    avatar.traverse(child => {
        if (!child.isMesh) return;
        const name = child.name.toLowerCase();
        if (name.includes('eye')   ||
            name.includes('hair')  ||
            name.includes('teeth') ||
            name.includes('eyebrow')) return;

        const orig = TexState.originalMaterials
                         .get(child.uuid);
        child.material = createClothMaterial(
            orig, texture
        );
        child.material.needsUpdate = true;
    });
}


// ════════════════════════════════════════════════════════
// PART 6 — RESTORE ORIGINAL MATERIALS (UNDO)
// ════════════════════════════════════════════════════════

function restoreOriginalMaterials(avatar) {
    /**
     * Restores all meshes to their original materials.
     * Called when user clicks "Remove Cloth".
     */

    if (!avatar || TexState.originalMaterials.size === 0) {
        console.log('[!] No originals to restore');
        return false;
    }

    let restored = 0;

    avatar.traverse(child => {
        if (!child.isMesh) return;

        const origMat = TexState.originalMaterials
                            .get(child.uuid);
        if (!origMat) return;

        // Dispose modified material to free GPU memory
        if (child.material &&
            child.material !== origMat) {
            if (Array.isArray(child.material)) {
                child.material.forEach(m => {
                    if (m.map) m.map.dispose();
                    m.dispose();
                });
            } else {
                if (child.material.map)
                    child.material.map.dispose();
                child.material.dispose();
            }
        }

        // Restore original
        child.material = origMat;
        child.material.needsUpdate = true;
        restored++;
    });

    // Dispose current texture
    if (TexState.currentTexture) {
        TexState.currentTexture.dispose();
        TexState.currentTexture = null;
    }

    TexState.applied = false;
    console.log(`[✓] Restored ${restored} original materials`);
    return true;
}


// ════════════════════════════════════════════════════════
// PART 7 — LIVE CONTROLS
// Scale, opacity, blend mode — applied instantly
// ════════════════════════════════════════════════════════

function updateTextureScale(scale) {
    /**
     * Changes how many times the texture tiles.
     * scale=1 → fills the mesh once (best for shirts)
     * scale=2 → tiles twice (for patterns)
     */

    TexState.scale = parseFloat(scale);

    if (!TexState.currentTexture) return;

    TexState.currentTexture.repeat.set(
        TexState.scale,
        TexState.scale
    );
    TexState.currentTexture.needsUpdate = true;

    console.log(`[~] Texture scale: ${TexState.scale}`);
}


function updateTextureOpacity(opacity) {
    /**
     * Makes the texture semi-transparent.
     * 1.0 = fully opaque, 0.0 = invisible
     */

    TexState.opacity = parseFloat(opacity);

    TexState.shirtMeshes.forEach(mesh => {
        const applyOpacity = mat => {
            mat.opacity     = TexState.opacity;
            mat.transparent = TexState.opacity < 1.0;
            mat.needsUpdate = true;
        };

        if (Array.isArray(mesh.material)) {
            mesh.material.forEach(applyOpacity);
        } else if (mesh.material) {
            applyOpacity(mesh.material);
        }
    });
}


function updateBlendMode(mode) {
    /**
     * Changes how the texture blends with original mesh.
     * Requires re-applying the texture.
     */

    TexState.blendMode = mode;

    // Re-apply with new blend mode if texture loaded
    if (TexState.currentTexture && State.avatar) {
        applyToShirtMeshes(TexState.currentTexture);
        console.log(`[~] Blend mode: ${mode}`);
    }
}


function updateTextureRotation(degrees) {
    /**
     * Rotates the texture on the mesh.
     * Useful for fixing upside-down logos etc.
     */

    if (!TexState.currentTexture) return;

    TexState.currentTexture.rotation =
        (degrees * Math.PI) / 180;
    TexState.currentTexture.needsUpdate = true;
}


function updateTextureOffset(u, v) {
    /**
     * Offsets the texture position on the mesh.
     * Useful for centering a logo/graphic.
     */

    if (!TexState.currentTexture) return;

    TexState.currentTexture.offset.set(
        parseFloat(u), parseFloat(v)
    );
    TexState.currentTexture.needsUpdate = true;
}


// ════════════════════════════════════════════════════════
// PART 8 — HIGHLIGHT SHIRT MESHES (DEBUG)
// Shows which meshes were detected as shirts
// ════════════════════════════════════════════════════════

function highlightShirtMeshes(avatar, highlight = true) {
    /**
     * Temporarily colors shirt meshes bright green
     * so you can see which parts were detected.
     * Set highlight=false to remove.
     */

    if (highlight) {
        TexState.shirtMeshes.forEach(mesh => {
            const hlMat = new THREE.MeshBasicMaterial({
                color    : 0x00ff88,
                wireframe: false,
                opacity  : 0.7,
                transparent: true,
            });
            mesh.material = hlMat;
        });
        console.log('[~] Shirt meshes highlighted green');
    } else {
        // Restore
        restoreOriginalMaterials(avatar);
    }
}


// ════════════════════════════════════════════════════════
// PART 9 — PROCESS CLOTH IMAGE
// Makes clothing image look better before applying
// ════════════════════════════════════════════════════════

function processClothImage(
    imageDataUrl,
    removeBackground = true
) {
    /**
     * Processes the clothing image before applying:
     *   1. Optionally removes white background
     *   2. Enhances contrast slightly
     *   3. Returns processed image as data URL
     *
     * We use a hidden canvas for pixel manipulation.
     */

    return new Promise(resolve => {
        const img    = new Image();
        img.onload = () => {
            const canvas  = document.createElement('canvas');
            const ctx     = canvas.getContext('2d');
            canvas.width  = img.width;
            canvas.height = img.height;

            // Draw image
            ctx.drawImage(img, 0, 0);

            if (!removeBackground) {
                resolve(canvas.toDataURL());
                return;
            }

            // Remove white/near-white background
            const imageData = ctx.getImageData(
                0, 0, canvas.width, canvas.height
            );
            const data = imageData.data;

            for (let i = 0; i < data.length; i += 4) {
                const r = data[i];
                const g = data[i+1];
                const b = data[i+2];

                // Is this pixel white/near-white?
                const isWhite = r > 230 &&
                                g > 230 &&
                                b > 230;

                // Is this pixel near-white grey?
                const isGrey  = r > 200 && g > 200 &&
                                b > 200 &&
                                Math.abs(r-g) < 20 &&
                                Math.abs(g-b) < 20;

                if (isWhite || isGrey) {
                    // Make transparent
                    data[i+3] = 0;
                } else if (data[i+3] < 50) {
                    // Already transparent
                    data[i+3] = 0;
                }
            }

            ctx.putImageData(imageData, 0, 0);
            resolve(canvas.toDataURL('image/png'));
        };

        img.onerror = () => resolve(imageDataUrl);
        img.src     = imageDataUrl;
        img.crossOrigin = 'anonymous';
    });
}


// ════════════════════════════════════════════════════════
// PART 10 — MAIN PUBLIC FUNCTIONS
// Called from avatar.html
// ════════════════════════════════════════════════════════

async function applyClothingTexture() {
    /**
     * Main "Apply to Avatar" button handler.
     * Gets image source, processes it, applies it.
     */

    if (!State.avatar) {
        showToast('❌ Load an avatar first!', 'error');
        return;
    }

    // Get the pending texture source
    const src = window._pendingTexture ||
                getUploadedClothingUrl();

    if (!src) {
        showToast('❌ Select a clothing image first!', 'error');
        return;
    }

    // Show progress
    const btn = document.getElementById('applyTextureBtn');
    if (btn) {
        btn.disabled    = true;
        btn.textContent = '⏳ Applying…';
    }

    showToast('🎨 Applying clothing texture…', 'info');
    updateStatus('texture', 'loading');

    try {
        // Process image (remove background)
        const removesBg = document.getElementById(
            'removeBgCheck'
        )?.checked ?? true;

        let processedSrc = src;
        if (removesBg && !src.includes('.glb')) {
            showToast('✂️ Removing background…', 'info');
            processedSrc = await processClothImage(
                src, true
            );
        }

        // Apply texture with progress
        await applyTextureToAvatar(
            State.avatar,
            processedSrc,
            pct => {
                const bar = document.getElementById(
                    'texProgressBar'
                );
                if (bar) bar.style.width = pct + '%';
            }
        );

        updateStatus('texture', 'on');
        showToast('✅ Outfit applied!', 'success');

        // Update UI
        const removeBtn = document.getElementById(
            'removeClothBtn'
        );
        if (removeBtn) removeBtn.disabled = false;

    } catch (err) {
        console.error('[✗] Texture apply failed:', err);
        updateStatus('texture', 'off');
        showToast(`❌ ${err.message}`, 'error');
    } finally {
        if (btn) {
            btn.disabled    = false;
            btn.textContent = '👔 Apply to Avatar';
        }

        // Reset progress bar
        setTimeout(() => {
            const bar = document.getElementById(
                'texProgressBar'
            );
            if (bar) bar.style.width = '0%';
        }, 1500);
    }
}


function removeClothingTexture() {
    /**
     * "Remove Cloth" button handler.
     * Restores original avatar outfit.
     */

    if (!State.avatar) return;

    const ok = restoreOriginalMaterials(State.avatar);

    if (ok) {
        updateStatus('texture', 'off');
        showToast('✓ Original outfit restored', 'info');

        const removeBtn = document.getElementById(
            'removeClothBtn'
        );
        if (removeBtn) removeBtn.disabled = true;
    }
}


function getUploadedClothingUrl() {
    /**
     * Gets clothing URL from the texture preview image.
     */

    const img = document.querySelector(
        '#texPreview img'
    );
    return img ? img.src : null;
}


function onTextureFileSelect(input) {
    /**
     * File input change handler.
     * Shows preview and stores source.
     */

    if (!input.files || !input.files[0]) return;

    const file   = input.files[0];
    const reader = new FileReader();

    reader.onload = e => {
        // Show preview
        const preview = document.getElementById('texPreview');
        if (preview) {
            preview.innerHTML =
                `<img src="${e.target.result}"
                      alt="Clothing texture"
                      style="width:100%;height:100%;
                             object-fit:contain;">`;
        }

        // Store for apply
        window._pendingTexture = e.target.result;

        // Enable apply button
        const btn = document.getElementById(
            'applyTextureBtn'
        );
        if (btn) btn.disabled = !State.avatar;

        showToast('👔 Image ready — click Apply!', 'info');
    };

    reader.readAsDataURL(file);
}


// Called when avatar loads — scan and save
function onAvatarReadyForTexture(avatar) {
    scanAvatarForShirtMeshes(avatar);
    saveOriginalMaterials(avatar);
    console.log('[✓] Avatar ready for texturing');
}