// static/js/scene_setup.js
// ─────────────────────────────────────────────────────
// STEP 2: Professional Scene Setup
//
// Handles:
//   1. Studio lighting (4 lights like photography)
//   2. Environment map (reflections on avatar)
//   3. Ground plane with subtle reflection
//   4. Particle system (floating background dots)
//   5. Auto-framing camera to any avatar
//   6. Smooth camera intro animation
// ─────────────────────────────────────────────────────

'use strict';

// ════════════════════════════════════════════════════════
// PART 1 — STUDIO LIGHTING
// 4-point lighting = how photographers light a person
// ════════════════════════════════════════════════════════

function setupStudioLighting(scene) {
    /**
     * 4-point lighting setup:
     *
     *   KEY LIGHT    → main light (front-left, bright)
     *   FILL LIGHT   → softens shadows (front-right, dim)
     *   RIM LIGHT    → outlines edges (behind, accent)
     *   AMBIENT      → fills dark areas (all directions)
     *
     * This makes the avatar look like it's in a studio,
     * just like Snapchat / Meta avatar renders!
     */

    // ── 1. Ambient light (base fill) ──────────────────
    // Soft overall illumination so no area is pitch black
    const ambient = new THREE.AmbientLight(
        0xffffff,   // white
        0.45        // intensity (0=dark, 1=bright)
    );
    scene.add(ambient);

    // ── 2. Key light (main light source) ──────────────
    // Positioned front-left at 45°, like a studio key
    const keyLight = new THREE.DirectionalLight(
        0xfff5e8,   // slightly warm white
        1.2
    );
    keyLight.position.set(-2.5, 4, 3);
    keyLight.castShadow = true;

    // Shadow quality settings
    keyLight.shadow.mapSize.width  = 2048;
    keyLight.shadow.mapSize.height = 2048;
    keyLight.shadow.camera.near    = 0.1;
    keyLight.shadow.camera.far     = 20;
    keyLight.shadow.camera.left    = -3;
    keyLight.shadow.camera.right   = 3;
    keyLight.shadow.camera.top     = 6;
    keyLight.shadow.camera.bottom  = -2;
    keyLight.shadow.bias           = -0.001;
    keyLight.shadow.normalBias     = 0.02;

    scene.add(keyLight);

    // ── 3. Fill light (soften shadows) ────────────────
    // Positioned front-right, dimmer than key
    const fillLight = new THREE.DirectionalLight(
        0xe8f0ff,   // slightly cool/blue
        0.5
    );
    fillLight.position.set(3, 3, 2);
    scene.add(fillLight);

    // ── 4. Rim light (outline/silhouette) ─────────────
    // Behind avatar, creates a glowing edge effect
    const rimLight = new THREE.DirectionalLight(
        0xb088ff,   // purple tint (like Meta avatars!)
        0.8
    );
    rimLight.position.set(0, 3, -4);
    scene.add(rimLight);

    // ── 5. Under-chin fill (subtle) ───────────────────
    // Lifts the chin shadows for a flattering look
    const chinLight = new THREE.PointLight(
        0xfff0e0,   // warm
        0.3,
        5           // range
    );
    chinLight.position.set(0, 0.5, 2);
    scene.add(chinLight);

    // ── 6. Hair highlight ─────────────────────────────
    // Top light makes hair look shiny
    const hairLight = new THREE.SpotLight(
        0xffffff,
        0.6,
        8,          // distance
        Math.PI / 6 // angle
    );
    hairLight.position.set(0, 5, 1);
    hairLight.target.position.set(0, 2, 0);
    scene.add(hairLight);
    scene.add(hairLight.target);

    console.log('[✓] Studio lighting setup: 6 lights');

    return {
        ambient, keyLight, fillLight,
        rimLight, chinLight, hairLight
    };
}


// ════════════════════════════════════════════════════════
// PART 2 — ENVIRONMENT MAP
// Fake reflections on shiny surfaces (skin, fabric)
// ════════════════════════════════════════════════════════

function setupEnvironmentMap(scene, renderer) {
    /**
     * Environment maps add reflections to materials.
     * We use a procedural gradient env map since we
     * can't load HDR files easily from CDN.
     *
     * This makes skin look natural and clothes look
     * like they have fabric texture.
     */

    // Create a simple gradient cube texture as env map
    const pmremGenerator = new THREE.PMREMGenerator(renderer);
    pmremGenerator.compileEquirectangularShader();

    // Build a gradient scene to use as environment
    const envScene = new THREE.Scene();

    // Top hemisphere: cool blue-grey sky
    envScene.background = new THREE.Color(0x1a1035);

    // Create gradient sphere to approximate studio env
    const envGeo = new THREE.SphereGeometry(10, 16, 16);
    const envMat = new THREE.MeshBasicMaterial({
        color   : 0x1a1035,
        side    : THREE.BackSide,
    });
    const envSphere = new THREE.Mesh(envGeo, envMat);
    envScene.add(envSphere);

    // Generate env map from this scene
    const envRT   = pmremGenerator.fromScene(
        new THREE.RoomEnvironment(), 0.04
    );
    scene.environment = envRT.texture;

    pmremGenerator.dispose();

    console.log('[✓] Environment map applied');
    return envRT;
}


// ════════════════════════════════════════════════════════
// PART 3 — GROUND PLANE
// Subtle floor with shadow + reflection
// ════════════════════════════════════════════════════════

function setupGroundPlane(scene) {
    /**
     * Creates a circular ground plane under the avatar.
     * Has two layers:
     *   1. Shadow receiver (catches key light shadow)
     *   2. Gradient disc (subtle radial glow)
     */

    // ── Shadow receiver ────────────────────────────────
    const shadowGeo = new THREE.PlaneGeometry(6, 6);
    const shadowMat = new THREE.ShadowMaterial({
        opacity    : 0.35,
        transparent: true,
    });

    const shadowPlane = new THREE.Mesh(shadowGeo, shadowMat);
    shadowPlane.rotation.x  = -Math.PI / 2;
    shadowPlane.position.y  = -1.0;
    shadowPlane.receiveShadow = true;
    scene.add(shadowPlane);

    // ── Gradient glow disc ─────────────────────────────
    // Circular gradient from purple → transparent
    // Creates a "stage spotlight" effect
    const discGeo = new THREE.CircleGeometry(1.8, 64);

    // Build gradient using vertex colors
    const posAttr  = discGeo.attributes.position;
    const colors   = [];
    const center   = new THREE.Vector3(0, 0, 0);
    const maxDist  = 1.8;

    for (let i = 0; i < posAttr.count; i++) {
        const v    = new THREE.Vector3().fromBufferAttribute(
            posAttr, i
        );
        const dist = v.distanceTo(center);
        const t    = dist / maxDist;  // 0 at center, 1 at edge

        // Center: purple glow / Edge: transparent black
        const r = 0.25 * (1 - t);
        const g = 0.08 * (1 - t);
        const b = 0.45 * (1 - t);
        colors.push(r, g, b);
    }

    discGeo.setAttribute(
        'color',
        new THREE.Float32BufferAttribute(colors, 3)
    );

    const discMat = new THREE.MeshBasicMaterial({
        vertexColors: true,
        transparent : true,
        opacity     : 0.6,
        depthWrite  : false,
    });

    const disc = new THREE.Mesh(discGeo, discMat);
    disc.rotation.x = -Math.PI / 2;
    disc.position.y = -0.999;  // just above shadow plane
    scene.add(disc);

    // ── Grid lines (very subtle) ───────────────────────
    const grid = new THREE.GridHelper(
        6,      // size
        20,     // divisions
        0x1a0f35,  // color 1
        0x120826   // color 2
    );
    grid.position.y = -1.0;
    grid.material.opacity    = 0.3;
    grid.material.transparent = true;
    scene.add(grid);

    console.log('[✓] Ground plane created');
    return { shadowPlane, disc, grid };
}


// ════════════════════════════════════════════════════════
// PART 4 — PARTICLE SYSTEM
// Floating background dots like Meta/Snapchat avatars
// ════════════════════════════════════════════════════════

function createParticleBackground(scene) {
    /**
     * Creates floating particle dots in the background.
     * Gives that "digital / virtual world" feel
     * you see in Meta Avatars and Snapchat Bitmoji.
     */

    const PARTICLE_COUNT = 180;
    const geo = new THREE.BufferGeometry();

    const positions = new Float32Array(PARTICLE_COUNT * 3);
    const colors    = new Float32Array(PARTICLE_COUNT * 3);
    const sizes     = new Float32Array(PARTICLE_COUNT);

    // Color palette for particles
    const palette = [
        new THREE.Color(0x7c3aed),   // purple
        new THREE.Color(0x4f46e5),   // blue-purple
        new THREE.Color(0xa78bfa),   // light purple
        new THREE.Color(0x60a5fa),   // light blue
        new THREE.Color(0xf472b6),   // pink
    ];

    for (let i = 0; i < PARTICLE_COUNT; i++) {
        const i3 = i * 3;

        // Spread particles in a sphere around avatar
        const theta  = Math.random() * Math.PI * 2;
        const phi    = Math.acos(2 * Math.random() - 1);
        const radius = 3 + Math.random() * 5;

        positions[i3]     = radius * Math.sin(phi) * Math.cos(theta);
        positions[i3 + 1] = (Math.random() - 0.3) * 6;
        positions[i3 + 2] = radius * Math.sin(phi) * Math.sin(theta);

        // Random color from palette
        const col = palette[
            Math.floor(Math.random() * palette.length)
        ];
        colors[i3]     = col.r;
        colors[i3 + 1] = col.g;
        colors[i3 + 2] = col.b;

        // Random size
        sizes[i] = 0.5 + Math.random() * 2.5;
    }

    geo.setAttribute('position',
        new THREE.BufferAttribute(positions, 3));
    geo.setAttribute('color',
        new THREE.BufferAttribute(colors, 3));
    geo.setAttribute('size',
        new THREE.BufferAttribute(sizes, 1));

    const mat = new THREE.PointsMaterial({
        size        : 0.04,
        vertexColors: true,
        transparent : true,
        opacity     : 0.55,
        sizeAttenuation: true,
        depthWrite  : false,
    });

    const particles = new THREE.Points(geo, mat);
    scene.add(particles);

    console.log(`[✓] Particles: ${PARTICLE_COUNT} dots`);

    // Return so we can animate them
    return particles;
}


// ════════════════════════════════════════════════════════
// PART 5 — AUTO-FRAMING CAMERA
// Perfectly frames ANY avatar (short/tall/wide)
// ════════════════════════════════════════════════════════

function autoFrameAvatar(camera, avatar, controls) {
    /**
     * Calculates the bounding box of the avatar and
     * positions the camera to perfectly show the full body.
     *
     * Works for any avatar height — no more cropped heads
     * or too-small figures!
     */

    if (!avatar) return;

    // Calculate bounding box (min/max extents of model)
    const box    = new THREE.Box3().setFromObject(avatar);
    const size   = box.getSize(new THREE.Vector3());
    const center = box.getCenter(new THREE.Vector3());

    console.log('[~] Avatar bounds:', {
        height: size.y.toFixed(2),
        width : size.x.toFixed(2),
        center: `(${center.x.toFixed(1)}, ${center.y.toFixed(1)})`
    });

    // Calculate ideal camera distance
    // Based on avatar height + field of view
    const fov       = camera.fov * (Math.PI / 180);
    const avatarH   = size.y;
    const padding   = 1.25;  // 25% padding around avatar
    const camDist   = (avatarH * padding) /
                      (2 * Math.tan(fov / 2));

    // Set camera to look at avatar center
    const targetY   = center.y;

    // Smooth animation to new position
    animateCameraTo(
        camera,
        { x: 0, y: targetY, z: camDist },
        { x: 0, y: targetY, z: 0       },
        controls
    );

    // Update OrbitControls target if available
    if (controls) {
        controls.target.set(0, targetY, 0);
        controls.update();
    }

    console.log(`[✓] Camera framed: distance=${camDist.toFixed(2)}, targetY=${targetY.toFixed(2)}`);
}


// ════════════════════════════════════════════════════════
// PART 6 — CAMERA ANIMATION
// Smooth intro fly-in when avatar loads
// ════════════════════════════════════════════════════════

function animateCameraTo(camera, targetPos,
                          lookAt, controls,
                          duration = 1200) {
    /**
     * Smoothly moves camera from current position
     * to a target position over `duration` ms.
     *
     * Uses easeOutCubic for natural deceleration.
     */

    const startPos = {
        x: camera.position.x,
        y: camera.position.y,
        z: camera.position.z,
    };

    const startLook = new THREE.Vector3();
    camera.getWorldDirection(startLook);

    const startTime = performance.now();

    function easeOutCubic(t) {
        return 1 - Math.pow(1 - t, 3);
    }

    function tick() {
        const elapsed = performance.now() - startTime;
        const t       = Math.min(elapsed / duration, 1);
        const eased   = easeOutCubic(t);

        // Interpolate position
        camera.position.x =
            startPos.x + (targetPos.x - startPos.x) * eased;
        camera.position.y =
            startPos.y + (targetPos.y - startPos.y) * eased;
        camera.position.z =
            startPos.z + (targetPos.z - startPos.z) * eased;

        // Look at target
        camera.lookAt(lookAt.x, lookAt.y, lookAt.z);

        if (t < 1) {
            requestAnimationFrame(tick);
        } else {
            // Animation complete
            if (controls) controls.update();
            console.log('[✓] Camera animation complete');
        }
    }

    requestAnimationFrame(tick);
}


// ════════════════════════════════════════════════════════
// PART 7 — AVATAR MATERIAL ENHANCER
// Makes RPM avatar look better after loading
// ════════════════════════════════════════════════════════

function enhanceAvatarMaterials(avatar) {
    /**
     * Improves the visual quality of loaded RPM avatars.
     *
     * RPM avatars use MeshStandardMaterial by default.
     * We enhance:
     *   - Roughness (how matte vs shiny)
     *   - Metalness (how reflective)
     *   - Env map intensity (reflection strength)
     *   - Shadow softness
     */

    let skinMeshes    = 0;
    let clothMeshes   = 0;
    let hairMeshes    = 0;
    let otherMeshes   = 0;

    avatar.traverse(child => {
        if (!child.isMesh) return;

        const mat  = child.material;
        const name = (child.name || '').toLowerCase();

        if (!mat) return;

        // ── Detect mesh type by name ───────────────────
        const isSkin  = name.includes('skin')  ||
                        name.includes('head')  ||
                        name.includes('face')  ||
                        name.includes('body');

        const isHair  = name.includes('hair')  ||
                        name.includes('beard') ||
                        name.includes('eyebrow');

        const isCloth = name.includes('outfit') ||
                        name.includes('shirt')  ||
                        name.includes('top')    ||
                        name.includes('bottom') ||
                        name.includes('shoe')   ||
                        name.includes('jacket');

        const isEye   = name.includes('eye') ||
                        name.includes('cornea');

        // ── Apply settings per type ────────────────────
        if (isSkin) {
            // Skin: slightly rough, not metallic
            if (mat.roughness !== undefined)
                mat.roughness = 0.65;
            if (mat.metalness !== undefined)
                mat.metalness = 0.0;
            if (mat.envMapIntensity !== undefined)
                mat.envMapIntensity = 0.4;
            skinMeshes++;

        } else if (isHair) {
            // Hair: medium gloss
            if (mat.roughness !== undefined)
                mat.roughness = 0.55;
            if (mat.metalness !== undefined)
                mat.metalness = 0.05;
            if (mat.envMapIntensity !== undefined)
                mat.envMapIntensity = 0.6;
            hairMeshes++;

        } else if (isCloth) {
            // Cloth: matte fabric look
            if (mat.roughness !== undefined)
                mat.roughness = 0.78;
            if (mat.metalness !== undefined)
                mat.metalness = 0.0;
            if (mat.envMapIntensity !== undefined)
                mat.envMapIntensity = 0.3;
            clothMeshes++;

        } else if (isEye) {
            // Eyes: shiny/reflective
            if (mat.roughness !== undefined)
                mat.roughness = 0.05;
            if (mat.metalness !== undefined)
                mat.metalness = 0.1;
            if (mat.envMapIntensity !== undefined)
                mat.envMapIntensity = 1.2;

        } else {
            // Default: generic enhancement
            if (mat.envMapIntensity !== undefined)
                mat.envMapIntensity = 0.5;
            otherMeshes++;
        }

        // All meshes: enable shadows
        child.castShadow    = true;
        child.receiveShadow = true;
        mat.needsUpdate     = true;
    });

    console.log(`[✓] Materials enhanced:
        skin=${skinMeshes} hair=${hairMeshes}
        cloth=${clothMeshes} other=${otherMeshes}`);
}


// ════════════════════════════════════════════════════════
// PART 8 — SCENE EXTRAS
// Stage ring, helper lines, avatar platform
// ════════════════════════════════════════════════════════

function addSceneExtras(scene) {
    /**
     * Adds visual extras that make the scene feel
     * more like a proper avatar display room:
     *   - Glowing stage ring under avatar feet
     *   - Subtle vignette sphere
     */

    // ── Glowing stage ring ─────────────────────────────
    // A thin torus (donut) at ground level
    const ringGeo = new THREE.TorusGeometry(
        0.7,    // outer radius
        0.015,  // tube radius (thin)
        8,      // radial segments
        80      // tubular segments
    );

    const ringMat = new THREE.MeshBasicMaterial({
        color      : 0x7c3aed,
        transparent: true,
        opacity    : 0.6,
    });

    const ring = new THREE.Mesh(ringGeo, ringMat);
    ring.rotation.x = -Math.PI / 2;
    ring.position.y = -0.995;
    scene.add(ring);

    // Animate ring glow (pulsing opacity)
    ring._pulsePhase = 0;

    // ── Second outer ring (dimmer) ─────────────────────
    const ring2Geo = new THREE.TorusGeometry(
        1.1, 0.008, 8, 80
    );
    const ring2Mat = new THREE.MeshBasicMaterial({
        color      : 0x4f46e5,
        transparent: true,
        opacity    : 0.25,
    });
    const ring2 = new THREE.Mesh(ring2Geo, ring2Mat);
    ring2.rotation.x = -Math.PI / 2;
    ring2.position.y = -0.997;
    scene.add(ring2);

    console.log('[✓] Scene extras added');
    return { ring, ring2 };
}


// ════════════════════════════════════════════════════════
// PART 9 — FULL SCENE INITIALIZER
// Call this once to set up everything
// ════════════════════════════════════════════════════════

function setupFullScene(scene, camera, renderer) {
    /**
     * Master setup function — call after initScene().
     * Sets up all visual elements in the correct order.
     */

    console.log('[~] Setting up full scene...');

    // 1. Lighting
    const lights = setupStudioLighting(scene);

    // 2. Environment map
    let envMap = null;
    try {
        envMap = setupEnvironmentMap(scene, renderer);
    } catch(e) {
        console.warn('[!] Env map failed (RoomEnvironment not available):', e.message);
    }

    // 3. Ground
    const ground = setupGroundPlane(scene);

    // 4. Particles
    const particles = createParticleBac