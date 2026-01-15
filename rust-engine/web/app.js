/**
 * Geometric Cognition Engine - Web Application
 *
 * This module handles:
 * - WASM module initialization
 * - WebGL rendering of projected polytopes
 * - UI interaction and state updates
 */

// Global state
let engine = null;
let gl = null;
let program = null;
let vertexBuffer = null;
let colorBuffer = null;
let indexBuffer = null;
let running = true;
let lastTime = 0;
let frameCount = 0;
let fpsCounter = 0;
let lastFpsTime = 0;

// Camera state (for mouse interaction)
let cameraRotX = 0;
let cameraRotY = 0;
let isDragging = false;
let lastMouseX = 0;
let lastMouseY = 0;
let zoom = 3.0;

// Vertex shader source
const vsSource = `
    attribute vec3 aPosition;
    attribute vec4 aColor;
    uniform mat4 uViewProjection;
    uniform mat4 uModel;
    varying vec4 vColor;

    void main() {
        gl_Position = uViewProjection * uModel * vec4(aPosition, 1.0);
        gl_PointSize = 6.0;
        vColor = aColor;
    }
`;

// Fragment shader source
const fsSource = `
    precision mediump float;
    varying vec4 vColor;

    void main() {
        gl_FragColor = vColor;
    }
`;

/**
 * Resize canvas to match display size
 */
function resizeCanvasToDisplaySize(canvas) {
    const displayWidth = canvas.clientWidth;
    const displayHeight = canvas.clientHeight;

    // Check if canvas is not the same size
    if (canvas.width !== displayWidth || canvas.height !== displayHeight) {
        canvas.width = displayWidth;
        canvas.height = displayHeight;
        console.log('Resized canvas to:', displayWidth, 'x', displayHeight);
        return true;
    }
    return false;
}

/**
 * Initialize WebGL context and shaders
 */
function initGL(canvas) {
    // Ensure canvas has a size
    if (canvas.width === 0 || canvas.height === 0) {
        canvas.width = 800;
        canvas.height = 600;
        console.log('Set default canvas size: 800x600');
    }

    gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
    if (!gl) {
        throw new Error('WebGL not supported');
    }

    // Compile shaders
    const vs = gl.createShader(gl.VERTEX_SHADER);
    gl.shaderSource(vs, vsSource);
    gl.compileShader(vs);
    if (!gl.getShaderParameter(vs, gl.COMPILE_STATUS)) {
        const err = 'VS error: ' + gl.getShaderInfoLog(vs);
        console.error(err);
        showDebugOnPage(err);
    }

    const fs = gl.createShader(gl.FRAGMENT_SHADER);
    gl.shaderSource(fs, fsSource);
    gl.compileShader(fs);
    if (!gl.getShaderParameter(fs, gl.COMPILE_STATUS)) {
        const err = 'FS error: ' + gl.getShaderInfoLog(fs);
        console.error(err);
        showDebugOnPage(err);
    }

    // Link program
    program = gl.createProgram();
    gl.attachShader(program, vs);
    gl.attachShader(program, fs);
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
        const err = 'Link error: ' + gl.getProgramInfoLog(program);
        console.error(err);
        showDebugOnPage(err);
    }

    // Verify attribute/uniform locations
    gl.useProgram(program);
    const posLoc = gl.getAttribLocation(program, 'aPosition');
    const colorLoc = gl.getAttribLocation(program, 'aColor');
    const vpLoc = gl.getUniformLocation(program, 'uViewProjection');
    const modelLoc = gl.getUniformLocation(program, 'uModel');
    console.log('Shader locations - pos:', posLoc, 'color:', colorLoc, 'vp:', vpLoc, 'model:', modelLoc);

    // Create buffers
    vertexBuffer = gl.createBuffer();
    colorBuffer = gl.createBuffer();
    indexBuffer = gl.createBuffer();

    // Setup GL state
    gl.disable(gl.DEPTH_TEST);  // Disabled temporarily for debugging
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.clearColor(0.02, 0.02, 0.05, 1.0);

    return gl;
}

/**
 * Create view-projection matrix
 */
function createViewProjection(width, height) {
    const aspect = width / height;
    const fov = Math.PI / 4;
    const near = 0.1;
    const far = 100.0;

    // Perspective projection matrix (column-major for WebGL)
    const f = 1.0 / Math.tan(fov / 2);
    const rangeInv = 1.0 / (near - far);

    // Column-major order: each group of 4 is one column
    const proj = [
        f / aspect, 0, 0, 0,                          // Column 0
        0, f, 0, 0,                                   // Column 1
        0, 0, (far + near) * rangeInv, -1,            // Column 2 (note: -1 at bottom for w=-z)
        0, 0, 2 * far * near * rangeInv, 0            // Column 3
    ];

    // View matrix (camera looking at origin)
    const cx = Math.cos(cameraRotX);
    const sx = Math.sin(cameraRotX);
    const cy = Math.cos(cameraRotY);
    const sy = Math.sin(cameraRotY);

    const eyeX = zoom * sy * cx;
    const eyeY = zoom * sx;
    const eyeZ = zoom * cy * cx;

    // Simple look-at
    const view = lookAt([eyeX, eyeY, eyeZ], [0, 0, 0], [0, 1, 0]);

    // For column-major matrices, multiplication order is reversed
    // We want: projection * view * model * vertex
    // So we compute: proj * view
    return multiplyMatricesColumnMajor(proj, view);
}

/**
 * Look-at matrix (column-major for WebGL)
 */
function lookAt(eye, target, up) {
    const zAxis = normalize(subtract(eye, target));
    const xAxis = normalize(cross(up, zAxis));
    const yAxis = cross(zAxis, xAxis);

    // Column-major order for WebGL
    return [
        xAxis[0], yAxis[0], zAxis[0], 0,              // Column 0
        xAxis[1], yAxis[1], zAxis[1], 0,              // Column 1
        xAxis[2], yAxis[2], zAxis[2], 0,              // Column 2
        -dot(xAxis, eye), -dot(yAxis, eye), -dot(zAxis, eye), 1  // Column 3
    ];
}

// Vector math helpers
function subtract(a, b) { return [a[0]-b[0], a[1]-b[1], a[2]-b[2]]; }
function cross(a, b) { return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]; }
function dot(a, b) { return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]; }
function normalize(v) { const l = Math.sqrt(dot(v, v)); return [v[0]/l, v[1]/l, v[2]/l]; }

// Matrix multiplication for column-major matrices (WebGL style)
function multiplyMatricesColumnMajor(a, b) {
    const result = new Array(16).fill(0);
    for (let col = 0; col < 4; col++) {
        for (let row = 0; row < 4; row++) {
            let sum = 0;
            for (let k = 0; k < 4; k++) {
                // a[col][k] * b[k][row] in column-major indexing
                sum += a[k * 4 + row] * b[col * 4 + k];
            }
            result[col * 4 + row] = sum;
        }
    }
    return result;
}

/**
 * Draw a test pattern to verify WebGL is working
 */
function drawTestPattern() {
    if (!gl || !program) return;

    // Simple triangle in NDC space
    const testPositions = new Float32Array([
        0.0,  0.5, 0.0,   // top
       -0.5, -0.5, 0.0,   // bottom left
        0.5, -0.5, 0.0    // bottom right
    ]);

    const testColors = new Float32Array([
        1.0, 0.0, 0.0, 1.0,  // red
        0.0, 1.0, 0.0, 1.0,  // green
        0.0, 0.0, 1.0, 1.0   // blue
    ]);

    gl.useProgram(program);

    // Identity matrices for test
    const identity = new Float32Array([
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1
    ]);

    gl.uniformMatrix4fv(gl.getUniformLocation(program, 'uViewProjection'), false, identity);
    gl.uniformMatrix4fv(gl.getUniformLocation(program, 'uModel'), false, identity);

    // Position
    gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, testPositions, gl.STATIC_DRAW);
    const posLoc = gl.getAttribLocation(program, 'aPosition');
    gl.enableVertexAttribArray(posLoc);
    gl.vertexAttribPointer(posLoc, 3, gl.FLOAT, false, 0, 0);

    // Color
    gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, testColors, gl.STATIC_DRAW);
    const colorLoc = gl.getAttribLocation(program, 'aColor');
    gl.enableVertexAttribArray(colorLoc);
    gl.vertexAttribPointer(colorLoc, 4, gl.FLOAT, false, 0, 0);

    gl.drawArrays(gl.TRIANGLES, 0, 3);
    console.log('Drew test triangle');
}

/**
 * Render the current frame
 */
let renderDebugLogged = false;
function render() {
    if (!engine || !gl) return;

    try {
        const canvas = gl.canvas;
        gl.viewport(0, 0, canvas.width, canvas.height);
        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

        // Get vertex and edge data from engine
        const vertices = engine.get_vertices();
        const edges = engine.get_edges();

        if (!renderDebugLogged) {
            showDebugOnPage('Render: ' + (vertices.length/7) + ' verts, ' + (edges.length/2) + ' edges');
            if (vertices.length > 0) {
                const v0 = 'V0:[' + vertices[0].toFixed(2) + ',' + vertices[1].toFixed(2) + ',' + vertices[2].toFixed(2) + ']';
                showDebugOnPage(v0);
                // Transform first vertex
                const vp = createViewProjection(canvas.width, canvas.height);
                let clip = [0,0,0,0];
                for (let r = 0; r < 4; r++) {
                    clip[r] = vp[r] * vertices[0] + vp[4+r] * vertices[1] + vp[8+r] * vertices[2] + vp[12+r];
                }
                const ndc = [clip[0]/clip[3], clip[1]/clip[3], clip[2]/clip[3]];
                showDebugOnPage('NDC:[' + ndc[0].toFixed(2) + ',' + ndc[1].toFixed(2) + ',' + ndc[2].toFixed(2) + '] w=' + clip[3].toFixed(2));
            }
        }

        if (vertices.length === 0) {
            console.warn('No vertices to render - drawing test pattern');
            // Draw a test triangle to verify WebGL works
            drawTestPattern();
            return;
        }

        // Parse vertex data (x,y,z,r,g,b,a per vertex)
        const positions = [];
        const colors = [];
        const vertexCount = vertices.length / 7;

        for (let i = 0; i < vertexCount; i++) {
            const base = i * 7;
            positions.push(vertices[base], vertices[base + 1], vertices[base + 2]);
            colors.push(vertices[base + 3], vertices[base + 4], vertices[base + 5], vertices[base + 6]);
        }

        // Upload vertex data
        gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(positions), gl.DYNAMIC_DRAW);

        gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(colors), gl.DYNAMIC_DRAW);

        // Use program
        gl.useProgram(program);

        // Set uniforms
        const viewProj = createViewProjection(canvas.width, canvas.height);
        const vpLoc = gl.getUniformLocation(program, 'uViewProjection');
        gl.uniformMatrix4fv(vpLoc, false, new Float32Array(viewProj));

        const modelLoc = gl.getUniformLocation(program, 'uModel');
        gl.uniformMatrix4fv(modelLoc, false, new Float32Array([
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1
        ]));

        // Position attribute
        const posLoc = gl.getAttribLocation(program, 'aPosition');
        gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer);
        gl.enableVertexAttribArray(posLoc);
        gl.vertexAttribPointer(posLoc, 3, gl.FLOAT, false, 0, 0);

        // Color attribute
        const colorLoc = gl.getAttribLocation(program, 'aColor');
        gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);
        gl.enableVertexAttribArray(colorLoc);
        gl.vertexAttribPointer(colorLoc, 4, gl.FLOAT, false, 0, 0);

        // Draw edges
        if (edges.length > 0) {
            gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indexBuffer);
            gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(edges), gl.DYNAMIC_DRAW);
            gl.drawElements(gl.LINES, edges.length, gl.UNSIGNED_SHORT, 0);
        }

        // Draw points
        gl.drawArrays(gl.POINTS, 0, vertexCount);

        if (!renderDebugLogged) {
            const error = gl.getError();
            if (error !== gl.NO_ERROR) {
                console.error('WebGL error:', error);
            } else {
                console.log('WebGL: No errors');
            }
            console.log('Drew', vertexCount, 'vertices and', edges.length, 'edge indices');

            // Log transformed position of first vertex
            const vp = createViewProjection(canvas.width, canvas.height);
            const pos = [positions[0], positions[1], positions[2], 1];
            let result = [0, 0, 0, 0];
            for (let i = 0; i < 4; i++) {
                for (let j = 0; j < 4; j++) {
                    result[i] += vp[i * 4 + j] * pos[j];
                }
            }
            console.log('First vertex after viewProj:', result);
            console.log('NDC:', result[0]/result[3], result[1]/result[3], result[2]/result[3]);
            console.log('=== END DEBUG ===');
            renderDebugLogged = true;
        }

    } catch (e) {
        console.error('Render error:', e);
    }
}

/**
 * Update UI with engine state
 */
function updateUI() {
    if (!engine) return;

    // Update Betti numbers
    try {
        const betti = JSON.parse(engine.get_betti_numbers());
        document.getElementById('betti-0').textContent = betti.b0;
        document.getElementById('betti-1').textContent = betti.b1;
        document.getElementById('betti-2').textContent = betti.b2;
        document.getElementById('euler').textContent = betti.euler;
    } catch (e) {}

    // Update Trinity state
    try {
        const trinity = JSON.parse(engine.get_trinity_state());
        document.getElementById('alpha-level').textContent = trinity.alpha.level.toFixed(2);
        document.getElementById('beta-level').textContent = trinity.beta.level.toFixed(2);
        document.getElementById('gamma-level').textContent = trinity.gamma.level.toFixed(2);
        document.getElementById('tension').textContent = trinity.tension.toFixed(3);
        document.getElementById('coherence').textContent = trinity.coherence.toFixed(3);
    } catch (e) {}

    // Update patterns
    try {
        const patterns = JSON.parse(engine.get_patterns());
        const container = document.getElementById('patterns');
        if (patterns.length > 0) {
            container.innerHTML = patterns.map(p => `<span class="pattern-tag">${p}</span>`).join('');
        } else {
            container.innerHTML = '<span class="pattern-tag">None</span>';
        }
    } catch (e) {}

    // Update status
    document.getElementById('frame-count').textContent = engine.frame_count();
    document.getElementById('current-mode').textContent = engine.get_mode();
    document.getElementById('synthesis-status').textContent =
        engine.is_synthesis_detected() ? 'Synthesis: ACTIVE' : 'Synthesis: Inactive';
}

/**
 * Animation loop
 */
function animate(time) {
    if (!lastTime) lastTime = time;
    const deltaTime = (time - lastTime) / 1000;
    lastTime = time;

    if (running && engine) {
        // Get rotation values from sliders
        const xy = parseFloat(document.getElementById('rot-xy').value);
        const xz = parseFloat(document.getElementById('rot-xz').value);
        const xw = parseFloat(document.getElementById('rot-xw').value);
        const yz = parseFloat(document.getElementById('rot-yz').value);
        const yw = parseFloat(document.getElementById('rot-yw').value);
        const zw = parseFloat(document.getElementById('rot-zw').value);

        engine.set_rotation_speeds(xy, xz, xw, yz, yw, zw);
        engine.update(deltaTime);
    }

    render();

    // Update FPS
    frameCount++;
    if (time - lastFpsTime >= 1000) {
        document.getElementById('fps').textContent = frameCount;
        frameCount = 0;
        lastFpsTime = time;
    }

    // Update UI every 10 frames
    if (engine && engine.frame_count() % 10 === 0) {
        updateUI();
    }

    requestAnimationFrame(animate);
}

/**
 * Setup UI event handlers
 */
function setupUI() {
    // Mode selector
    document.getElementById('mode-select').addEventListener('change', (e) => {
        if (engine) {
            engine.set_mode(e.target.value);
        }
    });

    // Pause button
    document.getElementById('btn-pause').addEventListener('click', () => {
        running = !running;
        document.getElementById('btn-pause').textContent = running ? 'Pause' : 'Resume';
    });

    // Reset button
    document.getElementById('btn-reset').addEventListener('click', () => {
        if (engine) {
            // Reset rotation sliders
            document.getElementById('rot-xy').value = 0.3;
            document.getElementById('rot-xz').value = 0.2;
            document.getElementById('rot-xw').value = 0.1;
            document.getElementById('rot-yz').value = 0.15;
            document.getElementById('rot-yw').value = 0.25;
            document.getElementById('rot-zw').value = 0.05;
            updateSliderLabels();
        }
    });

    // Rotation sliders
    const sliders = ['xy', 'xz', 'xw', 'yz', 'yw', 'zw'];
    sliders.forEach(plane => {
        const slider = document.getElementById(`rot-${plane}`);
        slider.addEventListener('input', () => updateSliderLabels());
    });

    // Canvas mouse interaction
    const canvas = document.getElementById('gce-canvas');

    canvas.addEventListener('mousedown', (e) => {
        isDragging = true;
        lastMouseX = e.clientX;
        lastMouseY = e.clientY;
    });

    canvas.addEventListener('mousemove', (e) => {
        if (isDragging) {
            const dx = e.clientX - lastMouseX;
            const dy = e.clientY - lastMouseY;
            cameraRotY += dx * 0.01;
            cameraRotX += dy * 0.01;
            cameraRotX = Math.max(-Math.PI/2 + 0.1, Math.min(Math.PI/2 - 0.1, cameraRotX));
            lastMouseX = e.clientX;
            lastMouseY = e.clientY;
        }
    });

    canvas.addEventListener('mouseup', () => isDragging = false);
    canvas.addEventListener('mouseleave', () => isDragging = false);

    canvas.addEventListener('wheel', (e) => {
        e.preventDefault();
        zoom += e.deltaY * 0.01;
        zoom = Math.max(1, Math.min(10, zoom));
    });
}

function updateSliderLabels() {
    const sliders = ['xy', 'xz', 'xw', 'yz', 'yw', 'zw'];
    sliders.forEach(plane => {
        const value = document.getElementById(`rot-${plane}`).value;
        document.getElementById(`${plane}-val`).textContent = value;
    });
}

/**
 * Show debug info on page (for mobile debugging)
 */
function showDebugOnPage(message) {
    let debugDiv = document.getElementById('debug-output');
    if (!debugDiv) {
        debugDiv = document.createElement('div');
        debugDiv.id = 'debug-output';
        debugDiv.style.cssText = 'position:fixed;bottom:0;left:0;right:0;background:rgba(0,0,0,0.9);color:#0f0;font-family:monospace;font-size:10px;max-height:150px;overflow-y:auto;padding:5px;z-index:9999;';
        document.body.appendChild(debugDiv);
    }
    const line = document.createElement('div');
    line.textContent = message;
    debugDiv.appendChild(line);
    debugDiv.scrollTop = debugDiv.scrollHeight;
}

/**
 * Main initialization
 */
async function main() {
    const loadingEl = document.getElementById('loading');
    const loadingText = loadingEl.querySelector('.loading-text');

    try {
        showDebugOnPage('Starting initialization...');
        loadingText.textContent = 'Initializing WebGL...';

        // Initialize WebGL
        const canvas = document.getElementById('gce-canvas');
        initGL(canvas);
        showDebugOnPage('WebGL OK: ' + canvas.width + 'x' + canvas.height);

        loadingText.textContent = 'Loading WASM module...';

        // Load WASM module
        const wasm = await import('./pkg/geometric_cognition.js');
        await wasm.default();
        showDebugOnPage('WASM loaded');

        showDebugOnPage('Version: ' + wasm.get_version());

        loadingText.textContent = 'Creating engine...';

        // Create engine instance
        engine = new wasm.WebEngine('gce-canvas');
        showDebugOnPage('Engine created');

        // Test vertex data
        const testVertices = engine.get_vertices();
        const vertCount = testVertices.length / 7;
        showDebugOnPage('Vertices: ' + vertCount + ' (array len: ' + testVertices.length + ')');

        const testEdges = engine.get_edges();
        showDebugOnPage('Edges: ' + (testEdges.length / 2));

        if (testVertices.length > 0) {
            showDebugOnPage('V0: [' + testVertices[0].toFixed(2) + ', ' + testVertices[1].toFixed(2) + ', ' + testVertices[2].toFixed(2) + ']');
        }

        // Setup UI
        setupUI();
        updateSliderLabels();

        // Hide loading screen
        loadingEl.classList.add('hidden');
        showDebugOnPage('Starting render loop...');

        // Start animation loop
        requestAnimationFrame(animate);

    } catch (error) {
        console.error('Initialization error:', error);
        console.error('Stack:', error.stack);
        loadingText.textContent = `Error: ${error.message}`;
        loadingText.style.color = '#ff6666';
    }
}

// Start
main();
