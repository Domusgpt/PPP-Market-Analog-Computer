// Test the matrix math locally

// Vector math helpers
function subtract(a, b) { return [a[0]-b[0], a[1]-b[1], a[2]-b[2]]; }
function cross(a, b) { return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]; }
function dot(a, b) { return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]; }
function normalize(v) { const l = Math.sqrt(dot(v, v)); return [v[0]/l, v[1]/l, v[2]/l]; }

function lookAt(eye, target, up) {
    const zAxis = normalize(subtract(eye, target));
    const xAxis = normalize(cross(up, zAxis));
    const yAxis = cross(zAxis, xAxis);

    return [
        xAxis[0], yAxis[0], zAxis[0], 0,
        xAxis[1], yAxis[1], zAxis[1], 0,
        xAxis[2], yAxis[2], zAxis[2], 0,
        -dot(xAxis, eye), -dot(yAxis, eye), -dot(zAxis, eye), 1
    ];
}

function multiplyMatricesColumnMajor(a, b) {
    const result = new Array(16).fill(0);
    for (let col = 0; col < 4; col++) {
        for (let row = 0; row < 4; row++) {
            let sum = 0;
            for (let k = 0; k < 4; k++) {
                sum += a[k * 4 + row] * b[col * 4 + k];
            }
            result[col * 4 + row] = sum;
        }
    }
    return result;
}

function createViewProjection(width, height) {
    const aspect = width / height;
    const fov = Math.PI / 4;
    const near = 0.1;
    const far = 100.0;
    const zoom = 3.0;
    const cameraRotX = 0;
    const cameraRotY = 0;

    const f = 1.0 / Math.tan(fov / 2);
    const rangeInv = 1.0 / (near - far);

    const proj = [
        f / aspect, 0, 0, 0,
        0, f, 0, 0,
        0, 0, (far + near) * rangeInv, -1,
        0, 0, 2 * far * near * rangeInv, 0
    ];

    const cx = Math.cos(cameraRotX);
    const sx = Math.sin(cameraRotX);
    const cy = Math.cos(cameraRotY);
    const sy = Math.sin(cameraRotY);

    const eyeX = zoom * sy * cx;
    const eyeY = zoom * sx;
    const eyeZ = zoom * cy * cx;

    const view = lookAt([eyeX, eyeY, eyeZ], [0, 0, 0], [0, 1, 0]);

    console.log("Eye position:", [eyeX, eyeY, eyeZ]);
    console.log("View matrix:", view);
    console.log("Proj matrix:", proj);

    return multiplyMatricesColumnMajor(proj, view);
}

// Transform a vertex through the viewProj matrix (column-major)
function transformVertex(viewProj, vertex) {
    const v = [vertex[0], vertex[1], vertex[2], 1];
    const result = [0, 0, 0, 0];

    for (let row = 0; row < 4; row++) {
        for (let col = 0; col < 4; col++) {
            result[row] += viewProj[col * 4 + row] * v[col];
        }
    }

    return result;
}

// Test with sample 24-cell vertices
const testVertices = [
    [-1, -1, 0],   // First vertex from Rust
    [1, 1, 0],
    [0, 0, 0.75],  // Last vertex from Rust
    [0.5, 0.5, 0.5],
];

console.log("=== Matrix Test ===\n");

const viewProj = createViewProjection(800, 600);
console.log("\nViewProj matrix:", viewProj);

console.log("\n=== Vertex Transformations ===\n");

for (const v of testVertices) {
    const transformed = transformVertex(viewProj, v);
    const w = transformed[3];
    const ndc = [transformed[0]/w, transformed[1]/w, transformed[2]/w];

    console.log(`Vertex ${JSON.stringify(v)}:`);
    console.log(`  Clip space: [${transformed.map(x => x.toFixed(3)).join(', ')}]`);
    console.log(`  NDC: [${ndc.map(x => x.toFixed(3)).join(', ')}]`);
    console.log(`  Visible: ${Math.abs(ndc[0]) <= 1 && Math.abs(ndc[1]) <= 1 && ndc[2] >= -1 && ndc[2] <= 1}`);
    console.log();
}

// Test origin point
console.log("\n=== Origin Test ===");
const origin = [0, 0, 0];
const transformedOrigin = transformVertex(viewProj, origin);
const wOrigin = transformedOrigin[3];
const ndcOrigin = [transformedOrigin[0]/wOrigin, transformedOrigin[1]/wOrigin, transformedOrigin[2]/wOrigin];
console.log("Origin clip space:", transformedOrigin);
console.log("Origin NDC:", ndcOrigin);
