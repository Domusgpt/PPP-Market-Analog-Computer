const DEFAULT_RADIUS = 2.2;

const create4dRotationMatrix = (angles) => {
    const planes = [
        ['x', 'y', angles.xy],
        ['x', 'z', angles.xz],
        ['x', 'w', angles.xw],
        ['y', 'z', angles.yz],
        ['y', 'w', angles.yw],
        ['z', 'w', angles.zw]
    ];
    const axes = ['x', 'y', 'z', 'w'];
    let matrix = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ];

    const multiply = (a, b) => a.map((row, i) => row.map((_, j) => row.reduce((sum, value, k) => sum + value * b[k][j], 0)));

    planes.forEach(([axisA, axisB, angle]) => {
        const cos = Math.cos(angle);
        const sin = Math.sin(angle);
        const rot = [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ];
        const indexA = axes.indexOf(axisA);
        const indexB = axes.indexOf(axisB);
        rot[indexA][indexA] = cos;
        rot[indexA][indexB] = -sin;
        rot[indexB][indexA] = sin;
        rot[indexB][indexB] = cos;
        matrix = multiply(rot, matrix);
    });

    return matrix;
};

const applyMatrix = (matrix, vector) => matrix.map((row) => row.reduce((sum, value, idx) => sum + value * vector[idx], 0));

const buildTesseract = (scale = 1) => {
    const vertices = [];
    for (let i = 0; i < 16; i += 1) {
        const bits = i.toString(2).padStart(4, '0').split('').map(Number);
        vertices.push(bits.map((bit) => (bit === 0 ? -scale : scale)));
    }
    const edges = [];
    vertices.forEach((vertex, i) => {
        vertices.forEach((other, j) => {
            if (j <= i) {
                return;
            }
            const diff = vertex.reduce((sum, value, idx) => sum + (value !== other[idx] ? 1 : 0), 0);
            if (diff === 1) {
                edges.push([i, j]);
            }
        });
    });
    return { vertices, edges };
};

const createAdjacency = (edges, vertexCount) => {
    const adjacency = Array.from({ length: vertexCount }, () => []);
    edges.forEach(([a, b]) => {
        adjacency[a].push(b);
        adjacency[b].push(a);
    });
    return adjacency;
};

export class VisualCognitionEngine {
    constructor({ radius = DEFAULT_RADIUS } = {}) {
        this.radius = radius;
        this.angles = { xy: 0, xz: 0, xw: 0, yz: 0, yw: 0, zw: 0 };
        this.geometry = buildTesseract(1);
        this.adjacency = createAdjacency(this.geometry.edges, this.geometry.vertices.length);
        this.stateVector = [0, 0, 0, 0];
        this.telemetry = {
            frame: 0,
            rotation: { ...this.angles },
            projection: { radius: this.radius, depthScale: 0 },
            geometry: { vertices: this.geometry.vertices.length, edges: this.geometry.edges.length, cells: 8 },
            telemetry: { opacityMean: 0, lineWidthMean: 0, overlapDensity: 0 }
        };
    }

    updateStateFromChannels(channels = []) {
        const padded = [0, 0, 0, 0];
        for (let i = 0; i < 4; i += 1) {
            const value = typeof channels[i] === 'number' ? channels[i] : 0.5;
            padded[i] = (value - 0.5) * 2;
        }
        this.stateVector = padded;
    }

    updateRotation(deltaAngles) {
        Object.keys(this.angles).forEach((key) => {
            if (typeof deltaAngles[key] === 'number') {
                this.angles[key] += deltaAngles[key];
            }
        });
    }

    projectVertices() {
        const matrix = create4dRotationMatrix(this.angles);
        const projected = this.geometry.vertices.map((vertex) => {
            const rotated = applyMatrix(matrix, vertex.map((value, idx) => value + this.stateVector[idx]));
            const scale = this.radius / (this.radius - rotated[3]);
            return {
                x: rotated[0] * scale,
                y: rotated[1] * scale,
                z: rotated[2] * scale,
                w: rotated[3]
            };
        });
        return projected;
    }

    renderToCanvas(canvas, context, options = {}) {
        const { width, height } = canvas;
        context.clearRect(0, 0, width, height);
        context.save();
        context.translate(width / 2, height / 2);
        const projected = this.projectVertices();
        const depthValues = projected.map((point) => point.z);
        const depthScale = depthValues.reduce((sum, value) => sum + Math.abs(value), 0) / depthValues.length;
        const scale = Math.min(width, height) * 0.22;
        context.lineWidth = 1.5;
        context.strokeStyle = 'rgba(120, 200, 255, 0.75)';
        this.geometry.edges.forEach(([a, b]) => {
            const p1 = projected[a];
            const p2 = projected[b];
            const z1 = 1 / (1 + Math.max(-2, Math.min(2, p1.z)));
            const z2 = 1 / (1 + Math.max(-2, Math.min(2, p2.z)));
            context.lineWidth = 1 + (z1 + z2) * 0.6;
            context.beginPath();
            context.moveTo(p1.x * scale, p1.y * scale);
            context.lineTo(p2.x * scale, p2.y * scale);
            context.stroke();
        });
        context.restore();

        this.telemetry = {
            ...this.telemetry,
            frame: this.telemetry.frame + 1,
            rotation: { ...this.angles },
            projection: { radius: this.radius, depthScale: Number(depthScale.toFixed(3)) },
            telemetry: {
                opacityMean: Number((0.35 + depthScale * 0.08).toFixed(3)),
                lineWidthMean: Number((1.4 + depthScale * 0.2).toFixed(3)),
                overlapDensity: Number((0.18 + Math.sin(this.angles.xw) * 0.05).toFixed(3))
            },
            ...options
        };

        return { projected, telemetry: this.telemetry };
    }

    exportTopology() {
        return {
            vertices: this.geometry.vertices,
            edges: this.geometry.edges,
            adjacency: this.adjacency
        };
    }
}
