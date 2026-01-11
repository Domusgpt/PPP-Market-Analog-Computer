/**
 * Simplex5 - 5-Cell (4-simplex) topology provider.
 *
 * The 5-cell is the simplest 4D polytope, representing maximal association.
 */

import {
    Vector4D,
    ConvexityResult,
    TopologyProvider
} from '../../types/index.js';


const SIMPLEX_VERTICES: Vector4D[] = [
    [1, 1, 1, 1],
    [1, -1, -1, -1],
    [-1, 1, -1, -1],
    [-1, -1, 1, -1],
    [-1, -1, -1, 1]
];

const SIMPLEX_RADIUS = Math.sqrt(
    SIMPLEX_VERTICES[0][0] ** 2 +
    SIMPLEX_VERTICES[0][1] ** 2 +
    SIMPLEX_VERTICES[0][2] ** 2 +
    SIMPLEX_VERTICES[0][3] ** 2
);

function solveLinearSystem4x4(matrix: number[][], vector: number[]): number[] {
    const a = matrix.map((row, rowIndex) => [...row, vector[rowIndex]]);
    const size = 4;

    for (let col = 0; col < size; col++) {
        let pivotRow = col;
        for (let row = col + 1; row < size; row++) {
            if (Math.abs(a[row][col]) > Math.abs(a[pivotRow][col])) {
                pivotRow = row;
            }
        }

        if (pivotRow !== col) {
            const temp = a[col];
            a[col] = a[pivotRow];
            a[pivotRow] = temp;
        }

        const pivot = a[col][col];
        if (Math.abs(pivot) < 1e-10) {
            return [0, 0, 0, 0];
        }

        for (let c = col; c <= size; c++) {
            a[col][c] /= pivot;
        }

        for (let row = 0; row < size; row++) {
            if (row === col) continue;
            const factor = a[row][col];
            for (let c = col; c <= size; c++) {
                a[row][c] -= factor * a[col][c];
            }
        }
    }

    return a.map(row => row[size]);
}

function computeBarycentric(point: Vector4D): number[] {
    const base = SIMPLEX_VERTICES[0];
    const columns = SIMPLEX_VERTICES.slice(1).map(vertex => [
        vertex[0] - base[0],
        vertex[1] - base[1],
        vertex[2] - base[2],
        vertex[3] - base[3]
    ]);

    const matrix = [
        [columns[0][0], columns[1][0], columns[2][0], columns[3][0]],
        [columns[0][1], columns[1][1], columns[2][1], columns[3][1]],
        [columns[0][2], columns[1][2], columns[2][2], columns[3][2]],
        [columns[0][3], columns[1][3], columns[2][3], columns[3][3]]
    ];

    const rhs = [
        point[0] - base[0],
        point[1] - base[1],
        point[2] - base[2],
        point[3] - base[3]
    ];

    const weights = solveLinearSystem4x4(matrix, rhs);
    const w0 = 1 - weights.reduce((sum, value) => sum + value, 0);
    return [w0, ...weights];
}

function findNearestVertex(point: Vector4D): { index: number; distance: number } {
    let nearestIndex = 0;
    let nearestDistance = Number.POSITIVE_INFINITY;

    SIMPLEX_VERTICES.forEach((vertex, index) => {
        const dist = Math.sqrt(
            (point[0] - vertex[0]) ** 2 +
            (point[1] - vertex[1]) ** 2 +
            (point[2] - vertex[2]) ** 2 +
            (point[3] - vertex[3]) ** 2
        );
        if (dist < nearestDistance) {
            nearestDistance = dist;
            nearestIndex = index;
        }
    });

    return { index: nearestIndex, distance: nearestDistance };
}

export class Simplex5 implements TopologyProvider {
    readonly name = 'Simplex5';
    readonly vertices = SIMPLEX_VERTICES;
    readonly neighbors = SIMPLEX_VERTICES.map((_, index) =>
        SIMPLEX_VERTICES.map((__, neighborIndex) => neighborIndex).filter(id => id !== index)
    );
    readonly circumradius = SIMPLEX_RADIUS;

    checkConvexity(point: Vector4D): ConvexityResult {
        const barycentric = computeBarycentric(point);
        const isValid = barycentric.every(weight => weight >= -1e-6);
        const nearest = findNearestVertex(point);
        const coherence = Math.max(0, 1 - nearest.distance / SIMPLEX_RADIUS);

        return {
            isValid,
            coherence,
            nearestVertex: nearest.index,
            distance: nearest.distance,
            centroid: this.vertices[nearest.index],
            activeVertices: [nearest.index]
        };
    }

    computeCoherence(point: Vector4D): number {
        return this.checkConvexity(point).coherence;
    }
}

export const simplex5 = new Simplex5();
