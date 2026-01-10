/**
 * Hypercube8 - 8-Cell (tesseract) topology provider.
 *
 * The 8-cell represents discrimination and orthogonal opposites.
 */

import {
    Vector4D,
    ConvexityResult,
    TopologyProvider
} from '../../types/index.js';


const HYPERCUBE_VERTICES: Vector4D[] = [];

for (const x of [-1, 1]) {
    for (const y of [-1, 1]) {
        for (const z of [-1, 1]) {
            for (const w of [-1, 1]) {
                HYPERCUBE_VERTICES.push([x, y, z, w]);
            }
        }
    }
}

const HYPERCUBE_RADIUS = Math.sqrt(
    HYPERCUBE_VERTICES[0][0] ** 2 +
    HYPERCUBE_VERTICES[0][1] ** 2 +
    HYPERCUBE_VERTICES[0][2] ** 2 +
    HYPERCUBE_VERTICES[0][3] ** 2
);

function findNearestVertex(point: Vector4D): { index: number; distance: number } {
    let nearestIndex = 0;
    let nearestDistance = Number.POSITIVE_INFINITY;

    HYPERCUBE_VERTICES.forEach((vertex, index) => {
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

export class Hypercube8 implements TopologyProvider {
    readonly name = 'Hypercube8';
    readonly vertices = HYPERCUBE_VERTICES;
    readonly neighbors = HYPERCUBE_VERTICES.map((vertex, index) => {
        const neighbors: number[] = [];
        HYPERCUBE_VERTICES.forEach((candidate, candidateIndex) => {
            if (index === candidateIndex) return;
            const diffCount =
                (vertex[0] !== candidate[0] ? 1 : 0) +
                (vertex[1] !== candidate[1] ? 1 : 0) +
                (vertex[2] !== candidate[2] ? 1 : 0) +
                (vertex[3] !== candidate[3] ? 1 : 0);
            if (diffCount === 1) {
                neighbors.push(candidateIndex);
            }
        });
        return neighbors;
    });
    readonly circumradius = HYPERCUBE_RADIUS;

    checkConvexity(point: Vector4D): ConvexityResult {
        const isValid = point.every(component => Math.abs(component) <= 1 + 1e-6);
        const nearest = findNearestVertex(point);
        const coherence = Math.max(0, 1 - nearest.distance / HYPERCUBE_RADIUS);

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

export const hypercube8 = new Hypercube8();
