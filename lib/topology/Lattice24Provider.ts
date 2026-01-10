/**
 * Lattice24Provider - adapts the 24-Cell lattice to the TopologyProvider interface.
 */

import { TopologyProvider, Vector4D } from '../../types/index.js';
import { getDefaultLattice } from './Lattice24.js';

export class Lattice24Provider implements TopologyProvider {
    private readonly _lattice = getDefaultLattice();

    readonly name = 'Lattice24';
    readonly vertices = this._lattice.vertices.map(vertex => vertex.coordinates);
    readonly neighbors = this._lattice.vertices.map(vertex => vertex.neighbors);
    readonly circumradius = this._lattice.circumradius;

    checkConvexity(point: Vector4D, kNearest?: number) {
        return this._lattice.checkConvexity(point, kNearest);
    }

    computeCoherence(point: Vector4D, kNearest?: number) {
        return this._lattice.checkConvexity(point, kNearest).coherence;
    }
}

export const lattice24Provider = new Lattice24Provider();
