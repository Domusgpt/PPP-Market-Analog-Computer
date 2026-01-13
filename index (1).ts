/**
 * @clear-seas/cpe Topology Module
 *
 * Re-exports topological structures for the Chronomorphic Polytopal Engine.
 *
 * Includes:
 * - Lattice24: The 24-cell (Orthocognitum) for CPE state space
 * - Lattice600: The 600-cell (Interaction Manifold) for three-body dynamics
 * - E8H4Folding: Moxness matrix and E8→H4 projection
 * - TrinityDecomposition: Standard Model particle mapping
 */

export * from './Lattice24.js';
export * from './Lattice600.js';
export * from './E8H4Folding.js';
export * from './TrinityDecomposition.js';
