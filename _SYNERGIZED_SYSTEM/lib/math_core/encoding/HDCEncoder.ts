/**
 * Hyperdimensional Computing (HDC) Encoder
 *
 * @package @clear-seas/cpe
 * @version 2.0.0
 * @license MIT
 *
 * Implements hyperdimensional computing for the Chronomorphic Polytopal Engine.
 * HDC uses high-dimensional vectors (hypervectors) for robust, noise-tolerant
 * representation and computation.
 *
 * Key Operations:
 * - Binding (⊗): Combines two concepts, like subject-verb binding
 * - Bundling (+): Creates a composite representation
 * - Permutation (π): Encodes sequence/order information
 *
 * Integration with PPP:
 * - HDC vectors can be projected onto polytope vertices
 * - Similarity measured via cosine distance
 * - Robust to noise and partial information
 *
 * Ported from CPE repo (Domusgpt/-Chronomorphic-Polytopal-Engine)
 * with import paths adapted for PPP math_core layout.
 */

import { Vector4D, MATH_CONSTANTS } from '../geometric_algebra/types.js';
import { Lattice24, getDefaultLattice } from '../geometric_algebra/Lattice24.js';

// =============================================================================
// TYPES
// =============================================================================

/**
 * A hypervector is a high-dimensional binary or bipolar vector.
 * We use Float32Array for efficient operations.
 */
export type Hypervector = Float32Array;

/**
 * HDC configuration options.
 */
export interface HDCConfig {
  dimensions: number;      // Number of dimensions (typically 1000-10000)
  bipolar: boolean;        // Use bipolar (-1, +1) vs binary (0, 1)
  seed?: number;           // Random seed for reproducibility
  memoryLimit?: number;    // Maximum number of items in memory (LRU cache)
}

/**
 * Default configuration.
 */
const DEFAULT_HDC_CONFIG: HDCConfig = {
  dimensions: 4096,
  bipolar: true,
  memoryLimit: 10000
};

// =============================================================================
// RANDOM NUMBER GENERATOR
// =============================================================================

/**
 * Seeded random number generator for reproducibility.
 */
class SeededRNG {
  private seed: number;

  constructor(seed: number = Date.now()) {
    this.seed = seed;
  }

  next(): number {
    this.seed = (this.seed * 1103515245 + 12345) & 0x7fffffff;
    return this.seed / 0x7fffffff;
  }

  nextBipolar(): number {
    return this.next() < 0.5 ? -1 : 1;
  }

  nextBinary(): number {
    return this.next() < 0.5 ? 0 : 1;
  }
}

// =============================================================================
// HYPERVECTOR OPERATIONS
// =============================================================================

/**
 * Create a random hypervector.
 */
function createRandomHypervector(
  dimensions: number,
  bipolar: boolean,
  rng: SeededRNG
): Hypervector {
  const hv = new Float32Array(dimensions);

  for (let i = 0; i < dimensions; i++) {
    hv[i] = bipolar ? rng.nextBipolar() : rng.nextBinary();
  }

  return hv;
}

/**
 * Compute cosine similarity between two hypervectors.
 */
function cosineSimilarity(a: Hypervector, b: Hypervector): number {
  if (a.length !== b.length) {
    throw new Error('Hypervector dimensions must match');
  }

  let dotProduct = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < a.length; i++) {
    dotProduct += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }

  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  if (denom < MATH_CONSTANTS.EPSILON) return 0;

  return dotProduct / denom;
}

/**
 * Binding operation (element-wise multiplication for bipolar).
 * Creates a representation of "A bound to B".
 */
function bind(a: Hypervector, b: Hypervector): Hypervector {
  if (a.length !== b.length) {
    throw new Error('Hypervector dimensions must match');
  }

  const result = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) {
    result[i] = a[i] * b[i];
  }
  return result;
}

/**
 * Bundling operation (element-wise addition).
 * Creates a composite representation.
 */
function bundle(vectors: Hypervector[]): Hypervector {
  if (vectors.length === 0) {
    throw new Error('Cannot bundle empty array');
  }

  const dimensions = vectors[0].length;
  const result = new Float32Array(dimensions);

  for (const hv of vectors) {
    if (hv.length !== dimensions) {
      throw new Error('All hypervectors must have same dimensions');
    }
    for (let i = 0; i < dimensions; i++) {
      result[i] += hv[i];
    }
  }

  return result;
}

/**
 * Normalize a hypervector (for bundled results).
 */
function normalizeHypervector(hv: Hypervector, bipolar: boolean): Hypervector {
  const result = new Float32Array(hv.length);

  if (bipolar) {
    // Threshold at 0
    for (let i = 0; i < hv.length; i++) {
      result[i] = hv[i] >= 0 ? 1 : -1;
    }
  } else {
    // Threshold at 0.5
    for (let i = 0; i < hv.length; i++) {
      result[i] = hv[i] >= 0.5 ? 1 : 0;
    }
  }

  return result;
}

/**
 * Permutation operation (cyclic shift).
 * Encodes sequence/position information.
 */
function permute(hv: Hypervector, shift: number = 1): Hypervector {
  const n = hv.length;
  const result = new Float32Array(n);
  const normalizedShift = ((shift % n) + n) % n;

  for (let i = 0; i < n; i++) {
    result[(i + normalizedShift) % n] = hv[i];
  }

  return result;
}

/**
 * Inverse permutation.
 */
function inversePermute(hv: Hypervector, shift: number = 1): Hypervector {
  return permute(hv, -shift);
}

// =============================================================================
// HDC ENCODER CLASS
// =============================================================================

/**
 * HDCEncoder provides hyperdimensional computing capabilities.
 *
 * Usage:
 * ```typescript
 * const encoder = new HDCEncoder();
 * const hvA = encoder.encode('concept_a');
 * const hvB = encoder.encode('concept_b');
 * const similarity = encoder.similarity(hvA, hvB);
 * ```
 */
export class HDCEncoder {
  private _config: HDCConfig;
  private _rng: SeededRNG;
  private _itemMemory: Map<string, Hypervector>;
  private _lattice: Lattice24;
  private _vertexHypervectors: Hypervector[];

  constructor(config: Partial<HDCConfig> = {}, lattice?: Lattice24) {
    this._config = { ...DEFAULT_HDC_CONFIG, ...config };
    this._rng = new SeededRNG(this._config.seed ?? 42);
    this._itemMemory = new Map();
    this._lattice = lattice ?? getDefaultLattice();
    this._vertexHypervectors = this._initializeVertexHypervectors();
  }

  /**
   * Initialize hypervectors for each lattice vertex.
   */
  private _initializeVertexHypervectors(): Hypervector[] {
    const hvs: Hypervector[] = [];

    for (let i = 0; i < this._lattice.vertexCount; i++) {
      hvs.push(createRandomHypervector(
        this._config.dimensions,
        this._config.bipolar,
        this._rng
      ));
    }

    return hvs;
  }

  // =========================================================================
  // ENCODING
  // =========================================================================

  /**
   * Encode a symbol/concept into a hypervector.
   * Creates a new random hypervector if not in memory.
   */
  encode(symbol: string): Hypervector {
    if (this._itemMemory.has(symbol)) {
      // LRU: Move to end (most recently used)
      const hv = this._itemMemory.get(symbol)!;
      this._itemMemory.delete(symbol);
      this._itemMemory.set(symbol, hv);
      return hv;
    }

    const hv = createRandomHypervector(
      this._config.dimensions,
      this._config.bipolar,
      this._rng
    );

    // LRU: Check size limit before adding
    const limit = this._config.memoryLimit ?? 10000;
    if (this._itemMemory.size >= limit) {
      // Remove oldest entry (first key in Map)
      const oldestKey = this._itemMemory.keys().next().value;
      if (oldestKey !== undefined) {
        this._itemMemory.delete(oldestKey);
      }
    }

    this._itemMemory.set(symbol, hv);
    return hv;
  }

  /**
   * Encode a numeric value into a hypervector using thermometer encoding.
   */
  encodeNumeric(value: number, min: number = 0, max: number = 1): Hypervector {
    const normalized = Math.max(0, Math.min(1, (value - min) / (max - min)));
    const levels = 100;
    const level = Math.floor(normalized * levels);

    // Create thermometer code: bundle first `level` hypervectors
    const hvs: Hypervector[] = [];
    for (let i = 0; i <= level; i++) {
      hvs.push(this.encode(`level_${i}`));
    }

    return hvs.length > 0 ? normalizeHypervector(bundle(hvs), this._config.bipolar) : this.encode('zero');
  }

  /**
   * Encode a 4D vector into a hypervector.
   */
  encodeVector4D(v: Vector4D): Hypervector {
    // Encode each component and bind with position
    const hvs: Hypervector[] = [];

    const components = ['x', 'y', 'z', 'w'];
    for (let i = 0; i < 4; i++) {
      const componentHV = this.encodeNumeric(v[i], -2, 2);
      const positionHV = this.encode(`pos_${components[i]}`);
      hvs.push(bind(componentHV, positionHV));
    }

    return normalizeHypervector(bundle(hvs), this._config.bipolar);
  }

  /**
   * Encode a sequence of symbols.
   */
  encodeSequence(symbols: string[]): Hypervector {
    if (symbols.length === 0) {
      return this.encode('empty');
    }

    const hvs: Hypervector[] = [];

    for (let i = 0; i < symbols.length; i++) {
      const symbolHV = this.encode(symbols[i]);
      // Apply i permutations to encode position
      hvs.push(permute(symbolHV, i));
    }

    return normalizeHypervector(bundle(hvs), this._config.bipolar);
  }

  // =========================================================================
  // OPERATIONS
  // =========================================================================

  /**
   * Bind two hypervectors.
   */
  bind(a: Hypervector, b: Hypervector): Hypervector {
    return bind(a, b);
  }

  /**
   * Bundle multiple hypervectors.
   */
  bundle(vectors: Hypervector[]): Hypervector {
    return normalizeHypervector(bundle(vectors), this._config.bipolar);
  }

  /**
   * Permute a hypervector.
   */
  permute(hv: Hypervector, shift: number = 1): Hypervector {
    return permute(hv, shift);
  }

  /**
   * Compute similarity between two hypervectors.
   */
  similarity(a: Hypervector, b: Hypervector): number {
    return cosineSimilarity(a, b);
  }

  // =========================================================================
  // LATTICE INTEGRATION
  // =========================================================================

  /**
   * Get the hypervector for a lattice vertex.
   */
  getVertexHypervector(vertexId: number): Hypervector {
    return this._vertexHypervectors[vertexId] ?? this._vertexHypervectors[0];
  }

  /**
   * Find the nearest lattice vertex to a hypervector.
   */
  findNearestVertex(hv: Hypervector): { vertexId: number; similarity: number } {
    let bestVertex = 0;
    let bestSimilarity = -Infinity;

    for (let i = 0; i < this._vertexHypervectors.length; i++) {
      const sim = this.similarity(hv, this._vertexHypervectors[i]);
      if (sim > bestSimilarity) {
        bestSimilarity = sim;
        bestVertex = i;
      }
    }

    return { vertexId: bestVertex, similarity: bestSimilarity };
  }

  /**
   * Project a hypervector to a 4D position using vertex similarity.
   */
  projectToPosition(hv: Hypervector): Vector4D {
    // Compute similarity to each vertex
    const similarities: number[] = [];
    let totalSim = 0;

    for (let i = 0; i < this._vertexHypervectors.length; i++) {
      const sim = Math.max(0, this.similarity(hv, this._vertexHypervectors[i]));
      similarities.push(sim);
      totalSim += sim;
    }

    // Weighted average of vertex positions
    const position: Vector4D = [0, 0, 0, 0];

    if (totalSim > MATH_CONSTANTS.EPSILON) {
      for (let i = 0; i < this._vertexHypervectors.length; i++) {
        const vertex = this._lattice.getVertex(i);
        if (vertex) {
          const weight = similarities[i] / totalSim;
          position[0] += vertex.coordinates[0] * weight;
          position[1] += vertex.coordinates[1] * weight;
          position[2] += vertex.coordinates[2] * weight;
          position[3] += vertex.coordinates[3] * weight;
        }
      }
    }

    return position;
  }

  /**
   * Encode a 4D position as a hypervector using vertex proximity.
   */
  encodePosition(position: Vector4D): Hypervector {
    const kNearest = this._lattice.findKNearest(position, 4);

    // Bundle nearest vertex hypervectors with distance weighting
    const hvs: Hypervector[] = [];

    for (const vId of kNearest) {
      hvs.push(this._vertexHypervectors[vId]);
    }

    return this.bundle(hvs);
  }

  // =========================================================================
  // MEMORY OPERATIONS
  // =========================================================================

  /**
   * Store a hypervector in item memory.
   */
  store(symbol: string, hv: Hypervector): void {
    this._itemMemory.set(symbol, hv);
  }

  /**
   * Retrieve a hypervector from item memory.
   */
  retrieve(symbol: string): Hypervector | undefined {
    const hv = this._itemMemory.get(symbol);
    if (hv) {
      // LRU: Move to end (most recently used)
      this._itemMemory.delete(symbol);
      this._itemMemory.set(symbol, hv);
    }
    return hv;
  }

  /**
   * Query item memory for most similar entries.
   */
  query(hv: Hypervector, topK: number = 5): { symbol: string; similarity: number }[] {
    const results: { symbol: string; similarity: number }[] = [];

    for (const [symbol, storedHV] of this._itemMemory) {
      const sim = this.similarity(hv, storedHV);
      results.push({ symbol, similarity: sim });
    }

    results.sort((a, b) => b.similarity - a.similarity);
    return results.slice(0, topK);
  }

  /**
   * Clear item memory.
   */
  clearMemory(): void {
    this._itemMemory.clear();
  }

  // =========================================================================
  // STATISTICS
  // =========================================================================

  get dimensions(): number {
    return this._config.dimensions;
  }

  get memorySize(): number {
    return this._itemMemory.size;
  }

  getStats(): Record<string, unknown> {
    return {
      dimensions: this._config.dimensions,
      bipolar: this._config.bipolar,
      memorySize: this._itemMemory.size,
      vertexCount: this._vertexHypervectors.length
    };
  }
}

// =============================================================================
// FACTORY FUNCTIONS
// =============================================================================

export function createHDCEncoder(config?: Partial<HDCConfig>): HDCEncoder {
  return new HDCEncoder(config);
}

// =============================================================================
// EXPORTS
// =============================================================================

export {
  Hypervector,
  HDCConfig,
  DEFAULT_HDC_CONFIG,
  cosineSimilarity,
  bind,
  bundle,
  permute,
  normalizeHypervector
};
