// File: src/core/hdc.ts
// Hyperdimensional Computing (HDC) / Vector Symbolic Architecture (VSA)
// Core primitives for high-dimensional distributed representations

/**
 * Default hypervector dimension
 * Higher dimensions = better orthogonality, but more compute
 * D=10000 gives ~0.01 expected cosine similarity between random vectors
 */
export const DEFAULT_DIMENSION = 10000;

/**
 * Hypervector - a high-dimensional distributed representation
 * In PPP, concepts are encoded as hypervectors
 */
export class Hypervector {
  readonly dimension: number;
  readonly data: Float32Array;

  constructor(dimension: number = DEFAULT_DIMENSION, data?: Float32Array) {
    this.dimension = dimension;
    this.data = data ?? new Float32Array(dimension);
  }

  /**
   * Create a random hypervector (approximately orthogonal to all others)
   * Uses bipolar representation: values in {-1, +1}
   */
  static random(dimension: number = DEFAULT_DIMENSION): Hypervector {
    const data = new Float32Array(dimension);
    for (let i = 0; i < dimension; i++) {
      data[i] = Math.random() < 0.5 ? -1 : 1;
    }
    return new Hypervector(dimension, data);
  }

  /**
   * Create a zero hypervector
   */
  static zero(dimension: number = DEFAULT_DIMENSION): Hypervector {
    return new Hypervector(dimension, new Float32Array(dimension));
  }

  /**
   * Create from array
   */
  static fromArray(arr: number[]): Hypervector {
    return new Hypervector(arr.length, Float32Array.from(arr));
  }

  /**
   * Clone this hypervector
   */
  clone(): Hypervector {
    return new Hypervector(this.dimension, Float32Array.from(this.data));
  }

  /**
   * Normalize to unit length
   */
  normalize(): Hypervector {
    const norm = this.norm();
    if (norm === 0) return this.clone();
    const data = new Float32Array(this.dimension);
    for (let i = 0; i < this.dimension; i++) {
      data[i] = this.data[i] / norm;
    }
    return new Hypervector(this.dimension, data);
  }

  /**
   * L2 norm (Euclidean length)
   */
  norm(): number {
    let sum = 0;
    for (let i = 0; i < this.dimension; i++) {
      sum += this.data[i] * this.data[i];
    }
    return Math.sqrt(sum);
  }

  /**
   * Cosine similarity with another hypervector
   * Returns value in [-1, 1], where 1 = identical, 0 = orthogonal, -1 = opposite
   */
  cosineSimilarity(other: Hypervector): number {
    if (this.dimension !== other.dimension) {
      throw new Error('Dimension mismatch');
    }
    let dot = 0;
    let normA = 0;
    let normB = 0;
    for (let i = 0; i < this.dimension; i++) {
      dot += this.data[i] * other.data[i];
      normA += this.data[i] * this.data[i];
      normB += other.data[i] * other.data[i];
    }
    const denom = Math.sqrt(normA) * Math.sqrt(normB);
    return denom === 0 ? 0 : dot / denom;
  }

  /**
   * Dot product
   */
  dot(other: Hypervector): number {
    if (this.dimension !== other.dimension) {
      throw new Error('Dimension mismatch');
    }
    let sum = 0;
    for (let i = 0; i < this.dimension; i++) {
      sum += this.data[i] * other.data[i];
    }
    return sum;
  }

  /**
   * Element-wise addition (for bundling preparation)
   */
  add(other: Hypervector): Hypervector {
    if (this.dimension !== other.dimension) {
      throw new Error('Dimension mismatch');
    }
    const data = new Float32Array(this.dimension);
    for (let i = 0; i < this.dimension; i++) {
      data[i] = this.data[i] + other.data[i];
    }
    return new Hypervector(this.dimension, data);
  }

  /**
   * Scalar multiplication
   */
  scale(scalar: number): Hypervector {
    const data = new Float32Array(this.dimension);
    for (let i = 0; i < this.dimension; i++) {
      data[i] = this.data[i] * scalar;
    }
    return new Hypervector(this.dimension, data);
  }

  /**
   * Binarize (threshold at 0)
   * Returns bipolar {-1, +1} representation
   */
  binarize(): Hypervector {
    const data = new Float32Array(this.dimension);
    for (let i = 0; i < this.dimension; i++) {
      data[i] = this.data[i] >= 0 ? 1 : -1;
    }
    return new Hypervector(this.dimension, data);
  }

  /**
   * Get a compact string representation (first/last few values)
   */
  toString(): string {
    const first = Array.from(this.data.slice(0, 3)).map(v => v.toFixed(2));
    const last = Array.from(this.data.slice(-3)).map(v => v.toFixed(2));
    return `HV[${this.dimension}](${first.join(',')}...${last.join(',')})`;
  }
}

/**
 * VSA Operations - The algebra of hyperdimensional computing
 */
export class VSA {
  readonly dimension: number;

  constructor(dimension: number = DEFAULT_DIMENSION) {
    this.dimension = dimension;
  }

  /**
   * BUNDLING (Superposition) - Creates a vector similar to all inputs
   * Geometric interpretation: Moves to the centroid of the polytope
   * defined by the input vectors
   *
   * C = A + B + ... (then normalized/binarized)
   */
  bundle(vectors: Hypervector[]): Hypervector {
    if (vectors.length === 0) {
      return Hypervector.zero(this.dimension);
    }

    const sum = Hypervector.zero(this.dimension);
    for (const v of vectors) {
      if (v.dimension !== this.dimension) {
        throw new Error('Dimension mismatch in bundle');
      }
      for (let i = 0; i < this.dimension; i++) {
        sum.data[i] += v.data[i];
      }
    }

    // Binarize for clean representation (majority vote)
    return sum.binarize();
  }

  /**
   * BINDING (Multiplication) - Creates a vector orthogonal to both inputs
   * Geometric interpretation: Maps to a new region of hyperspace
   * Creates associations/relations
   *
   * For bipolar vectors: element-wise multiplication
   * C = A ⊗ B (Hadamard product for bipolar)
   */
  bind(a: Hypervector, b: Hypervector): Hypervector {
    if (a.dimension !== this.dimension || b.dimension !== this.dimension) {
      throw new Error('Dimension mismatch in bind');
    }

    const data = new Float32Array(this.dimension);
    for (let i = 0; i < this.dimension; i++) {
      data[i] = a.data[i] * b.data[i];
    }
    return new Hypervector(this.dimension, data);
  }

  /**
   * UNBIND (Inverse binding) - Retrieves one component from a bound pair
   * For bipolar: binding is its own inverse (A ⊗ A = 1)
   *
   * If C = A ⊗ B, then A = C ⊗ B (approximately)
   */
  unbind(bound: Hypervector, key: Hypervector): Hypervector {
    // For bipolar vectors, bind is self-inverse
    return this.bind(bound, key);
  }

  /**
   * PERMUTATION (Rotation/Shift) - Creates a dissimilar but related vector
   * Geometric interpretation: Rotates the polytope
   * Used for encoding sequence/order
   *
   * Π(A) - cyclic shift of components
   */
  permute(v: Hypervector, shifts: number = 1): Hypervector {
    if (v.dimension !== this.dimension) {
      throw new Error('Dimension mismatch in permute');
    }

    const data = new Float32Array(this.dimension);
    const normalizedShift = ((shifts % this.dimension) + this.dimension) % this.dimension;

    for (let i = 0; i < this.dimension; i++) {
      const srcIdx = (i - normalizedShift + this.dimension) % this.dimension;
      data[i] = v.data[srcIdx];
    }
    return new Hypervector(this.dimension, data);
  }

  /**
   * INVERSE PERMUTATION
   */
  inversePermute(v: Hypervector, shifts: number = 1): Hypervector {
    return this.permute(v, -shifts);
  }

  /**
   * Create a sequence encoding: [A, B, C] -> Π²(A) + Π(B) + C
   * Position is encoded via permutation count
   */
  encodeSequence(vectors: Hypervector[]): Hypervector {
    if (vectors.length === 0) {
      return Hypervector.zero(this.dimension);
    }

    const permuted: Hypervector[] = vectors.map((v, i) =>
      this.permute(v, vectors.length - 1 - i)
    );

    return this.bundle(permuted);
  }

  /**
   * Create a record/struct encoding: {role1: filler1, role2: filler2}
   * Each role-filler pair is bound, then all are bundled
   */
  encodeRecord(pairs: Array<{ role: Hypervector; filler: Hypervector }>): Hypervector {
    const bindings = pairs.map(({ role, filler }) => this.bind(role, filler));
    return this.bundle(bindings);
  }

  /**
   * Query a record for a specific role
   */
  queryRecord(record: Hypervector, role: Hypervector): Hypervector {
    return this.unbind(record, role);
  }

  /**
   * Similarity search - find the most similar vector in a codebook
   */
  findMostSimilar(
    query: Hypervector,
    codebook: Map<string, Hypervector>
  ): { name: string; vector: Hypervector; similarity: number } | null {
    let best: { name: string; vector: Hypervector; similarity: number } | null = null;

    for (const [name, vector] of codebook) {
      const sim = query.cosineSimilarity(vector);
      if (!best || sim > best.similarity) {
        best = { name, vector, similarity: sim };
      }
    }

    return best;
  }

  /**
   * Create a random codebook of named concepts
   */
  createCodebook(names: string[]): Map<string, Hypervector> {
    const codebook = new Map<string, Hypervector>();
    for (const name of names) {
      codebook.set(name, Hypervector.random(this.dimension));
    }
    return codebook;
  }
}

/**
 * Semantic Memory - An associative memory using HDC
 * Stores concept-vector pairs and enables similarity-based retrieval
 */
export class SemanticMemory {
  private vsa: VSA;
  private items: Map<string, Hypervector> = new Map();

  constructor(dimension: number = DEFAULT_DIMENSION) {
    this.vsa = new VSA(dimension);
  }

  /**
   * Store a concept with its hypervector
   */
  store(name: string, vector: Hypervector): void {
    this.items.set(name, vector);
  }

  /**
   * Store a concept, generating a random vector
   */
  storeNew(name: string): Hypervector {
    const vector = Hypervector.random(this.vsa.dimension);
    this.items.set(name, vector);
    return vector;
  }

  /**
   * Retrieve by exact name
   */
  get(name: string): Hypervector | undefined {
    return this.items.get(name);
  }

  /**
   * Query by similarity - returns top-k matches
   */
  query(queryVector: Hypervector, topK: number = 5): Array<{ name: string; similarity: number }> {
    const results: Array<{ name: string; similarity: number }> = [];

    for (const [name, vector] of this.items) {
      results.push({
        name,
        similarity: queryVector.cosineSimilarity(vector),
      });
    }

    results.sort((a, b) => b.similarity - a.similarity);
    return results.slice(0, topK);
  }

  /**
   * Get all stored concepts
   */
  getAllNames(): string[] {
    return Array.from(this.items.keys());
  }

  /**
   * Get count
   */
  get size(): number {
    return this.items.size;
  }

  /**
   * Export as serializable object
   */
  export(): Record<string, number[]> {
    const result: Record<string, number[]> = {};
    for (const [name, vector] of this.items) {
      result[name] = Array.from(vector.data);
    }
    return result;
  }
}

export default { Hypervector, VSA, SemanticMemory, DEFAULT_DIMENSION };
