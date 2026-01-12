// File: src/core/fhrr.ts
// Fourier Holographic Reduced Representations (FHRR)
// Phasor-based Vector Symbolic Architecture for geometric reasoning

import { DEFAULT_DIMENSION } from './hdc';

/**
 * Complex number for phasor representation
 */
export interface Complex {
  re: number;
  im: number;
}

/**
 * Create complex from polar form (magnitude, phase)
 */
export function polar(magnitude: number, phase: number): Complex {
  return {
    re: magnitude * Math.cos(phase),
    im: magnitude * Math.sin(phase),
  };
}

/**
 * Create unit phasor from phase angle
 */
export function phasor(phase: number): Complex {
  return polar(1, phase);
}

/**
 * Complex multiplication
 */
export function cmul(a: Complex, b: Complex): Complex {
  return {
    re: a.re * b.re - a.im * b.im,
    im: a.re * b.im + a.im * b.re,
  };
}

/**
 * Complex conjugate
 */
export function conj(c: Complex): Complex {
  return { re: c.re, im: -c.im };
}

/**
 * Complex addition
 */
export function cadd(a: Complex, b: Complex): Complex {
  return { re: a.re + b.re, im: a.im + b.im };
}

/**
 * Complex magnitude
 */
export function cmag(c: Complex): number {
  return Math.sqrt(c.re * c.re + c.im * c.im);
}

/**
 * Complex phase angle
 */
export function cphase(c: Complex): number {
  return Math.atan2(c.im, c.re);
}

/**
 * Normalize to unit magnitude
 */
export function cnormalize(c: Complex): Complex {
  const mag = cmag(c);
  if (mag === 0) return { re: 1, im: 0 };
  return { re: c.re / mag, im: c.im / mag };
}

/**
 * FHRR Phasor Vector - represents concepts as arrays of unit phasors
 * Each element is e^{iθ_k} where θ_k is the phase angle
 *
 * Key insight: In FHRR, binding = element-wise multiplication = phase addition
 * This makes "reasoning by rotation" a native operation
 */
export class PhasorVector {
  readonly dimension: number;
  readonly phases: Float32Array; // Store only phases (magnitude is always 1)

  constructor(dimension: number = DEFAULT_DIMENSION, phases?: Float32Array) {
    this.dimension = dimension;
    this.phases = phases ?? new Float32Array(dimension);
  }

  /**
   * Create a random phasor vector (phases uniformly distributed in [0, 2π))
   */
  static random(dimension: number = DEFAULT_DIMENSION): PhasorVector {
    const phases = new Float32Array(dimension);
    for (let i = 0; i < dimension; i++) {
      phases[i] = Math.random() * 2 * Math.PI;
    }
    return new PhasorVector(dimension, phases);
  }

  /**
   * Create identity phasor vector (all phases = 0)
   */
  static identity(dimension: number = DEFAULT_DIMENSION): PhasorVector {
    return new PhasorVector(dimension, new Float32Array(dimension));
  }

  /**
   * Create from phase array
   */
  static fromPhases(phases: number[]): PhasorVector {
    return new PhasorVector(phases.length, Float32Array.from(phases));
  }

  /**
   * Clone this phasor vector
   */
  clone(): PhasorVector {
    return new PhasorVector(this.dimension, Float32Array.from(this.phases));
  }

  /**
   * Get complex representation at index
   */
  getComplex(index: number): Complex {
    return phasor(this.phases[index]);
  }

  /**
   * Convert to array of complex numbers
   */
  toComplexArray(): Complex[] {
    const result: Complex[] = new Array(this.dimension);
    for (let i = 0; i < this.dimension; i++) {
      result[i] = phasor(this.phases[i]);
    }
    return result;
  }

  /**
   * BINDING in FHRR: Element-wise multiplication = Phase addition
   *
   * If A has phase θ_A and B has phase θ_B,
   * then A ⊗ B has phase θ_A + θ_B
   *
   * Geometric interpretation: This IS rotation in the complex plane!
   */
  bind(other: PhasorVector): PhasorVector {
    if (this.dimension !== other.dimension) {
      throw new Error('Dimension mismatch in bind');
    }

    const phases = new Float32Array(this.dimension);
    for (let i = 0; i < this.dimension; i++) {
      // Phase addition (mod 2π)
      phases[i] = (this.phases[i] + other.phases[i]) % (2 * Math.PI);
    }
    return new PhasorVector(this.dimension, phases);
  }

  /**
   * UNBINDING: Element-wise division = Phase subtraction
   * Uses complex conjugate (negate phases)
   */
  unbind(key: PhasorVector): PhasorVector {
    if (this.dimension !== key.dimension) {
      throw new Error('Dimension mismatch in unbind');
    }

    const phases = new Float32Array(this.dimension);
    for (let i = 0; i < this.dimension; i++) {
      // Phase subtraction (mod 2π)
      let phase = this.phases[i] - key.phases[i];
      if (phase < 0) phase += 2 * Math.PI;
      phases[i] = phase;
    }
    return new PhasorVector(this.dimension, phases);
  }

  /**
   * Get inverse (conjugate) - negates all phases
   */
  inverse(): PhasorVector {
    const phases = new Float32Array(this.dimension);
    for (let i = 0; i < this.dimension; i++) {
      phases[i] = -this.phases[i];
      if (phases[i] < 0) phases[i] += 2 * Math.PI;
    }
    return new PhasorVector(this.dimension, phases);
  }

  /**
   * BUNDLING in FHRR: Circular mean of phasors
   * Average the complex values, then normalize back to unit phasor
   */
  static bundle(vectors: PhasorVector[]): PhasorVector {
    if (vectors.length === 0) {
      throw new Error('Cannot bundle empty list');
    }

    const dimension = vectors[0].dimension;
    const phases = new Float32Array(dimension);

    for (let i = 0; i < dimension; i++) {
      // Sum complex values
      let sumRe = 0;
      let sumIm = 0;
      for (const v of vectors) {
        sumRe += Math.cos(v.phases[i]);
        sumIm += Math.sin(v.phases[i]);
      }
      // Take the angle of the sum (circular mean)
      phases[i] = Math.atan2(sumIm, sumRe);
      if (phases[i] < 0) phases[i] += 2 * Math.PI;
    }

    return new PhasorVector(dimension, phases);
  }

  /**
   * PERMUTATION: Cyclic shift of elements
   */
  permute(shifts: number = 1): PhasorVector {
    const phases = new Float32Array(this.dimension);
    const normalizedShift = ((shifts % this.dimension) + this.dimension) % this.dimension;

    for (let i = 0; i < this.dimension; i++) {
      const srcIdx = (i - normalizedShift + this.dimension) % this.dimension;
      phases[i] = this.phases[srcIdx];
    }
    return new PhasorVector(this.dimension, phases);
  }

  /**
   * Cosine similarity using complex inner product
   * Returns real part of normalized complex dot product
   */
  similarity(other: PhasorVector): number {
    if (this.dimension !== other.dimension) {
      throw new Error('Dimension mismatch');
    }

    let sumRe = 0;
    let sumIm = 0;

    for (let i = 0; i < this.dimension; i++) {
      // Inner product: a * conj(b) = e^{i(θ_a - θ_b)}
      const phaseDiff = this.phases[i] - other.phases[i];
      sumRe += Math.cos(phaseDiff);
      sumIm += Math.sin(phaseDiff);
    }

    // Normalize by dimension
    // Real part gives cosine similarity
    return sumRe / this.dimension;
  }

  /**
   * Apply a rotation (add a constant phase to all elements)
   * This is a global rotation in the phasor space
   */
  rotate(angle: number): PhasorVector {
    const phases = new Float32Array(this.dimension);
    for (let i = 0; i < this.dimension; i++) {
      phases[i] = (this.phases[i] + angle) % (2 * Math.PI);
    }
    return new PhasorVector(this.dimension, phases);
  }

  /**
   * Apply a rotation rotor (element-wise angle addition from rotor)
   * This is the core "reasoning by rotation" operation in PPP
   */
  applyRotor(rotor: PhasorVector): PhasorVector {
    return this.bind(rotor); // In FHRR, binding IS rotation
  }

  /**
   * Get the total phase (sum of all phases, useful for analysis)
   */
  totalPhase(): number {
    let sum = 0;
    for (let i = 0; i < this.dimension; i++) {
      sum += this.phases[i];
    }
    return sum;
  }

  /**
   * Get phase coherence (how aligned are the phases)
   * Returns 0-1, where 1 = all phases equal, 0 = uniformly distributed
   */
  coherence(): number {
    let sumRe = 0;
    let sumIm = 0;
    for (let i = 0; i < this.dimension; i++) {
      sumRe += Math.cos(this.phases[i]);
      sumIm += Math.sin(this.phases[i]);
    }
    return Math.sqrt(sumRe * sumRe + sumIm * sumIm) / this.dimension;
  }

  /**
   * String representation
   */
  toString(): string {
    const first = Array.from(this.phases.slice(0, 3)).map(p => (p / Math.PI).toFixed(2) + 'π');
    const last = Array.from(this.phases.slice(-3)).map(p => (p / Math.PI).toFixed(2) + 'π');
    return `FHRR[${this.dimension}](${first.join(',')}...${last.join(',')})`;
  }
}

/**
 * FHRR-based VSA operations
 */
export class FHRR_VSA {
  readonly dimension: number;

  constructor(dimension: number = DEFAULT_DIMENSION) {
    this.dimension = dimension;
  }

  /**
   * Create a random concept vector
   */
  randomConcept(): PhasorVector {
    return PhasorVector.random(this.dimension);
  }

  /**
   * Create a rule (rotor) that transforms concept A into concept B
   * Rule = B ⊗ inv(A) = B ⊗ conj(A)
   *
   * Applying this rule to A gives B (approximately):
   * A ⊗ Rule = A ⊗ B ⊗ conj(A) = B
   */
  createRule(from: PhasorVector, to: PhasorVector): PhasorVector {
    return to.bind(from.inverse());
  }

  /**
   * Apply a rule to transform a concept
   * This is the core of "reasoning by rotation"
   */
  applyRule(concept: PhasorVector, rule: PhasorVector): PhasorVector {
    return concept.bind(rule);
  }

  /**
   * Chain multiple rules: Rule_total = Rule_1 ⊗ Rule_2 ⊗ ... ⊗ Rule_n
   */
  chainRules(rules: PhasorVector[]): PhasorVector {
    if (rules.length === 0) {
      return PhasorVector.identity(this.dimension);
    }
    let result = rules[0].clone();
    for (let i = 1; i < rules.length; i++) {
      result = result.bind(rules[i]);
    }
    return result;
  }

  /**
   * Encode a structure: { role1: filler1, role2: filler2, ... }
   */
  encodeStructure(
    bindings: Array<{ role: PhasorVector; filler: PhasorVector }>
  ): PhasorVector {
    const bound = bindings.map(({ role, filler }) => role.bind(filler));
    return PhasorVector.bundle(bound);
  }

  /**
   * Query a structure for a role
   */
  queryStructure(structure: PhasorVector, role: PhasorVector): PhasorVector {
    return structure.unbind(role);
  }

  /**
   * Create a codebook of named concepts
   */
  createCodebook(names: string[]): Map<string, PhasorVector> {
    const codebook = new Map<string, PhasorVector>();
    for (const name of names) {
      codebook.set(name, PhasorVector.random(this.dimension));
    }
    return codebook;
  }

  /**
   * Find the most similar concept in a codebook
   */
  lookup(
    query: PhasorVector,
    codebook: Map<string, PhasorVector>
  ): { name: string; similarity: number } | null {
    let best: { name: string; similarity: number } | null = null;

    for (const [name, vector] of codebook) {
      const sim = query.similarity(vector);
      if (!best || sim > best.similarity) {
        best = { name, similarity: sim };
      }
    }

    return best;
  }
}

/**
 * Resonator Network - iterative cleanup for noisy queries
 * Converges to the nearest clean concept in the codebook
 */
export class ResonatorNetwork {
  private vsa: FHRR_VSA;
  private codebook: Map<string, PhasorVector>;
  private maxIterations: number;
  private convergenceThreshold: number;

  constructor(
    codebook: Map<string, PhasorVector>,
    maxIterations: number = 100,
    convergenceThreshold: number = 0.99
  ) {
    if (codebook.size === 0) {
      throw new Error('Codebook must not be empty');
    }
    const firstEntry = codebook.values().next().value as PhasorVector;
    this.vsa = new FHRR_VSA(firstEntry.dimension);
    this.codebook = codebook;
    this.maxIterations = maxIterations;
    this.convergenceThreshold = convergenceThreshold;
  }

  /**
   * Resonate a noisy query to find the nearest clean concept
   */
  resonate(query: PhasorVector): {
    result: PhasorVector;
    name: string;
    iterations: number;
    similarity: number;
  } {
    let current = query.clone();
    let iterations = 0;
    let bestMatch = this.vsa.lookup(current, this.codebook);

    while (iterations < this.maxIterations) {
      // Compute weighted average based on similarity
      const similarities: Array<{ vector: PhasorVector; weight: number }> = [];
      let totalWeight = 0;

      for (const [, vector] of this.codebook) {
        const sim = current.similarity(vector);
        // Use softmax-like weighting
        const weight = Math.exp(sim * 10);
        similarities.push({ vector, weight });
        totalWeight += weight;
      }

      // Bundle with weights (circular weighted mean)
      const phases = new Float32Array(current.dimension);
      for (let i = 0; i < current.dimension; i++) {
        let sumRe = 0;
        let sumIm = 0;
        for (const { vector, weight } of similarities) {
          const normalizedWeight = weight / totalWeight;
          sumRe += normalizedWeight * Math.cos(vector.phases[i]);
          sumIm += normalizedWeight * Math.sin(vector.phases[i]);
        }
        phases[i] = Math.atan2(sumIm, sumRe);
        if (phases[i] < 0) phases[i] += 2 * Math.PI;
      }

      current = new PhasorVector(current.dimension, phases);
      iterations++;

      bestMatch = this.vsa.lookup(current, this.codebook);
      if (bestMatch && bestMatch.similarity >= this.convergenceThreshold) {
        break;
      }
    }

    const finalMatch = this.vsa.lookup(current, this.codebook)!;
    return {
      result: this.codebook.get(finalMatch.name)!,
      name: finalMatch.name,
      iterations,
      similarity: finalMatch.similarity,
    };
  }
}

export default { PhasorVector, FHRR_VSA, ResonatorNetwork, polar, phasor };
