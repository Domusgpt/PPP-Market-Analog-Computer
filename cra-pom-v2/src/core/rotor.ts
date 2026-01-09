// File: src/core/rotor.ts
// Rotors and Rules - Clifford Algebra for reasoning by rotation
// Implements the PPP principle: "logical rules are rotation operators"

import { Hypervector, DEFAULT_DIMENSION } from './hdc';
import { PhasorVector } from './fhrr';

/**
 * A Rotor represents a rotation in high-dimensional space
 * In PPP, rules are encoded as rotors that transform concept vectors
 *
 * Mathematical foundation: R = e^{-Bθ/2} (Clifford algebra)
 * Where B is the bivector (plane of rotation) and θ is the angle
 *
 * For high-D spaces, we use a more practical representation:
 * - For FHRR: Rotor is a PhasorVector where applying = binding
 * - For bipolar HDC: Rotor is a transformation matrix
 */
export class Rotor {
  readonly name: string;
  readonly dimension: number;
  readonly phasor: PhasorVector; // FHRR representation
  readonly metadata: Record<string, unknown>;

  constructor(
    name: string,
    phasor: PhasorVector,
    metadata: Record<string, unknown> = {}
  ) {
    this.name = name;
    this.dimension = phasor.dimension;
    this.phasor = phasor;
    this.metadata = metadata;
  }

  /**
   * Create a rotor from source and target concepts
   * This rotor transforms source → target (approximately)
   *
   * Rule = Target ⊗ inv(Source)
   * Applying to Source: Source ⊗ Rule = Source ⊗ Target ⊗ inv(Source) ≈ Target
   */
  static fromTransition(
    name: string,
    source: PhasorVector,
    target: PhasorVector
  ): Rotor {
    if (source.dimension !== target.dimension) {
      throw new Error('Dimension mismatch');
    }

    // Rule = Target ⊗ inverse(Source) = Target ⊗ conjugate(Source)
    const phasor = target.bind(source.inverse());

    return new Rotor(name, phasor, {
      type: 'transition',
      source: 'encoded',
      target: 'encoded',
    });
  }

  /**
   * Create a random rotor (random rotation)
   */
  static random(name: string, dimension: number = DEFAULT_DIMENSION): Rotor {
    return new Rotor(name, PhasorVector.random(dimension), { type: 'random' });
  }

  /**
   * Create identity rotor (no change)
   */
  static identity(dimension: number = DEFAULT_DIMENSION): Rotor {
    return new Rotor('identity', PhasorVector.identity(dimension), { type: 'identity' });
  }

  /**
   * Create a uniform rotation rotor (same angle for all dimensions)
   */
  static uniformRotation(
    name: string,
    angle: number,
    dimension: number = DEFAULT_DIMENSION
  ): Rotor {
    const phases = new Float32Array(dimension);
    for (let i = 0; i < dimension; i++) {
      phases[i] = angle;
    }
    return new Rotor(name, new PhasorVector(dimension, phases), {
      type: 'uniform',
      angle,
    });
  }

  /**
   * Create a sparse rotor (rotates only some dimensions)
   */
  static sparseRotation(
    name: string,
    rotations: Map<number, number>, // index -> angle
    dimension: number = DEFAULT_DIMENSION
  ): Rotor {
    const phases = new Float32Array(dimension);
    for (const [index, angle] of rotations) {
      if (index >= 0 && index < dimension) {
        phases[index] = angle;
      }
    }
    return new Rotor(name, new PhasorVector(dimension, phases), {
      type: 'sparse',
      nonZeroCount: rotations.size,
    });
  }

  /**
   * Apply this rotor to a phasor vector
   * This is the core "reasoning by rotation" operation
   */
  apply(vector: PhasorVector): PhasorVector {
    return vector.bind(this.phasor);
  }

  /**
   * Apply this rotor to a hypervector (converts to phasor, applies, converts back)
   */
  applyToHypervector(vector: Hypervector): Hypervector {
    // Convert hypervector to phasor (phases from values)
    const phasor = hypervectorToPhasor(vector);
    const rotated = this.apply(phasor);
    return phasorToHypervector(rotated);
  }

  /**
   * Get the inverse rotor (undo this rotation)
   */
  inverse(): Rotor {
    return new Rotor(
      `inv(${this.name})`,
      this.phasor.inverse(),
      { ...this.metadata, inverted: true }
    );
  }

  /**
   * Compose with another rotor: this ∘ other = apply this then other
   */
  compose(other: Rotor): Rotor {
    if (this.dimension !== other.dimension) {
      throw new Error('Dimension mismatch');
    }
    return new Rotor(
      `${this.name}∘${other.name}`,
      this.phasor.bind(other.phasor),
      { type: 'composition', components: [this.name, other.name] }
    );
  }

  /**
   * Scale the rotation (multiply all angles by factor)
   * factor < 1 = smaller rotation, factor > 1 = larger rotation
   */
  scale(factor: number): Rotor {
    const phases = new Float32Array(this.dimension);
    for (let i = 0; i < this.dimension; i++) {
      phases[i] = (this.phasor.phases[i] * factor) % (2 * Math.PI);
    }
    return new Rotor(
      `${this.name}*${factor.toFixed(2)}`,
      new PhasorVector(this.dimension, phases),
      { ...this.metadata, scaled: factor }
    );
  }

  /**
   * Interpolate between identity and this rotor
   * t=0 gives identity, t=1 gives this rotor
   */
  slerp(t: number): Rotor {
    return this.scale(t);
  }

  /**
   * Get the magnitude of rotation (average absolute phase)
   */
  magnitude(): number {
    let sum = 0;
    for (let i = 0; i < this.dimension; i++) {
      sum += Math.abs(this.phasor.phases[i]);
    }
    return sum / this.dimension;
  }

  /**
   * String representation
   */
  toString(): string {
    return `Rotor(${this.name}, mag=${this.magnitude().toFixed(3)})`;
  }
}

/**
 * Convert hypervector to phasor (value -> phase mapping)
 */
function hypervectorToPhasor(hv: Hypervector): PhasorVector {
  const phases = new Float32Array(hv.dimension);
  for (let i = 0; i < hv.dimension; i++) {
    // Map [-1, 1] to [0, 2π]
    phases[i] = (hv.data[i] + 1) * Math.PI;
  }
  return new PhasorVector(hv.dimension, phases);
}

/**
 * Convert phasor to hypervector (phase -> value mapping)
 */
function phasorToHypervector(pv: PhasorVector): Hypervector {
  const data = new Float32Array(pv.dimension);
  for (let i = 0; i < pv.dimension; i++) {
    // Map [0, 2π] to [-1, 1]
    data[i] = pv.phases[i] / Math.PI - 1;
  }
  return new Hypervector(pv.dimension, data);
}

/**
 * Rule - A named transformation with semantic meaning
 * Wraps a Rotor with additional semantic information
 */
export interface Rule {
  name: string;
  description: string;
  rotor: Rotor;
  sourcePolytope?: string; // Name of source concept polytope
  targetPolytope?: string; // Name of target concept polytope
  confidence: number; // 0-1, how reliable is this rule
  bidirectional: boolean; // Can be applied in reverse?
}

/**
 * RuleLibrary - Collection of learned rules
 * Implements the "rule as rotation" paradigm
 */
export class RuleLibrary {
  private rules: Map<string, Rule> = new Map();
  readonly dimension: number;

  constructor(dimension: number = DEFAULT_DIMENSION) {
    this.dimension = dimension;
  }

  /**
   * Add a rule to the library
   */
  addRule(rule: Rule): void {
    this.rules.set(rule.name, rule);
  }

  /**
   * Learn a rule from examples
   * Given pairs of (before, after) concept vectors, learn the transformation
   */
  learnRule(
    name: string,
    description: string,
    examples: Array<{ before: PhasorVector; after: PhasorVector }>,
    bidirectional: boolean = false
  ): Rule {
    if (examples.length === 0) {
      throw new Error('Need at least one example');
    }

    // Average the rotors from each example
    const rotors = examples.map(({ before, after }) =>
      Rotor.fromTransition(name, before, after)
    );

    // Bundle the rotor phasors
    const avgPhasor = PhasorVector.bundle(rotors.map((r) => r.phasor));

    const rule: Rule = {
      name,
      description,
      rotor: new Rotor(name, avgPhasor, { learnedFromExamples: examples.length }),
      confidence: this.computeConfidence(examples, avgPhasor),
      bidirectional,
    };

    this.addRule(rule);
    return rule;
  }

  /**
   * Compute confidence based on how well the learned rotor fits examples
   */
  private computeConfidence(
    examples: Array<{ before: PhasorVector; after: PhasorVector }>,
    rotor: PhasorVector
  ): number {
    let totalSim = 0;

    for (const { before, after } of examples) {
      const predicted = before.bind(rotor);
      totalSim += predicted.similarity(after);
    }

    return totalSim / examples.length;
  }

  /**
   * Get a rule by name
   */
  getRule(name: string): Rule | undefined {
    return this.rules.get(name);
  }

  /**
   * Apply a rule to a concept
   */
  applyRule(ruleName: string, concept: PhasorVector): PhasorVector | null {
    const rule = this.rules.get(ruleName);
    if (!rule) return null;
    return rule.rotor.apply(concept);
  }

  /**
   * Find applicable rules (rules whose source matches the query)
   */
  findApplicableRules(
    _query: PhasorVector,
    sourcePolytope?: string
  ): Array<{ rule: Rule; applicability: number }> {
    const results: Array<{ rule: Rule; applicability: number }> = [];

    for (const rule of this.rules.values()) {
      // If source polytope is specified, check it matches
      if (sourcePolytope && rule.sourcePolytope && rule.sourcePolytope !== sourcePolytope) {
        continue;
      }

      // Applicability based on rule confidence
      results.push({
        rule,
        applicability: rule.confidence,
      });
    }

    return results.sort((a, b) => b.applicability - a.applicability);
  }

  /**
   * Chain multiple rules
   */
  chainRules(ruleNames: string[]): Rotor | null {
    const rotors: Rotor[] = [];

    for (const name of ruleNames) {
      const rule = this.rules.get(name);
      if (!rule) return null;
      rotors.push(rule.rotor);
    }

    if (rotors.length === 0) return Rotor.identity();

    let result = rotors[0];
    for (let i = 1; i < rotors.length; i++) {
      result = result.compose(rotors[i]);
    }

    return result;
  }

  /**
   * Get all rule names
   */
  getRuleNames(): string[] {
    return Array.from(this.rules.keys());
  }

  /**
   * Export rules as serializable object
   */
  export(): Record<string, unknown> {
    const result: Record<string, unknown> = {};
    for (const [name, rule] of this.rules) {
      result[name] = {
        description: rule.description,
        confidence: rule.confidence,
        bidirectional: rule.bidirectional,
        sourcePolytope: rule.sourcePolytope,
        targetPolytope: rule.targetPolytope,
        magnitude: rule.rotor.magnitude(),
      };
    }
    return result;
  }
}

/**
 * InferenceChain - A sequence of rule applications forming a proof
 */
export interface InferenceStep {
  rule: Rule;
  inputConcept: string;
  outputConcept: string;
  beforeVector: PhasorVector;
  afterVector: PhasorVector;
  confidence: number;
}

export class InferenceChain {
  readonly steps: InferenceStep[] = [];

  /**
   * Add a step to the chain
   */
  addStep(step: InferenceStep): void {
    this.steps.push(step);
  }

  /**
   * Get the overall confidence (product of step confidences)
   */
  getOverallConfidence(): number {
    if (this.steps.length === 0) return 1;
    return this.steps.reduce((acc, step) => acc * step.confidence, 1);
  }

  /**
   * Get the composite rotor (all rules chained)
   */
  getCompositeRotor(): Rotor | null {
    if (this.steps.length === 0) return null;

    let result = this.steps[0].rule.rotor;
    for (let i = 1; i < this.steps.length; i++) {
      result = result.compose(this.steps[i].rule.rotor);
    }

    return result;
  }

  /**
   * Get a human-readable proof trace
   */
  getProofTrace(): string[] {
    return this.steps.map(
      (step, i) =>
        `${i + 1}. ${step.inputConcept} --[${step.rule.name}]--> ${step.outputConcept} (conf: ${(step.confidence * 100).toFixed(1)}%)`
    );
  }

  /**
   * Verify the chain (check that outputs connect to inputs)
   */
  verify(): boolean {
    for (let i = 1; i < this.steps.length; i++) {
      const prev = this.steps[i - 1];
      const curr = this.steps[i];

      // Check that previous output matches current input (approximately)
      const similarity = prev.afterVector.similarity(curr.beforeVector);
      if (similarity < 0.5) {
        return false;
      }
    }
    return true;
  }
}

export default { Rotor, RuleLibrary, InferenceChain };
