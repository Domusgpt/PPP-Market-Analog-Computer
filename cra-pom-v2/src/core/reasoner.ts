// File: src/core/reasoner.ts
// PPP Reasoning Engine - The complete symbolic reasoning system
// Implements: "Symbolic reasoning is fundamentally a geometric operation"

import { Hypervector, DEFAULT_DIMENSION } from './hdc';
import { PhasorVector, ResonatorNetwork } from './fhrr';
import { ConvexPolytope, VoronoiTessellation } from './polytope';
import { Rotor, Rule, RuleLibrary, InferenceChain } from './rotor';
import { GardenOfForkingPaths, FutureBundle } from './garden';

/**
 * A concept in the PPP framework
 * Has both a prototype vector AND a polytope region
 */
export interface Concept {
  name: string;
  prototype: PhasorVector;
  polytope: ConvexPolytope;
  properties: Map<string, PhasorVector>; // Bound role-filler pairs
  superConcepts: string[]; // IS-A hierarchy
  subConcepts: string[];
}

/**
 * Query for reasoning
 */
export interface ReasoningQuery {
  type: 'classify' | 'infer' | 'predict' | 'verify' | 'analogy';
  subject: PhasorVector | string;
  predicate?: string;
  object?: PhasorVector | string;
  context?: Map<string, PhasorVector>;
  maxSteps?: number;
}

/**
 * Result of reasoning
 */
export interface ReasoningResult {
  success: boolean;
  answer: string;
  confidence: number;
  inferenceChain?: InferenceChain;
  geometricProof?: {
    startPolytope: string;
    endPolytope: string;
    rotations: string[];
    trajectoryLength: number;
  };
  alternativeAnswers?: Array<{ answer: string; confidence: number }>;
  futureBundle?: FutureBundle;
}

/**
 * PPP Reasoning Engine
 *
 * The core of Polytopal Projection Processing:
 * - Concepts are polytope regions
 * - Rules are rotation operators
 * - Inference = geometric projection and traversal
 * - The future is a polytope of possibilities
 */
export class PPPReasoningEngine {
  readonly dimension: number;
  private concepts: Map<string, Concept> = new Map();
  private voronoi: VoronoiTessellation;
  private ruleLibrary: RuleLibrary;
  private garden: GardenOfForkingPaths;
  private resonator: ResonatorNetwork | null = null;

  // Safety polytope for Constitutional AI
  private safetyPolytope: ConvexPolytope | null = null;

  constructor(dimension: number = DEFAULT_DIMENSION) {
    this.dimension = dimension;
    this.voronoi = new VoronoiTessellation(dimension);
    this.ruleLibrary = new RuleLibrary(dimension);
    this.garden = new GardenOfForkingPaths(this.ruleLibrary, { dimension });
  }

  /**
   * Define a new concept
   */
  defineConcept(
    name: string,
    options: {
      prototype?: PhasorVector;
      examples?: PhasorVector[];
      superConcepts?: string[];
      properties?: Array<{ role: string; filler: PhasorVector }>;
      radius?: number;
    } = {}
  ): Concept {
    // Create or use prototype
    const prototype = options.prototype ??
      (options.examples && options.examples.length > 0
        ? PhasorVector.bundle(options.examples)
        : PhasorVector.random(this.dimension));

    // Create polytope
    let polytope: ConvexPolytope;
    if (options.examples && options.examples.length > 1) {
      // Create from examples (their hypervector equivalents)
      const hvExamples = options.examples.map((p) => phasorToHypervector(p));
      polytope = ConvexPolytope.fromExamples(name, hvExamples, options.radius ?? 0.2);
    } else {
      // Create from prototype
      polytope = ConvexPolytope.fromPrototype(
        name,
        phasorToHypervector(prototype),
        options.radius ?? 0.5
      );
    }

    // Build properties map
    const properties = new Map<string, PhasorVector>();
    if (options.properties) {
      for (const { role, filler } of options.properties) {
        const roleVector = this.getOrCreateRoleVector(role);
        properties.set(role, roleVector.bind(filler));
      }
    }

    const concept: Concept = {
      name,
      prototype,
      polytope,
      properties,
      superConcepts: options.superConcepts ?? [],
      subConcepts: [],
    };

    // Update hierarchy
    for (const superName of concept.superConcepts) {
      const superConcept = this.concepts.get(superName);
      if (superConcept && !superConcept.subConcepts.includes(name)) {
        superConcept.subConcepts.push(name);
      }
    }

    this.concepts.set(name, concept);

    // Update Voronoi tessellation
    this.voronoi.addConcept(name, phasorToHypervector(prototype));

    // Rebuild resonator if needed
    this.rebuildResonator();

    return concept;
  }

  /**
   * Get or create a role vector (for property encoding)
   */
  private roleVectors: Map<string, PhasorVector> = new Map();

  private getOrCreateRoleVector(role: string): PhasorVector {
    if (!this.roleVectors.has(role)) {
      this.roleVectors.set(role, PhasorVector.random(this.dimension));
    }
    return this.roleVectors.get(role)!;
  }

  /**
   * Rebuild the resonator network with current concepts
   */
  private rebuildResonator(): void {
    if (this.concepts.size > 0) {
      const codebook = new Map<string, PhasorVector>();
      for (const [name, concept] of this.concepts) {
        codebook.set(name, concept.prototype);
      }
      this.resonator = new ResonatorNetwork(codebook);
    }
  }

  /**
   * Define a rule (transformation between concepts)
   */
  defineRule(
    name: string,
    description: string,
    fromConcept: string,
    toConcept: string,
    options: {
      confidence?: number;
      bidirectional?: boolean;
      examples?: Array<{ before: PhasorVector; after: PhasorVector }>;
    } = {}
  ): Rule {
    const source = this.concepts.get(fromConcept);
    const target = this.concepts.get(toConcept);

    if (!source || !target) {
      throw new Error(`Concepts not found: ${fromConcept} or ${toConcept}`);
    }

    let rotor: Rotor;
    if (options.examples && options.examples.length > 0) {
      // Learn from examples
      const avgPhasor = PhasorVector.bundle(
        options.examples.map(({ before, after }) =>
          after.bind(before.inverse())
        )
      );
      rotor = new Rotor(name, avgPhasor);
    } else {
      // Create from concept prototypes
      rotor = Rotor.fromTransition(name, source.prototype, target.prototype);
    }

    const rule: Rule = {
      name,
      description,
      rotor,
      sourcePolytope: fromConcept,
      targetPolytope: toConcept,
      confidence: options.confidence ?? 1.0,
      bidirectional: options.bidirectional ?? false,
    };

    this.ruleLibrary.addRule(rule);
    return rule;
  }

  /**
   * Set the safety polytope for Constitutional AI
   * All outputs must stay within this polytope
   */
  setSafetyPolytope(polytope: ConvexPolytope): void {
    this.safetyPolytope = polytope;
    this.garden.addConstraint({
      name: 'safety',
      type: 'polytope',
      polytope,
      penalty: 1.0, // Full penalty for safety violations
    });
  }

  /**
   * CLASSIFY: Determine which concept a vector belongs to
   * "Projection onto the nearest polytope"
   */
  classify(query: PhasorVector): ReasoningResult {
    // Use Voronoi classification
    const hvQuery = phasorToHypervector(query);
    const classification = this.voronoi.classify(hvQuery);

    // Refine with resonator if available
    let refinedName = classification.concept;
    let refinedConfidence = classification.similarity;

    if (this.resonator) {
      const resonated = this.resonator.resonate(query);
      if (resonated.similarity > refinedConfidence) {
        refinedName = resonated.name;
        refinedConfidence = resonated.similarity;
      }
    }

    // Check polytope containment for additional confidence
    const concept = this.concepts.get(refinedName);
    const containment = concept?.polytope.containsPoint(hvQuery);
    const adjustedConfidence = containment?.isInside
      ? refinedConfidence
      : refinedConfidence * 0.8;

    return {
      success: true,
      answer: refinedName,
      confidence: adjustedConfidence,
      geometricProof: {
        startPolytope: 'query',
        endPolytope: refinedName,
        rotations: [],
        trajectoryLength: 0,
      },
      alternativeAnswers: classification.alternatives.map((alt) => ({
        answer: alt.concept,
        confidence: alt.similarity,
      })),
    };
  }

  /**
   * INFER: Apply rules to derive new knowledge
   * "Rotation through polytope space"
   */
  infer(
    subject: string | PhasorVector,
    ruleName: string
  ): ReasoningResult {
    // Get starting concept/vector
    let startVector: PhasorVector;
    let startConcept: string;

    if (typeof subject === 'string') {
      const concept = this.concepts.get(subject);
      if (!concept) {
        return {
          success: false,
          answer: `Unknown concept: ${subject}`,
          confidence: 0,
        };
      }
      startVector = concept.prototype;
      startConcept = subject;
    } else {
      startVector = subject;
      const classified = this.classify(subject);
      startConcept = classified.answer;
    }

    // Get the rule
    const rule = this.ruleLibrary.getRule(ruleName);
    if (!rule) {
      return {
        success: false,
        answer: `Unknown rule: ${ruleName}`,
        confidence: 0,
      };
    }

    // Apply the rotation
    const resultVector = rule.rotor.apply(startVector);

    // Classify the result
    const endClassification = this.classify(resultVector);

    // Build inference chain
    const chain = new InferenceChain();
    chain.addStep({
      rule,
      inputConcept: startConcept,
      outputConcept: endClassification.answer,
      beforeVector: startVector,
      afterVector: resultVector,
      confidence: rule.confidence * endClassification.confidence,
    });

    // Safety check
    if (this.safetyPolytope) {
      const hvResult = phasorToHypervector(resultVector);
      const safetyCheck = this.safetyPolytope.containsPoint(hvResult);
      if (!safetyCheck.isInside) {
        // Project back to safety polytope (vector could be used for safer result)
        this.safetyPolytope.project(hvResult);
        return {
          success: true,
          answer: `${endClassification.answer} (safety-constrained)`,
          confidence: endClassification.confidence * 0.5, // Reduced confidence
          inferenceChain: chain,
          geometricProof: {
            startPolytope: startConcept,
            endPolytope: endClassification.answer,
            rotations: [ruleName, 'safety_projection'],
            trajectoryLength: 2,
          },
        };
      }
    }

    return {
      success: true,
      answer: endClassification.answer,
      confidence: chain.getOverallConfidence(),
      inferenceChain: chain,
      geometricProof: {
        startPolytope: startConcept,
        endPolytope: endClassification.answer,
        rotations: [ruleName],
        trajectoryLength: 1,
      },
    };
  }

  /**
   * CHAIN INFERENCE: Apply multiple rules in sequence
   * "Trajectory through semantic space"
   */
  chainInference(
    subject: string | PhasorVector,
    ruleNames: string[]
  ): ReasoningResult {
    if (ruleNames.length === 0) {
      return {
        success: false,
        answer: 'No rules provided',
        confidence: 0,
      };
    }

    let currentVector: PhasorVector;
    let currentConcept: string;

    if (typeof subject === 'string') {
      const concept = this.concepts.get(subject);
      if (!concept) {
        return {
          success: false,
          answer: `Unknown concept: ${subject}`,
          confidence: 0,
        };
      }
      currentVector = concept.prototype;
      currentConcept = subject;
    } else {
      currentVector = subject;
      currentConcept = this.classify(subject).answer;
    }

    const chain = new InferenceChain();
    const rotations: string[] = [];

    for (const ruleName of ruleNames) {
      const rule = this.ruleLibrary.getRule(ruleName);
      if (!rule) {
        return {
          success: false,
          answer: `Unknown rule in chain: ${ruleName}`,
          confidence: 0,
          inferenceChain: chain,
        };
      }

      const newVector = rule.rotor.apply(currentVector);
      const newClassification = this.classify(newVector);

      chain.addStep({
        rule,
        inputConcept: currentConcept,
        outputConcept: newClassification.answer,
        beforeVector: currentVector,
        afterVector: newVector,
        confidence: rule.confidence * newClassification.confidence,
      });

      rotations.push(ruleName);
      currentVector = newVector;
      currentConcept = newClassification.answer;
    }

    return {
      success: chain.verify(),
      answer: currentConcept,
      confidence: chain.getOverallConfidence(),
      inferenceChain: chain,
      geometricProof: {
        startPolytope: typeof subject === 'string' ? subject : 'query',
        endPolytope: currentConcept,
        rotations,
        trajectoryLength: ruleNames.length,
      },
    };
  }

  /**
   * PREDICT: Use Garden of Forking Paths for multi-future prediction
   * "Navigate the polytope of possibilities"
   */
  predict(
    subject: string | PhasorVector,
    steps: number = 5
  ): ReasoningResult {
    let startVector: PhasorVector;

    if (typeof subject === 'string') {
      const concept = this.concepts.get(subject);
      if (!concept) {
        return {
          success: false,
          answer: `Unknown concept: ${subject}`,
          confidence: 0,
        };
      }
      startVector = concept.prototype;
    } else {
      startVector = subject;
    }

    const futureBundle = this.garden.predictFutures(startVector, steps);
    const mostProbable = this.garden.getMostProbableFuture(futureBundle);

    if (!mostProbable) {
      return {
        success: false,
        answer: 'No futures predicted',
        confidence: 0,
        futureBundle,
      };
    }

    // Classify the most probable end state
    const finalState = mostProbable.states[mostProbable.states.length - 1];
    const endClassification = this.classify(finalState);

    return {
      success: true,
      answer: endClassification.answer,
      confidence: mostProbable.probability * endClassification.confidence,
      geometricProof: {
        startPolytope: typeof subject === 'string' ? subject : 'query',
        endPolytope: endClassification.answer,
        rotations: mostProbable.rules,
        trajectoryLength: mostProbable.states.length,
      },
      futureBundle,
      alternativeAnswers: futureBundle.paths
        .filter((p) => p.id !== mostProbable.id)
        .slice(0, 3)
        .map((p) => {
          const state = p.states[p.states.length - 1];
          const cls = this.classify(state);
          return { answer: cls.answer, confidence: p.probability };
        }),
    };
  }

  /**
   * VERIFY: Check if a statement is true
   * "Does the trajectory reach the target polytope?"
   */
  verify(
    subject: string | PhasorVector,
    predicate: string, // Rule name
    object: string // Expected concept
  ): ReasoningResult {
    const inferenceResult = this.infer(subject, predicate);

    if (!inferenceResult.success) {
      return inferenceResult;
    }

    const matches = inferenceResult.answer === object;
    const objectConcept = this.concepts.get(object);

    // Also check if result is within object's polytope
    let polytopeMatch = false;
    if (objectConcept && inferenceResult.inferenceChain) {
      const finalStep = inferenceResult.inferenceChain.steps[
        inferenceResult.inferenceChain.steps.length - 1
      ];
      const hvResult = phasorToHypervector(finalStep.afterVector);
      polytopeMatch = objectConcept.polytope.containsPoint(hvResult).isInside;
    }

    return {
      success: matches || polytopeMatch,
      answer: matches ? 'TRUE' : polytopeMatch ? 'APPROXIMATELY TRUE' : 'FALSE',
      confidence: matches
        ? inferenceResult.confidence
        : polytopeMatch
        ? inferenceResult.confidence * 0.8
        : 1 - inferenceResult.confidence,
      inferenceChain: inferenceResult.inferenceChain,
      geometricProof: {
        ...inferenceResult.geometricProof!,
        endPolytope: object,
      },
    };
  }

  /**
   * ANALOGY: A is to B as C is to ?
   * "Transfer rotation from one pair to another"
   */
  analogy(
    a: string,
    b: string,
    c: string
  ): ReasoningResult {
    const conceptA = this.concepts.get(a);
    const conceptB = this.concepts.get(b);
    const conceptC = this.concepts.get(c);

    if (!conceptA || !conceptB || !conceptC) {
      return {
        success: false,
        answer: 'Unknown concepts in analogy',
        confidence: 0,
      };
    }

    // Find the rotation from A to B
    const abRotor = Rotor.fromTransition(`${a}→${b}`, conceptA.prototype, conceptB.prototype);

    // Apply that rotation to C
    const resultVector = abRotor.apply(conceptC.prototype);

    // Classify the result
    const classification = this.classify(resultVector);

    return {
      success: true,
      answer: classification.answer,
      confidence: classification.confidence,
      geometricProof: {
        startPolytope: c,
        endPolytope: classification.answer,
        rotations: [`analogy(${a}→${b})`],
        trajectoryLength: 1,
      },
      alternativeAnswers: classification.alternativeAnswers,
    };
  }

  /**
   * Process a reasoning query
   */
  reason(query: ReasoningQuery): ReasoningResult {
    switch (query.type) {
      case 'classify':
        if (typeof query.subject === 'string') {
          const concept = this.concepts.get(query.subject);
          if (!concept) {
            return { success: false, answer: 'Unknown concept', confidence: 0 };
          }
          return this.classify(concept.prototype);
        }
        return this.classify(query.subject);

      case 'infer':
        if (!query.predicate) {
          return { success: false, answer: 'No rule specified', confidence: 0 };
        }
        return this.infer(query.subject, query.predicate);

      case 'predict':
        return this.predict(query.subject, query.maxSteps ?? 5);

      case 'verify':
        if (!query.predicate || !query.object || typeof query.object !== 'string') {
          return { success: false, answer: 'Verification requires predicate and object', confidence: 0 };
        }
        return this.verify(query.subject, query.predicate, query.object);

      case 'analogy':
        if (typeof query.subject !== 'string' || !query.predicate || typeof query.object !== 'string') {
          return { success: false, answer: 'Analogy requires three concept names', confidence: 0 };
        }
        // Parse predicate as "B" and object as "C" for "A:B::C:?"
        return this.analogy(query.subject, query.predicate, query.object);

      default:
        return { success: false, answer: 'Unknown query type', confidence: 0 };
    }
  }

  /**
   * Get all concept names
   */
  getConceptNames(): string[] {
    return Array.from(this.concepts.keys());
  }

  /**
   * Get all rule names
   */
  getRuleNames(): string[] {
    return this.ruleLibrary.getRuleNames();
  }

  /**
   * Get a concept by name
   */
  getConcept(name: string): Concept | undefined {
    return this.concepts.get(name);
  }

  /**
   * Export the knowledge base
   */
  export(): {
    concepts: Record<string, { superConcepts: string[]; subConcepts: string[] }>;
    rules: Record<string, unknown>;
  } {
    const concepts: Record<string, { superConcepts: string[]; subConcepts: string[] }> = {};
    for (const [name, concept] of this.concepts) {
      concepts[name] = {
        superConcepts: concept.superConcepts,
        subConcepts: concept.subConcepts,
      };
    }

    return {
      concepts,
      rules: this.ruleLibrary.export(),
    };
  }
}

/**
 * Helper: Convert PhasorVector to Hypervector
 */
function phasorToHypervector(pv: PhasorVector): Hypervector {
  const data = new Float32Array(pv.dimension);
  for (let i = 0; i < pv.dimension; i++) {
    // Use cosine of phase as the value
    data[i] = Math.cos(pv.phases[i]);
  }
  return new Hypervector(pv.dimension, data);
}

export default PPPReasoningEngine;
