// File: src/core/state.ts
// PPP System State Definition for Machine Consumption
// Provides serialization, deserialization, and a machine-readable API

import { Hypervector, DEFAULT_DIMENSION } from './hdc';
import { PhasorVector } from './fhrr';
import { ConvexPolytope, HalfSpace } from './polytope';
import { Rotor, Rule, InferenceChain } from './rotor';
import { FutureBundle, FutureConstraint } from './garden';
import PPPReasoningEngine, { ReasoningResult } from './reasoner';

// ============================================================================
// JSON Schema Type Definitions
// These define the serialized format for machine consumption
// ============================================================================

/**
 * Serialized Hypervector
 */
export interface SerializedHypervector {
  type: 'hypervector';
  dimension: number;
  data: number[]; // Float32Array -> number[]
  encoding: 'dense' | 'sparse';
  sparseIndices?: number[]; // For sparse encoding
  sparseValues?: number[];
}

/**
 * Serialized PhasorVector (FHRR)
 */
export interface SerializedPhasorVector {
  type: 'phasor';
  dimension: number;
  phases: number[]; // Float32Array -> number[]
  coherence: number; // Pre-computed coherence metric
}

/**
 * Serialized Convex Polytope
 */
export interface SerializedPolytope {
  type: 'polytope';
  name: string;
  dimension: number;
  constraints: Array<{
    normal: number[];
    offset: number;
    name?: string;
  }>;
  centroid: number[];
}

/**
 * Serialized Rotor (Rule transformation)
 */
export interface SerializedRotor {
  type: 'rotor';
  name: string;
  dimension: number;
  phasor: SerializedPhasorVector;
  magnitude: number;
  metadata: Record<string, unknown>;
}

/**
 * Serialized Rule
 */
export interface SerializedRule {
  name: string;
  description: string;
  rotor: SerializedRotor;
  sourcePolytope?: string;
  targetPolytope?: string;
  confidence: number;
  bidirectional: boolean;
}

/**
 * Serialized Concept
 */
export interface SerializedConcept {
  name: string;
  prototype: SerializedPhasorVector;
  polytope: SerializedPolytope;
  properties: Record<string, SerializedPhasorVector>;
  superConcepts: string[];
  subConcepts: string[];
}

/**
 * Serialized Inference Chain
 */
export interface SerializedInferenceChain {
  steps: Array<{
    ruleName: string;
    inputConcept: string;
    outputConcept: string;
    confidence: number;
  }>;
  overallConfidence: number;
  proofTrace: string[];
}

/**
 * Serialized Future Path
 */
export interface SerializedFuturePath {
  id: string;
  probability: number;
  stateCount: number;
  rules: string[];
  terminated: boolean;
  terminationReason?: string;
  finalStateCoherence: number;
}

/**
 * Serialized Future Bundle
 */
export interface SerializedFutureBundle {
  pathCount: number;
  forkCount: number;
  totalProbability: number;
  paths: SerializedFuturePath[];
  hasConvergence: boolean;
  futureSpread: number;
}

/**
 * Complete PPP System State
 * This is the full serializable state for machine consumption
 */
export interface PPPSystemState {
  // Schema metadata
  schema: {
    version: string;
    type: 'ppp-system-state';
    created: string;
    checksum?: string;
  };

  // Configuration
  config: {
    dimension: number;
    maxPaths: number;
    probabilityThreshold: number;
    maxDepth: number;
  };

  // Knowledge Base
  knowledge: {
    concepts: Record<string, SerializedConcept>;
    rules: Record<string, SerializedRule>;
    roleVectors: Record<string, SerializedPhasorVector>;
  };

  // Current Context (optional, for session state)
  context?: {
    currentQuery?: SerializedPhasorVector;
    lastResult?: SerializedReasoningResult;
    activeInferenceChain?: SerializedInferenceChain;
    futurePrediction?: SerializedFutureBundle;
  };

  // Constraints
  constraints: {
    safety?: SerializedPolytope;
    futureConstraints: Array<{
      name: string;
      type: 'polytope' | 'similarity' | 'energy';
      polytope?: SerializedPolytope;
      targetVector?: SerializedPhasorVector;
      threshold?: number;
      penalty: number;
    }>;
  };

  // Statistics
  statistics: {
    conceptCount: number;
    ruleCount: number;
    totalInferences: number;
    averageConfidence: number;
  };
}

/**
 * Serialized Reasoning Result
 */
export interface SerializedReasoningResult {
  success: boolean;
  answer: string;
  confidence: number;
  inferenceChain?: SerializedInferenceChain;
  geometricProof?: {
    startPolytope: string;
    endPolytope: string;
    rotations: string[];
    trajectoryLength: number;
  };
  alternativeAnswers?: Array<{ answer: string; confidence: number }>;
  futureBundle?: SerializedFutureBundle;
}

/**
 * Machine API Request
 */
export interface MachineAPIRequest {
  action:
    | 'define_concept'
    | 'define_rule'
    | 'classify'
    | 'infer'
    | 'predict'
    | 'verify'
    | 'analogy'
    | 'chain_inference'
    | 'export_state'
    | 'import_state'
    | 'get_concepts'
    | 'get_rules'
    | 'reset';

  params?: {
    // For define_concept
    conceptName?: string;
    prototype?: SerializedPhasorVector;
    examples?: SerializedPhasorVector[];
    superConcepts?: string[];
    properties?: Array<{ role: string; filler: SerializedPhasorVector }>;
    radius?: number;

    // For define_rule
    ruleName?: string;
    description?: string;
    fromConcept?: string;
    toConcept?: string;
    confidence?: number;
    bidirectional?: boolean;

    // For classify, infer, predict
    subject?: string | SerializedPhasorVector;

    // For infer
    predicate?: string;

    // For verify
    object?: string;

    // For analogy
    a?: string;
    b?: string;
    c?: string;

    // For chain_inference
    ruleNames?: string[];

    // For predict
    steps?: number;

    // For import_state
    state?: PPPSystemState;
  };
}

/**
 * Machine API Response
 */
export interface MachineAPIResponse {
  success: boolean;
  action: string;
  timestamp: string;

  // Response data (varies by action)
  data?: {
    result?: SerializedReasoningResult;
    state?: PPPSystemState;
    concepts?: string[];
    rules?: string[];
    concept?: SerializedConcept;
    rule?: SerializedRule;
  };

  // Error information
  error?: {
    code: string;
    message: string;
    details?: unknown;
  };
}

// ============================================================================
// Serialization Functions
// ============================================================================

/**
 * Serialize a Hypervector to JSON-compatible format
 */
export function serializeHypervector(hv: Hypervector): SerializedHypervector {
  // Check if sparse representation is more efficient
  const nonZeroIndices: number[] = [];
  const nonZeroValues: number[] = [];

  for (let i = 0; i < hv.dimension; i++) {
    if (Math.abs(hv.data[i]) > 1e-6) {
      nonZeroIndices.push(i);
      nonZeroValues.push(hv.data[i]);
    }
  }

  const useSparse = nonZeroIndices.length < hv.dimension * 0.1;

  if (useSparse) {
    return {
      type: 'hypervector',
      dimension: hv.dimension,
      data: [],
      encoding: 'sparse',
      sparseIndices: nonZeroIndices,
      sparseValues: nonZeroValues,
    };
  }

  return {
    type: 'hypervector',
    dimension: hv.dimension,
    data: Array.from(hv.data),
    encoding: 'dense',
  };
}

/**
 * Deserialize a Hypervector from JSON format
 */
export function deserializeHypervector(data: SerializedHypervector): Hypervector {
  const arr = new Float32Array(data.dimension);

  if (data.encoding === 'sparse' && data.sparseIndices && data.sparseValues) {
    for (let i = 0; i < data.sparseIndices.length; i++) {
      arr[data.sparseIndices[i]] = data.sparseValues[i];
    }
  } else {
    for (let i = 0; i < data.data.length; i++) {
      arr[i] = data.data[i];
    }
  }

  return new Hypervector(data.dimension, arr);
}

/**
 * Serialize a PhasorVector to JSON-compatible format
 */
export function serializePhasorVector(pv: PhasorVector): SerializedPhasorVector {
  return {
    type: 'phasor',
    dimension: pv.dimension,
    phases: Array.from(pv.phases),
    coherence: pv.coherence(),
  };
}

/**
 * Deserialize a PhasorVector from JSON format
 */
export function deserializePhasorVector(data: SerializedPhasorVector): PhasorVector {
  return new PhasorVector(data.dimension, Float32Array.from(data.phases));
}

/**
 * Serialize a ConvexPolytope to JSON-compatible format
 */
export function serializePolytope(polytope: ConvexPolytope): SerializedPolytope {
  return {
    type: 'polytope',
    name: polytope.name,
    dimension: polytope.dimension,
    constraints: polytope.constraints.map((c) => ({
      normal: Array.from(c.normal),
      offset: c.offset,
      name: c.name,
    })),
    centroid: Array.from(polytope.centroid),
  };
}

/**
 * Deserialize a ConvexPolytope from JSON format
 */
export function deserializePolytope(data: SerializedPolytope): ConvexPolytope {
  const constraints: HalfSpace[] = data.constraints.map((c) => ({
    normal: Float32Array.from(c.normal),
    offset: c.offset,
    name: c.name,
  }));

  return new ConvexPolytope(
    data.name,
    data.dimension,
    constraints,
    Float32Array.from(data.centroid)
  );
}

/**
 * Serialize a Rotor to JSON-compatible format
 */
export function serializeRotor(rotor: Rotor): SerializedRotor {
  return {
    type: 'rotor',
    name: rotor.name,
    dimension: rotor.dimension,
    phasor: serializePhasorVector(rotor.phasor),
    magnitude: rotor.magnitude(),
    metadata: rotor.metadata,
  };
}

/**
 * Deserialize a Rotor from JSON format
 */
export function deserializeRotor(data: SerializedRotor): Rotor {
  return new Rotor(
    data.name,
    deserializePhasorVector(data.phasor),
    data.metadata
  );
}

/**
 * Serialize a Rule to JSON-compatible format
 */
export function serializeRule(rule: Rule): SerializedRule {
  return {
    name: rule.name,
    description: rule.description,
    rotor: serializeRotor(rule.rotor),
    sourcePolytope: rule.sourcePolytope,
    targetPolytope: rule.targetPolytope,
    confidence: rule.confidence,
    bidirectional: rule.bidirectional,
  };
}

/**
 * Deserialize a Rule from JSON format
 */
export function deserializeRule(data: SerializedRule): Rule {
  return {
    name: data.name,
    description: data.description,
    rotor: deserializeRotor(data.rotor),
    sourcePolytope: data.sourcePolytope,
    targetPolytope: data.targetPolytope,
    confidence: data.confidence,
    bidirectional: data.bidirectional,
  };
}

/**
 * Serialize an InferenceChain to JSON-compatible format
 */
export function serializeInferenceChain(chain: InferenceChain): SerializedInferenceChain {
  return {
    steps: chain.steps.map((step) => ({
      ruleName: step.rule.name,
      inputConcept: step.inputConcept,
      outputConcept: step.outputConcept,
      confidence: step.confidence,
    })),
    overallConfidence: chain.getOverallConfidence(),
    proofTrace: chain.getProofTrace(),
  };
}

/**
 * Serialize a FutureBundle to JSON-compatible format
 */
export function serializeFutureBundle(bundle: FutureBundle): SerializedFutureBundle {
  return {
    pathCount: bundle.paths.length,
    forkCount: bundle.forks.length,
    totalProbability: bundle.totalProbability,
    paths: bundle.paths.map((path) => ({
      id: path.id,
      probability: path.probability,
      stateCount: path.states.length,
      rules: path.rules,
      terminated: path.terminated,
      terminationReason: path.terminationReason,
      finalStateCoherence: path.states.length > 0
        ? path.states[path.states.length - 1].coherence()
        : 0,
    })),
    hasConvergence: bundle.convergencePoint !== undefined,
    futureSpread: bundle.paths.length > 0
      ? 1 - bundle.paths.reduce((sum, p) => sum + p.probability * p.probability, 0)
      : 0,
  };
}

/**
 * Serialize a ReasoningResult to JSON-compatible format
 */
export function serializeReasoningResult(result: ReasoningResult): SerializedReasoningResult {
  return {
    success: result.success,
    answer: result.answer,
    confidence: result.confidence,
    inferenceChain: result.inferenceChain
      ? serializeInferenceChain(result.inferenceChain)
      : undefined,
    geometricProof: result.geometricProof,
    alternativeAnswers: result.alternativeAnswers,
    futureBundle: result.futureBundle
      ? serializeFutureBundle(result.futureBundle)
      : undefined,
  };
}

// ============================================================================
// PPP State Manager - Machine-Readable Interface
// ============================================================================

/**
 * PPPStateManager provides a machine-readable interface to the PPP system
 * This is the primary interface for machine consumption
 */
export class PPPStateManager {
  private engine: PPPReasoningEngine;
  private totalInferences: number = 0;
  private confidenceSum: number = 0;
  private futureConstraints: FutureConstraint[] = [];

  constructor(dimension: number = DEFAULT_DIMENSION) {
    this.engine = new PPPReasoningEngine(dimension);
  }

  /**
   * Process a machine API request
   * This is the main entry point for machine interaction
   */
  processRequest(request: MachineAPIRequest): MachineAPIResponse {
    const timestamp = new Date().toISOString();

    try {
      switch (request.action) {
        case 'define_concept':
          return this.handleDefineConcept(request, timestamp);
        case 'define_rule':
          return this.handleDefineRule(request, timestamp);
        case 'classify':
          return this.handleClassify(request, timestamp);
        case 'infer':
          return this.handleInfer(request, timestamp);
        case 'predict':
          return this.handlePredict(request, timestamp);
        case 'verify':
          return this.handleVerify(request, timestamp);
        case 'analogy':
          return this.handleAnalogy(request, timestamp);
        case 'chain_inference':
          return this.handleChainInference(request, timestamp);
        case 'export_state':
          return this.handleExportState(timestamp);
        case 'import_state':
          return this.handleImportState(request, timestamp);
        case 'get_concepts':
          return this.handleGetConcepts(timestamp);
        case 'get_rules':
          return this.handleGetRules(timestamp);
        case 'reset':
          return this.handleReset(timestamp);
        default:
          return {
            success: false,
            action: request.action,
            timestamp,
            error: {
              code: 'UNKNOWN_ACTION',
              message: `Unknown action: ${request.action}`,
            },
          };
      }
    } catch (err) {
      return {
        success: false,
        action: request.action,
        timestamp,
        error: {
          code: 'EXECUTION_ERROR',
          message: err instanceof Error ? err.message : 'Unknown error',
          details: err,
        },
      };
    }
  }

  /**
   * Export the complete system state
   */
  exportState(): PPPSystemState {
    const concepts: Record<string, SerializedConcept> = {};
    const rules: Record<string, SerializedRule> = {};

    // Serialize concepts
    for (const name of this.engine.getConceptNames()) {
      const concept = this.engine.getConcept(name);
      if (concept) {
        const properties: Record<string, SerializedPhasorVector> = {};
        for (const [role, pv] of concept.properties) {
          properties[role] = serializePhasorVector(pv);
        }

        concepts[name] = {
          name: concept.name,
          prototype: serializePhasorVector(concept.prototype),
          polytope: serializePolytope(concept.polytope),
          properties,
          superConcepts: concept.superConcepts,
          subConcepts: concept.subConcepts,
        };
      }
    }

    // Serialize rules (via export method)
    const exportedKB = this.engine.export();
    for (const [name, ruleData] of Object.entries(exportedKB.rules)) {
      const rd = ruleData as Record<string, unknown>;
      rules[name] = {
        name,
        description: rd.description as string ?? '',
        rotor: {
          type: 'rotor',
          name,
          dimension: this.engine.dimension,
          phasor: {
            type: 'phasor',
            dimension: this.engine.dimension,
            phases: [],
            coherence: 0,
          },
          magnitude: rd.magnitude as number ?? 0,
          metadata: {},
        },
        sourcePolytope: rd.sourcePolytope as string | undefined,
        targetPolytope: rd.targetPolytope as string | undefined,
        confidence: rd.confidence as number ?? 1,
        bidirectional: rd.bidirectional as boolean ?? false,
      };
    }

    // Compute statistics
    const avgConfidence = this.totalInferences > 0
      ? this.confidenceSum / this.totalInferences
      : 0;

    return {
      schema: {
        version: '1.0.0',
        type: 'ppp-system-state',
        created: new Date().toISOString(),
      },
      config: {
        dimension: this.engine.dimension,
        maxPaths: 8,
        probabilityThreshold: 0.01,
        maxDepth: 10,
      },
      knowledge: {
        concepts,
        rules,
        roleVectors: {}, // Could be extended to serialize role vectors
      },
      constraints: {
        futureConstraints: this.futureConstraints.map((c) => ({
          name: c.name,
          type: c.type,
          polytope: c.polytope ? serializePolytope(c.polytope) : undefined,
          targetVector: c.targetVector ? serializePhasorVector(c.targetVector) : undefined,
          threshold: c.threshold,
          penalty: c.penalty,
        })),
      },
      statistics: {
        conceptCount: Object.keys(concepts).length,
        ruleCount: Object.keys(rules).length,
        totalInferences: this.totalInferences,
        averageConfidence: avgConfidence,
      },
    };
  }

  /**
   * Import system state from serialized format
   */
  importState(state: PPPSystemState): void {
    // Reset engine
    this.engine = new PPPReasoningEngine(state.config.dimension);

    // Import concepts
    for (const [, conceptData] of Object.entries(state.knowledge.concepts)) {
      this.engine.defineConcept(conceptData.name, {
        prototype: deserializePhasorVector(conceptData.prototype),
        superConcepts: conceptData.superConcepts,
        radius: 0.5, // Default radius
      });
    }

    // Import rules
    for (const [, ruleData] of Object.entries(state.knowledge.rules)) {
      if (ruleData.sourcePolytope && ruleData.targetPolytope) {
        this.engine.defineRule(
          ruleData.name,
          ruleData.description,
          ruleData.sourcePolytope,
          ruleData.targetPolytope,
          {
            confidence: ruleData.confidence,
            bidirectional: ruleData.bidirectional,
          }
        );
      }
    }

    // Import statistics
    this.totalInferences = state.statistics.totalInferences;
    this.confidenceSum = state.statistics.averageConfidence * state.statistics.totalInferences;
  }

  /**
   * Get the underlying reasoning engine
   */
  getEngine(): PPPReasoningEngine {
    return this.engine;
  }

  // ============================================================================
  // Private Request Handlers
  // ============================================================================

  private handleDefineConcept(
    request: MachineAPIRequest,
    timestamp: string
  ): MachineAPIResponse {
    const { conceptName, prototype, examples, superConcepts, properties, radius } = request.params ?? {};

    if (!conceptName) {
      return {
        success: false,
        action: 'define_concept',
        timestamp,
        error: { code: 'MISSING_PARAM', message: 'conceptName is required' },
      };
    }

    const options: Parameters<PPPReasoningEngine['defineConcept']>[1] = {
      superConcepts,
      radius,
    };

    if (prototype) {
      options.prototype = deserializePhasorVector(prototype);
    }

    if (examples) {
      options.examples = examples.map(deserializePhasorVector);
    }

    if (properties) {
      options.properties = properties.map((p) => ({
        role: p.role,
        filler: deserializePhasorVector(p.filler),
      }));
    }

    const concept = this.engine.defineConcept(conceptName, options);

    const props: Record<string, SerializedPhasorVector> = {};
    for (const [role, pv] of concept.properties) {
      props[role] = serializePhasorVector(pv);
    }

    return {
      success: true,
      action: 'define_concept',
      timestamp,
      data: {
        concept: {
          name: concept.name,
          prototype: serializePhasorVector(concept.prototype),
          polytope: serializePolytope(concept.polytope),
          properties: props,
          superConcepts: concept.superConcepts,
          subConcepts: concept.subConcepts,
        },
      },
    };
  }

  private handleDefineRule(
    request: MachineAPIRequest,
    timestamp: string
  ): MachineAPIResponse {
    const { ruleName, description, fromConcept, toConcept, confidence, bidirectional } = request.params ?? {};

    if (!ruleName || !description || !fromConcept || !toConcept) {
      return {
        success: false,
        action: 'define_rule',
        timestamp,
        error: {
          code: 'MISSING_PARAM',
          message: 'ruleName, description, fromConcept, and toConcept are required'
        },
      };
    }

    const rule = this.engine.defineRule(ruleName, description, fromConcept, toConcept, {
      confidence,
      bidirectional,
    });

    return {
      success: true,
      action: 'define_rule',
      timestamp,
      data: {
        rule: serializeRule(rule),
      },
    };
  }

  private handleClassify(
    request: MachineAPIRequest,
    timestamp: string
  ): MachineAPIResponse {
    const { subject } = request.params ?? {};

    if (!subject) {
      return {
        success: false,
        action: 'classify',
        timestamp,
        error: { code: 'MISSING_PARAM', message: 'subject is required' },
      };
    }

    const query = typeof subject === 'string'
      ? this.engine.getConcept(subject)?.prototype
      : deserializePhasorVector(subject);

    if (!query) {
      return {
        success: false,
        action: 'classify',
        timestamp,
        error: { code: 'INVALID_SUBJECT', message: 'Could not resolve subject' },
      };
    }

    const result = this.engine.classify(query);
    this.recordInference(result.confidence);

    return {
      success: true,
      action: 'classify',
      timestamp,
      data: {
        result: serializeReasoningResult(result),
      },
    };
  }

  private handleInfer(
    request: MachineAPIRequest,
    timestamp: string
  ): MachineAPIResponse {
    const { subject, predicate } = request.params ?? {};

    if (!subject || !predicate) {
      return {
        success: false,
        action: 'infer',
        timestamp,
        error: { code: 'MISSING_PARAM', message: 'subject and predicate are required' },
      };
    }

    const subjectValue = typeof subject === 'string'
      ? subject
      : deserializePhasorVector(subject);

    const result = this.engine.infer(subjectValue, predicate);
    this.recordInference(result.confidence);

    return {
      success: true,
      action: 'infer',
      timestamp,
      data: {
        result: serializeReasoningResult(result),
      },
    };
  }

  private handlePredict(
    request: MachineAPIRequest,
    timestamp: string
  ): MachineAPIResponse {
    const { subject, steps } = request.params ?? {};

    if (!subject) {
      return {
        success: false,
        action: 'predict',
        timestamp,
        error: { code: 'MISSING_PARAM', message: 'subject is required' },
      };
    }

    const subjectValue = typeof subject === 'string'
      ? subject
      : deserializePhasorVector(subject);

    const result = this.engine.predict(subjectValue, steps ?? 5);
    this.recordInference(result.confidence);

    return {
      success: true,
      action: 'predict',
      timestamp,
      data: {
        result: serializeReasoningResult(result),
      },
    };
  }

  private handleVerify(
    request: MachineAPIRequest,
    timestamp: string
  ): MachineAPIResponse {
    const { subject, predicate, object } = request.params ?? {};

    if (!subject || !predicate || !object) {
      return {
        success: false,
        action: 'verify',
        timestamp,
        error: { code: 'MISSING_PARAM', message: 'subject, predicate, and object are required' },
      };
    }

    const subjectValue = typeof subject === 'string'
      ? subject
      : deserializePhasorVector(subject);

    const result = this.engine.verify(subjectValue, predicate, object);
    this.recordInference(result.confidence);

    return {
      success: true,
      action: 'verify',
      timestamp,
      data: {
        result: serializeReasoningResult(result),
      },
    };
  }

  private handleAnalogy(
    request: MachineAPIRequest,
    timestamp: string
  ): MachineAPIResponse {
    const { a, b, c } = request.params ?? {};

    if (!a || !b || !c) {
      return {
        success: false,
        action: 'analogy',
        timestamp,
        error: { code: 'MISSING_PARAM', message: 'a, b, and c are required' },
      };
    }

    const result = this.engine.analogy(a, b, c);
    this.recordInference(result.confidence);

    return {
      success: true,
      action: 'analogy',
      timestamp,
      data: {
        result: serializeReasoningResult(result),
      },
    };
  }

  private handleChainInference(
    request: MachineAPIRequest,
    timestamp: string
  ): MachineAPIResponse {
    const { subject, ruleNames } = request.params ?? {};

    if (!subject || !ruleNames) {
      return {
        success: false,
        action: 'chain_inference',
        timestamp,
        error: { code: 'MISSING_PARAM', message: 'subject and ruleNames are required' },
      };
    }

    const subjectValue = typeof subject === 'string'
      ? subject
      : deserializePhasorVector(subject);

    const result = this.engine.chainInference(subjectValue, ruleNames);
    this.recordInference(result.confidence);

    return {
      success: true,
      action: 'chain_inference',
      timestamp,
      data: {
        result: serializeReasoningResult(result),
      },
    };
  }

  private handleExportState(timestamp: string): MachineAPIResponse {
    return {
      success: true,
      action: 'export_state',
      timestamp,
      data: {
        state: this.exportState(),
      },
    };
  }

  private handleImportState(
    request: MachineAPIRequest,
    timestamp: string
  ): MachineAPIResponse {
    const { state } = request.params ?? {};

    if (!state) {
      return {
        success: false,
        action: 'import_state',
        timestamp,
        error: { code: 'MISSING_PARAM', message: 'state is required' },
      };
    }

    this.importState(state);

    return {
      success: true,
      action: 'import_state',
      timestamp,
    };
  }

  private handleGetConcepts(timestamp: string): MachineAPIResponse {
    return {
      success: true,
      action: 'get_concepts',
      timestamp,
      data: {
        concepts: this.engine.getConceptNames(),
      },
    };
  }

  private handleGetRules(timestamp: string): MachineAPIResponse {
    return {
      success: true,
      action: 'get_rules',
      timestamp,
      data: {
        rules: this.engine.getRuleNames(),
      },
    };
  }

  private handleReset(timestamp: string): MachineAPIResponse {
    this.engine = new PPPReasoningEngine(this.engine.dimension);
    this.totalInferences = 0;
    this.confidenceSum = 0;
    this.futureConstraints = [];

    return {
      success: true,
      action: 'reset',
      timestamp,
    };
  }

  private recordInference(confidence: number): void {
    this.totalInferences++;
    this.confidenceSum += confidence;
  }
}

// ============================================================================
// JSON Schema Definition (for external validation)
// ============================================================================

/**
 * JSON Schema for PPPSystemState
 * Can be used for external validation tools
 */
export const PPP_SYSTEM_STATE_SCHEMA = {
  $schema: 'http://json-schema.org/draft-07/schema#',
  $id: 'https://ppp.schema/system-state/v1',
  title: 'PPP System State',
  description: 'Complete serialized state of a PPP Reasoning Engine',
  type: 'object',
  required: ['schema', 'config', 'knowledge', 'constraints', 'statistics'],
  properties: {
    schema: {
      type: 'object',
      required: ['version', 'type', 'created'],
      properties: {
        version: { type: 'string', pattern: '^\\d+\\.\\d+\\.\\d+$' },
        type: { const: 'ppp-system-state' },
        created: { type: 'string', format: 'date-time' },
        checksum: { type: 'string' },
      },
    },
    config: {
      type: 'object',
      required: ['dimension', 'maxPaths', 'probabilityThreshold', 'maxDepth'],
      properties: {
        dimension: { type: 'integer', minimum: 1 },
        maxPaths: { type: 'integer', minimum: 1 },
        probabilityThreshold: { type: 'number', minimum: 0, maximum: 1 },
        maxDepth: { type: 'integer', minimum: 1 },
      },
    },
    knowledge: {
      type: 'object',
      required: ['concepts', 'rules', 'roleVectors'],
      properties: {
        concepts: {
          type: 'object',
          additionalProperties: { $ref: '#/definitions/SerializedConcept' },
        },
        rules: {
          type: 'object',
          additionalProperties: { $ref: '#/definitions/SerializedRule' },
        },
        roleVectors: {
          type: 'object',
          additionalProperties: { $ref: '#/definitions/SerializedPhasorVector' },
        },
      },
    },
    constraints: {
      type: 'object',
      properties: {
        safety: { $ref: '#/definitions/SerializedPolytope' },
        futureConstraints: {
          type: 'array',
          items: { $ref: '#/definitions/FutureConstraint' },
        },
      },
    },
    statistics: {
      type: 'object',
      required: ['conceptCount', 'ruleCount', 'totalInferences', 'averageConfidence'],
      properties: {
        conceptCount: { type: 'integer', minimum: 0 },
        ruleCount: { type: 'integer', minimum: 0 },
        totalInferences: { type: 'integer', minimum: 0 },
        averageConfidence: { type: 'number', minimum: 0, maximum: 1 },
      },
    },
  },
  definitions: {
    SerializedPhasorVector: {
      type: 'object',
      required: ['type', 'dimension', 'phases', 'coherence'],
      properties: {
        type: { const: 'phasor' },
        dimension: { type: 'integer', minimum: 1 },
        phases: { type: 'array', items: { type: 'number' } },
        coherence: { type: 'number', minimum: 0, maximum: 1 },
      },
    },
    SerializedPolytope: {
      type: 'object',
      required: ['type', 'name', 'dimension', 'constraints', 'centroid'],
      properties: {
        type: { const: 'polytope' },
        name: { type: 'string' },
        dimension: { type: 'integer', minimum: 1 },
        constraints: {
          type: 'array',
          items: {
            type: 'object',
            required: ['normal', 'offset'],
            properties: {
              normal: { type: 'array', items: { type: 'number' } },
              offset: { type: 'number' },
              name: { type: 'string' },
            },
          },
        },
        centroid: { type: 'array', items: { type: 'number' } },
      },
    },
    SerializedConcept: {
      type: 'object',
      required: ['name', 'prototype', 'polytope', 'properties', 'superConcepts', 'subConcepts'],
      properties: {
        name: { type: 'string' },
        prototype: { $ref: '#/definitions/SerializedPhasorVector' },
        polytope: { $ref: '#/definitions/SerializedPolytope' },
        properties: {
          type: 'object',
          additionalProperties: { $ref: '#/definitions/SerializedPhasorVector' },
        },
        superConcepts: { type: 'array', items: { type: 'string' } },
        subConcepts: { type: 'array', items: { type: 'string' } },
      },
    },
    SerializedRule: {
      type: 'object',
      required: ['name', 'description', 'rotor', 'confidence', 'bidirectional'],
      properties: {
        name: { type: 'string' },
        description: { type: 'string' },
        rotor: { $ref: '#/definitions/SerializedRotor' },
        sourcePolytope: { type: 'string' },
        targetPolytope: { type: 'string' },
        confidence: { type: 'number', minimum: 0, maximum: 1 },
        bidirectional: { type: 'boolean' },
      },
    },
    SerializedRotor: {
      type: 'object',
      required: ['type', 'name', 'dimension', 'phasor', 'magnitude', 'metadata'],
      properties: {
        type: { const: 'rotor' },
        name: { type: 'string' },
        dimension: { type: 'integer', minimum: 1 },
        phasor: { $ref: '#/definitions/SerializedPhasorVector' },
        magnitude: { type: 'number' },
        metadata: { type: 'object' },
      },
    },
    FutureConstraint: {
      type: 'object',
      required: ['name', 'type', 'penalty'],
      properties: {
        name: { type: 'string' },
        type: { enum: ['polytope', 'similarity', 'energy'] },
        polytope: { $ref: '#/definitions/SerializedPolytope' },
        targetVector: { $ref: '#/definitions/SerializedPhasorVector' },
        threshold: { type: 'number' },
        penalty: { type: 'number', minimum: 0, maximum: 1 },
      },
    },
  },
} as const;

export default PPPStateManager;
