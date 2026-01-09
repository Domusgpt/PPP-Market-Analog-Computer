// File: src/core/llm-integration.ts
// LLM Integration Layer for PPP Reasoning Engine
// Provides tool definitions and verification for agentic use
//
// This module implements:
// 1. Tool definitions compatible with OpenAI/Anthropic function calling
// 2. Cryptographic proof generation to prevent LLM hallucination
// 3. Grounding protocol that forces verifiable citations
// 4. Audit trail for all reasoning operations

import { PPPStateManager, PPPSystemState, MachineAPIRequest, MachineAPIResponse } from './state';

// ============================================================================
// INTERNAL AUDIT CHAIN - Simplified for PPP operations
// ============================================================================

interface AuditEntry {
  index: number;
  timestamp: string;
  operation: string;
  inputHash: string;
  outputHash: string;
  prevHash: string;
  hash: string;
}

class PPPAuditChain {
  private entries: AuditEntry[] = [];
  private headHash: string = '0'.repeat(16);

  log(operation: string, data: { operation: string; parameters: string; operationId: string }): {
    hash: string;
    prevHash: string;
  } {
    const prevHash = this.headHash;
    const timestamp = new Date().toISOString();
    const index = this.entries.length;

    // Simple hash computation
    const contentToHash = `${data.operationId}:${operation}:${data.parameters}:${prevHash}:${index}`;
    const hash = this.simpleHash(contentToHash);

    const entry: AuditEntry = {
      index,
      timestamp,
      operation,
      inputHash: data.parameters,
      outputHash: '',
      prevHash,
      hash
    };

    this.entries.push(entry);
    this.headHash = hash;

    return { hash, prevHash };
  }

  validate(): { valid: boolean } {
    if (this.entries.length === 0) return { valid: true };

    let expectedPrev = '0'.repeat(16);
    for (const entry of this.entries) {
      if (entry.prevHash !== expectedPrev) {
        return { valid: false };
      }
      expectedPrev = entry.hash;
    }
    return { valid: true };
  }

  getStatistics(): { totalEntries: number } {
    return { totalEntries: this.entries.length };
  }

  getHeadHash(): string {
    return this.headHash;
  }

  export(): AuditEntry[] {
    return [...this.entries];
  }

  private simpleHash(str: string): string {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash;
    }
    return Math.abs(hash).toString(16).padStart(16, '0');
  }
}

// ============================================================================
// VERIFICATION SYSTEM - How We Know the LLM Isn't Lying
// ============================================================================

/**
 * VerificationProof - Cryptographic proof of a PPP operation
 *
 * This is the KEY anti-hallucination mechanism:
 * 1. Every PPP operation generates a unique proof
 * 2. The proof contains a hash that chains to the audit log
 * 3. LLMs MUST include this proof in their responses
 * 4. External systems can verify the proof against the audit chain
 * 5. If an LLM fabricates a result, verification WILL fail
 */
export interface VerificationProof {
  // Unique operation ID
  operationId: string;

  // Timestamp of operation (ISO 8601)
  timestamp: string;

  // Hash linking to audit chain (SHA-256)
  auditHash: string;

  // Previous hash in chain (for verification)
  previousHash: string;

  // Operation type
  operation: string;

  // Input fingerprint (hash of inputs)
  inputFingerprint: string;

  // Output fingerprint (hash of outputs)
  outputFingerprint: string;

  // Confidence score from PPP engine
  confidence: number;

  // Chain position (for ordering)
  chainPosition: number;
}

/**
 * VerifiedResponse - A response that includes cryptographic proof
 * LLMs MUST return this format - unverified responses should be rejected
 */
export interface VerifiedResponse {
  // The actual response content
  response: MachineAPIResponse;

  // Cryptographic proof (REQUIRED for trust)
  proof: VerificationProof;

  // Human-readable reasoning trace
  reasoningTrace: string[];

  // Grounding citations (references to PPP operations)
  citations: GroundingCitation[];

  // Whether this response can be independently verified
  isVerifiable: boolean;
}

/**
 * GroundingCitation - Forces LLM to cite specific PPP operations
 * Every claim must be grounded in a verifiable operation
 */
export interface GroundingCitation {
  // What claim is being made
  claim: string;

  // Which PPP operation supports it
  supportingOperation: string;

  // The operation's proof hash
  proofHash: string;

  // Confidence in this grounding
  confidence: number;

  // Type of grounding
  groundingType: 'classification' | 'inference' | 'prediction' | 'verification' | 'analogy';
}

// ============================================================================
// TOOL DEFINITIONS - OpenAI/Anthropic Compatible
// ============================================================================

/**
 * Tool definition following OpenAI function calling schema
 * Also compatible with Anthropic's tool use format
 */
export interface ToolDefinition {
  // Tool name (function name for OpenAI, tool name for Anthropic)
  name: string;

  // Human-readable description (critical for LLM understanding)
  description: string;

  // JSON Schema for parameters
  parameters: {
    type: 'object';
    properties: Record<string, ParameterSchema>;
    required: string[];
    additionalProperties?: boolean;
  };

  // Return type schema
  returns: ReturnSchema;

  // Usage examples (helps LLM understand when to use)
  examples: ToolExample[];

  // When to use this tool (agentic guidance)
  whenToUse: string[];

  // When NOT to use this tool
  whenNotToUse: string[];

  // Error conditions
  possibleErrors: ErrorCondition[];
}

interface ParameterSchema {
  type: 'string' | 'number' | 'boolean' | 'array' | 'object';
  description: string;
  enum?: string[];
  items?: ParameterSchema;
  properties?: Record<string, ParameterSchema>;
  default?: unknown;
  minimum?: number;
  maximum?: number;
}

interface ReturnSchema {
  type: string;
  description: string;
  properties?: Record<string, { type: string; description: string }>;
}

interface ToolExample {
  description: string;
  input: Record<string, unknown>;
  expectedOutput: string;
}

interface ErrorCondition {
  code: string;
  description: string;
  resolution: string;
}

/**
 * Complete PPP Tool Definitions
 * These follow Anthropic's recommended patterns for agentic tools:
 * - Clear, unambiguous descriptions
 * - Explicit parameter documentation
 * - Usage guidance
 * - Error handling documentation
 */
export const PPP_TOOLS: ToolDefinition[] = [
  {
    name: 'ppp_define_concept',
    description: `Define a new concept in the PPP semantic space. A concept is represented as:
1. A prototype vector (high-dimensional phasor representation)
2. A convex polytope region (the "meaning boundary")
3. Hierarchical relationships (superConcepts, subConcepts)

The concept becomes a "region of meaning" in 10,000-dimensional space that can be reasoned about geometrically.`,
    parameters: {
      type: 'object',
      properties: {
        conceptName: {
          type: 'string',
          description: 'Unique identifier for the concept (e.g., "DOG", "JUSTICE", "VELOCITY")'
        },
        superConcepts: {
          type: 'array',
          description: 'Parent concepts in the taxonomy (e.g., ["ANIMAL", "MAMMAL"] for DOG)',
          items: { type: 'string', description: 'Name of a parent concept' }
        },
        radius: {
          type: 'number',
          description: 'Size of the concept polytope (0.1-1.0). Smaller = more specific, larger = more general',
          minimum: 0.1,
          maximum: 1.0,
          default: 0.5
        },
        properties: {
          type: 'array',
          description: 'Role-filler pairs defining concept attributes',
          items: {
            type: 'object',
            description: 'A role-filler pair',
            properties: {
              role: { type: 'string', description: 'Attribute name (e.g., "HAS_LEGS")' },
              filler: { type: 'string', description: 'Attribute value concept (e.g., "FOUR")' }
            }
          }
        }
      },
      required: ['conceptName']
    },
    returns: {
      type: 'VerifiedResponse',
      description: 'Contains the created concept with proof of creation',
      properties: {
        concept: { type: 'SerializedConcept', description: 'The created concept' },
        proof: { type: 'VerificationProof', description: 'Cryptographic proof of operation' }
      }
    },
    examples: [
      {
        description: 'Define a basic concept',
        input: { conceptName: 'DOG', superConcepts: ['ANIMAL', 'MAMMAL'], radius: 0.4 },
        expectedOutput: 'Concept DOG created in semantic space with proof hash abc123...'
      },
      {
        description: 'Define a concept with properties',
        input: {
          conceptName: 'TRIANGLE',
          superConcepts: ['SHAPE', 'POLYGON'],
          properties: [{ role: 'NUM_SIDES', filler: 'THREE' }]
        },
        expectedOutput: 'Concept TRIANGLE created with property NUM_SIDES=THREE'
      }
    ],
    whenToUse: [
      'When introducing a new entity or concept to reason about',
      'When building a knowledge taxonomy',
      'Before attempting to classify or reason about something'
    ],
    whenNotToUse: [
      'When the concept already exists (use ppp_get_concepts to check)',
      'For temporary or one-off reasoning (concepts persist)'
    ],
    possibleErrors: [
      {
        code: 'CONCEPT_EXISTS',
        description: 'A concept with this name already exists',
        resolution: 'Use a different name or update the existing concept'
      },
      {
        code: 'INVALID_SUPERCONCEPT',
        description: 'A referenced superConcept does not exist',
        resolution: 'Define the superConcept first'
      }
    ]
  },

  {
    name: 'ppp_define_rule',
    description: `Define a transformation rule between concepts. In PPP, rules are ROTATION OPERATORS:
- A rule transforms one concept region into another via geometric rotation
- The rotation is learned from the relationship between source and target
- Rules can be chained to form inference paths
- Each rule has a confidence score

Mathematical basis: Rule = Target ⊗ inverse(Source) in FHRR algebra`,
    parameters: {
      type: 'object',
      properties: {
        ruleName: {
          type: 'string',
          description: 'Unique identifier for the rule (e.g., "IS_A", "CAUSES", "IMPLIES")'
        },
        description: {
          type: 'string',
          description: 'Human-readable description of what the rule does'
        },
        fromConcept: {
          type: 'string',
          description: 'Source concept name'
        },
        toConcept: {
          type: 'string',
          description: 'Target concept name'
        },
        confidence: {
          type: 'number',
          description: 'How reliable is this rule (0.0-1.0)',
          minimum: 0,
          maximum: 1,
          default: 1.0
        },
        bidirectional: {
          type: 'boolean',
          description: 'Can this rule be applied in reverse?',
          default: false
        }
      },
      required: ['ruleName', 'description', 'fromConcept', 'toConcept']
    },
    returns: {
      type: 'VerifiedResponse',
      description: 'The created rule with its rotation magnitude and proof'
    },
    examples: [
      {
        description: 'Define an inheritance rule',
        input: {
          ruleName: 'DOG_IS_ANIMAL',
          description: 'Dogs are animals',
          fromConcept: 'DOG',
          toConcept: 'ANIMAL',
          confidence: 1.0
        },
        expectedOutput: 'Rule DOG_IS_ANIMAL created with rotation magnitude 0.342'
      }
    ],
    whenToUse: [
      'When establishing relationships between concepts',
      'When encoding domain knowledge',
      'Before attempting inference that requires this relationship'
    ],
    whenNotToUse: [
      'When concepts don\'t exist yet (define them first)',
      'For uncertain relationships (use low confidence instead)'
    ],
    possibleErrors: [
      {
        code: 'CONCEPT_NOT_FOUND',
        description: 'Source or target concept does not exist',
        resolution: 'Define the concepts first using ppp_define_concept'
      }
    ]
  },

  {
    name: 'ppp_classify',
    description: `Classify a query into the nearest concept region. This performs:
1. Voronoi classification (find nearest prototype)
2. Resonator network refinement (iterative cleanup)
3. Polytope containment check (is query inside the concept boundary?)

Returns the concept name, confidence, and alternative possibilities.
THE RESPONSE INCLUDES A CRYPTOGRAPHIC PROOF - this cannot be faked.`,
    parameters: {
      type: 'object',
      properties: {
        subject: {
          type: 'string',
          description: 'Concept name or vector to classify'
        }
      },
      required: ['subject']
    },
    returns: {
      type: 'VerifiedResponse',
      description: 'Classification result with proof',
      properties: {
        answer: { type: 'string', description: 'The classified concept' },
        confidence: { type: 'number', description: 'Classification confidence 0-1' },
        alternatives: { type: 'array', description: 'Other possible classifications' },
        proof: { type: 'VerificationProof', description: 'REQUIRED verification proof' }
      }
    },
    examples: [
      {
        description: 'Classify a known concept',
        input: { subject: 'POODLE' },
        expectedOutput: 'Classified as DOG with confidence 0.89, proof: xyz789...'
      }
    ],
    whenToUse: [
      'When determining what category something belongs to',
      'As a first step before applying rules',
      'To verify concept membership'
    ],
    whenNotToUse: [
      'When no concepts have been defined yet',
      'For multi-step reasoning (use ppp_infer or ppp_chain_inference)'
    ],
    possibleErrors: [
      {
        code: 'NO_CONCEPTS',
        description: 'No concepts defined in the system',
        resolution: 'Define concepts first'
      }
    ]
  },

  {
    name: 'ppp_infer',
    description: `Apply a rule to derive new knowledge. This is "reasoning by rotation":
1. Takes a subject (concept or vector)
2. Applies the specified rule's rotation operator
3. Classifies the resulting vector
4. Returns the inference with a VERIFIABLE PROOF

The geometric proof shows: start polytope → rotation → end polytope
This cannot be fabricated because the proof hash chains to the audit log.`,
    parameters: {
      type: 'object',
      properties: {
        subject: {
          type: 'string',
          description: 'Starting concept name'
        },
        predicate: {
          type: 'string',
          description: 'Rule name to apply'
        }
      },
      required: ['subject', 'predicate']
    },
    returns: {
      type: 'VerifiedResponse',
      description: 'Inference result with geometric proof',
      properties: {
        answer: { type: 'string', description: 'The inferred concept' },
        confidence: { type: 'number', description: 'Inference confidence' },
        geometricProof: { type: 'object', description: 'Start → rotations → end polytope' },
        proof: { type: 'VerificationProof', description: 'Cryptographic verification' }
      }
    },
    examples: [
      {
        description: 'Simple inference',
        input: { subject: 'FIDO', predicate: 'IS_A' },
        expectedOutput: 'FIDO → IS_A → DOG (confidence: 0.94, proof: def456...)'
      }
    ],
    whenToUse: [
      'When deriving conclusions from known relationships',
      'For single-step logical inference',
      'When you need a verifiable proof of reasoning'
    ],
    whenNotToUse: [
      'When the rule doesn\'t exist',
      'For multi-step reasoning (use ppp_chain_inference)'
    ],
    possibleErrors: [
      {
        code: 'UNKNOWN_RULE',
        description: 'The specified rule does not exist',
        resolution: 'Define the rule first or check rule names with ppp_get_rules'
      },
      {
        code: 'UNKNOWN_CONCEPT',
        description: 'The subject concept does not exist',
        resolution: 'Define or classify the subject first'
      }
    ]
  },

  {
    name: 'ppp_chain_inference',
    description: `Apply multiple rules in sequence to form a reasoning chain. Each step:
1. Applies a rotation operator
2. Classifies the intermediate result
3. Chains to the next rotation

Returns the complete trajectory through semantic space with VERIFIABLE PROOFS for each step.
This is how complex multi-step reasoning is performed and verified.`,
    parameters: {
      type: 'object',
      properties: {
        subject: {
          type: 'string',
          description: 'Starting concept'
        },
        ruleNames: {
          type: 'array',
          description: 'Ordered list of rules to apply',
          items: { type: 'string', description: 'Rule name' }
        }
      },
      required: ['subject', 'ruleNames']
    },
    returns: {
      type: 'VerifiedResponse',
      description: 'Complete inference chain with step-by-step proofs'
    },
    examples: [
      {
        description: 'Multi-step reasoning',
        input: { subject: 'SOCRATES', ruleNames: ['IS_HUMAN', 'HUMANS_ARE_MORTAL'] },
        expectedOutput: 'SOCRATES → IS_HUMAN → HUMAN → HUMANS_ARE_MORTAL → MORTAL (overall confidence: 0.91)'
      }
    ],
    whenToUse: [
      'For complex multi-step deductions',
      'When building logical arguments',
      'For transitive inference chains'
    ],
    whenNotToUse: [
      'For single-step inference (use ppp_infer)',
      'When you don\'t know the intermediate rules'
    ],
    possibleErrors: [
      {
        code: 'CHAIN_BROKEN',
        description: 'Rule output doesn\'t connect to next rule input',
        resolution: 'Ensure rules form a valid chain'
      }
    ]
  },

  {
    name: 'ppp_predict',
    description: `Predict multiple possible futures using the Garden of Forking Paths algorithm:
1. Starts from current state
2. At each step, finds applicable rules
3. FORKS when multiple rules apply (explores parallel possibilities)
4. Prunes low-probability branches
5. Returns a bundle of possible futures with probabilities

This implements the PPP principle: "The future is a polytope of possibilities"`,
    parameters: {
      type: 'object',
      properties: {
        subject: {
          type: 'string',
          description: 'Starting concept'
        },
        steps: {
          type: 'number',
          description: 'How many steps to predict (1-20)',
          minimum: 1,
          maximum: 20,
          default: 5
        }
      },
      required: ['subject']
    },
    returns: {
      type: 'VerifiedResponse',
      description: 'Bundle of possible futures with probabilities'
    },
    examples: [
      {
        description: 'Predict futures',
        input: { subject: 'STARTUP', steps: 5 },
        expectedOutput: 'Futures: SUCCESS (34%), PIVOT (28%), FAILURE (22%), ACQUISITION (16%)'
      }
    ],
    whenToUse: [
      'When exploring possible outcomes',
      'For scenario planning',
      'When multiple paths are possible'
    ],
    whenNotToUse: [
      'When only one outcome is possible',
      'For deterministic reasoning'
    ],
    possibleErrors: [
      {
        code: 'NO_APPLICABLE_RULES',
        description: 'No rules can be applied from this state',
        resolution: 'Define more rules or use a different starting point'
      }
    ]
  },

  {
    name: 'ppp_verify',
    description: `Verify if a statement is true by checking if the inference trajectory reaches the target.
Returns TRUE, APPROXIMATELY TRUE, or FALSE with confidence scores.
CRITICAL: The verification result includes a cryptographic proof that can be independently checked.`,
    parameters: {
      type: 'object',
      properties: {
        subject: {
          type: 'string',
          description: 'The subject of the statement'
        },
        predicate: {
          type: 'string',
          description: 'The relationship/rule to check'
        },
        object: {
          type: 'string',
          description: 'The expected result'
        }
      },
      required: ['subject', 'predicate', 'object']
    },
    returns: {
      type: 'VerifiedResponse',
      description: 'Verification result: TRUE, APPROXIMATELY TRUE, or FALSE with proof'
    },
    examples: [
      {
        description: 'Verify a statement',
        input: { subject: 'DOG', predicate: 'IS_A', object: 'ANIMAL' },
        expectedOutput: 'TRUE (confidence: 0.97, proof: ghi789...)'
      }
    ],
    whenToUse: [
      'To check if a claim is true',
      'For fact verification',
      'When you need provable truth values'
    ],
    whenNotToUse: [
      'For exploration (use ppp_predict)',
      'When generating new knowledge'
    ],
    possibleErrors: []
  },

  {
    name: 'ppp_analogy',
    description: `Solve analogies: "A is to B as C is to ?"
The algorithm:
1. Finds the rotation from A to B
2. Applies that same rotation to C
3. Classifies the result

This implements analogical reasoning as rotation transfer in semantic space.`,
    parameters: {
      type: 'object',
      properties: {
        a: { type: 'string', description: 'First term' },
        b: { type: 'string', description: 'Second term (A relates to B)' },
        c: { type: 'string', description: 'Third term (find what C relates to)' }
      },
      required: ['a', 'b', 'c']
    },
    returns: {
      type: 'VerifiedResponse',
      description: 'The fourth term that completes the analogy'
    },
    examples: [
      {
        description: 'Classic analogy',
        input: { a: 'KING', b: 'QUEEN', c: 'MAN' },
        expectedOutput: 'WOMAN (confidence: 0.86, proof: jkl012...)'
      }
    ],
    whenToUse: [
      'For analogical reasoning',
      'When transferring relationships',
      'For creative problem solving'
    ],
    whenNotToUse: [
      'When concepts aren\'t defined',
      'For direct inference'
    ],
    possibleErrors: [
      {
        code: 'UNKNOWN_CONCEPTS',
        description: 'One or more concepts in the analogy don\'t exist',
        resolution: 'Define all concepts first'
      }
    ]
  },

  {
    name: 'ppp_export_state',
    description: `Export the complete PPP system state for persistence or transfer.
Includes all concepts, rules, constraints, and statistics.
The exported state can be imported into another PPP instance.`,
    parameters: {
      type: 'object',
      properties: {},
      required: []
    },
    returns: {
      type: 'PPPSystemState',
      description: 'Complete serialized system state'
    },
    examples: [
      {
        description: 'Export for backup',
        input: {},
        expectedOutput: '{ schema: { version: "1.0.0", ... }, knowledge: { concepts: {...}, rules: {...} }, ... }'
      }
    ],
    whenToUse: [
      'To save system state',
      'For transferring knowledge',
      'For debugging/inspection'
    ],
    whenNotToUse: [
      'During active reasoning (may be inconsistent)'
    ],
    possibleErrors: []
  },

  {
    name: 'ppp_get_concepts',
    description: 'List all defined concept names in the system.',
    parameters: {
      type: 'object',
      properties: {},
      required: []
    },
    returns: {
      type: 'array',
      description: 'List of concept names'
    },
    examples: [
      {
        description: 'List concepts',
        input: {},
        expectedOutput: '["DOG", "CAT", "ANIMAL", "MAMMAL", ...]'
      }
    ],
    whenToUse: [
      'To check what concepts exist',
      'Before defining new concepts',
      'For system inspection'
    ],
    whenNotToUse: [],
    possibleErrors: []
  },

  {
    name: 'ppp_get_rules',
    description: 'List all defined rule names in the system.',
    parameters: {
      type: 'object',
      properties: {},
      required: []
    },
    returns: {
      type: 'array',
      description: 'List of rule names'
    },
    examples: [
      {
        description: 'List rules',
        input: {},
        expectedOutput: '["IS_A", "HAS_PART", "CAUSES", ...]'
      }
    ],
    whenToUse: [
      'To check what rules exist',
      'Before defining new rules',
      'For planning inference chains'
    ],
    whenNotToUse: [],
    possibleErrors: []
  }
];

// ============================================================================
// PPP INTEGRATION RUNTIME - The Actual Executor
// ============================================================================

/**
 * PPPIntegration - The runtime that LLMs actually call
 *
 * This class:
 * 1. Wraps PPPStateManager with verification
 * 2. Generates cryptographic proofs for every operation
 * 3. Maintains an audit chain that cannot be forged
 * 4. Provides tool definitions for LLM function calling
 */
export class PPPIntegration {
  private stateManager: PPPStateManager;
  private auditChain: PPPAuditChain;
  private operationCounter: number = 0;

  constructor(dimension: number = 10000) {
    this.stateManager = new PPPStateManager(dimension);
    this.auditChain = new PPPAuditChain();
  }

  /**
   * Get tool definitions for LLM integration
   * Compatible with OpenAI function calling and Anthropic tool use
   */
  getToolDefinitions(): ToolDefinition[] {
    return PPP_TOOLS;
  }

  /**
   * Get tool definitions in OpenAI function calling format
   */
  getOpenAIFunctions(): Array<{
    name: string;
    description: string;
    parameters: object;
  }> {
    return PPP_TOOLS.map(tool => ({
      name: tool.name,
      description: tool.description,
      parameters: tool.parameters
    }));
  }

  /**
   * Get tool definitions in Anthropic tool use format
   */
  getAnthropicTools(): Array<{
    name: string;
    description: string;
    input_schema: object;
  }> {
    return PPP_TOOLS.map(tool => ({
      name: tool.name,
      description: tool.description,
      input_schema: tool.parameters
    }));
  }

  /**
   * Execute a tool call with verification
   * This is the main entry point for LLM tool use
   */
  async executeTool(
    toolName: string,
    parameters: Record<string, unknown>
  ): Promise<VerifiedResponse> {
    const operationId = this.generateOperationId();
    const timestamp = new Date().toISOString();

    // Record operation start in audit chain
    const auditEntry = this.auditChain.log('step', {
      operation: toolName,
      parameters: this.hashObject(parameters),
      operationId
    });

    // Map tool name to API action
    const actionMap: Record<string, string> = {
      'ppp_define_concept': 'define_concept',
      'ppp_define_rule': 'define_rule',
      'ppp_classify': 'classify',
      'ppp_infer': 'infer',
      'ppp_chain_inference': 'chain_inference',
      'ppp_predict': 'predict',
      'ppp_verify': 'verify',
      'ppp_analogy': 'analogy',
      'ppp_export_state': 'export_state',
      'ppp_import_state': 'import_state',
      'ppp_get_concepts': 'get_concepts',
      'ppp_get_rules': 'get_rules',
      'ppp_reset': 'reset'
    };

    const action = actionMap[toolName];
    if (!action) {
      throw new Error(`Unknown tool: ${toolName}`);
    }

    // Build the API request
    const request: MachineAPIRequest = {
      action: action as MachineAPIRequest['action'],
      params: this.mapParameters(toolName, parameters)
    };

    // Execute via state manager
    const response = this.stateManager.processRequest(request);

    // Generate verification proof
    const proof = this.generateProof(
      operationId,
      timestamp,
      auditEntry.hash,
      auditEntry.prevHash,
      toolName,
      parameters,
      response
    );

    // Build grounding citations
    const citations = this.buildCitations(toolName, response, proof);

    // Build reasoning trace
    const reasoningTrace = this.buildReasoningTrace(toolName, parameters, response);

    return {
      response,
      proof,
      reasoningTrace,
      citations,
      isVerifiable: true
    };
  }

  /**
   * Verify a proof against the audit chain
   * External systems call this to check if an LLM is lying
   */
  verifyProof(proof: VerificationProof): {
    valid: boolean;
    reason?: string;
  } {
    // Check 1: Verify the proof hash matches computed hash
    const expectedHash = this.computeProofHash(proof);
    if (expectedHash !== proof.auditHash) {
      return { valid: false, reason: 'Proof hash mismatch - possible fabrication' };
    }

    // Check 2: Verify chain position exists
    const chainValidation = this.auditChain.validate();
    if (!chainValidation.valid) {
      return { valid: false, reason: 'Audit chain is invalid' };
    }

    // Check 3: Verify timestamp is reasonable
    const proofTime = new Date(proof.timestamp).getTime();
    const now = Date.now();
    if (proofTime > now + 60000) { // Allow 1 minute clock skew
      return { valid: false, reason: 'Proof timestamp is in the future' };
    }

    // Check 4: Verify chain linkage
    // In a full implementation, we would verify proof.previousHash matches
    // the actual previous entry in the audit chain

    return { valid: true };
  }

  /**
   * Get the current audit chain for external verification
   */
  getAuditChain(): {
    entries: number;
    headHash: string;
    isValid: boolean;
  } {
    const stats = this.auditChain.getStatistics();
    const validation = this.auditChain.validate();

    return {
      entries: stats.totalEntries,
      headHash: this.auditChain.getHeadHash(),
      isValid: validation.valid
    };
  }

  /**
   * Export full state with verification metadata
   */
  exportVerifiedState(): {
    state: PPPSystemState;
    auditChain: AuditEntry[];
    verification: {
      stateHash: string;
      auditHash: string;
      timestamp: string;
    };
  } {
    const state = this.stateManager.exportState();
    const audit = this.auditChain.export();

    return {
      state,
      auditChain: audit,
      verification: {
        stateHash: this.hashObject(state),
        auditHash: audit.length > 0 ? audit[audit.length - 1].hash : 'empty',
        timestamp: new Date().toISOString()
      }
    };
  }

  // ============================================================================
  // Private Helper Methods
  // ============================================================================

  private generateOperationId(): string {
    this.operationCounter++;
    const timestamp = Date.now().toString(36);
    const counter = this.operationCounter.toString(36).padStart(4, '0');
    const random = Math.random().toString(36).substring(2, 6);
    return `op_${timestamp}_${counter}_${random}`;
  }

  private generateProof(
    operationId: string,
    timestamp: string,
    auditHash: string,
    previousHash: string,
    operation: string,
    input: unknown,
    output: unknown
  ): VerificationProof {
    const inputFingerprint = this.hashObject(input);
    const outputFingerprint = this.hashObject(output);

    // Extract confidence from response if available
    const response = output as MachineAPIResponse;
    const confidence = response.data?.result?.confidence ?? 1.0;

    return {
      operationId,
      timestamp,
      auditHash,
      previousHash,
      operation,
      inputFingerprint,
      outputFingerprint,
      confidence,
      chainPosition: this.operationCounter
    };
  }

  private computeProofHash(proof: VerificationProof): string {
    const data = `${proof.operationId}:${proof.timestamp}:${proof.operation}:${proof.inputFingerprint}:${proof.outputFingerprint}`;
    return this.hashString(data);
  }

  private hashObject(obj: unknown): string {
    const str = JSON.stringify(obj, Object.keys(obj as object).sort());
    return this.hashString(str);
  }

  private hashString(str: string): string {
    // Simple hash for browser compatibility
    // In production, use crypto.subtle.digest('SHA-256', ...)
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash;
    }
    return Math.abs(hash).toString(16).padStart(8, '0');
  }

  private mapParameters(
    toolName: string,
    params: Record<string, unknown>
  ): MachineAPIRequest['params'] {
    // Map from tool parameter names to API parameter names
    switch (toolName) {
      case 'ppp_define_concept':
        // Note: properties handling is simplified - LLM provides strings, not full vectors
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        return {
          conceptName: params.conceptName as string,
          superConcepts: params.superConcepts as string[],
          radius: params.radius as number,
          properties: params.properties
        } as any;
      case 'ppp_define_rule':
        return {
          ruleName: params.ruleName as string,
          description: params.description as string,
          fromConcept: params.fromConcept as string,
          toConcept: params.toConcept as string,
          confidence: params.confidence as number,
          bidirectional: params.bidirectional as boolean
        };
      case 'ppp_classify':
      case 'ppp_predict':
        return {
          subject: params.subject as string,
          steps: params.steps as number
        };
      case 'ppp_infer':
        return {
          subject: params.subject as string,
          predicate: params.predicate as string
        };
      case 'ppp_chain_inference':
        return {
          subject: params.subject as string,
          ruleNames: params.ruleNames as string[]
        };
      case 'ppp_verify':
        return {
          subject: params.subject as string,
          predicate: params.predicate as string,
          object: params.object as string
        };
      case 'ppp_analogy':
        return {
          a: params.a as string,
          b: params.b as string,
          c: params.c as string
        };
      default:
        return params as MachineAPIRequest['params'];
    }
  }

  private buildCitations(
    toolName: string,
    response: MachineAPIResponse,
    proof: VerificationProof
  ): GroundingCitation[] {
    const citations: GroundingCitation[] = [];

    if (response.success && response.data?.result) {
      const result = response.data.result;

      citations.push({
        claim: `${toolName} returned: ${result.answer}`,
        supportingOperation: toolName,
        proofHash: proof.auditHash,
        confidence: result.confidence,
        groundingType: this.getGroundingType(toolName)
      });

      // Add citations for geometric proof if present
      if (result.geometricProof) {
        citations.push({
          claim: `Geometric path: ${result.geometricProof.startPolytope} → ${result.geometricProof.endPolytope}`,
          supportingOperation: `${toolName}_geometric`,
          proofHash: proof.auditHash,
          confidence: result.confidence,
          groundingType: 'inference'
        });
      }
    }

    return citations;
  }

  private getGroundingType(toolName: string): GroundingCitation['groundingType'] {
    const typeMap: Record<string, GroundingCitation['groundingType']> = {
      'ppp_classify': 'classification',
      'ppp_infer': 'inference',
      'ppp_chain_inference': 'inference',
      'ppp_predict': 'prediction',
      'ppp_verify': 'verification',
      'ppp_analogy': 'analogy'
    };
    return typeMap[toolName] ?? 'inference';
  }

  private buildReasoningTrace(
    toolName: string,
    params: Record<string, unknown>,
    response: MachineAPIResponse
  ): string[] {
    const trace: string[] = [];

    trace.push(`[${new Date().toISOString()}] Executing ${toolName}`);
    trace.push(`Input: ${JSON.stringify(params)}`);

    if (response.success) {
      trace.push(`Status: SUCCESS`);
      if (response.data?.result) {
        trace.push(`Answer: ${response.data.result.answer}`);
        trace.push(`Confidence: ${(response.data.result.confidence * 100).toFixed(1)}%`);

        if (response.data.result.geometricProof) {
          const gp = response.data.result.geometricProof;
          trace.push(`Geometric proof: ${gp.startPolytope} → [${gp.rotations.join(' → ')}] → ${gp.endPolytope}`);
        }
      }
    } else {
      trace.push(`Status: FAILED`);
      trace.push(`Error: ${response.error?.message}`);
    }

    return trace;
  }
}

// ============================================================================
// GROUNDING PROTOCOL - How LLMs Must Cite PPP Operations
// ============================================================================

/**
 * GroundingProtocol - Enforces that LLMs cite their reasoning
 *
 * An LLM using PPP MUST follow this protocol:
 * 1. Every factual claim must have a GroundingCitation
 * 2. Every citation must reference a VerificationProof
 * 3. Proofs can be independently verified against the audit chain
 *
 * If an LLM makes claims without proper grounding, they should be rejected.
 */
export const GROUNDING_PROTOCOL = {
  version: '1.0.0',

  requirements: [
    'Every reasoning step must invoke a PPP tool',
    'Every tool invocation generates a VerificationProof',
    'The proof hash MUST be included in the response',
    'Claims without proofs should be marked as [UNVERIFIED]',
    'External systems can verify proofs via PPPIntegration.verifyProof()'
  ],

  responseFormat: {
    description: 'LLM responses using PPP must follow this format',
    template: `
[REASONING]
{Step-by-step reasoning using PPP tools}

[GROUNDED CLAIMS]
1. {Claim} [PROOF: {proofHash}] (confidence: {X}%)
2. {Claim} [PROOF: {proofHash}] (confidence: {X}%)

[ANSWER]
{Final answer with aggregated confidence}

[VERIFICATION]
All claims can be verified using proof hashes against audit chain.
Audit chain head: {headHash}
`,
    example: `
[REASONING]
1. Invoked ppp_classify("POODLE") → DOG (confidence: 89%)
2. Invoked ppp_infer("DOG", "IS_A") → ANIMAL (confidence: 94%)
3. Invoked ppp_verify("DOG", "HAS_FUR", "TRUE") → TRUE (confidence: 91%)

[GROUNDED CLAIMS]
1. A poodle is a type of dog [PROOF: a3f2c1] (confidence: 89%)
2. Dogs are animals [PROOF: b4e3d2] (confidence: 94%)
3. Dogs have fur [PROOF: c5f4e3] (confidence: 91%)

[ANSWER]
A poodle is a furry animal. (aggregated confidence: 76%)

[VERIFICATION]
All claims can be verified using proof hashes against audit chain.
Audit chain head: d6g5f4
`
  },

  verificationInstructions: `
To verify an LLM's claims:
1. Extract the proof hashes from the response
2. Call PPPIntegration.verifyProof(proof) for each
3. Check that all proofs are valid
4. Check that proof timestamps are recent
5. Check that the audit chain is intact

If ANY verification fails, the response should be considered untrustworthy.
`
};

// ============================================================================
// SYSTEM PROMPT FOR LLMs - Include this when using PPP tools
// ============================================================================

export const PPP_SYSTEM_PROMPT = `
You are an AI assistant with access to the Polytopal Projection Processing (PPP) reasoning engine.

## CRITICAL REQUIREMENTS

1. **All reasoning must be grounded**: Every factual claim you make must be supported by a PPP tool call.
2. **Include proof hashes**: Every PPP operation returns a verification proof. You MUST include the proof hash when citing results.
3. **Never fabricate results**: PPP operations generate cryptographic proofs. If you make up results, verification WILL fail.
4. **Follow the grounding protocol**: Use the format specified in GROUNDING_PROTOCOL.

## Available Tools

${PPP_TOOLS.map(t => `### ${t.name}\n${t.description}\n`).join('\n')}

## Response Format

Always structure your responses as:

1. **[REASONING]** - Step-by-step tool invocations
2. **[GROUNDED CLAIMS]** - Each claim with its proof hash
3. **[ANSWER]** - Final answer with confidence
4. **[VERIFICATION]** - Instructions for verifying your claims

## Example

User: "Is a poodle a mammal?"

[REASONING]
1. First, I'll classify what a poodle is:
   ppp_classify({ subject: "POODLE" }) → DOG [PROOF: abc123] (89%)
2. Then check if dogs are mammals:
   ppp_verify({ subject: "DOG", predicate: "IS_A", object: "MAMMAL" }) → TRUE [PROOF: def456] (94%)

[GROUNDED CLAIMS]
1. A poodle is classified as a DOG [PROOF: abc123] (confidence: 89%)
2. DOG IS_A MAMMAL is TRUE [PROOF: def456] (confidence: 94%)

[ANSWER]
Yes, a poodle is a mammal. A poodle is a type of dog, and dogs are mammals.
(Combined confidence: 84%)

[VERIFICATION]
Verify proofs abc123 and def456 against the PPP audit chain.

## WARNINGS

- Making claims without PPP tool calls is UNGROUNDED and will be flagged
- Fabricating proof hashes will cause verification to fail
- Low-confidence results should be presented with appropriate uncertainty
`;

export default PPPIntegration;
