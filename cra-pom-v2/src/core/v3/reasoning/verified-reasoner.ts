/**
 * PPP v3 Verified Reasoning Engine
 *
 * HONEST DOCUMENTATION:
 * This is a reasoning engine where every operation is logged to a
 * cryptographically-signed audit chain in an isolated worker.
 *
 * WHAT THIS PROVIDES:
 * - Tamper-evident reasoning trace
 * - Verifiable operation sequence
 * - Semantic concept retrieval (when using real embeddings)
 * - HDC/VSA operations for concept manipulation
 *
 * WHAT THIS DOES NOT PROVIDE:
 * - Proof that the LLM actually used this for reasoning
 * - Semantic correctness (math doesn't equal truth)
 * - Protection against the LLM ignoring results
 *
 * TRUST MODEL:
 * - All operations are signed by the isolated verification worker
 * - External verifiers can check the entire reasoning trace
 * - The LLM cannot forge entries in the audit chain
 */

import { VerificationClient, getVerificationClient } from '../isolation';
import { ConceptStore, getConceptStore } from '../embeddings';
import type { SignedData } from '../crypto';

// ============================================================================
// Types
// ============================================================================

export interface ReasoningStep {
  /** Step number in the reasoning chain */
  stepNumber: number;

  /** Type of operation */
  operation: ReasoningOperation;

  /** Natural language description */
  description: string;

  /** Input concepts/values */
  inputs: string[];

  /** Output concepts/values */
  outputs: string[];

  /** Confidence score (0-1) */
  confidence: number;

  /** Citations/grounding */
  citations: string[];

  /** Timestamp */
  timestamp: string;
}

export type ReasoningOperation =
  | 'CONCEPT_LOOKUP' // Retrieved a concept from the store
  | 'CONCEPT_BIND' // Bound concepts together (HDC bind)
  | 'CONCEPT_BUNDLE' // Bundled concepts (HDC bundle)
  | 'SIMILARITY_QUERY' // Found similar concepts
  | 'INFERENCE' // Made an inference
  | 'COMPOSITION' // Composed concepts (vector arithmetic)
  | 'HYPOTHESIS' // Generated a hypothesis
  | 'VERIFICATION' // Verified a claim
  | 'CONCLUSION'; // Reached a conclusion

export interface ReasoningSession {
  /** Unique session ID */
  sessionId: string;

  /** When the session started */
  startedAt: string;

  /** Query/task that initiated the session */
  initialQuery: string;

  /** All steps taken */
  steps: SignedReasoningStep[];

  /** Final conclusion if reached */
  conclusion?: SignedConclusion;

  /** Grounding status */
  grounding: {
    semanticEmbeddings: boolean;
    embeddingSource: string;
    conceptsUsed: number;
  };
}

export interface SignedReasoningStep extends SignedData<ReasoningStep> {}

export interface Conclusion {
  /** The conclusion reached */
  statement: string;

  /** Confidence (0-1) */
  confidence: number;

  /** Supporting steps */
  supportingSteps: number[];

  /** Caveats/limitations */
  caveats: string[];

  /** Whether this is grounded in semantic embeddings */
  semanticallyGrounded: boolean;
}

export interface SignedConclusion extends SignedData<Conclusion> {}

export interface ReasoningQuery {
  /** The question or task */
  query: string;

  /** Concepts to consider */
  concepts?: string[];

  /** Maximum steps */
  maxSteps?: number;

  /** Minimum confidence threshold */
  minConfidence?: number;
}

export interface ReasoningResult {
  /** The session that was created */
  session: ReasoningSession;

  /** Whether a conclusion was reached */
  concluded: boolean;

  /** Summary */
  summary: string;

  /** Verification info */
  verification: {
    chainValid: boolean;
    signaturesValid: boolean;
    publicKey: JsonWebKey;
  };
}

// ============================================================================
// Implementation
// ============================================================================

export class VerifiedReasoner {
  private verificationClient: VerificationClient | null = null;
  private conceptStore: ConceptStore;
  private currentSession: ReasoningSession | null = null;
  private stepCounter = 0;

  constructor(conceptStore?: ConceptStore) {
    this.conceptStore = conceptStore || getConceptStore();
  }

  /**
   * Initialize the reasoner
   *
   * This sets up the connection to the verification worker.
   */
  async initialize(): Promise<{ publicKey: JsonWebKey }> {
    this.verificationClient = await getVerificationClient();
    const publicKey = await this.verificationClient.getPublicKey();
    return { publicKey };
  }

  /**
   * Is the reasoner initialized?
   */
  isInitialized(): boolean {
    return this.verificationClient !== null && this.verificationClient.isInitialized();
  }

  /**
   * Start a new reasoning session
   */
  async startSession(query: string): Promise<ReasoningSession> {
    this.ensureInitialized();

    const sessionId = this.generateSessionId();
    const grounding = this.conceptStore.getGroundingStatus();

    this.currentSession = {
      sessionId,
      startedAt: new Date().toISOString(),
      initialQuery: query,
      steps: [],
      grounding: {
        semanticEmbeddings: grounding.embeddingSource !== 'deterministic_fallback',
        embeddingSource: grounding.embeddingSource,
        conceptsUsed: 0,
      },
    };

    this.stepCounter = 0;

    // Log session start to the audit chain
    await this.verificationClient!.appendToChain('SESSION_START', {
      sessionId,
      query,
      grounding,
    });

    return this.currentSession;
  }

  /**
   * Look up a concept
   */
  async lookupConcept(conceptName: string): Promise<SignedReasoningStep> {
    this.ensureSession();

    const concept = this.conceptStore.getConceptByName(conceptName);
    const retrieval = await this.conceptStore.retrieve(conceptName, 1);

    const step: ReasoningStep = {
      stepNumber: ++this.stepCounter,
      operation: 'CONCEPT_LOOKUP',
      description: `Looking up concept: ${conceptName}`,
      inputs: [conceptName],
      outputs: concept
        ? [`Found: ${concept.name} - ${concept.description}`]
        : retrieval.results.length > 0
          ? [`Nearest: ${retrieval.results[0].concept.name} (similarity: ${retrieval.results[0].similarity.toFixed(3)})`]
          : ['Not found'],
      confidence: concept ? 1.0 : retrieval.results[0]?.similarity || 0,
      citations: concept ? [concept.id] : [],
      timestamp: new Date().toISOString(),
    };

    const signed = await this.signAndRecordStep(step);
    this.currentSession!.grounding.conceptsUsed++;

    return signed;
  }

  /**
   * Query for similar concepts
   */
  async querySimilar(query: string, topK = 5): Promise<SignedReasoningStep> {
    this.ensureSession();

    const results = await this.conceptStore.retrieve(query, topK);

    const step: ReasoningStep = {
      stepNumber: ++this.stepCounter,
      operation: 'SIMILARITY_QUERY',
      description: `Finding concepts similar to: ${query}`,
      inputs: [query],
      outputs: results.results.map(
        (r) => `${r.concept.name}: ${r.similarity.toFixed(3)}`
      ),
      confidence: results.results[0]?.similarity || 0,
      citations: results.results.map((r) => r.concept.id),
      timestamp: new Date().toISOString(),
    };

    const signed = await this.signAndRecordStep(step);

    // Warn if not semantically meaningful
    if (!results.semanticallyMeaningful) {
      console.warn(
        'WARNING: Similarity query used fallback embeddings. Results are NOT semantically meaningful.'
      );
    }

    return signed;
  }

  /**
   * Compose concepts using vector arithmetic
   */
  async composeConcepts(
    operations: Array<{
      concept: string;
      operation: 'add' | 'subtract';
      weight?: number;
    }>
  ): Promise<SignedReasoningStep> {
    this.ensureSession();

    const result = await this.conceptStore.compose(operations);

    const step: ReasoningStep = {
      stepNumber: ++this.stepCounter,
      operation: 'COMPOSITION',
      description: `Composing concepts: ${operations
        .map((o) => `${o.operation === 'add' ? '+' : '-'}${o.concept}`)
        .join(' ')}`,
      inputs: operations.map((o) => o.concept),
      outputs: result.nearest.map(
        (n) => `${n.concept.name}: ${n.similarity.toFixed(3)}`
      ),
      confidence: result.nearest[0]?.similarity || 0,
      citations: result.nearest.map((n) => n.concept.id),
      timestamp: new Date().toISOString(),
    };

    const signed = await this.signAndRecordStep(step);

    if (!result.semanticallyMeaningful) {
      console.warn(
        'WARNING: Concept composition used fallback embeddings. Results are NOT semantically meaningful.'
      );
    }

    return signed;
  }

  /**
   * Make an inference
   */
  async makeInference(
    premises: string[],
    inference: string,
    confidence: number
  ): Promise<SignedReasoningStep> {
    this.ensureSession();

    const step: ReasoningStep = {
      stepNumber: ++this.stepCounter,
      operation: 'INFERENCE',
      description: `Inferring from ${premises.length} premises: ${inference}`,
      inputs: premises,
      outputs: [inference],
      confidence: Math.min(1, Math.max(0, confidence)),
      citations: [],
      timestamp: new Date().toISOString(),
    };

    return this.signAndRecordStep(step);
  }

  /**
   * Generate a hypothesis
   */
  async generateHypothesis(
    observations: string[],
    hypothesis: string,
    confidence: number
  ): Promise<SignedReasoningStep> {
    this.ensureSession();

    const step: ReasoningStep = {
      stepNumber: ++this.stepCounter,
      operation: 'HYPOTHESIS',
      description: `Generating hypothesis: ${hypothesis}`,
      inputs: observations,
      outputs: [hypothesis],
      confidence: Math.min(1, Math.max(0, confidence)),
      citations: [],
      timestamp: new Date().toISOString(),
    };

    return this.signAndRecordStep(step);
  }

  /**
   * Reach a conclusion
   */
  async conclude(
    statement: string,
    confidence: number,
    supportingSteps: number[],
    caveats: string[] = []
  ): Promise<SignedConclusion> {
    this.ensureSession();

    const grounding = this.currentSession!.grounding;

    const conclusion: Conclusion = {
      statement,
      confidence: Math.min(1, Math.max(0, confidence)),
      supportingSteps,
      caveats: [
        ...caveats,
        ...(grounding.semanticEmbeddings
          ? []
          : ['Using fallback embeddings - semantic grounding limited']),
      ],
      semanticallyGrounded: grounding.semanticEmbeddings,
    };

    // Sign the conclusion
    const signed = (await this.verificationClient!.signData(
      conclusion
    )) as SignedConclusion;

    // Record in the audit chain
    await this.verificationClient!.appendToChain('CONCLUSION', conclusion);

    // Attach to session
    this.currentSession!.conclusion = signed;

    return signed;
  }

  /**
   * End the session and get the complete result
   */
  async endSession(): Promise<ReasoningResult> {
    this.ensureSession();

    // Validate the chain
    const chainValidation = await this.verificationClient!.validateChain();
    const publicKey = await this.verificationClient!.getPublicKey();

    const result: ReasoningResult = {
      session: this.currentSession!,
      concluded: !!this.currentSession!.conclusion,
      summary: this.generateSummary(),
      verification: {
        chainValid: chainValidation.valid,
        signaturesValid: chainValidation.details.signaturesValid,
        publicKey,
      },
    };

    // Log session end
    await this.verificationClient!.appendToChain('SESSION_END', {
      sessionId: this.currentSession!.sessionId,
      stepCount: this.currentSession!.steps.length,
      concluded: result.concluded,
    });

    // Clear current session
    this.currentSession = null;
    this.stepCounter = 0;

    return result;
  }

  /**
   * Get the current session (if active)
   */
  getCurrentSession(): ReasoningSession | null {
    return this.currentSession;
  }

  /**
   * Export the audit chain for external verification
   */
  async exportAuditChain(): Promise<unknown> {
    this.ensureInitialized();
    return this.verificationClient!.exportChain();
  }

  // ============================================================================
  // Private Methods
  // ============================================================================

  private ensureInitialized(): void {
    if (!this.verificationClient) {
      throw new Error('Reasoner not initialized. Call initialize() first.');
    }
  }

  private ensureSession(): void {
    this.ensureInitialized();
    if (!this.currentSession) {
      throw new Error('No active session. Call startSession() first.');
    }
  }

  private generateSessionId(): string {
    return `session-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  }

  private async signAndRecordStep(step: ReasoningStep): Promise<SignedReasoningStep> {
    // Sign the step
    const signed = (await this.verificationClient!.signData(
      step
    )) as SignedReasoningStep;

    // Record in the audit chain
    await this.verificationClient!.appendToChain('REASONING_STEP', step);

    // Add to session
    this.currentSession!.steps.push(signed);

    return signed;
  }

  private generateSummary(): string {
    if (!this.currentSession) return '';

    const session = this.currentSession;
    const lines: string[] = [];

    lines.push(`Reasoning Session: ${session.sessionId}`);
    lines.push(`Query: ${session.initialQuery}`);
    lines.push(`Steps: ${session.steps.length}`);
    lines.push(
      `Grounding: ${session.grounding.semanticEmbeddings ? 'Semantic' : 'Fallback (non-semantic)'}`
    );

    if (session.conclusion) {
      lines.push(`Conclusion: ${session.conclusion.payload.statement}`);
      lines.push(`Confidence: ${(session.conclusion.payload.confidence * 100).toFixed(1)}%`);
    } else {
      lines.push('No conclusion reached');
    }

    return lines.join('\n');
  }
}

// ============================================================================
// Singleton
// ============================================================================

let globalReasoner: VerifiedReasoner | null = null;

export async function getVerifiedReasoner(): Promise<VerifiedReasoner> {
  if (!globalReasoner) {
    globalReasoner = new VerifiedReasoner();
    await globalReasoner.initialize();
  }
  return globalReasoner;
}

export function resetVerifiedReasoner(): void {
  globalReasoner = null;
}
