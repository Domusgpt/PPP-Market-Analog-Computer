/**
 * PPP v3 - Polytopal Projection Processing
 *
 * A redesigned reasoning system with honest documentation, real cryptography,
 * process isolation, and semantic grounding.
 *
 * =============================================================================
 * WHAT THIS SYSTEM IS
 * =============================================================================
 *
 * PPP v3 is a framework for creating auditable reasoning traces. It provides:
 *
 * 1. CRYPTOGRAPHIC FOUNDATION
 *    - Real SHA-256 hashing via Web Crypto API
 *    - ECDSA P-256 signatures for non-repudiation
 *    - Signed hash chain for tamper-evident audit trail
 *
 * 2. PROCESS ISOLATION
 *    - Verification service runs in a Web Worker
 *    - Private key is non-extractable and trapped in worker
 *    - LLM in main thread CANNOT access signing keys
 *    - All communication via message passing
 *
 * 3. SEMANTIC GROUNDING
 *    - Real embeddings when API configured
 *    - Honest fallback with clear warnings when not
 *    - Concept store for semantic retrieval
 *
 * 4. VERIFIED REASONING
 *    - Every reasoning step is signed and logged
 *    - Audit chain can be exported and verified externally
 *    - Sessions track grounding status
 *
 * =============================================================================
 * WHAT THIS SYSTEM IS NOT
 * =============================================================================
 *
 * This system does NOT:
 * - Prove that an LLM actually used these tools for reasoning
 * - Validate the semantic truth of claims
 * - Prevent an LLM from ignoring results
 * - Provide AGI or true understanding
 *
 * The cryptographic proofs only show that certain operations were requested
 * through the proper channels and were logged. They do not prove the LLM
 * incorporated this into its actual reasoning.
 *
 * =============================================================================
 * USAGE
 * =============================================================================
 *
 * ```typescript
 * import { getVerifiedReasoner, configureEmbeddings } from './core/v3';
 *
 * // Optional: Configure real embeddings
 * configureEmbeddings('https://api.openai.com/v1/embeddings', apiKey, 'text-embedding-3-small');
 *
 * // Get reasoner
 * const reasoner = await getVerifiedReasoner();
 *
 * // Start session
 * await reasoner.startSession('What is the relationship between X and Y?');
 *
 * // Perform reasoning steps (each is signed and logged)
 * await reasoner.lookupConcept('X');
 * await reasoner.querySimilar('Y', 5);
 * await reasoner.makeInference(['X implies Z', 'Y is similar to X'], 'Y may imply Z', 0.7);
 *
 * // Conclude and get result
 * await reasoner.conclude('Y probably implies Z', 0.7, [1, 2, 3], ['Based on limited data']);
 * const result = await reasoner.endSession();
 *
 * // Result includes verification status
 * console.log(result.verification.chainValid); // true
 * console.log(result.verification.signaturesValid); // true
 *
 * // Export for external verification
 * const auditChain = await reasoner.exportAuditChain();
 * ```
 *
 * =============================================================================
 */

// Cryptographic primitives
export {
  CryptoService,
  getCryptoService,
  resetCryptoService,
  SignedHashChain,
  createSignedHashChain,
  type KeyPair,
  type ExportedKeyPair,
  type Signature,
  type HashResult,
  type SignedData,
  type ChainEntry,
  type SignedChainEntry,
  type ChainValidationResult,
} from './crypto';

// Process isolation
export {
  VerificationClient,
  getVerificationClient,
  resetVerificationClient,
  signWithIsolation,
  verifyWithIsolation,
  appendToAuditChain,
  verifyAttestation,
  type TrustedOperations,
  type UntrustedOperations,
  type SignatureRequest,
  type SignedPayload,
  type VerificationOutcome,
  type ChainAppendResult,
  type Attestation,
  type AttestationType,
  type ReasoningStepClaim,
  type ConceptLookupClaim,
  type InferenceClaim,
} from './isolation';

// Semantic embeddings
export {
  EmbeddingService,
  getEmbeddingService,
  resetEmbeddingService,
  configureEmbeddings,
  ConceptStore,
  getConceptStore,
  resetConceptStore,
  initializeWithBasicConcepts,
  LOGIC_CONCEPTS,
  REASONING_CONCEPTS,
  type EmbeddingResult,
  type EmbeddingSource,
  type EmbeddingConfig,
  type SimilarityResult,
  type StoredConcept,
  type ConceptRetrievalResult,
  type ConceptCompositionResult,
} from './embeddings';

// Verified reasoning
export {
  VerifiedReasoner,
  getVerifiedReasoner,
  resetVerifiedReasoner,
  type ReasoningStep,
  type ReasoningOperation,
  type ReasoningSession,
  type SignedReasoningStep,
  type Conclusion,
  type SignedConclusion,
  type ReasoningQuery,
  type ReasoningResult,
} from './reasoning';

// Visualization integration
export {
  VisualizationBridge,
  createVisualizationBridge,
  getVisualizationBridge,
  resetVisualizationBridge,
  type VisualizationState,
  type ConceptNode,
  type ConceptEdge,
  type VisualizationEvent,
  type UseVisualizationBridgeState,
} from './integration';

// Version info
export const PPP_VERSION = '3.0.0';
export const PPP_CODENAME = 'Honest Geometric Cognition';
