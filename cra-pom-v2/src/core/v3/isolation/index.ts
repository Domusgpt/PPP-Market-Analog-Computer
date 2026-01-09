/**
 * PPP v3 Process Isolation Module
 *
 * This module provides the trust boundary between the LLM (untrusted)
 * and the verification service (trusted).
 *
 * KEY EXPORTS:
 * - VerificationClient: Main-thread interface to the worker
 * - Trust boundary types and attestation verification
 * - Convenience functions for common operations
 */

// Client for communicating with the verification worker
export {
  VerificationClient,
  getVerificationClient,
  resetVerificationClient,
  signWithIsolation,
  verifyWithIsolation,
  appendToAuditChain,
} from './verification-client';

// Trust boundary definitions and attestation types
export {
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
} from './trust-boundary';

// Note: The verification-worker.ts is bundled separately and loaded by VerificationClient
