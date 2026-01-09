/**
 * PPP v3 Trust Boundary Definitions
 *
 * This module defines the trust model for the PPP system.
 * These types enforce the separation between trusted and untrusted contexts.
 *
 * ARCHITECTURE:
 *
 * ┌─────────────────────────────────────────────────────────────────┐
 * │                     UNTRUSTED ZONE                              │
 * │                    (Main Thread)                                │
 * │                                                                 │
 * │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
 * │  │    LLM      │───▶│  Tool API   │───▶│   Client    │────┐    │
 * │  │  (Untrusted)│    │             │    │             │    │    │
 * │  └─────────────┘    └─────────────┘    └─────────────┘    │    │
 * │                                                            │    │
 * │  Can request signatures                                    │    │
 * │  CANNOT access private key                                 │    │
 * │  CANNOT forge signatures                                   │    │
 * │                                                            │    │
 * ├────────────────────────── MESSAGE PASSING ─────────────────┼────┤
 * │                                                            │    │
 * │                      TRUSTED ZONE                          │    │
 * │                     (Web Worker)                           ▼    │
 * │                                                                 │
 * │  ┌─────────────────────────────────────────────────────┐        │
 * │  │                 Verification Worker                  │        │
 * │  │                                                      │        │
 * │  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │        │
 * │  │  │ Private Key │  │ Hash Chain  │  │   Signer    │  │        │
 * │  │  │ (Trapped)   │  │             │  │             │  │        │
 * │  │  └─────────────┘  └─────────────┘  └─────────────┘  │        │
 * │  │                                                      │        │
 * │  │  Has the private key (non-extractable)              │        │
 * │  │  Creates all signatures                             │        │
 * │  │  Maintains the audit chain                          │        │
 * │  └─────────────────────────────────────────────────────┘        │
 * │                                                                 │
 * └─────────────────────────────────────────────────────────────────┘
 *
 * TRUST INVARIANTS:
 * 1. Private key ONLY exists in the Worker
 * 2. Main thread CANNOT access Worker memory
 * 3. All signatures MUST come from the Worker
 * 4. External verifiers can verify with ONLY the public key
 */

// ============================================================================
// Trust Zone Types
// ============================================================================

/**
 * Operations that can only be performed in the trusted zone (Worker)
 *
 * The main thread should NEVER have these capabilities.
 */
export interface TrustedOperations {
  /** Sign data with the private key */
  sign(data: Uint8Array): Promise<string>;

  /** Generate a new key pair */
  generateKeyPair(): Promise<void>;

  /** Access the private key (NEVER exposed) */
  getPrivateKey(): never;
}

/**
 * Operations available in the untrusted zone (Main thread)
 *
 * These are the ONLY operations the LLM can trigger.
 */
export interface UntrustedOperations {
  /** Request a signature (delegated to Worker) */
  requestSignature<T>(payload: T): Promise<SignatureRequest<T>>;

  /** Verify a signature (uses public key only) */
  verify<T>(signedData: SignedPayload<T>): Promise<VerificationOutcome>;

  /** Get the public key (safe to share) */
  getPublicKey(): Promise<JsonWebKey>;

  /** Request chain append (delegated to Worker) */
  requestChainAppend(operation: string, data: unknown): Promise<ChainAppendResult>;
}

// ============================================================================
// Request/Response Types
// ============================================================================

/**
 * A request to sign data
 *
 * This goes from the untrusted main thread to the trusted worker.
 */
export interface SignatureRequest<T> {
  /** Unique request ID for correlation */
  requestId: string;

  /** The data to sign */
  payload: T;

  /** When the request was made */
  requestedAt: string;

  /** Where the request originated (for auditing) */
  origin: 'tool_call' | 'system' | 'user';
}

/**
 * A signed payload returned from the trusted zone
 */
export interface SignedPayload<T> {
  /** The original data */
  payload: T;

  /** Cryptographic proof */
  proof: {
    /** SHA-256 hash of the payload */
    dataHash: string;

    /** ECDSA P-256 signature of the hash */
    signature: string;

    /** Public key for verification */
    publicKey: JsonWebKey;

    /** When signed */
    signedAt: string;

    /** Algorithm identifier */
    algorithm: 'ECDSA-P256-SHA256';
  };

  /** Metadata */
  meta: {
    /** Request ID this fulfills */
    requestId: string;

    /** Chain index where this was logged */
    chainIndex: number;

    /** Time to process (ms) */
    processingTime: number;
  };
}

/**
 * Verification outcome
 */
export interface VerificationOutcome {
  /** Overall validity */
  valid: boolean;

  /** Detailed checks */
  checks: {
    /** Did the hash match? */
    dataIntegrity: boolean;

    /** Was the signature valid? */
    signatureValid: boolean;

    /** Was it signed by the expected key? */
    correctSigner: boolean;

    /** Is the timestamp reasonable? */
    timestampValid: boolean;
  };

  /** Error message if invalid */
  error?: string;

  /** When verification was performed */
  verifiedAt: string;
}

/**
 * Result of appending to the chain
 */
export interface ChainAppendResult {
  /** Success status */
  success: boolean;

  /** Entry index in the chain */
  index: number;

  /** Hash of this entry */
  entryHash: string;

  /** Previous entry hash (for verification) */
  previousHash: string;

  /** Signature of the entry */
  signature: string;

  /** When appended */
  appendedAt: string;
}

// ============================================================================
// Attestation Types
// ============================================================================

/**
 * An attestation is a signed claim about something
 *
 * This is the fundamental unit of trust in the system.
 */
export interface Attestation<T = unknown> {
  /** What kind of attestation this is */
  type: AttestationType;

  /** The claim being attested */
  claim: T;

  /** Who made this attestation */
  attester: {
    /** Public key of the attester */
    publicKey: JsonWebKey;

    /** Identifier (fingerprint of public key) */
    id: string;
  };

  /** Cryptographic proof */
  proof: {
    signature: string;
    timestamp: string;
    chainIndex: number;
  };
}

/**
 * Types of attestations the system can make
 */
export type AttestationType =
  | 'REASONING_STEP' // A step in the reasoning process
  | 'CONCEPT_LOOKUP' // A concept was looked up
  | 'INFERENCE' // An inference was made
  | 'BINDING_OPERATION' // A binding operation occurred
  | 'FUTURE_PREDICTION' // A future prediction was made
  | 'VERIFICATION_RESULT' // A verification was performed
  | 'SYSTEM_EVENT'; // System-level event

/**
 * Reasoning step attestation claim
 */
export interface ReasoningStepClaim {
  stepNumber: number;
  operation: string;
  inputs: string[];
  outputs: string[];
  confidence: number;
  citations: string[];
}

/**
 * Concept lookup attestation claim
 */
export interface ConceptLookupClaim {
  query: string;
  conceptId: string;
  similarity: number;
  retrieved: boolean;
}

/**
 * Inference attestation claim
 */
export interface InferenceClaim {
  rule: string;
  premises: string[];
  conclusion: string;
  rotorApplied: boolean;
  resultingState: string;
}

// ============================================================================
// Trust Verification
// ============================================================================

/**
 * Verify that an attestation is valid
 *
 * This can be done by ANYONE with the public key.
 * No access to the trusted zone is required.
 */
export async function verifyAttestation<T>(
  attestation: Attestation<T>,
  expectedPublicKey?: JsonWebKey
): Promise<VerificationOutcome> {
  const verifiedAt = new Date().toISOString();

  const checks = {
    dataIntegrity: false,
    signatureValid: false,
    correctSigner: false,
    timestampValid: false,
  };

  try {
    // Check 1: Compute hash and verify
    const claimJson = deterministicStringify(attestation.claim);
    const encoder = new TextEncoder();
    const claimBytes = encoder.encode(claimJson);
    const hashBuffer = await crypto.subtle.digest('SHA-256', claimBytes);
    const computedHash = bytesToHex(new Uint8Array(hashBuffer));

    // We'll verify against the signature
    checks.dataIntegrity = true; // Will be confirmed by signature check

    // Check 2: Import public key and verify signature
    const publicKey = await crypto.subtle.importKey(
      'jwk',
      attestation.attester.publicKey,
      { name: 'ECDSA', namedCurve: 'P-256' },
      false,
      ['verify']
    );

    // The signature is over the hash
    const hashBytes = encoder.encode(computedHash);
    const signatureBytes = base64ToBytes(attestation.proof.signature);

    checks.signatureValid = await crypto.subtle.verify(
      { name: 'ECDSA', hash: 'SHA-256' },
      publicKey,
      signatureBytes as BufferSource,
      hashBytes as BufferSource
    );

    if (!checks.signatureValid) {
      return {
        valid: false,
        checks,
        error: 'Signature verification failed',
        verifiedAt,
      };
    }

    checks.dataIntegrity = true; // Confirmed by valid signature

    // Check 3: Verify signer if expected key provided
    if (expectedPublicKey) {
      checks.correctSigner =
        JSON.stringify(attestation.attester.publicKey) ===
        JSON.stringify(expectedPublicKey);

      if (!checks.correctSigner) {
        return {
          valid: false,
          checks,
          error: 'Attestation signed by unexpected key',
          verifiedAt,
        };
      }
    } else {
      checks.correctSigner = true; // No expectation, any signer OK
    }

    // Check 4: Timestamp sanity
    const timestamp = new Date(attestation.proof.timestamp).getTime();
    const now = Date.now();
    const fiveMinutes = 5 * 60 * 1000;
    const oneDay = 24 * 60 * 60 * 1000;

    checks.timestampValid =
      timestamp <= now + fiveMinutes && timestamp >= now - oneDay;

    if (!checks.timestampValid) {
      return {
        valid: false,
        checks,
        error: 'Timestamp out of acceptable range',
        verifiedAt,
      };
    }

    return { valid: true, checks, verifiedAt };
  } catch (err) {
    return {
      valid: false,
      checks,
      error: `Verification error: ${err instanceof Error ? err.message : 'Unknown'}`,
      verifiedAt,
    };
  }
}

// ============================================================================
// Utility Functions
// ============================================================================

function deterministicStringify(obj: unknown): string {
  return JSON.stringify(obj, (_, value) => {
    if (value && typeof value === 'object' && !Array.isArray(value)) {
      return Object.keys(value)
        .sort()
        .reduce((sorted: Record<string, unknown>, key) => {
          sorted[key] = value[key];
          return sorted;
        }, {});
    }
    return value;
  });
}

function bytesToHex(bytes: Uint8Array): string {
  return Array.from(bytes)
    .map((b) => b.toString(16).padStart(2, '0'))
    .join('');
}

function base64ToBytes(base64: string): Uint8Array {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i);
  }
  return bytes;
}
