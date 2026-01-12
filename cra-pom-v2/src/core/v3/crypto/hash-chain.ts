/**
 * PPP v3 Hash Chain
 *
 * HONEST DOCUMENTATION:
 * This is a cryptographically-signed append-only log of operations.
 *
 * WHAT THIS PROVIDES:
 * - Tamper-evident record of operations
 * - Verifiable ordering via hash chain
 * - Non-repudiation via signatures
 *
 * WHAT THIS DOES NOT PROVIDE:
 * - Proof that the LLM used this for reasoning (LLM can ignore it)
 * - Semantic truth of operations (signing doesn't validate meaning)
 * - Protection against the signer lying (signer controls content)
 *
 * TRUST MODEL:
 * This chain is trustworthy IF AND ONLY IF:
 * 1. The signing key is held by a trusted party
 * 2. That party runs in a process the LLM cannot access
 * 3. External verification is performed
 */

import { CryptoService, SignedData } from './crypto-service';

// ============================================================================
// Types
// ============================================================================

export interface ChainEntry<T = unknown> {
  // Sequence number (0-indexed)
  index: number;

  // ISO 8601 timestamp
  timestamp: string;

  // Type of operation
  operationType: string;

  // The actual operation data
  data: T;

  // Hash of previous entry (creates the chain)
  previousHash: string;

  // Hash of this entry's content (before signing)
  contentHash: string;
}

export interface SignedChainEntry<T = unknown> extends SignedData<ChainEntry<T>> {
  // Inherited from SignedData:
  // payload: ChainEntry<T>
  // proof: { dataHash, signature, publicKeyJwk, timestamp, algorithm }
}

export interface ChainValidationResult {
  valid: boolean;
  length: number;
  brokenAt?: number;
  error?: string;
  details: {
    hashChainIntact: boolean;
    signaturesValid: boolean;
    sequenceCorrect: boolean;
  };
}

// ============================================================================
// Constants
// ============================================================================

const GENESIS_HASH = '0'.repeat(64); // 64 hex chars = 256 bits

// ============================================================================
// Implementation
// ============================================================================

export class SignedHashChain {
  private entries: SignedChainEntry[] = [];
  private headHash: string = GENESIS_HASH;
  private crypto: CryptoService;
  private initialized: boolean = false;

  constructor(crypto: CryptoService) {
    this.crypto = crypto;
  }

  /**
   * Initialize the chain with a genesis entry
   */
  async initialize(): Promise<SignedChainEntry> {
    if (this.initialized) {
      throw new Error('Chain already initialized');
    }

    const entry = await this.append('GENESIS', {
      message: 'Chain initialized',
      version: '3.0.0',
      algorithm: 'ECDSA-P256-SHA256',
    });

    this.initialized = true;
    return entry;
  }

  /**
   * Append a new entry to the chain
   *
   * Process:
   * 1. Create entry with previous hash link
   * 2. Compute content hash
   * 3. Sign the entry
   * 4. Append to chain
   * 5. Update head hash
   */
  async append<T>(operationType: string, data: T): Promise<SignedChainEntry<T>> {
    const index = this.entries.length;
    const timestamp = new Date().toISOString();

    // Create the entry (without hashes yet)
    const entryContent = {
      index,
      timestamp,
      operationType,
      data,
      previousHash: this.headHash,
      contentHash: '', // Will be filled
    };

    // Compute content hash (of everything except contentHash itself)
    const contentForHashing = {
      ...entryContent,
      contentHash: undefined,
    };
    const contentHash = await this.crypto.hashObject(contentForHashing);
    entryContent.contentHash = contentHash.hex;

    // Sign the complete entry
    const signedEntry = await this.crypto.signData(entryContent) as SignedChainEntry<T>;

    // Append and update head
    this.entries.push(signedEntry as SignedChainEntry);
    this.headHash = signedEntry.proof.dataHash;

    return signedEntry;
  }

  /**
   * Get the current head hash
   */
  getHeadHash(): string {
    return this.headHash;
  }

  /**
   * Get chain length
   */
  get length(): number {
    return this.entries.length;
  }

  /**
   * Get entry by index
   */
  getEntry(index: number): SignedChainEntry | undefined {
    return this.entries[index];
  }

  /**
   * Get the last N entries
   */
  getRecent(count: number): SignedChainEntry[] {
    const start = Math.max(0, this.entries.length - count);
    return this.entries.slice(start);
  }

  /**
   * Validate the entire chain
   *
   * Checks:
   * 1. Hash chain integrity (each entry links to previous)
   * 2. All signatures are valid
   * 3. Sequence numbers are correct
   */
  async validate(): Promise<ChainValidationResult> {
    const details = {
      hashChainIntact: true,
      signaturesValid: true,
      sequenceCorrect: true,
    };

    if (this.entries.length === 0) {
      return { valid: true, length: 0, details };
    }

    let expectedPrevHash = GENESIS_HASH;

    for (let i = 0; i < this.entries.length; i++) {
      const entry = this.entries[i];

      // Check 1: Sequence number
      if (entry.payload.index !== i) {
        details.sequenceCorrect = false;
        return {
          valid: false,
          length: this.entries.length,
          brokenAt: i,
          error: `Sequence mismatch at index ${i}: expected ${i}, got ${entry.payload.index}`,
          details,
        };
      }

      // Check 2: Previous hash link
      if (entry.payload.previousHash !== expectedPrevHash) {
        details.hashChainIntact = false;
        return {
          valid: false,
          length: this.entries.length,
          brokenAt: i,
          error: `Hash chain broken at index ${i}`,
          details,
        };
      }

      // Check 3: Signature validity
      const verification = await this.crypto.verifySignedData(entry);
      if (!verification.valid) {
        details.signaturesValid = false;
        return {
          valid: false,
          length: this.entries.length,
          brokenAt: i,
          error: `Invalid signature at index ${i}: ${verification.error}`,
          details,
        };
      }

      // Update expected previous hash for next iteration
      expectedPrevHash = entry.proof.dataHash;
    }

    return { valid: true, length: this.entries.length, details };
  }

  /**
   * Export the chain for external verification
   *
   * The exported data includes everything needed to verify independently:
   * - All entries with signatures
   * - The public key
   * - Chain metadata
   */
  export(): {
    version: string;
    publicKey: JsonWebKey;
    genesisHash: string;
    headHash: string;
    length: number;
    entries: SignedChainEntry[];
  } {
    return {
      version: '3.0.0',
      publicKey: this.crypto.getPublicKeyJwk(),
      genesisHash: GENESIS_HASH,
      headHash: this.headHash,
      length: this.entries.length,
      entries: [...this.entries],
    };
  }

  /**
   * Verify an exported chain (static method for external use)
   *
   * This can be used by external systems that only have the exported chain
   * and the public key. They don't need access to the crypto service that
   * created the signatures.
   */
  static async verifyExported(
    exported: ReturnType<SignedHashChain['export']>
  ): Promise<ChainValidationResult> {
    const details = {
      hashChainIntact: true,
      signaturesValid: true,
      sequenceCorrect: true,
    };

    if (exported.entries.length === 0) {
      return { valid: true, length: 0, details };
    }

    // Create a temporary crypto service for verification only
    const tempCrypto = new CryptoService();
    await tempCrypto.initialize();

    let expectedPrevHash = exported.genesisHash;

    for (let i = 0; i < exported.entries.length; i++) {
      const entry = exported.entries[i];

      // Check sequence
      if (entry.payload.index !== i) {
        details.sequenceCorrect = false;
        return {
          valid: false,
          length: exported.entries.length,
          brokenAt: i,
          error: `Sequence mismatch at index ${i}`,
          details,
        };
      }

      // Check hash chain
      if (entry.payload.previousHash !== expectedPrevHash) {
        details.hashChainIntact = false;
        return {
          valid: false,
          length: exported.entries.length,
          brokenAt: i,
          error: `Hash chain broken at index ${i}`,
          details,
        };
      }

      // Verify signature using the exported public key
      const verification = await tempCrypto.verifySignedData(entry);
      if (!verification.valid) {
        details.signaturesValid = false;
        return {
          valid: false,
          length: exported.entries.length,
          brokenAt: i,
          error: `Invalid signature at index ${i}: ${verification.error}`,
          details,
        };
      }

      expectedPrevHash = entry.proof.dataHash;
    }

    // Verify head hash matches
    if (expectedPrevHash !== exported.headHash) {
      details.hashChainIntact = false;
      return {
        valid: false,
        length: exported.entries.length,
        error: 'Head hash mismatch',
        details,
      };
    }

    return { valid: true, length: exported.entries.length, details };
  }
}

// ============================================================================
// Factory
// ============================================================================

export async function createSignedHashChain(
  crypto: CryptoService
): Promise<SignedHashChain> {
  const chain = new SignedHashChain(crypto);
  await chain.initialize();
  return chain;
}
