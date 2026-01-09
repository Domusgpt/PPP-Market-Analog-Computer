/**
 * PPP v3 Cryptographic Service
 *
 * HONEST DOCUMENTATION:
 * This module provides REAL cryptographic operations using Web Crypto API.
 * - SHA-256 hashing (not a simple 32-bit hash)
 * - ECDSA P-256 signatures (real asymmetric cryptography)
 *
 * WHAT THIS PROVIDES:
 * - Proof that data hasn't been tampered with (hash integrity)
 * - Proof that a signature came from the holder of the private key
 *
 * WHAT THIS DOES NOT PROVIDE:
 * - Proof that an LLM didn't hallucinate (that requires process isolation)
 * - Semantic truth of the content (cryptography doesn't verify meaning)
 * - Protection against replay attacks (requires additional mechanisms)
 */

// ============================================================================
// Types
// ============================================================================

export interface KeyPair {
  publicKey: CryptoKey;
  privateKey: CryptoKey;
}

export interface ExportedKeyPair {
  publicKey: JsonWebKey;
  privateKeyFingerprint: string; // We don't export private keys
}

export interface Signature {
  data: Uint8Array;
  base64: string;
}

export interface HashResult {
  bytes: Uint8Array;
  hex: string;
  base64: string;
}

export interface SignedData<T> {
  payload: T;
  proof: {
    dataHash: string;        // SHA-256 of payload
    signature: string;       // Base64 ECDSA signature
    publicKeyJwk: JsonWebKey; // Public key for verification
    timestamp: string;       // ISO 8601
    algorithm: 'ECDSA-P256-SHA256';
  };
}

// ============================================================================
// Implementation
// ============================================================================

/**
 * CryptoService - Real cryptographic operations
 *
 * Uses Web Crypto API which provides:
 * - Hardware-accelerated operations where available
 * - Cryptographically secure random number generation
 * - Constant-time comparison (timing attack resistant)
 */
export class CryptoService {
  private keyPair: KeyPair | null = null;
  private publicKeyJwk: JsonWebKey | null = null;

  /**
   * Initialize with a new key pair
   * Call this once at startup
   */
  async initialize(): Promise<void> {
    this.keyPair = await this.generateKeyPair();
    this.publicKeyJwk = await crypto.subtle.exportKey('jwk', this.keyPair.publicKey);
  }

  /**
   * Check if initialized
   */
  isInitialized(): boolean {
    return this.keyPair !== null;
  }

  /**
   * Get public key for external verification
   * This is safe to share - it's the PUBLIC key
   */
  getPublicKeyJwk(): JsonWebKey {
    if (!this.publicKeyJwk) {
      throw new Error('CryptoService not initialized. Call initialize() first.');
    }
    return this.publicKeyJwk;
  }

  /**
   * Generate a new ECDSA P-256 key pair
   *
   * P-256 chosen because:
   * - Widely supported in Web Crypto
   * - 128-bit security level (sufficient for this use case)
   * - Fast signature verification
   */
  async generateKeyPair(): Promise<KeyPair> {
    const keyPair = await crypto.subtle.generateKey(
      {
        name: 'ECDSA',
        namedCurve: 'P-256',
      },
      true, // extractable (for public key export)
      ['sign', 'verify']
    );

    return {
      publicKey: keyPair.publicKey,
      privateKey: keyPair.privateKey,
    };
  }

  /**
   * Compute SHA-256 hash of data
   *
   * This is a REAL cryptographic hash:
   * - 256-bit output
   * - Collision resistant
   * - One-way function
   */
  async hash(data: Uint8Array): Promise<HashResult> {
    const hashBuffer = await crypto.subtle.digest('SHA-256', data as BufferSource);
    const bytes = new Uint8Array(hashBuffer);

    return {
      bytes,
      hex: this.bytesToHex(bytes),
      base64: this.bytesToBase64(bytes),
    };
  }

  /**
   * Hash a string (UTF-8 encoded)
   */
  async hashString(str: string): Promise<HashResult> {
    const encoder = new TextEncoder();
    return this.hash(encoder.encode(str));
  }

  /**
   * Hash an object (JSON serialized, keys sorted for determinism)
   */
  async hashObject(obj: unknown): Promise<HashResult> {
    const json = this.deterministicStringify(obj);
    return this.hashString(json);
  }

  /**
   * Sign data with the private key
   *
   * WHAT THIS PROVES:
   * - The signature was created by someone with access to the private key
   * - The data has not been modified since signing
   *
   * WHAT THIS DOES NOT PROVE:
   * - The content is true or accurate
   * - The signer is trustworthy
   */
  async sign(data: Uint8Array): Promise<Signature> {
    if (!this.keyPair) {
      throw new Error('CryptoService not initialized');
    }

    const signatureBuffer = await crypto.subtle.sign(
      {
        name: 'ECDSA',
        hash: 'SHA-256',
      },
      this.keyPair.privateKey,
      data as BufferSource
    );

    const signatureBytes = new Uint8Array(signatureBuffer);

    return {
      data: signatureBytes,
      base64: this.bytesToBase64(signatureBytes),
    };
  }

  /**
   * Sign a string
   */
  async signString(str: string): Promise<Signature> {
    const encoder = new TextEncoder();
    return this.sign(encoder.encode(str));
  }

  /**
   * Verify a signature
   *
   * This can be done with ONLY the public key.
   * External verifiers can verify without access to private key.
   */
  async verify(
    data: Uint8Array,
    signature: Uint8Array,
    publicKey: CryptoKey
  ): Promise<boolean> {
    try {
      return await crypto.subtle.verify(
        {
          name: 'ECDSA',
          hash: 'SHA-256',
        },
        publicKey,
        signature as BufferSource,
        data as BufferSource
      );
    } catch {
      return false;
    }
  }

  /**
   * Verify a signature using a JWK public key
   * This is what external verifiers will use
   */
  async verifyWithJwk(
    data: Uint8Array,
    signatureBase64: string,
    publicKeyJwk: JsonWebKey
  ): Promise<boolean> {
    try {
      const publicKey = await crypto.subtle.importKey(
        'jwk',
        publicKeyJwk,
        {
          name: 'ECDSA',
          namedCurve: 'P-256',
        },
        false,
        ['verify']
      );

      const signature = this.base64ToBytes(signatureBase64);
      return this.verify(data, signature, publicKey);
    } catch {
      return false;
    }
  }

  /**
   * Create a signed wrapper around any data
   *
   * The result includes everything needed for external verification:
   * - The original payload
   * - Hash of the payload
   * - Signature of the hash
   * - Public key for verification
   * - Timestamp
   */
  async signData<T>(payload: T): Promise<SignedData<T>> {
    if (!this.publicKeyJwk) {
      throw new Error('CryptoService not initialized');
    }

    const payloadJson = this.deterministicStringify(payload);
    const hash = await this.hashString(payloadJson);
    const signature = await this.signString(hash.hex);

    return {
      payload,
      proof: {
        dataHash: hash.hex,
        signature: signature.base64,
        publicKeyJwk: this.publicKeyJwk,
        timestamp: new Date().toISOString(),
        algorithm: 'ECDSA-P256-SHA256',
      },
    };
  }

  /**
   * Verify signed data
   *
   * Returns detailed verification result
   */
  async verifySignedData<T>(signedData: SignedData<T>): Promise<{
    valid: boolean;
    checks: {
      hashMatches: boolean;
      signatureValid: boolean;
      timestampReasonable: boolean;
    };
    error?: string;
  }> {
    const checks = {
      hashMatches: false,
      signatureValid: false,
      timestampReasonable: false,
    };

    try {
      // Check 1: Recompute hash and compare
      const payloadJson = this.deterministicStringify(signedData.payload);
      const computedHash = await this.hashString(payloadJson);
      checks.hashMatches = computedHash.hex === signedData.proof.dataHash;

      if (!checks.hashMatches) {
        return {
          valid: false,
          checks,
          error: 'Hash mismatch - data may have been tampered with',
        };
      }

      // Check 2: Verify signature
      const encoder = new TextEncoder();
      const hashBytes = encoder.encode(signedData.proof.dataHash);

      checks.signatureValid = await this.verifyWithJwk(
        hashBytes,
        signedData.proof.signature,
        signedData.proof.publicKeyJwk
      );

      if (!checks.signatureValid) {
        return {
          valid: false,
          checks,
          error: 'Invalid signature - data may be forged',
        };
      }

      // Check 3: Timestamp sanity check
      const timestamp = new Date(signedData.proof.timestamp).getTime();
      const now = Date.now();
      const fiveMinutes = 5 * 60 * 1000;

      // Allow 5 minutes of clock skew
      checks.timestampReasonable =
        timestamp <= now + fiveMinutes && timestamp >= now - 24 * 60 * 60 * 1000;

      if (!checks.timestampReasonable) {
        return {
          valid: false,
          checks,
          error: 'Timestamp out of acceptable range',
        };
      }

      return { valid: true, checks };
    } catch (err) {
      return {
        valid: false,
        checks,
        error: `Verification error: ${err instanceof Error ? err.message : 'Unknown'}`,
      };
    }
  }

  // ============================================================================
  // Helper Methods
  // ============================================================================

  /**
   * Deterministic JSON stringify (sorted keys)
   * Essential for consistent hashing
   */
  private deterministicStringify(obj: unknown): string {
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

  /**
   * Convert bytes to hex string
   */
  private bytesToHex(bytes: Uint8Array): string {
    return Array.from(bytes)
      .map((b) => b.toString(16).padStart(2, '0'))
      .join('');
  }

  /**
   * Convert bytes to base64
   */
  private bytesToBase64(bytes: Uint8Array): string {
    // Browser-compatible base64 encoding
    const binary = Array.from(bytes)
      .map((b) => String.fromCharCode(b))
      .join('');
    return btoa(binary);
  }

  /**
   * Convert base64 to bytes
   */
  private base64ToBytes(base64: string): Uint8Array {
    const binary = atob(base64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) {
      bytes[i] = binary.charCodeAt(i);
    }
    return bytes;
  }
}

// ============================================================================
// Singleton for convenience
// ============================================================================

let globalCryptoService: CryptoService | null = null;

export async function getCryptoService(): Promise<CryptoService> {
  if (!globalCryptoService) {
    globalCryptoService = new CryptoService();
    await globalCryptoService.initialize();
  }
  return globalCryptoService;
}

export function resetCryptoService(): void {
  globalCryptoService = null;
}
