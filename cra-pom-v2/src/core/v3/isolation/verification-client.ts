/**
 * PPP v3 Verification Client
 *
 * This is the main-thread interface to the isolated verification worker.
 *
 * TRUST MODEL:
 * - This client runs in the main thread (where the LLM operates)
 * - All signing operations are delegated to the worker
 * - The LLM CANNOT access the private key because it's in the worker
 *
 * SECURITY PROPERTIES:
 * - Message-passing only (no shared memory)
 * - Request/response correlation via UUIDs
 * - Timeout handling for hung workers
 *
 * WHAT THIS PROVIDES:
 * - Clean async API for verification operations
 * - Type-safe request/response handling
 * - Automatic worker lifecycle management
 *
 * WHAT THIS DOES NOT PROVIDE:
 * - The ability to forge signatures (that's the point!)
 * - Access to the private key
 */

import type { SignedData } from '../crypto';

// ============================================================================
// Types
// ============================================================================

type RequestType =
  | 'INITIALIZE'
  | 'SIGN_DATA'
  | 'VERIFY_DATA'
  | 'GET_PUBLIC_KEY'
  | 'APPEND_TO_CHAIN'
  | 'VALIDATE_CHAIN'
  | 'EXPORT_CHAIN'
  | 'GET_STATS';

interface WorkerRequest {
  id: string;
  type: RequestType;
  payload?: unknown;
}

interface WorkerResponse {
  id: string;
  success: boolean;
  result?: unknown;
  error?: string;
}

interface ChainEntry<T = unknown> {
  index: number;
  timestamp: string;
  operationType: string;
  data: T;
  previousHash: string;
  contentHash: string;
}

interface SignedChainEntry<T = unknown> extends SignedData<ChainEntry<T>> {}

interface ChainValidationResult {
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

interface ExportedChain {
  version: string;
  publicKey: JsonWebKey | null;
  genesisHash: string;
  headHash: string;
  length: number;
  entries: SignedChainEntry[];
}

interface VerificationResult {
  valid: boolean;
  checks: {
    hashMatches: boolean;
    signatureValid: boolean;
    timestampReasonable: boolean;
  };
  error?: string;
}

interface WorkerStats {
  initialized: boolean;
  chainLength: number;
  signingRequests: number;
  verificationRequests: number;
  lastActivity: string;
  uptime: number;
}

// ============================================================================
// Client Implementation
// ============================================================================

/**
 * VerificationClient - Main thread interface to the verification worker
 *
 * This class manages the Web Worker and provides a clean async API.
 * All cryptographic operations are performed in the isolated worker.
 */
export class VerificationClient {
  private worker: Worker | null = null;
  private pendingRequests: Map<
    string,
    { resolve: (value: unknown) => void; reject: (error: Error) => void }
  > = new Map();
  private initialized = false;
  private readyPromise: Promise<void> | null = null;
  private workerUrl: string;

  // Default timeout for worker operations (30 seconds)
  private static readonly DEFAULT_TIMEOUT = 30000;

  constructor(workerUrl?: string) {
    // Default to the verification worker in the same directory
    this.workerUrl = workerUrl || new URL('./verification-worker.ts', import.meta.url).href;
  }

  /**
   * Initialize the verification client
   *
   * This:
   * 1. Creates the Web Worker
   * 2. Waits for the worker to be ready
   * 3. Initializes the cryptographic key pair in the worker
   */
  async initialize(): Promise<{ publicKey: JsonWebKey }> {
    if (this.initialized) {
      throw new Error('VerificationClient already initialized');
    }

    // Create worker
    this.worker = new Worker(this.workerUrl, { type: 'module' });

    // Set up message handler
    this.worker.onmessage = this.handleMessage.bind(this);
    this.worker.onerror = this.handleError.bind(this);

    // Wait for worker ready signal
    await this.waitForWorkerReady();

    // Initialize crypto in worker
    const result = await this.sendRequest<{ publicKey: JsonWebKey }>(
      'INITIALIZE'
    );

    this.initialized = true;
    return result;
  }

  /**
   * Check if initialized
   */
  isInitialized(): boolean {
    return this.initialized;
  }

  /**
   * Get the public key (safe to share)
   */
  async getPublicKey(): Promise<JsonWebKey> {
    this.ensureInitialized();
    return this.sendRequest<JsonWebKey>('GET_PUBLIC_KEY');
  }

  /**
   * Sign arbitrary data
   *
   * The signing happens in the isolated worker.
   * The LLM in the main thread cannot forge this signature.
   */
  async signData<T>(payload: T): Promise<SignedData<T>> {
    this.ensureInitialized();
    return this.sendRequest<SignedData<T>>('SIGN_DATA', payload);
  }

  /**
   * Verify signed data
   *
   * Can verify signatures from this worker or any other source
   * that uses the same algorithm (ECDSA P-256).
   */
  async verifyData<T>(signedData: SignedData<T>): Promise<VerificationResult> {
    this.ensureInitialized();
    return this.sendRequest<VerificationResult>('VERIFY_DATA', signedData);
  }

  /**
   * Append an entry to the hash chain
   *
   * The chain provides:
   * - Tamper-evident ordering
   * - Signed entries that can be verified externally
   */
  async appendToChain<T>(
    operationType: string,
    data: T
  ): Promise<SignedChainEntry<T>> {
    this.ensureInitialized();
    return this.sendRequest<SignedChainEntry<T>>('APPEND_TO_CHAIN', {
      operationType,
      data,
    });
  }

  /**
   * Validate the entire hash chain
   */
  async validateChain(): Promise<ChainValidationResult> {
    this.ensureInitialized();
    return this.sendRequest<ChainValidationResult>('VALIDATE_CHAIN');
  }

  /**
   * Export the chain for external verification
   */
  async exportChain(): Promise<ExportedChain> {
    this.ensureInitialized();
    return this.sendRequest<ExportedChain>('EXPORT_CHAIN');
  }

  /**
   * Get worker statistics
   */
  async getStats(): Promise<WorkerStats> {
    this.ensureInitialized();
    return this.sendRequest<WorkerStats>('GET_STATS');
  }

  /**
   * Terminate the worker
   *
   * After this, the private key is destroyed and no more
   * signatures can be created.
   */
  terminate(): void {
    if (this.worker) {
      this.worker.terminate();
      this.worker = null;
      this.initialized = false;

      // Reject all pending requests
      for (const [id, { reject }] of this.pendingRequests) {
        reject(new Error('Worker terminated'));
        this.pendingRequests.delete(id);
      }
    }
  }

  // ============================================================================
  // Private Methods
  // ============================================================================

  private ensureInitialized(): void {
    if (!this.initialized || !this.worker) {
      throw new Error('VerificationClient not initialized. Call initialize() first.');
    }
  }

  private waitForWorkerReady(): Promise<void> {
    if (this.readyPromise) {
      return this.readyPromise;
    }

    this.readyPromise = new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        reject(new Error('Worker ready timeout'));
      }, 10000);

      const handler = (event: MessageEvent) => {
        if (event.data?.type === 'WORKER_READY') {
          clearTimeout(timeout);
          this.worker?.removeEventListener('message', handler);
          resolve();
        }
      };

      this.worker?.addEventListener('message', handler);
    });

    return this.readyPromise;
  }

  private generateRequestId(): string {
    // Generate a unique ID for request correlation
    return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  }

  private sendRequest<T>(
    type: RequestType,
    payload?: unknown,
    timeout = VerificationClient.DEFAULT_TIMEOUT
  ): Promise<T> {
    return new Promise((resolve, reject) => {
      const id = this.generateRequestId();

      // Set up timeout
      const timeoutId = setTimeout(() => {
        this.pendingRequests.delete(id);
        reject(new Error(`Request timeout: ${type}`));
      }, timeout);

      // Store resolver
      this.pendingRequests.set(id, {
        resolve: (result) => {
          clearTimeout(timeoutId);
          resolve(result as T);
        },
        reject: (error) => {
          clearTimeout(timeoutId);
          reject(error);
        },
      });

      // Send request
      const request: WorkerRequest = { id, type, payload };
      this.worker?.postMessage(request);
    });
  }

  private handleMessage(event: MessageEvent<WorkerResponse>): void {
    const { id, success, result, error } = event.data;

    // Skip non-response messages (like WORKER_READY)
    if (!id) return;

    const pending = this.pendingRequests.get(id);
    if (!pending) {
      console.warn('Received response for unknown request:', id);
      return;
    }

    this.pendingRequests.delete(id);

    if (success) {
      pending.resolve(result);
    } else {
      pending.reject(new Error(error || 'Unknown worker error'));
    }
  }

  private handleError(error: ErrorEvent): void {
    console.error('Worker error:', error);

    // Reject all pending requests
    for (const [id, { reject }] of this.pendingRequests) {
      reject(new Error(`Worker error: ${error.message}`));
      this.pendingRequests.delete(id);
    }
  }
}

// ============================================================================
// Singleton for convenience
// ============================================================================

let globalClient: VerificationClient | null = null;

/**
 * Get the global verification client
 *
 * This provides a single shared client for the entire application.
 * The client (and its worker) are lazily initialized on first use.
 */
export async function getVerificationClient(): Promise<VerificationClient> {
  if (!globalClient) {
    globalClient = new VerificationClient();
    await globalClient.initialize();
  }
  return globalClient;
}

/**
 * Reset the global client (for testing)
 */
export function resetVerificationClient(): void {
  if (globalClient) {
    globalClient.terminate();
    globalClient = null;
  }
}

// ============================================================================
// Convenience Functions
// ============================================================================

/**
 * Sign data using the global verification client
 *
 * This is the primary API for the rest of the application.
 * The LLM CANNOT call this directly to forge signatures - it can only
 * request signatures through the proper tool interface.
 */
export async function signWithIsolation<T>(payload: T): Promise<SignedData<T>> {
  const client = await getVerificationClient();
  return client.signData(payload);
}

/**
 * Verify signed data
 */
export async function verifyWithIsolation<T>(
  signedData: SignedData<T>
): Promise<VerificationResult> {
  const client = await getVerificationClient();
  return client.verifyData(signedData);
}

/**
 * Append to the audit chain
 */
export async function appendToAuditChain<T>(
  operationType: string,
  data: T
): Promise<SignedChainEntry<T>> {
  const client = await getVerificationClient();
  return client.appendToChain(operationType, data);
}
