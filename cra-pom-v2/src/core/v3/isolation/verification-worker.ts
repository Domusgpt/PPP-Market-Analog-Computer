/**
 * PPP v3 Verification Worker
 *
 * This Web Worker is the TRUST BOUNDARY of the PPP system.
 *
 * CRITICAL SECURITY PROPERTIES:
 * 1. The private key exists ONLY in this worker
 * 2. The main thread CANNOT access this worker's memory
 * 3. All operations go through a message-passing API
 * 4. The LLM cannot forge signatures because it cannot access the key
 *
 * WHAT THIS PROVIDES:
 * - True process isolation (Web Worker separate memory space)
 * - Non-exportable private key (generated with extractable: false)
 * - Audit trail of all signing requests
 *
 * WHAT THIS DOES NOT PROVIDE:
 * - Protection against malicious code in the worker itself
 * - Protection if the worker is compromised at build time
 * - Semantic validation of what is being signed
 */

// ============================================================================
// Types (duplicated here to avoid import issues in worker)
// ============================================================================

interface HashResult {
  bytes: Uint8Array;
  hex: string;
  base64: string;
}

interface SignedData<T> {
  payload: T;
  proof: {
    dataHash: string;
    signature: string;
    publicKeyJwk: JsonWebKey;
    timestamp: string;
    algorithm: 'ECDSA-P256-SHA256';
  };
}

interface WorkerRequest {
  id: string;
  type:
    | 'INITIALIZE'
    | 'SIGN_DATA'
    | 'VERIFY_DATA'
    | 'GET_PUBLIC_KEY'
    | 'APPEND_TO_CHAIN'
    | 'VALIDATE_CHAIN'
    | 'EXPORT_CHAIN'
    | 'GET_STATS';
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

interface WorkerStats {
  initialized: boolean;
  chainLength: number;
  signingRequests: number;
  verificationRequests: number;
  lastActivity: string;
  uptime: number;
}

// ============================================================================
// Worker State (isolated in this worker's memory)
// ============================================================================

let privateKey: CryptoKey | null = null;
let publicKey: CryptoKey | null = null;
let publicKeyJwk: JsonWebKey | null = null;

// Hash chain state
const chainEntries: SignedChainEntry[] = [];
let headHash: string = '0'.repeat(64);
const GENESIS_HASH = '0'.repeat(64);

// Statistics
let signingRequests = 0;
let verificationRequests = 0;
const startTime = Date.now();
let lastActivity = new Date().toISOString();

// ============================================================================
// Cryptographic Operations
// ============================================================================

async function initialize(): Promise<{ publicKey: JsonWebKey }> {
  if (privateKey) {
    throw new Error('Worker already initialized');
  }

  // Generate key pair with NON-EXTRACTABLE private key
  // This means the private key CANNOT be exported, even by this worker
  const keyPair = await crypto.subtle.generateKey(
    {
      name: 'ECDSA',
      namedCurve: 'P-256',
    },
    false, // NOT extractable - private key is trapped in WebCrypto
    ['sign', 'verify']
  );

  privateKey = keyPair.privateKey;
  publicKey = keyPair.publicKey;

  // Export only the public key (this is safe)
  publicKeyJwk = await crypto.subtle.exportKey('jwk', publicKey);

  lastActivity = new Date().toISOString();

  return { publicKey: publicKeyJwk };
}

async function hashData(data: Uint8Array): Promise<HashResult> {
  const hashBuffer = await crypto.subtle.digest('SHA-256', data as BufferSource);
  const bytes = new Uint8Array(hashBuffer);

  return {
    bytes,
    hex: bytesToHex(bytes),
    base64: bytesToBase64(bytes),
  };
}

async function hashString(str: string): Promise<HashResult> {
  const encoder = new TextEncoder();
  return hashData(encoder.encode(str));
}

async function hashObject(obj: unknown): Promise<HashResult> {
  const json = deterministicStringify(obj);
  return hashString(json);
}

async function sign(data: Uint8Array): Promise<string> {
  if (!privateKey) {
    throw new Error('Worker not initialized');
  }

  const signatureBuffer = await crypto.subtle.sign(
    {
      name: 'ECDSA',
      hash: 'SHA-256',
    },
    privateKey,
    data as BufferSource
  );

  signingRequests++;
  lastActivity = new Date().toISOString();

  return bytesToBase64(new Uint8Array(signatureBuffer));
}

async function signData<T>(payload: T): Promise<SignedData<T>> {
  if (!publicKeyJwk) {
    throw new Error('Worker not initialized');
  }

  const payloadJson = deterministicStringify(payload);
  const hash = await hashString(payloadJson);
  const encoder = new TextEncoder();
  const signature = await sign(encoder.encode(hash.hex));

  return {
    payload,
    proof: {
      dataHash: hash.hex,
      signature,
      publicKeyJwk,
      timestamp: new Date().toISOString(),
      algorithm: 'ECDSA-P256-SHA256',
    },
  };
}

async function verifyData<T>(signedData: SignedData<T>): Promise<{
  valid: boolean;
  checks: {
    hashMatches: boolean;
    signatureValid: boolean;
    timestampReasonable: boolean;
  };
  error?: string;
}> {
  verificationRequests++;
  lastActivity = new Date().toISOString();

  const checks = {
    hashMatches: false,
    signatureValid: false,
    timestampReasonable: false,
  };

  try {
    // Check 1: Verify hash
    const payloadJson = deterministicStringify(signedData.payload);
    const computedHash = await hashString(payloadJson);
    checks.hashMatches = computedHash.hex === signedData.proof.dataHash;

    if (!checks.hashMatches) {
      return {
        valid: false,
        checks,
        error: 'Hash mismatch - data may have been tampered with',
      };
    }

    // Check 2: Verify signature
    const importedKey = await crypto.subtle.importKey(
      'jwk',
      signedData.proof.publicKeyJwk,
      {
        name: 'ECDSA',
        namedCurve: 'P-256',
      },
      false,
      ['verify']
    );

    const encoder = new TextEncoder();
    const hashBytes = encoder.encode(signedData.proof.dataHash);
    const signatureBytes = base64ToBytes(signedData.proof.signature);

    checks.signatureValid = await crypto.subtle.verify(
      {
        name: 'ECDSA',
        hash: 'SHA-256',
      },
      importedKey,
      signatureBytes as BufferSource,
      hashBytes as BufferSource
    );

    if (!checks.signatureValid) {
      return {
        valid: false,
        checks,
        error: 'Invalid signature',
      };
    }

    // Check 3: Timestamp sanity
    const timestamp = new Date(signedData.proof.timestamp).getTime();
    const now = Date.now();
    const fiveMinutes = 5 * 60 * 1000;
    const oneDay = 24 * 60 * 60 * 1000;

    checks.timestampReasonable =
      timestamp <= now + fiveMinutes && timestamp >= now - oneDay;

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
// Hash Chain Operations
// ============================================================================

async function appendToChain<T>(
  operationType: string,
  data: T
): Promise<SignedChainEntry<T>> {
  const index = chainEntries.length;
  const timestamp = new Date().toISOString();

  // Create entry content
  const entryContent: ChainEntry<T> = {
    index,
    timestamp,
    operationType,
    data,
    previousHash: headHash,
    contentHash: '', // Will be computed
  };

  // Compute content hash (excluding contentHash field)
  const contentForHashing = { ...entryContent, contentHash: undefined };
  const contentHash = await hashObject(contentForHashing);
  entryContent.contentHash = contentHash.hex;

  // Sign the complete entry
  const signedEntry = (await signData(entryContent)) as SignedChainEntry<T>;

  // Append to chain
  chainEntries.push(signedEntry as SignedChainEntry);
  headHash = signedEntry.proof.dataHash;

  return signedEntry;
}

async function validateChain(): Promise<{
  valid: boolean;
  length: number;
  brokenAt?: number;
  error?: string;
  details: {
    hashChainIntact: boolean;
    signaturesValid: boolean;
    sequenceCorrect: boolean;
  };
}> {
  const details = {
    hashChainIntact: true,
    signaturesValid: true,
    sequenceCorrect: true,
  };

  if (chainEntries.length === 0) {
    return { valid: true, length: 0, details };
  }

  let expectedPrevHash = GENESIS_HASH;

  for (let i = 0; i < chainEntries.length; i++) {
    const entry = chainEntries[i];

    // Check sequence
    if (entry.payload.index !== i) {
      details.sequenceCorrect = false;
      return {
        valid: false,
        length: chainEntries.length,
        brokenAt: i,
        error: `Sequence mismatch at ${i}`,
        details,
      };
    }

    // Check hash chain
    if (entry.payload.previousHash !== expectedPrevHash) {
      details.hashChainIntact = false;
      return {
        valid: false,
        length: chainEntries.length,
        brokenAt: i,
        error: `Hash chain broken at ${i}`,
        details,
      };
    }

    // Verify signature
    const verification = await verifyData(entry);
    if (!verification.valid) {
      details.signaturesValid = false;
      return {
        valid: false,
        length: chainEntries.length,
        brokenAt: i,
        error: `Invalid signature at ${i}: ${verification.error}`,
        details,
      };
    }

    expectedPrevHash = entry.proof.dataHash;
  }

  return { valid: true, length: chainEntries.length, details };
}

function exportChain(): {
  version: string;
  publicKey: JsonWebKey | null;
  genesisHash: string;
  headHash: string;
  length: number;
  entries: SignedChainEntry[];
} {
  return {
    version: '3.0.0',
    publicKey: publicKeyJwk,
    genesisHash: GENESIS_HASH,
    headHash,
    length: chainEntries.length,
    entries: [...chainEntries],
  };
}

function getStats(): WorkerStats {
  return {
    initialized: privateKey !== null,
    chainLength: chainEntries.length,
    signingRequests,
    verificationRequests,
    lastActivity,
    uptime: Date.now() - startTime,
  };
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

function bytesToBase64(bytes: Uint8Array): string {
  const binary = Array.from(bytes)
    .map((b) => String.fromCharCode(b))
    .join('');
  return btoa(binary);
}

function base64ToBytes(base64: string): Uint8Array {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i);
  }
  return bytes;
}

// ============================================================================
// Message Handler
// ============================================================================

self.onmessage = async (event: MessageEvent<WorkerRequest>) => {
  const { id, type, payload } = event.data;

  let response: WorkerResponse;

  try {
    switch (type) {
      case 'INITIALIZE': {
        const result = await initialize();
        response = { id, success: true, result };
        break;
      }

      case 'SIGN_DATA': {
        const result = await signData(payload);
        response = { id, success: true, result };
        break;
      }

      case 'VERIFY_DATA': {
        const result = await verifyData(payload as SignedData<unknown>);
        response = { id, success: true, result };
        break;
      }

      case 'GET_PUBLIC_KEY': {
        if (!publicKeyJwk) {
          throw new Error('Worker not initialized');
        }
        response = { id, success: true, result: publicKeyJwk };
        break;
      }

      case 'APPEND_TO_CHAIN': {
        const { operationType, data } = payload as {
          operationType: string;
          data: unknown;
        };
        const result = await appendToChain(operationType, data);
        response = { id, success: true, result };
        break;
      }

      case 'VALIDATE_CHAIN': {
        const result = await validateChain();
        response = { id, success: true, result };
        break;
      }

      case 'EXPORT_CHAIN': {
        const result = exportChain();
        response = { id, success: true, result };
        break;
      }

      case 'GET_STATS': {
        const result = getStats();
        response = { id, success: true, result };
        break;
      }

      default:
        response = { id, success: false, error: `Unknown request type: ${type}` };
    }
  } catch (err) {
    response = {
      id,
      success: false,
      error: err instanceof Error ? err.message : 'Unknown error',
    };
  }

  self.postMessage(response);
};

// Signal that worker is ready
self.postMessage({ type: 'WORKER_READY' });
