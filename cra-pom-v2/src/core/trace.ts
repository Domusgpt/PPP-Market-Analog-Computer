// File: src/core/trace.ts
// TRACE - Hash-Chained Telemetry for Geometric Cognition
// Implements cryptographic audit logging using Web Crypto API

import type { GeometricState, ConvexityStatus } from './geometry';

/**
 * Event types for the audit log
 */
export type AuditEventType =
  | 'SYSTEM_INIT'
  | 'INFERENCE_STEP'
  | 'ENTROPY_INJECT'
  | 'CONVEXITY_WARNING'
  | 'CONVEXITY_VIOLATION'
  | 'POSITION_RESET'
  | 'CONSTRAINT_APPLIED'
  | 'ROTOR_UPDATE'
  | 'PROJECTION_UPDATE'
  | 'USER_ACTION';

/**
 * A single log entry in the hash-chain
 */
export interface LogEntry {
  /** Sequential entry index */
  index: number;

  /** ISO timestamp */
  timestamp: string;

  /** Event type classification */
  eventType: AuditEventType;

  /** The actual geometric state at this point */
  geometricState: GeometricState;

  /** Hash of the previous entry (creates the chain) */
  previousHash: string;

  /** SHA-256 hash of this entry's content */
  hash: string;

  /** Additional metadata */
  metadata?: Record<string, unknown>;
}

/**
 * Compact representation for display
 */
export interface LogEntryCompact {
  index: number;
  timestamp: string;
  eventType: AuditEventType;
  convexityStatus: ConvexityStatus;
  positionNorm: number;
  hash: string;
  previousHash: string;
}

/**
 * Chain validation result
 */
export interface ChainValidation {
  isValid: boolean;
  totalEntries: number;
  brokenAt?: number;
  expectedHash?: string;
  actualHash?: string;
}

/**
 * Genesis hash constant - the initial hash for the first entry
 */
const GENESIS_HASH = '0'.repeat(64);

/**
 * Convert ArrayBuffer to hex string
 */
function arrayBufferToHex(buffer: ArrayBuffer): string {
  const byteArray = new Uint8Array(buffer);
  return Array.from(byteArray)
    .map((byte) => byte.toString(16).padStart(2, '0'))
    .join('');
}

/**
 * Hash content using SHA-256
 */
async function sha256(content: string): Promise<string> {
  const encoder = new TextEncoder();
  const data = encoder.encode(content);
  const hashBuffer = await crypto.subtle.digest('SHA-256', data);
  return arrayBufferToHex(hashBuffer);
}

/**
 * Serialize geometric state to deterministic string
 */
function serializeState(state: GeometricState): string {
  // Use a deterministic serialization with fixed precision
  const pos = state.position;
  const conv = state.convexity;

  return JSON.stringify({
    p: [
      pos.w.toFixed(10),
      pos.x.toFixed(10),
      pos.y.toFixed(10),
      pos.z.toFixed(10),
    ],
    n: pos.norm.toFixed(10),
    s: state.stepCount,
    c: {
      status: conv.status,
      inside: conv.isInside,
      depth: conv.penetrationDepth.toFixed(10),
      dist: conv.distanceFromOrigin.toFixed(10),
    },
    lr: [
      state.leftRotor.w.toFixed(10),
      state.leftRotor.x.toFixed(10),
      state.leftRotor.y.toFixed(10),
      state.leftRotor.z.toFixed(10),
    ],
    rr: [
      state.rightRotor.w.toFixed(10),
      state.rightRotor.x.toFixed(10),
      state.rightRotor.y.toFixed(10),
      state.rightRotor.z.toFixed(10),
    ],
  });
}

/**
 * AuditChain - Immutable cryptographic ledger of cognitive trajectory
 *
 * Each entry contains:
 * 1. The complete geometric state
 * 2. A hash of the previous entry (creating an immutable chain)
 * 3. A SHA-256 hash of the current entry
 *
 * This creates a tamper-evident log where any modification to historical
 * entries will invalidate all subsequent hashes.
 */
export class AuditChain {
  private _entries: LogEntry[] = [];
  private _chainHead: string = GENESIS_HASH;
  private _isInitialized = false;

  /**
   * Get all entries
   */
  get entries(): readonly LogEntry[] {
    return this._entries;
  }

  /**
   * Get the current chain head (latest hash)
   */
  get chainHead(): string {
    return this._chainHead;
  }

  /**
   * Get the number of entries
   */
  get length(): number {
    return this._entries.length;
  }

  /**
   * Get initialization status
   */
  get isInitialized(): boolean {
    return this._isInitialized;
  }

  /**
   * Log an event with geometric state
   *
   * Formula: hash_i = SHA256(JSON(state) || previousHash)
   *
   * @param state - Current geometric state
   * @param eventType - Classification of the event
   * @param metadata - Optional additional data
   */
  async logEvent(
    state: GeometricState,
    eventType: AuditEventType,
    metadata?: Record<string, unknown>
  ): Promise<LogEntry> {
    const index = this._entries.length;
    const timestamp = new Date().toISOString();
    const previousHash = this._chainHead;

    // Create the content to be hashed
    const serializedState = serializeState(state);
    const contentToHash = `${serializedState}|${previousHash}|${index}|${timestamp}|${eventType}`;

    // Compute SHA-256 hash
    const hash = await sha256(contentToHash);

    // Create the entry
    const entry: LogEntry = {
      index,
      timestamp,
      eventType,
      geometricState: structuredClone(state),
      previousHash,
      hash,
      metadata,
    };

    // Update chain state
    this._entries.push(entry);
    this._chainHead = hash;

    if (!this._isInitialized && eventType === 'SYSTEM_INIT') {
      this._isInitialized = true;
    }

    return entry;
  }

  /**
   * Initialize the chain with a genesis entry
   */
  async initialize(initialState: GeometricState): Promise<LogEntry> {
    if (this._isInitialized) {
      throw new Error('Chain already initialized');
    }

    return this.logEvent(initialState, 'SYSTEM_INIT', {
      version: '2.0.0',
      algorithm: 'SHA-256',
      genesisHash: GENESIS_HASH,
    });
  }

  /**
   * Get a compact representation of recent entries
   */
  getRecentCompact(count = 10): LogEntryCompact[] {
    const start = Math.max(0, this._entries.length - count);
    return this._entries.slice(start).map((entry) => ({
      index: entry.index,
      timestamp: entry.timestamp,
      eventType: entry.eventType,
      convexityStatus: entry.geometricState.convexity.status,
      positionNorm: entry.geometricState.position.norm,
      hash: entry.hash,
      previousHash: entry.previousHash,
    }));
  }

  /**
   * Get entry by index
   */
  getEntry(index: number): LogEntry | undefined {
    return this._entries[index];
  }

  /**
   * Validate the entire chain integrity
   */
  async validateChain(): Promise<ChainValidation> {
    if (this._entries.length === 0) {
      return { isValid: true, totalEntries: 0 };
    }

    let expectedPreviousHash = GENESIS_HASH;

    for (let i = 0; i < this._entries.length; i++) {
      const entry = this._entries[i];

      // Check previous hash link
      if (entry.previousHash !== expectedPreviousHash) {
        return {
          isValid: false,
          totalEntries: this._entries.length,
          brokenAt: i,
          expectedHash: expectedPreviousHash,
          actualHash: entry.previousHash,
        };
      }

      // Recompute hash to verify
      const serializedState = serializeState(entry.geometricState);
      const contentToHash = `${serializedState}|${entry.previousHash}|${entry.index}|${entry.timestamp}|${entry.eventType}`;
      const recomputedHash = await sha256(contentToHash);

      if (entry.hash !== recomputedHash) {
        return {
          isValid: false,
          totalEntries: this._entries.length,
          brokenAt: i,
          expectedHash: recomputedHash,
          actualHash: entry.hash,
        };
      }

      expectedPreviousHash = entry.hash;
    }

    return { isValid: true, totalEntries: this._entries.length };
  }

  /**
   * Export the chain as JSON for external verification
   */
  exportChain(): string {
    return JSON.stringify(
      {
        version: '2.0.0',
        algorithm: 'SHA-256',
        genesisHash: GENESIS_HASH,
        chainHead: this._chainHead,
        entryCount: this._entries.length,
        entries: this._entries,
      },
      null,
      2
    );
  }

  /**
   * Get chain statistics
   */
  getStatistics(): ChainStatistics {
    const eventCounts: Record<AuditEventType, number> = {
      SYSTEM_INIT: 0,
      INFERENCE_STEP: 0,
      ENTROPY_INJECT: 0,
      CONVEXITY_WARNING: 0,
      CONVEXITY_VIOLATION: 0,
      POSITION_RESET: 0,
      CONSTRAINT_APPLIED: 0,
      ROTOR_UPDATE: 0,
      PROJECTION_UPDATE: 0,
      USER_ACTION: 0,
    };

    let safeCount = 0;
    let warningCount = 0;
    let violationCount = 0;

    for (const entry of this._entries) {
      eventCounts[entry.eventType]++;

      switch (entry.geometricState.convexity.status) {
        case 'SAFE':
          safeCount++;
          break;
        case 'WARNING':
          warningCount++;
          break;
        case 'VIOLATION':
          violationCount++;
          break;
      }
    }

    return {
      totalEntries: this._entries.length,
      chainHead: this._chainHead,
      eventCounts,
      convexityStats: {
        safe: safeCount,
        warning: warningCount,
        violation: violationCount,
      },
      firstEntry: this._entries[0]?.timestamp,
      lastEntry: this._entries[this._entries.length - 1]?.timestamp,
    };
  }

  /**
   * Clear the chain (for testing/reset purposes)
   */
  clear(): void {
    this._entries = [];
    this._chainHead = GENESIS_HASH;
    this._isInitialized = false;
  }
}

/**
 * Chain statistics
 */
export interface ChainStatistics {
  totalEntries: number;
  chainHead: string;
  eventCounts: Record<AuditEventType, number>;
  convexityStats: {
    safe: number;
    warning: number;
    violation: number;
  };
  firstEntry?: string;
  lastEntry?: string;
}

/**
 * Create a singleton audit chain instance
 */
let globalAuditChain: AuditChain | null = null;

export function getGlobalAuditChain(): AuditChain {
  if (!globalAuditChain) {
    globalAuditChain = new AuditChain();
  }
  return globalAuditChain;
}

export function resetGlobalAuditChain(): void {
  globalAuditChain = new AuditChain();
}
