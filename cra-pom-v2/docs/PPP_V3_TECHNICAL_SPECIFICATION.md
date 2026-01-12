# PPP v3 Technical Specification

**Document Version:** 1.0.0
**Last Updated:** 2026-01-09
**Status:** Complete Implementation
**Authors:** Claude (Anthropic)

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [Cryptographic Foundation](#3-cryptographic-foundation)
4. [Process Isolation](#4-process-isolation)
5. [Semantic Embeddings](#5-semantic-embeddings)
6. [Verified Reasoning Engine](#6-verified-reasoning-engine)
7. [Visualization Integration](#7-visualization-integration)
8. [Security Analysis](#8-security-analysis)
9. [API Reference](#9-api-reference)
10. [Limitations](#10-limitations)

---

## 1. Overview

### 1.1 Purpose

PPP v3 (Polytopal Projection Processing, version 3) is a framework for creating **auditable reasoning traces** with cryptographic guarantees. It is designed for use with Large Language Models (LLMs) where transparency and verifiability of the reasoning process is desired.

### 1.2 Version History

| Version | Date | Codename | Key Changes |
|---------|------|----------|-------------|
| 1.0 | 2025-Q3 | Original | Initial random vector implementation |
| 2.0 | 2025-Q4 | HDC/VSA | Added hyperdimensional computing operations |
| **3.0** | **2026-01-09** | **Honest Geometric Cognition** | Real cryptography, process isolation, honest documentation |

### 1.3 Design Principles

1. **Honesty First** - Documentation accurately reflects capabilities and limitations
2. **Real Cryptography** - No toy implementations; uses Web Crypto API standards
3. **Process Isolation** - Trust boundaries enforced via Web Workers
4. **Semantic Grounding** - Real embeddings when available, honest fallback when not
5. **External Verifiability** - Audit chains can be verified by any party with public key

### 1.4 What This System Does

- Creates signed, tamper-evident audit trails of reasoning operations
- Provides semantic concept storage and retrieval (when properly configured)
- Enables external verification of reasoning traces
- Isolates signing keys from untrusted code (LLM)

### 1.5 What This System Does NOT Do

- **Does NOT** prove an LLM actually used these tools for reasoning
- **Does NOT** validate semantic truth or correctness of claims
- **Does NOT** prevent hallucination
- **Does NOT** provide AGI or "true understanding"

---

## 2. Architecture

### 2.1 System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              APPLICATION                                     │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         MAIN THREAD                                  │   │
│  │                        (Untrusted Zone)                             │   │
│  │                                                                      │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │   │
│  │  │     LLM      │  │   Concept    │  │    Verified Reasoner     │  │   │
│  │  │   (Tools)    │──│    Store     │──│    (Orchestration)       │  │   │
│  │  └──────────────┘  └──────────────┘  └────────────┬─────────────┘  │   │
│  │                                                    │                │   │
│  │  ┌──────────────┐  ┌──────────────┐              │                │   │
│  │  │  Embedding   │  │ Verification │◄─────────────┘                │   │
│  │  │   Service    │  │    Client    │                               │   │
│  │  └──────────────┘  └──────┬───────┘                               │   │
│  │                           │                                        │   │
│  └───────────────────────────┼────────────────────────────────────────┘   │
│                              │ Message Passing                             │
│  ┌───────────────────────────┼────────────────────────────────────────┐   │
│  │                    WEB WORKER                                       │   │
│  │                   (Trusted Zone)                                    │   │
│  │                           ▼                                         │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │   │
│  │  │  Private Key │  │   Crypto     │  │      Hash Chain          │  │   │
│  │  │  (Trapped)   │  │   Service    │  │      (Signed)            │  │   │
│  │  └──────────────┘  └──────────────┘  └──────────────────────────┘  │   │
│  │                                                                      │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Module Structure

```
src/core/v3/
├── index.ts                    # Main exports
├── example.ts                  # Working example
│
├── crypto/                     # Cryptographic Foundation
│   ├── index.ts
│   ├── crypto-service.ts       # SHA-256, ECDSA P-256
│   └── hash-chain.ts           # Signed append-only log
│
├── isolation/                  # Process Isolation
│   ├── index.ts
│   ├── verification-worker.ts  # Web Worker (trusted zone)
│   ├── verification-client.ts  # Main thread client
│   └── trust-boundary.ts       # Trust model types
│
├── embeddings/                 # Semantic Grounding
│   ├── index.ts
│   ├── embedding-service.ts    # Embedding providers
│   └── concept-store.ts        # Concept storage/retrieval
│
├── reasoning/                  # Verified Reasoning
│   ├── index.ts
│   └── verified-reasoner.ts    # Reasoning engine
│
└── integration/                # External Integration
    ├── index.ts
    └── visualization-bridge.ts # Connect to visualization
```

### 2.3 Data Flow

1. **LLM Request** → Tool interface receives reasoning request
2. **Concept Lookup** → EmbeddingService retrieves/computes embeddings
3. **Reasoning Step** → VerifiedReasoner creates step record
4. **Signing Request** → VerificationClient sends to Worker
5. **Signature Created** → Worker signs with trapped private key
6. **Chain Appended** → Entry added to hash chain
7. **Response Returned** → Signed step returned to LLM

---

## 3. Cryptographic Foundation

### 3.1 Algorithms Used

| Purpose | Algorithm | Standard | Security Level |
|---------|-----------|----------|----------------|
| Hashing | SHA-256 | FIPS 180-4 | 128-bit collision resistance |
| Signatures | ECDSA P-256 | FIPS 186-4 | 128-bit security |
| Key Generation | Web Crypto API | W3C | Platform-native |
| Random Numbers | `crypto.getRandomValues()` | W3C | CSPRNG |

### 3.2 CryptoService API

```typescript
class CryptoService {
  // Initialize with new key pair
  async initialize(): Promise<void>;

  // Check initialization status
  isInitialized(): boolean;

  // Get public key (safe to share)
  getPublicKeyJwk(): JsonWebKey;

  // Hash operations
  async hash(data: Uint8Array): Promise<HashResult>;
  async hashString(str: string): Promise<HashResult>;
  async hashObject(obj: unknown): Promise<HashResult>;

  // Sign operations
  async sign(data: Uint8Array): Promise<Signature>;
  async signString(str: string): Promise<Signature>;
  async signData<T>(payload: T): Promise<SignedData<T>>;

  // Verify operations
  async verify(data: Uint8Array, signature: Uint8Array, publicKey: CryptoKey): Promise<boolean>;
  async verifyWithJwk(data: Uint8Array, signatureBase64: string, publicKeyJwk: JsonWebKey): Promise<boolean>;
  async verifySignedData<T>(signedData: SignedData<T>): Promise<VerificationResult>;
}
```

### 3.3 SignedData Structure

```typescript
interface SignedData<T> {
  payload: T;                    // Original data
  proof: {
    dataHash: string;           // SHA-256 of payload (hex)
    signature: string;          // ECDSA signature (base64)
    publicKeyJwk: JsonWebKey;   // Public key for verification
    timestamp: string;          // ISO 8601 timestamp
    algorithm: 'ECDSA-P256-SHA256';
  };
}
```

### 3.4 Hash Chain Structure

```typescript
interface ChainEntry<T> {
  index: number;                // Position in chain (0-indexed)
  timestamp: string;            // ISO 8601
  operationType: string;        // Type of operation
  data: T;                      // Entry payload
  previousHash: string;         // Hash of previous entry
  contentHash: string;          // Hash of this entry's content
}

interface SignedChainEntry<T> extends SignedData<ChainEntry<T>> {}
```

### 3.5 Deterministic Serialization

All objects are serialized with sorted keys for consistent hashing:

```typescript
function deterministicStringify(obj: unknown): string {
  return JSON.stringify(obj, (_, value) => {
    if (value && typeof value === 'object' && !Array.isArray(value)) {
      return Object.keys(value)
        .sort()
        .reduce((sorted, key) => {
          sorted[key] = value[key];
          return sorted;
        }, {});
    }
    return value;
  });
}
```

---

## 4. Process Isolation

### 4.1 Trust Model

| Zone | Location | Capabilities | Trust Level |
|------|----------|--------------|-------------|
| Untrusted | Main Thread | Request signatures, verify, read public key | None |
| Trusted | Web Worker | Sign, access private key, maintain chain | Full |

### 4.2 Why Web Workers?

1. **Separate Memory Space** - Worker memory is not accessible from main thread
2. **No Shared State** - Communication only via message passing
3. **Non-Extractable Keys** - Web Crypto can generate keys that cannot be exported

### 4.3 Worker Communication Protocol

**Request Format:**
```typescript
interface WorkerRequest {
  id: string;                   // UUID for correlation
  type: RequestType;            // Operation type
  payload?: unknown;            // Operation data
}

type RequestType =
  | 'INITIALIZE'
  | 'SIGN_DATA'
  | 'VERIFY_DATA'
  | 'GET_PUBLIC_KEY'
  | 'APPEND_TO_CHAIN'
  | 'VALIDATE_CHAIN'
  | 'EXPORT_CHAIN'
  | 'GET_STATS';
```

**Response Format:**
```typescript
interface WorkerResponse {
  id: string;                   // Matches request ID
  success: boolean;
  result?: unknown;
  error?: string;
}
```

### 4.4 Key Protection

```typescript
// In Worker: Key is generated as NON-EXTRACTABLE
const keyPair = await crypto.subtle.generateKey(
  { name: 'ECDSA', namedCurve: 'P-256' },
  false,  // NOT extractable - key is trapped
  ['sign', 'verify']
);
```

---

## 5. Semantic Embeddings

### 5.1 Embedding Sources

| Source | Semantic? | Description |
|--------|-----------|-------------|
| `external_api` | ✅ Yes | Real transformer embeddings (OpenAI, etc.) |
| `local_model` | ✅ Yes | Browser-based transformer (future) |
| `cache` | ✅ Yes | Pre-computed from real model |
| `deterministic_fallback` | ❌ **NO** | Hash-based, NOT semantic |

### 5.2 EmbeddingService API

```typescript
class EmbeddingService {
  // Get current source
  getSource(): EmbeddingSource;

  // Is current source semantically meaningful?
  isSemantic(): boolean;

  // Configure external API
  configureApi(url: string, apiKey: string, model: string): void;

  // Get embeddings
  async embed(text: string): Promise<EmbeddingResult>;
  async embedBatch(texts: string[]): Promise<EmbeddingResult[]>;

  // Similarity operations
  async similarity(textA: string, textB: string): Promise<SimilarityResult>;
  async findMostSimilar(query: string, candidates: string[], topK?: number): Promise<...>;
}
```

### 5.3 ConceptStore API

```typescript
class ConceptStore {
  // Add concepts
  async addConcept(name: string, description: string, metadata?: Record<string, unknown>): Promise<StoredConcept>;

  // Retrieve concepts
  getConcept(id: string): StoredConcept | undefined;
  getConceptByName(name: string): StoredConcept | undefined;

  // Semantic search
  async retrieve(query: string, topK?: number): Promise<ConceptRetrievalResult>;

  // Vector composition (e.g., king - man + woman ≈ queen)
  async compose(operations: Array<{concept: string; operation: 'add'|'subtract'; weight?: number}>): Promise<ConceptCompositionResult>;

  // Grounding status
  getGroundingStatus(): {...};
}
```

### 5.4 Fallback Behavior

When real embeddings are unavailable, the system:
1. Uses deterministic hash-based vectors
2. **Clearly warns** that results are not semantically meaningful
3. Reports `semanticallyMeaningful: false` in all results
4. Still functions for testing/development

---

## 6. Verified Reasoning Engine

### 6.1 VerifiedReasoner API

```typescript
class VerifiedReasoner {
  // Initialize (connects to verification worker)
  async initialize(): Promise<{ publicKey: JsonWebKey }>;

  // Session management
  async startSession(query: string): Promise<ReasoningSession>;
  async endSession(): Promise<ReasoningResult>;
  getCurrentSession(): ReasoningSession | null;

  // Reasoning operations (all signed)
  async lookupConcept(conceptName: string): Promise<SignedReasoningStep>;
  async querySimilar(query: string, topK?: number): Promise<SignedReasoningStep>;
  async composeConcepts(operations: Array<...>): Promise<SignedReasoningStep>;
  async makeInference(premises: string[], inference: string, confidence: number): Promise<SignedReasoningStep>;
  async generateHypothesis(observations: string[], hypothesis: string, confidence: number): Promise<SignedReasoningStep>;
  async conclude(statement: string, confidence: number, supportingSteps: number[], caveats?: string[]): Promise<SignedConclusion>;

  // Export for external verification
  async exportAuditChain(): Promise<ExportedChain>;
}
```

### 6.2 ReasoningStep Structure

```typescript
interface ReasoningStep {
  stepNumber: number;           // Sequential step number
  operation: ReasoningOperation; // Type of operation
  description: string;          // Human-readable description
  inputs: string[];             // Input concepts/values
  outputs: string[];            // Output concepts/values
  confidence: number;           // 0-1 confidence score
  citations: string[];          // Concept IDs referenced
  timestamp: string;            // ISO 8601
}

type ReasoningOperation =
  | 'CONCEPT_LOOKUP'
  | 'CONCEPT_BIND'
  | 'CONCEPT_BUNDLE'
  | 'SIMILARITY_QUERY'
  | 'INFERENCE'
  | 'COMPOSITION'
  | 'HYPOTHESIS'
  | 'VERIFICATION'
  | 'CONCLUSION';
```

### 6.3 Session Lifecycle

```
┌─────────────┐
│ startSession│ → Creates session, logs to chain
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Steps     │ → Each operation signed and logged
│ (repeated)  │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  conclude   │ → Final conclusion signed
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ endSession  │ → Returns complete result with verification
└─────────────┘
```

---

## 7. Visualization Integration

### 7.1 VisualizationBridge API

```typescript
class VisualizationBridge {
  // Connect to services
  connect(conceptStore: ConceptStore, verificationClient: VerificationClient): void;

  // Get current state
  getState(): VisualizationState;

  // Subscribe to events
  subscribe(listener: EventListener): () => void;

  // Event handlers (called by VerifiedReasoner)
  onSessionStart(session: ReasoningSession): void;
  onReasoningStep(step: SignedReasoningStep): void;
  onConclusion(conclusion: SignedConclusion): void;
  onSessionEnd(summary: string): void;
}
```

### 7.2 Visualization State

```typescript
interface VisualizationState {
  session: {
    id: string;
    query: string;
    stepCount: number;
    isActive: boolean;
  } | null;

  currentStep: {
    number: number;
    operation: ReasoningOperation;
    description: string;
    confidence: number;
    signed: boolean;
  } | null;

  conceptNodes: ConceptNode[];
  conceptEdges: ConceptEdge[];

  auditStatus: {
    valid: boolean;
    chainLength: number;
    lastHash: string;
  };

  grounding: {
    semantic: boolean;
    source: string;
    warning?: string;
  };
}
```

---

## 8. Security Analysis

### 8.1 Threat Model

| Threat | Mitigated? | Mechanism |
|--------|------------|-----------|
| LLM forges signature | ✅ Yes | Private key in separate process |
| LLM accesses private key | ✅ Yes | Non-extractable + Worker isolation |
| LLM modifies past entries | ✅ Yes | Hash chain + signatures |
| Tampering detected post-hoc | ✅ Yes | External verification |
| LLM ignores results | ❌ No | Cannot be prevented |
| LLM lies about reasoning | ❌ No | Only logs what's requested |
| Build-time compromise | ❌ No | Requires separate toolchain security |
| Side-channel attacks | ⚠️ Partial | Web Crypto uses constant-time ops |

### 8.2 Attack Surface

1. **Worker Code** - If malicious code is injected into the worker at build time, security is compromised
2. **Message Passing** - Messages could theoretically be intercepted, but signatures would fail verification
3. **Timing Attacks** - Mitigated by Web Crypto's constant-time implementations
4. **Browser Vulnerabilities** - Depends on browser security model

### 8.3 Verification Process

External verifiers can validate an exported chain with ONLY the public key:

```typescript
// Anyone can verify
const verification = await SignedHashChain.verifyExported(exportedChain);

// Checks performed:
// 1. Hash chain integrity (each entry links to previous)
// 2. Signature validity (ECDSA verification)
// 3. Sequence correctness (indices are sequential)
// 4. Content integrity (payload matches hash)
```

---

## 9. API Reference

### 9.1 Quick Reference

| Module | Key Exports |
|--------|-------------|
| `crypto` | `CryptoService`, `SignedHashChain`, `getCryptoService()` |
| `isolation` | `VerificationClient`, `getVerificationClient()`, `signWithIsolation()` |
| `embeddings` | `EmbeddingService`, `ConceptStore`, `getEmbeddingService()`, `getConceptStore()` |
| `reasoning` | `VerifiedReasoner`, `getVerifiedReasoner()` |
| `integration` | `VisualizationBridge`, `getVisualizationBridge()` |

### 9.2 Singleton Pattern

All major services use a singleton pattern for convenience:

```typescript
// Get global instance (creates if needed)
const reasoner = await getVerifiedReasoner();

// Reset for testing
resetVerifiedReasoner();
```

### 9.3 Type Exports

All types are exported for TypeScript usage:

```typescript
import type {
  SignedData,
  ChainEntry,
  ReasoningStep,
  StoredConcept,
  VisualizationState,
  // ... etc
} from './core/v3';
```

---

## 10. Limitations

### 10.1 Fundamental Limitations

| Limitation | Reason | Mitigation |
|------------|--------|------------|
| Cannot prove LLM used results | LLM is a black box | Audit trail shows what was available |
| Cannot verify semantic truth | Math ≠ truth | Document confidence levels |
| Cannot prevent hallucination | Fundamental LLM limitation | Signing doesn't create truth |
| Performance overhead | Crypto operations take time | ~1-2ms per signature |

### 10.2 Technical Limitations

| Limitation | Workaround |
|------------|------------|
| Requires Web Workers | Use Node.js worker polyfill |
| Requires Web Crypto | Use Node.js crypto module |
| No real-time collaboration | Would need distributed consensus |
| Single key per session | Could extend to multi-party signing |

### 10.3 When NOT to Use This System

- ❌ When you need **guaranteed** correctness (use formal verification)
- ❌ When you need to **prove** no hallucination (not currently possible)
- ❌ When you need sub-millisecond latency (crypto adds overhead)
- ❌ When trust isn't a concern (simpler solutions exist)

### 10.4 When This System IS Useful

- ✅ Creating auditable reasoning trails
- ✅ Non-repudiation of tool calls
- ✅ Tamper detection in logs
- ✅ Transparency for stakeholders
- ✅ Research into verified AI systems

---

## Document History

| Date | Version | Changes |
|------|---------|---------|
| 2026-01-09 | 1.0.0 | Initial specification for PPP v3 |

---

*This document is part of the PPP v3 distribution and should be kept in sync with the implementation.*
