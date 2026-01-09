# PPP v3 API Reference

**Document Version:** 1.0.0
**Last Updated:** 2026-01-09
**PPP Version:** 3.0.0

---

## Table of Contents

1. [Module: crypto](#1-module-crypto)
2. [Module: isolation](#2-module-isolation)
3. [Module: embeddings](#3-module-embeddings)
4. [Module: reasoning](#4-module-reasoning)
5. [Module: integration](#5-module-integration)
6. [Types Reference](#6-types-reference)
7. [Constants](#7-constants)

---

## 1. Module: crypto

**Import:** `import { ... } from './core/v3/crypto'`

### Classes

#### CryptoService

Provides real cryptographic operations using Web Crypto API.

```typescript
class CryptoService {
  constructor();
  async initialize(): Promise<void>;
  isInitialized(): boolean;
  getPublicKeyJwk(): JsonWebKey;
  async generateKeyPair(): Promise<KeyPair>;
  async hash(data: Uint8Array): Promise<HashResult>;
  async hashString(str: string): Promise<HashResult>;
  async hashObject(obj: unknown): Promise<HashResult>;
  async sign(data: Uint8Array): Promise<Signature>;
  async signString(str: string): Promise<Signature>;
  async signData<T>(payload: T): Promise<SignedData<T>>;
  async verify(data: Uint8Array, signature: Uint8Array, publicKey: CryptoKey): Promise<boolean>;
  async verifyWithJwk(data: Uint8Array, signatureBase64: string, publicKeyJwk: JsonWebKey): Promise<boolean>;
  async verifySignedData<T>(signedData: SignedData<T>): Promise<VerificationResult>;
}
```

**Methods:**

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `initialize()` | none | `Promise<void>` | Initialize with new key pair |
| `isInitialized()` | none | `boolean` | Check if initialized |
| `getPublicKeyJwk()` | none | `JsonWebKey` | Get public key (safe to share) |
| `generateKeyPair()` | none | `Promise<KeyPair>` | Generate new ECDSA P-256 key pair |
| `hash(data)` | `Uint8Array` | `Promise<HashResult>` | SHA-256 hash of bytes |
| `hashString(str)` | `string` | `Promise<HashResult>` | SHA-256 hash of UTF-8 string |
| `hashObject(obj)` | `unknown` | `Promise<HashResult>` | SHA-256 hash of deterministic JSON |
| `sign(data)` | `Uint8Array` | `Promise<Signature>` | ECDSA signature of bytes |
| `signString(str)` | `string` | `Promise<Signature>` | ECDSA signature of string |
| `signData<T>(payload)` | `T` | `Promise<SignedData<T>>` | Sign and wrap any data |
| `verify(data, sig, key)` | `Uint8Array, Uint8Array, CryptoKey` | `Promise<boolean>` | Verify signature |
| `verifyWithJwk(...)` | `Uint8Array, string, JsonWebKey` | `Promise<boolean>` | Verify with JWK |
| `verifySignedData<T>(...)` | `SignedData<T>` | `Promise<VerificationResult>` | Full verification |

---

#### SignedHashChain

Append-only cryptographically-signed log.

```typescript
class SignedHashChain {
  constructor(cryptoService: CryptoService);
  async initialize(): Promise<SignedChainEntry>;
  async append<T>(operationType: string, data: T): Promise<SignedChainEntry<T>>;
  async validate(): Promise<ChainValidationResult>;
  getHeadHash(): string;
  getLength(): number;
  export(): ExportedChain;
  static async verifyExported(exported: ExportedChain): Promise<ChainValidationResult>;
}
```

**Methods:**

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `initialize()` | none | `Promise<SignedChainEntry>` | Create genesis entry |
| `append<T>(type, data)` | `string, T` | `Promise<SignedChainEntry<T>>` | Add entry to chain |
| `validate()` | none | `Promise<ChainValidationResult>` | Validate entire chain |
| `getHeadHash()` | none | `string` | Current head hash |
| `getLength()` | none | `number` | Number of entries |
| `export()` | none | `ExportedChain` | Export for external use |
| `verifyExported(...)` | `ExportedChain` | `Promise<ChainValidationResult>` | *Static* External verification |

---

### Functions

#### getCryptoService()

```typescript
async function getCryptoService(): Promise<CryptoService>
```

Returns the global CryptoService instance, creating and initializing if needed.

#### resetCryptoService()

```typescript
function resetCryptoService(): void
```

Reset the global instance (for testing).

#### createSignedHashChain()

```typescript
async function createSignedHashChain(): Promise<SignedHashChain>
```

Create a new SignedHashChain with the global CryptoService.

---

## 2. Module: isolation

**Import:** `import { ... } from './core/v3/isolation'`

### Classes

#### VerificationClient

Main thread interface to the verification worker.

```typescript
class VerificationClient {
  constructor(workerUrl?: string);
  async initialize(): Promise<{ publicKey: JsonWebKey }>;
  isInitialized(): boolean;
  async getPublicKey(): Promise<JsonWebKey>;
  async signData<T>(payload: T): Promise<SignedData<T>>;
  async verifyData<T>(signedData: SignedData<T>): Promise<VerificationResult>;
  async appendToChain<T>(operationType: string, data: T): Promise<SignedChainEntry<T>>;
  async validateChain(): Promise<ChainValidationResult>;
  async exportChain(): Promise<ExportedChain>;
  async getStats(): Promise<WorkerStats>;
  terminate(): void;
}
```

**Methods:**

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `initialize()` | none | `Promise<{publicKey}>` | Start worker and init crypto |
| `isInitialized()` | none | `boolean` | Check if ready |
| `getPublicKey()` | none | `Promise<JsonWebKey>` | Get public key |
| `signData<T>(payload)` | `T` | `Promise<SignedData<T>>` | Sign via worker |
| `verifyData<T>(...)` | `SignedData<T>` | `Promise<VerificationResult>` | Verify signature |
| `appendToChain<T>(...)` | `string, T` | `Promise<SignedChainEntry<T>>` | Add to audit chain |
| `validateChain()` | none | `Promise<ChainValidationResult>` | Validate chain |
| `exportChain()` | none | `Promise<ExportedChain>` | Export chain |
| `getStats()` | none | `Promise<WorkerStats>` | Get worker stats |
| `terminate()` | none | `void` | Stop worker |

---

### Functions

#### getVerificationClient()

```typescript
async function getVerificationClient(): Promise<VerificationClient>
```

Get global VerificationClient, creating if needed.

#### resetVerificationClient()

```typescript
function resetVerificationClient(): void
```

Reset and terminate the global client.

#### signWithIsolation()

```typescript
async function signWithIsolation<T>(payload: T): Promise<SignedData<T>>
```

Convenience function to sign data using global client.

#### verifyWithIsolation()

```typescript
async function verifyWithIsolation<T>(signedData: SignedData<T>): Promise<VerificationResult>
```

Convenience function to verify data using global client.

#### appendToAuditChain()

```typescript
async function appendToAuditChain<T>(operationType: string, data: T): Promise<SignedChainEntry<T>>
```

Convenience function to append to chain using global client.

#### verifyAttestation()

```typescript
async function verifyAttestation<T>(
  attestation: Attestation<T>,
  expectedPublicKey?: JsonWebKey
): Promise<VerificationOutcome>
```

Verify an attestation (signed claim) externally.

---

## 3. Module: embeddings

**Import:** `import { ... } from './core/v3/embeddings'`

### Classes

#### EmbeddingService

Provides semantic embeddings from various sources.

```typescript
class EmbeddingService {
  constructor(config?: Partial<EmbeddingConfig>);
  getSource(): EmbeddingSource;
  isSemantic(): boolean;
  configureApi(url: string, apiKey: string, model: string): void;
  async embed(text: string): Promise<EmbeddingResult>;
  async embedBatch(texts: string[]): Promise<EmbeddingResult[]>;
  async similarity(textA: string, textB: string): Promise<SimilarityResult>;
  async findMostSimilar(query: string, candidates: string[], topK?: number): Promise<SimilarityMatch[]>;
  getDimension(): number;
  getStats(): EmbeddingStats;
}
```

**Methods:**

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `getSource()` | none | `EmbeddingSource` | Current embedding source |
| `isSemantic()` | none | `boolean` | Is source semantically meaningful? |
| `configureApi(...)` | `string, string, string` | `void` | Configure external API |
| `embed(text)` | `string` | `Promise<EmbeddingResult>` | Get embedding for text |
| `embedBatch(texts)` | `string[]` | `Promise<EmbeddingResult[]>` | Batch embeddings |
| `similarity(a, b)` | `string, string` | `Promise<SimilarityResult>` | Compute similarity |
| `findMostSimilar(...)` | `string, string[], number?` | `Promise<SimilarityMatch[]>` | Find top-K similar |
| `getDimension()` | none | `number` | Embedding dimension |
| `getStats()` | none | `EmbeddingStats` | Service statistics |

---

#### ConceptStore

Stores and retrieves concepts with embeddings.

```typescript
class ConceptStore {
  constructor(embeddingService?: EmbeddingService);
  async addConcept(name: string, description: string, metadata?: Record<string, unknown>): Promise<StoredConcept>;
  getConcept(id: string): StoredConcept | undefined;
  getConceptByName(name: string): StoredConcept | undefined;
  hasConcept(nameOrId: string): boolean;
  async retrieve(query: string, topK?: number): Promise<ConceptRetrievalResult>;
  async compose(operations: CompositionOperation[]): Promise<ConceptCompositionResult>;
  async findNearestToVector(vector: Float32Array, topK?: number): Promise<NearestMatch[]>;
  getAllConcepts(): StoredConcept[];
  get size(): number;
  getGroundingStatus(): GroundingStatus;
  export(): ExportedConceptStore;
  import(data: ExportedConceptStore): void;
  clear(): void;
}
```

**Methods:**

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `addConcept(...)` | `string, string, Record?` | `Promise<StoredConcept>` | Add a concept |
| `getConcept(id)` | `string` | `StoredConcept \| undefined` | Get by ID |
| `getConceptByName(name)` | `string` | `StoredConcept \| undefined` | Get by name |
| `hasConcept(nameOrId)` | `string` | `boolean` | Check existence |
| `retrieve(query, topK?)` | `string, number?` | `Promise<ConceptRetrievalResult>` | Semantic search |
| `compose(operations)` | `CompositionOperation[]` | `Promise<ConceptCompositionResult>` | Vector arithmetic |
| `findNearestToVector(...)` | `Float32Array, number?` | `Promise<NearestMatch[]>` | Find nearest |
| `getAllConcepts()` | none | `StoredConcept[]` | Get all concepts |
| `size` | getter | `number` | Number of concepts |
| `getGroundingStatus()` | none | `GroundingStatus` | Grounding info |
| `export()` | none | `ExportedConceptStore` | Export data |
| `import(data)` | `ExportedConceptStore` | `void` | Import data |
| `clear()` | none | `void` | Clear all concepts |

---

### Functions

#### getEmbeddingService()

```typescript
function getEmbeddingService(): EmbeddingService
```

Get global EmbeddingService instance.

#### resetEmbeddingService()

```typescript
function resetEmbeddingService(): void
```

Reset global instance.

#### configureEmbeddings()

```typescript
function configureEmbeddings(url: string, apiKey: string, model: string): void
```

Configure global service with external API.

#### getConceptStore()

```typescript
function getConceptStore(): ConceptStore
```

Get global ConceptStore instance.

#### resetConceptStore()

```typescript
function resetConceptStore(): void
```

Reset global instance.

#### initializeWithBasicConcepts()

```typescript
async function initializeWithBasicConcepts(store: ConceptStore): Promise<void>
```

Load LOGIC_CONCEPTS and REASONING_CONCEPTS into store.

---

### Constants

#### LOGIC_CONCEPTS

```typescript
const LOGIC_CONCEPTS: Array<{name: string; description: string}>
```

Basic logical concepts: true, false, and, or, not, implies, equivalent, contradiction, tautology.

#### REASONING_CONCEPTS

```typescript
const REASONING_CONCEPTS: Array<{name: string; description: string}>
```

Reasoning concepts: premise, conclusion, inference, deduction, induction, abduction, hypothesis, evidence, certainty, uncertainty.

---

## 4. Module: reasoning

**Import:** `import { ... } from './core/v3/reasoning'`

### Classes

#### VerifiedReasoner

Reasoning engine with cryptographic verification.

```typescript
class VerifiedReasoner {
  constructor(conceptStore?: ConceptStore);
  async initialize(): Promise<{ publicKey: JsonWebKey }>;
  isInitialized(): boolean;
  async startSession(query: string): Promise<ReasoningSession>;
  async lookupConcept(conceptName: string): Promise<SignedReasoningStep>;
  async querySimilar(query: string, topK?: number): Promise<SignedReasoningStep>;
  async composeConcepts(operations: CompositionOperation[]): Promise<SignedReasoningStep>;
  async makeInference(premises: string[], inference: string, confidence: number): Promise<SignedReasoningStep>;
  async generateHypothesis(observations: string[], hypothesis: string, confidence: number): Promise<SignedReasoningStep>;
  async conclude(statement: string, confidence: number, supportingSteps: number[], caveats?: string[]): Promise<SignedConclusion>;
  async endSession(): Promise<ReasoningResult>;
  getCurrentSession(): ReasoningSession | null;
  async exportAuditChain(): Promise<ExportedChain>;
}
```

**Methods:**

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `initialize()` | none | `Promise<{publicKey}>` | Initialize reasoner |
| `isInitialized()` | none | `boolean` | Check if ready |
| `startSession(query)` | `string` | `Promise<ReasoningSession>` | Start new session |
| `lookupConcept(name)` | `string` | `Promise<SignedReasoningStep>` | Look up concept |
| `querySimilar(query, k?)` | `string, number?` | `Promise<SignedReasoningStep>` | Semantic search |
| `composeConcepts(ops)` | `CompositionOperation[]` | `Promise<SignedReasoningStep>` | Vector arithmetic |
| `makeInference(...)` | `string[], string, number` | `Promise<SignedReasoningStep>` | Record inference |
| `generateHypothesis(...)` | `string[], string, number` | `Promise<SignedReasoningStep>` | Record hypothesis |
| `conclude(...)` | `string, number, number[], string[]?` | `Promise<SignedConclusion>` | Record conclusion |
| `endSession()` | none | `Promise<ReasoningResult>` | End and verify session |
| `getCurrentSession()` | none | `ReasoningSession \| null` | Get active session |
| `exportAuditChain()` | none | `Promise<ExportedChain>` | Export chain |

---

### Functions

#### getVerifiedReasoner()

```typescript
async function getVerifiedReasoner(): Promise<VerifiedReasoner>
```

Get global VerifiedReasoner, creating and initializing if needed.

#### resetVerifiedReasoner()

```typescript
function resetVerifiedReasoner(): void
```

Reset global instance.

---

## 5. Module: integration

**Import:** `import { ... } from './core/v3/integration'`

### Classes

#### VisualizationBridge

Connects reasoning to visualization.

```typescript
class VisualizationBridge {
  constructor();
  connect(conceptStore: ConceptStore, verificationClient: VerificationClient): void;
  getState(): VisualizationState;
  subscribe(listener: EventListener): () => void;
  onSessionStart(session: ReasoningSession): void;
  onReasoningStep(step: SignedReasoningStep): void;
  onConclusion(conclusion: SignedConclusion): void;
  onSessionEnd(summary: string): void;
  async updateAuditStatus(): Promise<void>;
  addConceptNode(concept: StoredConcept, role: ConceptNodeRole): void;
  addConceptEdge(fromId: string, toId: string, type: EdgeType, strength: number): void;
  fadeInactiveConcepts(decayRate?: number): void;
  reset(): void;
}
```

**Methods:**

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `connect(...)` | `ConceptStore, VerificationClient` | `void` | Connect to services |
| `getState()` | none | `VisualizationState` | Get current state |
| `subscribe(listener)` | `EventListener` | `() => void` | Subscribe to events |
| `onSessionStart(session)` | `ReasoningSession` | `void` | Handle session start |
| `onReasoningStep(step)` | `SignedReasoningStep` | `void` | Handle step |
| `onConclusion(conclusion)` | `SignedConclusion` | `void` | Handle conclusion |
| `onSessionEnd(summary)` | `string` | `void` | Handle session end |
| `updateAuditStatus()` | none | `Promise<void>` | Update audit status |
| `addConceptNode(...)` | `StoredConcept, ConceptNodeRole` | `void` | Add visualization node |
| `addConceptEdge(...)` | `string, string, EdgeType, number` | `void` | Add edge |
| `fadeInactiveConcepts(rate?)` | `number?` | `void` | Fade old nodes |
| `reset()` | none | `void` | Reset state |

---

### Functions

#### getVisualizationBridge()

```typescript
function getVisualizationBridge(): VisualizationBridge
```

Get global VisualizationBridge instance.

#### resetVisualizationBridge()

```typescript
function resetVisualizationBridge(): void
```

Reset global instance.

#### createVisualizationBridge()

```typescript
function createVisualizationBridge(): VisualizationBridge
```

Create a new instance (not singleton).

---

## 6. Types Reference

### Crypto Types

```typescript
interface KeyPair {
  publicKey: CryptoKey;
  privateKey: CryptoKey;
}

interface HashResult {
  bytes: Uint8Array;
  hex: string;
  base64: string;
}

interface Signature {
  data: Uint8Array;
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

interface ChainEntry<T> {
  index: number;
  timestamp: string;
  operationType: string;
  data: T;
  previousHash: string;
  contentHash: string;
}

interface SignedChainEntry<T> extends SignedData<ChainEntry<T>> {}

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
```

### Embedding Types

```typescript
type EmbeddingSource =
  | 'external_api'
  | 'local_model'
  | 'cache'
  | 'deterministic_fallback';

interface EmbeddingResult {
  text: string;
  vector: Float32Array;
  source: EmbeddingSource;
  model?: string;
  meta: {
    dimension: number;
    normalized: boolean;
    timestamp: string;
    cached: boolean;
  };
}

interface SimilarityResult {
  textA: string;
  textB: string;
  similarity: number;
  semanticallyMeaningful: boolean;
  source: EmbeddingSource;
}

interface StoredConcept {
  id: string;
  name: string;
  description: string;
  embedding: Float32Array;
  embeddingSource: EmbeddingSource;
  createdAt: string;
  metadata: Record<string, unknown>;
}

interface ConceptRetrievalResult {
  results: Array<{
    concept: StoredConcept;
    similarity: number;
  }>;
  query: string;
  semanticallyMeaningful: boolean;
  embeddingSource: EmbeddingSource;
  retrievedAt: string;
}
```

### Reasoning Types

```typescript
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

interface ReasoningStep {
  stepNumber: number;
  operation: ReasoningOperation;
  description: string;
  inputs: string[];
  outputs: string[];
  confidence: number;
  citations: string[];
  timestamp: string;
}

interface SignedReasoningStep extends SignedData<ReasoningStep> {}

interface Conclusion {
  statement: string;
  confidence: number;
  supportingSteps: number[];
  caveats: string[];
  semanticallyGrounded: boolean;
}

interface SignedConclusion extends SignedData<Conclusion> {}

interface ReasoningSession {
  sessionId: string;
  startedAt: string;
  initialQuery: string;
  steps: SignedReasoningStep[];
  conclusion?: SignedConclusion;
  grounding: {
    semanticEmbeddings: boolean;
    embeddingSource: string;
    conceptsUsed: number;
  };
}

interface ReasoningResult {
  session: ReasoningSession;
  concluded: boolean;
  summary: string;
  verification: {
    chainValid: boolean;
    signaturesValid: boolean;
    publicKey: JsonWebKey;
  };
}
```

### Visualization Types

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

interface ConceptNode {
  id: string;
  name: string;
  position4D: [number, number, number, number];
  role: 'input' | 'output' | 'intermediate' | 'retrieved' | 'inactive';
  strength: number;
  semantic: boolean;
}

interface ConceptEdge {
  from: string;
  to: string;
  type: 'similarity' | 'inference' | 'binding' | 'composition';
  strength: number;
}

type VisualizationEvent =
  | { type: 'SESSION_STARTED'; session: ReasoningSession }
  | { type: 'STEP_ADDED'; step: SignedReasoningStep }
  | { type: 'CONCLUSION_REACHED'; conclusion: SignedConclusion }
  | { type: 'SESSION_ENDED'; summary: string }
  | { type: 'CHAIN_VALIDATED'; valid: boolean }
  | { type: 'GROUNDING_CHANGED'; semantic: boolean; source: string }
  | { type: 'STATE_UPDATED'; state: VisualizationState };
```

---

## 7. Constants

```typescript
// Version information
export const PPP_VERSION = '3.0.0';
export const PPP_CODENAME = 'Honest Geometric Cognition';

// Default embedding dimension
const DEFAULT_DIMENSION = 384;

// Genesis hash for new chains
const GENESIS_HASH = '0'.repeat(64);

// Supported algorithms
const HASH_ALGORITHM = 'SHA-256';
const SIGNATURE_ALGORITHM = 'ECDSA';
const CURVE_NAME = 'P-256';
```

---

## Document History

| Date | Version | Changes |
|------|---------|---------|
| 2026-01-09 | 1.0.0 | Initial API reference for PPP v3 |

---

*This document is auto-generated from the PPP v3 source code and should be kept in sync with implementation changes.*
