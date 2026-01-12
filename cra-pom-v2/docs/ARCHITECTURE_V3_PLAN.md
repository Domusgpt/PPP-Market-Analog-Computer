# PPP v3 Architecture - Ground-Up Redesign

## Executive Summary

This document outlines a complete redesign of the PPP (Polytopal Projection Processing) system addressing fundamental flaws identified in v2:

1. **Security theater → Real cryptography**
2. **Random vectors → Grounded embeddings**
3. **Same-process trust → Isolation architecture**
4. **Marketing claims → Honest documentation**

---

## Part 1: Problem Analysis

### What v2 Got Wrong

| Claim | Reality | Impact |
|-------|---------|--------|
| "SHA-256 hashes" | 32-bit simple hash | Trivially forgeable |
| "Cannot be forged" | Same-process access | LLM can compute valid hashes |
| "Semantic reasoning" | Random orthogonal vectors | No actual meaning |
| "Cryptographic proofs" | No signatures | No non-repudiation |
| "Prevents lying" | No enforcement | LLM can ignore tools entirely |

### Root Causes

1. **No threat model**: We never defined who the adversary is
2. **Conflated demo with security**: Built a demo, documented it as secure
3. **Symbolic ≠ Semantic**: Vector operations don't create meaning
4. **Trust boundary confusion**: LLM and verifier in same process

---

## Part 2: Threat Model

### Adversary: The LLM Itself

The primary threat is an LLM that:
- **Hallucinates**: Makes up facts that sound plausible
- **Fabricates proofs**: If it can access the hash function, it can create valid-looking proofs
- **Ignores tools**: Simply doesn't call PPP and makes claims anyway
- **Manipulates context**: Refers to fake previous operations

### Trust Boundaries

```
┌─────────────────────────────────────────────────────────────────────┐
│                         UNTRUSTED ZONE                               │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                         LLM                                  │   │
│  │  - Can generate any text                                     │   │
│  │  - Can claim anything                                        │   │
│  │  - Has no inherent trustworthiness                          │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                │ API calls only (no internal access)
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         TRUST BOUNDARY                               │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    PPP Service (Isolated)                    │   │
│  │  - Runs in separate process/container                        │   │
│  │  - Signs all outputs with private key                        │   │
│  │  - LLM cannot access internals                              │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                │ Signed responses
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         TRUSTED ZONE                                 │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                      Verifier                                │   │
│  │  - Has PPP service's public key                             │   │
│  │  - Verifies signatures on all claims                        │   │
│  │  - Rejects unsigned or invalid claims                       │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### What We CAN Guarantee (Honest Assessment)

| Guarantee | Mechanism | Limitation |
|-----------|-----------|------------|
| PPP was actually called | Signed proof from isolated service | Requires process isolation |
| Inputs/outputs match | Hash in signature | Cannot prevent LLM from ignoring results |
| Temporal ordering | Timestamp in signature | Clock must be trusted |
| Chain integrity | Hash chain with signatures | Only for PPP operations, not LLM claims |

### What We CANNOT Guarantee

- LLM uses PPP for all claims (it can still hallucinate)
- LLM correctly interprets PPP results
- Semantic truth of the reasoning (PPP is symbolic, not semantic)
- LLM doesn't paraphrase incorrectly

---

## Part 3: Redesigned Architecture

### Layer 1: Cryptographic Foundation

```typescript
// Use Web Crypto API for real cryptography
interface CryptoService {
  // Asymmetric signing (Ed25519 via ECDSA P-256 in Web Crypto)
  generateKeyPair(): Promise<CryptoKeyPair>;
  sign(data: Uint8Array, privateKey: CryptoKey): Promise<Uint8Array>;
  verify(data: Uint8Array, signature: Uint8Array, publicKey: CryptoKey): Promise<boolean>;

  // Hashing (real SHA-256)
  hash(data: Uint8Array): Promise<Uint8Array>;

  // Key export for verification
  exportPublicKey(key: CryptoKey): Promise<JsonWebKey>;
}
```

### Layer 2: Isolated PPP Service

```typescript
interface PPPServiceConfig {
  // Process isolation mode
  isolation: 'worker' | 'iframe' | 'subprocess' | 'remote';

  // Signing key (generated at startup, public key shared)
  keyPair: CryptoKeyPair;

  // No direct access to internals from outside
  readonly: true;
}

interface SignedResponse<T> {
  // The actual data
  data: T;

  // Cryptographic proof
  proof: {
    // SHA-256 of data
    dataHash: string;

    // Ed25519/ECDSA signature of dataHash
    signature: string;

    // Public key fingerprint (for key rotation)
    keyId: string;

    // Timestamp (ISO 8601)
    timestamp: string;

    // Monotonic sequence number
    sequence: number;

    // Hash of previous response (chain)
    previousHash: string;
  };
}
```

### Layer 3: Semantic Grounding (NEW)

**The fundamental change**: Instead of random vectors, we ground concepts in real embeddings.

```typescript
interface EmbeddingService {
  // Get embedding from text (uses real model)
  embed(text: string): Promise<Float32Array>;

  // Similarity between embeddings
  similarity(a: Float32Array, b: Float32Array): number;
}

// Options for embedding source:
// 1. Local: TensorFlow.js with Universal Sentence Encoder
// 2. Remote: OpenAI embeddings API
// 3. Hybrid: Cache common concepts, query for new ones
```

### Layer 4: Grounded Concept Store

```typescript
interface GroundedConcept {
  name: string;

  // Real embedding from actual text/definition
  embedding: Float32Array;

  // Source of the embedding
  groundingSource: {
    type: 'definition' | 'examples' | 'external';
    text: string;        // The text that was embedded
    model: string;       // Which embedding model
    timestamp: string;   // When it was computed
  };

  // Polytope is derived from embedding, not random
  polytope: DerivedPolytope;

  // Relationships learned from embedding similarity
  relations: Map<string, number>; // concept -> similarity
}
```

### Layer 5: Honest Confidence Scores

```typescript
interface CalibratedConfidence {
  // Raw similarity score
  raw: number;

  // Calibrated probability (based on validation set)
  calibrated: number;

  // Uncertainty bounds
  lower: number;
  upper: number;

  // Basis for confidence
  basis: 'embedding_similarity' | 'rule_chain' | 'prior' | 'unknown';

  // Warning if confidence is unreliable
  warning?: string;
}
```

---

## Part 4: Implementation Plan

### Phase 1: Cryptographic Foundation (Week 1)

```
src/core/v3/
├── crypto.ts          # Web Crypto wrappers
├── signed-response.ts # Signed response types
├── hash-chain.ts      # Proper hash chain with signatures
└── key-manager.ts     # Key generation and storage
```

**Deliverables**:
- Real SHA-256 hashing
- ECDSA P-256 signing (Web Crypto compatible)
- Verifiable signatures
- Proper hash chain

### Phase 2: Process Isolation (Week 2)

```
src/core/v3/
├── isolation/
│   ├── worker-service.ts   # Web Worker isolation
│   ├── iframe-service.ts   # Iframe sandbox isolation
│   ├── message-protocol.ts # Secure message passing
│   └── service-proxy.ts    # Proxy for isolated service
```

**Deliverables**:
- PPP runs in Web Worker (no shared memory)
- All communication via postMessage
- Signatures on all responses
- LLM cannot access internal state

### Phase 3: Semantic Grounding (Week 3)

```
src/core/v3/
├── embeddings/
│   ├── embedding-service.ts  # Abstraction over embedding sources
│   ├── local-embeddings.ts   # TensorFlow.js USE
│   ├── remote-embeddings.ts  # OpenAI/other API
│   └── embedding-cache.ts    # Cache for performance
├── concepts/
│   ├── grounded-concept.ts   # Concepts with real embeddings
│   ├── concept-store.ts      # Storage and retrieval
│   └── relation-learner.ts   # Learn relations from similarity
```

**Deliverables**:
- Real embeddings from text
- Concepts grounded in meaning
- Similarity-based relations
- Not random vectors

### Phase 4: Reasoning Engine (Week 4)

```
src/core/v3/
├── reasoning/
│   ├── classifier.ts       # Classification with calibration
│   ├── inference.ts        # Rule-based inference
│   ├── confidence.ts       # Calibrated confidence scores
│   └── explainer.ts        # Explanation generation
```

**Deliverables**:
- Classification based on embedding similarity
- Inference with confidence propagation
- Calibrated confidence (not arbitrary multiplication)
- Explanations based on actual operations

### Phase 5: LLM Integration (Week 5)

```
src/core/v3/
├── llm/
│   ├── tool-definitions.ts  # Tool defs for OpenAI/Anthropic
│   ├── constrained-output.ts # Structured output enforcement
│   ├── grounding-checker.ts  # Verify claims have proofs
│   └── response-parser.ts    # Parse and verify LLM responses
```

**Deliverables**:
- Tool definitions that work
- Output parsing that extracts proofs
- Verification that all claims are grounded
- Rejection of ungrounded claims

### Phase 6: HypercubeCore Integration (Week 6)

```
src/core/v3/
├── visualization/
│   ├── hypercube-bridge.ts  # Bridge to HypercubeCore
│   ├── state-mapper.ts      # Map PPP state to 64 channels
│   └── glitch-controller.ts # Map violations to glitch
```

**Deliverables**:
- PPP state drives visualization
- Semantic violations cause glitches
- Real-time thought visualization
- Integration with existing HypercubeCore

---

## Part 5: Honest Documentation Strategy

### Principle: Document What IS, Not What We WISH

**Before** (v2 style):
> "The PPP system provides cryptographic proofs that cannot be forged, ensuring LLM truthfulness."

**After** (v3 style):
> "The PPP system provides signed proofs that a specific operation occurred. This proves the operation happened but does NOT prevent the LLM from ignoring results or making ungrounded claims. External verification is required."

### Documentation Structure

```
docs/
├── THREAT_MODEL.md           # Honest threat model
├── SECURITY_GUARANTEES.md    # What we can/cannot guarantee
├── ARCHITECTURE.md           # Technical architecture
├── LIMITATIONS.md            # Known limitations
├── INTEGRATION_GUIDE.md      # How to integrate
├── VERIFICATION_GUIDE.md     # How to verify claims
└── EXAMPLES/
    ├── basic_usage.ts        # Working code
    ├── verification.ts       # How to verify
    └── integration_test.ts   # Full integration test
```

### Required Sections in Each Doc

1. **What This Does** - Accurate description
2. **What This Does NOT Do** - Explicit limitations
3. **Assumptions** - What must be true for this to work
4. **Failure Modes** - What can go wrong
5. **Working Example** - Copy-paste code that runs

---

## Part 6: Integration with HypercubeCore

### Current HypercubeCore Architecture

Based on the context provided:
- Vanilla JavaScript/WebGL visualization
- 64-channel data input
- Glitch intensity uniform for visual effects
- Real-time "thought" visualization

### Integration Points

```javascript
// GeometricKernel.js becomes the bridge between PPP v3 and HypercubeCore

import { PPPServiceProxy } from './ppp-v3/isolation/service-proxy.js';
import { EmbeddingService } from './ppp-v3/embeddings/embedding-service.js';

export class ChronomorphicEngine {
  constructor() {
    // PPP runs in isolated worker
    this.ppp = new PPPServiceProxy('worker');

    // Real embeddings
    this.embeddings = new EmbeddingService('local'); // or 'remote'

    // State
    this.position = new Float32Array(4); // 4D thought vector
    this.target = new Float32Array(4);
    this.proofChain = [];
  }

  async injectThought(text) {
    // 1. Get real embedding
    const embedding = await this.embeddings.embed(text);

    // 2. Classify via isolated PPP
    const result = await this.ppp.classify(embedding);

    // 3. Verify signature
    const valid = await this.ppp.verifySignature(result);
    if (!valid) throw new Error('Invalid signature');

    // 4. Store proof
    this.proofChain.push(result.proof);

    // 5. Project to 4D for visualization
    this.target = this.projectTo4D(embedding);

    return result;
  }

  tick(speed = 0.05) {
    // Interpolate toward target
    for (let i = 0; i < 4; i++) {
      this.position[i] += (this.target[i] - this.position[i]) * speed;
    }

    // Normalize
    const mag = Math.sqrt(this.position.reduce((s, v) => s + v*v, 0));
    for (let i = 0; i < 4; i++) {
      this.position[i] /= mag;
    }

    // Check semantic validity (based on actual embedding distances)
    const tension = this.computeTension();
    const chi = tension > 0.5 ? 1 : 0; // Topology violation

    return {
      vector: this.position,
      chi,
      tension,
      proofCount: this.proofChain.length,
      lastProof: this.proofChain[this.proofChain.length - 1]
    };
  }

  projectTo4D(embedding) {
    // PCA or learned projection from high-D to 4D
    // This is a real projection, not random
    // ... implementation
  }

  computeTension() {
    // Distance from nearest grounded concept
    // Uses real embedding similarity
    // ... implementation
  }
}
```

### Visual Feedback Mapping

| PPP State | Visual Effect |
|-----------|---------------|
| Valid classification (high confidence) | Stable geometry, low glitch |
| Uncertain classification | Moderate glitch, color shift |
| No matching concept | High glitch, topology violation |
| Invalid proof | Maximum glitch, red alert |
| Ungrounded claim | Different visual treatment (no PPP backing) |

---

## Part 7: Migration Path

### For Existing Users

1. **v2 to v3 migration script**: Convert v2 state to v3 format
2. **Compatibility layer**: v2 API that wraps v3 (with warnings)
3. **Deprecation timeline**: v2 supported for 6 months, then removed

### Breaking Changes

| v2 Feature | v3 Change | Reason |
|------------|-----------|--------|
| `PPPStateManager` | Replaced by `PPPServiceProxy` | Process isolation |
| Simple hash | Real SHA-256 + signatures | Security |
| Random concept vectors | Grounded embeddings | Semantic meaning |
| Synchronous operations | All async (worker communication) | Isolation |
| In-process verification | External verification required | Trust boundary |

---

## Part 8: Success Criteria

### Security

- [ ] All operations signed with ECDSA P-256
- [ ] Signatures verifiable with public key only
- [ ] No internal state accessible from outside worker
- [ ] Hash chain verifiable independently

### Semantic Grounding

- [ ] Concepts have real embeddings from text
- [ ] Similarity scores correlate with human judgment
- [ ] Classification accuracy measurable on test set
- [ ] Not random vectors

### Documentation

- [ ] All claims in docs are testable
- [ ] Working copy-paste examples
- [ ] Explicit limitation sections
- [ ] Failure mode documentation

### Integration

- [ ] Works with OpenAI function calling
- [ ] Works with Anthropic tool use
- [ ] HypercubeCore integration complete
- [ ] End-to-end demo runs

---

## Appendix A: Technology Choices

### Cryptography

**Web Crypto API** (native browser):
- ECDSA P-256 for signatures (widely supported)
- SHA-256 for hashing
- No external dependencies
- Hardware acceleration where available

**Why not Ed25519?**
- Not in Web Crypto standard yet
- P-256 is sufficient for our threat model
- Better browser support

### Embeddings

**Option 1: TensorFlow.js + Universal Sentence Encoder**
- Pros: Runs locally, no API costs, offline capable
- Cons: ~30MB model download, slower than remote

**Option 2: OpenAI Embeddings API**
- Pros: High quality, fast
- Cons: Requires API key, costs money, network dependency

**Recommendation**: Support both, default to local for privacy

### Process Isolation

**Web Workers** (primary):
- Native browser support
- No shared memory
- postMessage for communication
- Can't access main thread variables

**Alternatives considered**:
- iframe sandbox: More complex, CSP issues
- Subprocess: Not browser-compatible
- Remote service: Latency, availability concerns

---

## Appendix B: Comparison to v2

| Aspect | v2 | v3 |
|--------|----|----|
| Hash function | Simple 32-bit | SHA-256 (Web Crypto) |
| Signatures | None | ECDSA P-256 |
| Process isolation | None (same process) | Web Worker |
| Concept grounding | Random vectors | Real embeddings |
| Confidence | Arbitrary multiplication | Calibrated |
| Documentation | Marketing | Technical + honest |
| Verification | Same-process (useless) | External (real) |

---

## Next Steps

1. Review and approve this plan
2. Begin Phase 1 implementation (crypto foundation)
3. Set up test infrastructure
4. Implement phases 2-6
5. Write honest documentation
6. Integration testing
7. Migration guide for v2 users
