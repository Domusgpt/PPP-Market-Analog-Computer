# PPP v3: Honest Documentation

## What This Document Is

This is the official documentation for PPP v3 (Polytopal Projection Processing). Unlike typical marketing documentation, this document is **brutally honest** about what the system does and does not do.

## Executive Summary

PPP v3 is a framework for creating **auditable reasoning traces**. It provides cryptographic signing, process isolation, and semantic embeddings.

**It does NOT:**
- Prove that an LLM actually used the reasoning tools
- Validate truth or correctness of claims
- Prevent hallucination
- Provide "true" AI reasoning or understanding

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          MAIN THREAD (Untrusted)                        │
│                                                                         │
│  ┌─────────────┐     ┌─────────────────┐     ┌─────────────────────┐   │
│  │     LLM     │────▶│  Tool Interface │────▶│ Verification Client │   │
│  │ (Untrusted) │     │   (API Layer)   │     │   (Message Pass)    │   │
│  └─────────────┘     └─────────────────┘     └──────────┬──────────┘   │
│                                                          │              │
│  The LLM CAN request signatures                          │              │
│  The LLM CANNOT access the private key                   │              │
│  The LLM CANNOT forge signatures                         │              │
│                                                          │              │
├──────────────────────────────────────── MESSAGE PASSING ─┼──────────────┤
│                                                          │              │
│                          WEB WORKER (Trusted)            ▼              │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    Verification Worker                           │   │
│  │                                                                  │   │
│  │  ┌─────────────────┐  ┌─────────────┐  ┌──────────────────────┐ │   │
│  │  │   Private Key   │  │  Hash Chain │  │   Signing Service    │ │   │
│  │  │ (Non-exportable)│  │  (Signed)   │  │                      │ │   │
│  │  └─────────────────┘  └─────────────┘  └──────────────────────┘ │   │
│  │                                                                  │   │
│  │  The private key exists ONLY here                               │   │
│  │  Cannot be extracted even by this worker                        │   │
│  │  All signatures originate here                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Trust Model

### What We Can Prove

1. **A signature was created** - We can prove a specific piece of data was signed by the holder of the private key.

2. **Data hasn't been tampered with** - The hash chain provides integrity guarantees.

3. **Operations were logged** - Every operation requested through proper channels is recorded.

4. **Sequence of operations** - The hash chain proves the order of operations.

### What We CANNOT Prove

1. **LLM actually used the results** - The LLM can request a concept lookup, receive the result, and completely ignore it.

2. **Reasoning is correct** - Math on vectors doesn't equal truth. Similar vectors ≠ correct answer.

3. **No hallucination** - Signing a hallucinated statement makes it a signed hallucinated statement.

4. **Completeness** - We can't prove the LLM didn't do other reasoning it didn't log.

## Cryptographic Foundation

### Real Cryptography (Not Toy Implementations)

| Component | Implementation | Security Level |
|-----------|---------------|----------------|
| Hash Function | SHA-256 via Web Crypto | 256-bit preimage resistance |
| Signatures | ECDSA P-256 via Web Crypto | 128-bit security |
| Key Storage | Non-extractable CryptoKey | Hardware-backed where available |
| Random Numbers | crypto.getRandomValues() | CSPRNG |

### Why This Matters

Previous implementations often use "toy" cryptography:
- CRC32 or simple hash functions (not collision resistant)
- No actual signatures (just append-only with wishful thinking)
- Keys stored in JavaScript variables (accessible to any code)

PPP v3 uses **real cryptography** from the Web Crypto API, which is:
- Implemented in native code (not JavaScript)
- Hardware-accelerated where available
- Audited and battle-tested

## Process Isolation

### Why Workers Matter

The private key exists **only** in the Web Worker. This is critical because:

1. **Separate memory space** - The main thread JavaScript cannot read Worker memory
2. **No shared mutable state** - Communication is only via message passing
3. **Non-extractable keys** - Web Crypto can generate keys that cannot be exported

### Attack Surface Analysis

| Attack | Mitigated? | How |
|--------|------------|-----|
| LLM forges signature | ✅ Yes | Private key in separate process |
| LLM accesses private key | ✅ Yes | Non-extractable + Worker isolation |
| LLM modifies past entries | ✅ Yes | Hash chain + signatures |
| LLM ignores results | ❌ No | Cannot be prevented |
| LLM lies about what it did | ❌ No | We only log what's requested |
| Malicious code in Worker | ❌ No | Build-time compromise |

## Semantic Embeddings

### Honest Grounding Status

PPP v3 reports **exactly** what kind of embeddings are being used:

| Source | Semantic? | Description |
|--------|-----------|-------------|
| `external_api` | ✅ Yes | Real transformer embeddings (e.g., OpenAI) |
| `local_model` | ✅ Yes | Browser-based transformer (when implemented) |
| `cache` | ✅ Yes | Pre-computed from real model |
| `deterministic_fallback` | ❌ **NO** | Hash-based, NOT semantic |

### What "Semantic" Means

With **real** embeddings:
- `similarity(king, queen) > similarity(king, potato)` ✅
- `king - man + woman ≈ queen` (analogy) ✅
- Similar concepts cluster together ✅

With **fallback** embeddings:
- Vectors are deterministic (same input → same output)
- But have **NO semantic relationship**
- `similarity(king, queen)` is essentially random
- Analogy operations produce meaningless results

**The system always warns when using fallback mode.**

## Tool Definitions (For Agentic Use)

### OpenAI-Compatible Tool Schema

```json
{
  "type": "function",
  "function": {
    "name": "ppp_lookup_concept",
    "description": "Look up a concept in the semantic store. Returns the concept if found, or nearest matches. All lookups are logged to the signed audit chain.",
    "parameters": {
      "type": "object",
      "properties": {
        "concept": {
          "type": "string",
          "description": "The concept name to look up"
        }
      },
      "required": ["concept"]
    }
  }
}
```

### Available Tools

| Tool | Description | Logged? |
|------|-------------|---------|
| `ppp_lookup_concept` | Look up a concept by name | ✅ |
| `ppp_query_similar` | Find similar concepts | ✅ |
| `ppp_compose_concepts` | Vector arithmetic on concepts | ✅ |
| `ppp_make_inference` | Record an inference | ✅ |
| `ppp_conclude` | Record a conclusion | ✅ |
| `ppp_get_audit_chain` | Export the audit chain | ✅ |
| `ppp_verify_chain` | Validate the chain integrity | ✅ |

## Verification Protocol

### How External Verifiers Work

Anyone can verify an exported audit chain with **only the public key**:

```typescript
// Export from the system
const exported = await reasoner.exportAuditChain();

// Verify independently (anywhere, anytime)
const result = await SignedHashChain.verifyExported(exported);

console.log(result.valid); // true/false
console.log(result.details.hashChainIntact); // Were any entries modified?
console.log(result.details.signaturesValid); // Are all signatures valid?
console.log(result.details.sequenceCorrect); // Is ordering preserved?
```

### What Verification Proves

✅ **Proves:**
- Data hasn't been modified
- Signatures are valid
- Order is preserved
- Came from holder of public key

❌ **Does NOT prove:**
- Data is true
- LLM used the data
- Reasoning was correct
- All operations were logged

## Limitations and Caveats

### Fundamental Limitations

1. **We cannot read minds** - If the LLM reasons internally and doesn't call our tools, we have no record.

2. **Signing doesn't create truth** - A cryptographic signature on a false statement is just a verifiable false statement.

3. **Similarity ≠ correctness** - Finding that "dog" is similar to "cat" doesn't tell you anything about whether your conclusion is correct.

4. **Security depends on isolation** - If the Worker is compromised at build time, all bets are off.

### When NOT to Use This System

- ❌ When you need **guaranteed** correctness (use formal verification instead)
- ❌ When you need to **prove** the LLM didn't hallucinate (not currently possible)
- ❌ When you need sub-millisecond performance (crypto adds overhead)
- ❌ When you're in an environment without Web Workers

### When This System IS Useful

- ✅ Creating auditable trails of reasoning requests
- ✅ Non-repudiation of what tools were called
- ✅ Detecting tampering with logs after the fact
- ✅ Providing transparency into the reasoning process
- ✅ Research into verified AI systems

## Code Examples

### Basic Usage

```typescript
import {
  getVerifiedReasoner,
  configureEmbeddings,
  initializeWithBasicConcepts,
  getConceptStore
} from './core/v3';

// 1. Configure real embeddings (optional but recommended)
configureEmbeddings(
  'https://api.openai.com/v1/embeddings',
  process.env.OPENAI_API_KEY!,
  'text-embedding-3-small'
);

// 2. Initialize concept store
const store = getConceptStore();
await initializeWithBasicConcepts(store);
await store.addConcept('dog', 'A domesticated carnivorous mammal');
await store.addConcept('cat', 'A small domesticated carnivorous mammal');

// 3. Get reasoner
const reasoner = await getVerifiedReasoner();

// 4. Start reasoning session
await reasoner.startSession('What is the relationship between dogs and cats?');

// 5. Perform reasoning (each step is signed)
const lookup1 = await reasoner.lookupConcept('dog');
const lookup2 = await reasoner.lookupConcept('cat');
const similar = await reasoner.querySimilar('pet', 5);

// 6. Make inference
await reasoner.makeInference(
  ['Dogs are domesticated mammals', 'Cats are domesticated mammals'],
  'Both dogs and cats are domesticated mammals',
  0.95
);

// 7. Conclude
await reasoner.conclude(
  'Dogs and cats are both domesticated pet mammals',
  0.9,
  [1, 2, 3, 4],
  ['Based on concept definitions only']
);

// 8. Get result with verification
const result = await reasoner.endSession();

console.log('Chain valid:', result.verification.chainValid);
console.log('Signatures valid:', result.verification.signaturesValid);
console.log('Summary:', result.summary);
```

### Verifying Externally

```typescript
import { SignedHashChain } from './core/v3';

// Receive exported chain from any source
const exportedChain = JSON.parse(receivedData);

// Verify it
const verification = await SignedHashChain.verifyExported(exportedChain);

if (verification.valid) {
  console.log('Chain is valid and unmodified');
  console.log(`Contains ${verification.length} entries`);
} else {
  console.error('Chain verification failed:', verification.error);
  console.log('Failed at entry:', verification.brokenAt);
}
```

## Version History

| Version | Codename | Key Changes |
|---------|----------|-------------|
| 1.0 | Original | Random vectors, no real crypto |
| 2.0 | HDC/VSA | Added VSA operations, still no real crypto |
| **3.0** | **Honest** | Real crypto, process isolation, honest docs |

## FAQ

### Q: Can this prevent AI hallucination?
**A: No.** Nothing can currently prevent LLM hallucination. This system can only log what operations were requested.

### Q: Does this make AI reasoning trustworthy?
**A: No.** It makes the **audit trail** trustworthy. The reasoning itself is still an LLM doing LLM things.

### Q: Why use this instead of just logging?
**A: Cryptographic guarantees.** Regular logs can be modified. Signed hash chains cannot be modified without detection.

### Q: Is this overkill for most applications?
**A: Probably yes.** If you trust your infrastructure and don't need external verification, simple logging is fine.

### Q: What's the performance overhead?
**A: Measurable but manageable.** ECDSA signing is ~1-2ms per operation. SHA-256 is ~0.1ms per hash. Worker communication adds latency.

## Contributing

When contributing to PPP v3, please:

1. **Be honest** - Don't overclaim what the system does
2. **Document limitations** - Every new feature should document what it doesn't do
3. **Test verification** - Ensure signatures actually verify
4. **Maintain isolation** - Don't leak Worker capabilities to main thread

## License

MIT - Do whatever you want, but don't claim this does things it doesn't.
