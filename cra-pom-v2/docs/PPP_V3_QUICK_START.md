# PPP v3 Quick Start Guide

**Document Version:** 1.0.0
**Last Updated:** 2026-01-09
**Applies To:** PPP v3.0.0

---

## Prerequisites

- Modern browser with Web Workers support (Chrome 70+, Firefox 65+, Safari 12+)
- TypeScript 4.5+ (if using TypeScript)
- Node.js 18+ (for build tools)

---

## Installation

PPP v3 is part of the cra-pom-v2 project. No separate installation required.

```typescript
// Import from the v3 module
import {
  getVerifiedReasoner,
  configureEmbeddings,
  getConceptStore,
  initializeWithBasicConcepts,
} from './core/v3';
```

---

## 5-Minute Quick Start

### Step 1: Initialize the Concept Store

```typescript
import { getConceptStore, initializeWithBasicConcepts } from './core/v3';

// Get the global concept store
const store = getConceptStore();

// Load basic logic and reasoning concepts
await initializeWithBasicConcepts(store);

// Add your own concepts
await store.addConcept('climate', 'The weather conditions in an area over time');
await store.addConcept('temperature', 'A measure of how hot or cold something is');
```

### Step 2: (Optional) Configure Real Embeddings

```typescript
import { configureEmbeddings } from './core/v3';

// For semantic grounding, configure an embedding API
// Without this, the system uses fallback embeddings (non-semantic)
configureEmbeddings(
  'https://api.openai.com/v1/embeddings',
  'your-api-key',
  'text-embedding-3-small'
);
```

### Step 3: Get the Verified Reasoner

```typescript
import { getVerifiedReasoner } from './core/v3';

// Initialize the reasoner (creates Web Worker, generates keys)
const reasoner = await getVerifiedReasoner();
```

### Step 4: Run a Reasoning Session

```typescript
// Start a session with a query
await reasoner.startSession('How does temperature affect climate?');

// Look up concepts (each step is cryptographically signed)
const step1 = await reasoner.lookupConcept('temperature');
const step2 = await reasoner.lookupConcept('climate');

// Query for similar concepts
const step3 = await reasoner.querySimilar('weather patterns', 5);

// Make an inference
const step4 = await reasoner.makeInference(
  ['Temperature is a measure of heat', 'Climate includes temperature patterns'],
  'Temperature directly influences climate patterns',
  0.85
);

// Reach a conclusion
await reasoner.conclude(
  'Temperature is a fundamental component of climate, affecting weather patterns and long-term climate conditions.',
  0.8,
  [1, 2, 3, 4],  // Supporting steps
  ['Based on conceptual relationships only']
);

// End session and get results
const result = await reasoner.endSession();

console.log('Conclusion reached:', result.concluded);
console.log('Chain valid:', result.verification.chainValid);
console.log('Signatures valid:', result.verification.signaturesValid);
```

### Step 5: Export for External Verification

```typescript
// Export the audit chain
const auditChain = await reasoner.exportAuditChain();

// Save to file or send to verifier
const json = JSON.stringify(auditChain, null, 2);
console.log('Audit chain:', json);
```

---

## Complete Working Example

```typescript
import {
  getVerifiedReasoner,
  getConceptStore,
  initializeWithBasicConcepts,
  SignedHashChain,
} from './core/v3';

async function main() {
  console.log('PPP v3 Quick Start Example');
  console.log('==========================\n');

  // 1. Set up concept store
  console.log('1. Setting up concept store...');
  const store = getConceptStore();
  await initializeWithBasicConcepts(store);
  await store.addConcept('dog', 'A domesticated canine companion');
  await store.addConcept('cat', 'A domesticated feline companion');
  await store.addConcept('pet', 'An animal kept for companionship');
  console.log(`   Loaded ${store.size} concepts\n`);

  // 2. Get reasoner
  console.log('2. Initializing verified reasoner...');
  const reasoner = await getVerifiedReasoner();
  console.log('   Reasoner ready (Web Worker active)\n');

  // 3. Start session
  console.log('3. Starting reasoning session...');
  const session = await reasoner.startSession('Are dogs and cats both pets?');
  console.log(`   Session: ${session.sessionId}\n`);

  // 4. Perform reasoning
  console.log('4. Performing reasoning steps...');

  const s1 = await reasoner.lookupConcept('dog');
  console.log(`   Step 1: ${s1.payload.operation} - ${s1.payload.outputs[0]}`);

  const s2 = await reasoner.lookupConcept('cat');
  console.log(`   Step 2: ${s2.payload.operation} - ${s2.payload.outputs[0]}`);

  const s3 = await reasoner.querySimilar('pet', 3);
  console.log(`   Step 3: ${s3.payload.operation} - Found ${s3.payload.outputs.length} similar`);

  const s4 = await reasoner.makeInference(
    ['Dogs are domesticated companions', 'Cats are domesticated companions', 'Pets are companions'],
    'Both dogs and cats qualify as pets',
    0.9
  );
  console.log(`   Step 4: ${s4.payload.operation} - Confidence: ${s4.payload.confidence}\n`);

  // 5. Conclude
  console.log('5. Reaching conclusion...');
  const conclusion = await reasoner.conclude(
    'Yes, both dogs and cats are pets as they are domesticated animals kept for companionship.',
    0.9,
    [1, 2, 3, 4],
    []
  );
  console.log(`   Conclusion: ${conclusion.payload.statement.substring(0, 50)}...`);
  console.log(`   Confidence: ${conclusion.payload.confidence * 100}%\n`);

  // 6. End session
  console.log('6. Ending session and verifying...');
  const result = await reasoner.endSession();
  console.log(`   Chain valid: ${result.verification.chainValid ? '✓' : '✗'}`);
  console.log(`   Signatures valid: ${result.verification.signaturesValid ? '✓' : '✗'}\n`);

  // 7. Export and verify externally
  console.log('7. External verification...');
  const exported = await reasoner.exportAuditChain();
  const verification = await SignedHashChain.verifyExported(exported);
  console.log(`   External verification: ${verification.valid ? 'PASSED' : 'FAILED'}`);
  console.log(`   Chain length: ${verification.length} entries\n`);

  console.log('Done! The reasoning session has been cryptographically verified.');
}

main().catch(console.error);
```

---

## Common Operations

### Looking Up Concepts

```typescript
const step = await reasoner.lookupConcept('democracy');
console.log(step.payload.outputs[0]); // "Found: democracy - A system of government..."
console.log(step.proof.signature);     // Base64 ECDSA signature
```

### Semantic Search

```typescript
const results = await reasoner.querySimilar('liberty', 5);
for (const output of results.payload.outputs) {
  console.log(output); // "freedom: 0.892", "equality: 0.756", etc.
}
```

### Concept Composition

```typescript
// king - man + woman ≈ queen
const composition = await reasoner.composeConcepts([
  { concept: 'king', operation: 'add' },
  { concept: 'man', operation: 'subtract' },
  { concept: 'woman', operation: 'add' },
]);
console.log(composition.payload.outputs); // Nearest concepts to result
```

### Making Inferences

```typescript
const inference = await reasoner.makeInference(
  ['All humans are mortal', 'Socrates is human'],  // Premises
  'Socrates is mortal',                             // Conclusion
  0.99                                              // Confidence
);
```

### Checking Grounding Status

```typescript
const store = getConceptStore();
const status = store.getGroundingStatus();

console.log(`Source: ${status.embeddingSource}`);
console.log(`Semantic: ${status.percentSemantic}%`);

if (status.embeddingSource === 'deterministic_fallback') {
  console.warn('WARNING: Using non-semantic fallback embeddings');
}
```

---

## Verification

### Verify Current Chain

```typescript
const client = await getVerificationClient();
const validation = await client.validateChain();

console.log('Valid:', validation.valid);
console.log('Length:', validation.length);
console.log('Hash chain intact:', validation.details.hashChainIntact);
console.log('Signatures valid:', validation.details.signaturesValid);
```

### External Verification

```typescript
import { SignedHashChain } from './core/v3';

// Anyone can verify with only the exported data
const exportedChain = /* received from another source */;
const result = await SignedHashChain.verifyExported(exportedChain);

if (result.valid) {
  console.log('Chain is authentic and unmodified');
} else {
  console.error('Verification failed:', result.error);
  console.log('Broken at entry:', result.brokenAt);
}
```

---

## Environment Detection

The system automatically detects the environment:

```typescript
// Browser with Web Workers: Full functionality
// Browser without Workers: Error on initialization
// Node.js: Requires Web Worker polyfill

if (typeof Worker === 'undefined') {
  console.error('Web Workers required');
}
```

---

## Troubleshooting

### "Worker not initialized"

```typescript
// Make sure to await initialization
const reasoner = await getVerifiedReasoner(); // Don't forget await!
```

### "No active session"

```typescript
// Start a session before performing operations
await reasoner.startSession('Your query here');
// Now you can call lookupConcept, makeInference, etc.
```

### "Using fallback embeddings" Warning

```typescript
// Configure real embeddings to remove warning
configureEmbeddings(url, apiKey, model);

// Or accept the warning for testing
// (results won't be semantically meaningful)
```

### Chain Validation Failed

```typescript
const validation = await client.validateChain();
if (!validation.valid) {
  console.log('Error:', validation.error);
  console.log('Failed at entry:', validation.brokenAt);
  console.log('Details:', validation.details);
}
```

---

## Next Steps

1. **Read the full documentation**: `docs/PPP_V3_HONEST_DOCUMENTATION.md`
2. **Understand the architecture**: `docs/PPP_V3_TECHNICAL_SPECIFICATION.md`
3. **Review the example**: `src/core/v3/example.ts`
4. **Check the changelog**: `docs/CHANGELOG.md`

---

## Important Notes

1. **This system does NOT prevent hallucination** - It only logs what operations were requested

2. **Signatures don't create truth** - A signed false statement is just a verifiable false statement

3. **Fallback embeddings are NOT semantic** - Configure real embeddings for meaningful similarity

4. **The audit trail is only as good as what's logged** - We can't log what the LLM does internally

---

*Last updated: 2026-01-09*
