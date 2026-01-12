# PPP Reasoning Engine - Agentic Documentation

## Overview

The Polytopal Projection Processing (PPP) Reasoning Engine provides a geometric foundation for AI reasoning with **verifiable proofs**. This documentation follows the agentic tool design standards recommended by Anthropic and OpenAI.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Architecture](#architecture)
3. [Tool Reference](#tool-reference)
4. [Verification System](#verification-system)
5. [Integration Guide](#integration-guide)
6. [Response Format](#response-format)
7. [Error Handling](#error-handling)
8. [Best Practices](#best-practices)
9. [Examples](#examples)
10. [API Reference](#api-reference)

---

## Quick Start

### Installation

```typescript
import { PPPIntegration } from './core/llm-integration';

const ppp = new PPPIntegration(10000); // 10,000-dimensional space
```

### Get Tool Definitions

```typescript
// For OpenAI function calling
const openAIFunctions = ppp.getOpenAIFunctions();

// For Anthropic tool use
const anthropicTools = ppp.getAnthropicTools();
```

### Execute a Tool

```typescript
const response = await ppp.executeTool('ppp_classify', {
  subject: 'POODLE'
});

// Response includes:
// - response: The classification result
// - proof: Cryptographic verification proof
// - citations: Grounding citations
// - reasoningTrace: Step-by-step trace
```

### Verify a Proof

```typescript
const verification = ppp.verifyProof(response.proof);
if (!verification.valid) {
  console.error('LLM may be lying:', verification.reason);
}
```

---

## Architecture

### Core Principle: Reasoning as Geometry

PPP represents reasoning as geometric operations in high-dimensional space:

```
┌─────────────────────────────────────────────────────────────────┐
│                    PPP REASONING ARCHITECTURE                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  CONCEPTS = Convex Polytopes (regions of meaning)               │
│      │                                                          │
│      ▼                                                          │
│  RULES = Rotation Operators (transform one concept to another)  │
│      │                                                          │
│      ▼                                                          │
│  INFERENCE = Trajectories (paths through semantic space)        │
│      │                                                          │
│      ▼                                                          │
│  PROOFS = Cryptographic Hashes (verifiable audit trail)         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Component Stack

```
┌─────────────────────────────────────────┐
│           LLM Interface Layer           │  ← Tool definitions, prompts
├─────────────────────────────────────────┤
│         PPPIntegration Runtime          │  ← Execution + verification
├─────────────────────────────────────────┤
│          PPPStateManager API            │  ← Request/response handling
├─────────────────────────────────────────┤
│        PPPReasoningEngine Core          │  ← Geometric reasoning
├─────────────────────────────────────────┤
│   HDC │ FHRR │ Polytopes │ Rotors │ Garden │  ← Mathematical primitives
└─────────────────────────────────────────┘
```

### How Verification Prevents Lying

```
┌─────────────────────────────────────────────────────────────────┐
│                    VERIFICATION FLOW                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. LLM calls PPP tool                                          │
│       │                                                         │
│       ▼                                                         │
│  2. PPP executes operation                                      │
│       │                                                         │
│       ▼                                                         │
│  3. Audit chain records operation with SHA-256 hash             │
│       │                                                         │
│       ▼                                                         │
│  4. Verification proof generated (includes hash)                │
│       │                                                         │
│       ▼                                                         │
│  5. LLM receives result + proof                                 │
│       │                                                         │
│       ▼                                                         │
│  6. LLM MUST include proof hash in response                     │
│       │                                                         │
│       ▼                                                         │
│  7. External system verifies proof against audit chain          │
│       │                                                         │
│       ▼                                                         │
│  8. If LLM fabricated result → HASH MISMATCH → DETECTED         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Tool Reference

### ppp_define_concept

**Purpose**: Define a new concept in the semantic space.

**When to Use**:
- Introducing new entities or concepts
- Building knowledge taxonomies
- Before reasoning about something new

**Parameters**:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `conceptName` | string | Yes | Unique identifier (e.g., "DOG") |
| `superConcepts` | string[] | No | Parent concepts in taxonomy |
| `radius` | number | No | Size of concept region (0.1-1.0, default: 0.5) |
| `properties` | array | No | Role-filler pairs for attributes |

**Returns**: `VerifiedResponse` with created concept and proof.

**Example**:
```json
{
  "conceptName": "DOG",
  "superConcepts": ["ANIMAL", "MAMMAL"],
  "radius": 0.4
}
```

---

### ppp_define_rule

**Purpose**: Define a transformation rule between concepts.

**When to Use**:
- Establishing relationships
- Encoding domain knowledge
- Before inference that requires this relationship

**Parameters**:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `ruleName` | string | Yes | Unique rule identifier |
| `description` | string | Yes | Human-readable description |
| `fromConcept` | string | Yes | Source concept name |
| `toConcept` | string | Yes | Target concept name |
| `confidence` | number | No | Reliability (0-1, default: 1.0) |
| `bidirectional` | boolean | No | Can be reversed? (default: false) |

**Returns**: `VerifiedResponse` with rule and rotation magnitude.

**Example**:
```json
{
  "ruleName": "DOG_IS_ANIMAL",
  "description": "Dogs are animals",
  "fromConcept": "DOG",
  "toConcept": "ANIMAL",
  "confidence": 1.0
}
```

---

### ppp_classify

**Purpose**: Classify a query into the nearest concept.

**When to Use**:
- Determining category membership
- First step before applying rules
- Verifying concept membership

**Parameters**:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `subject` | string | Yes | Concept name or description to classify |

**Returns**: `VerifiedResponse` with:
- `answer`: Classified concept name
- `confidence`: Classification confidence (0-1)
- `alternatives`: Other possible classifications
- `proof`: Verification proof (REQUIRED)

**Example**:
```json
{
  "subject": "POODLE"
}
```

**Response**:
```json
{
  "answer": "DOG",
  "confidence": 0.89,
  "alternatives": [
    { "answer": "PET", "confidence": 0.72 },
    { "answer": "ANIMAL", "confidence": 0.68 }
  ],
  "proof": {
    "operationId": "op_abc123",
    "auditHash": "3f2a1b4c",
    "confidence": 0.89
  }
}
```

---

### ppp_infer

**Purpose**: Apply a rule to derive new knowledge.

**When to Use**:
- Deriving conclusions from relationships
- Single-step logical inference
- When verifiable proof is needed

**Parameters**:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `subject` | string | Yes | Starting concept |
| `predicate` | string | Yes | Rule name to apply |

**Returns**: `VerifiedResponse` with:
- `answer`: Inferred concept
- `confidence`: Inference confidence
- `geometricProof`: Start → rotations → end
- `proof`: Verification proof

**Example**:
```json
{
  "subject": "FIDO",
  "predicate": "IS_A"
}
```

---

### ppp_chain_inference

**Purpose**: Apply multiple rules in sequence.

**When to Use**:
- Complex multi-step deductions
- Building logical arguments
- Transitive inference chains

**Parameters**:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `subject` | string | Yes | Starting concept |
| `ruleNames` | string[] | Yes | Ordered list of rules |

**Returns**: Complete inference chain with step-by-step proofs.

**Example**:
```json
{
  "subject": "SOCRATES",
  "ruleNames": ["IS_HUMAN", "HUMANS_ARE_MORTAL"]
}
```

---

### ppp_predict

**Purpose**: Predict multiple possible futures.

**When to Use**:
- Exploring possible outcomes
- Scenario planning
- When multiple paths are possible

**Parameters**:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `subject` | string | Yes | Starting concept |
| `steps` | number | No | Prediction steps (1-20, default: 5) |

**Returns**: Bundle of possible futures with probabilities.

---

### ppp_verify

**Purpose**: Check if a statement is true.

**When to Use**:
- Fact verification
- Checking claims
- When provable truth values are needed

**Parameters**:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `subject` | string | Yes | Subject of statement |
| `predicate` | string | Yes | Relationship to check |
| `object` | string | Yes | Expected result |

**Returns**: `TRUE`, `APPROXIMATELY TRUE`, or `FALSE` with confidence.

---

### ppp_analogy

**Purpose**: Solve "A is to B as C is to ?" analogies.

**When to Use**:
- Analogical reasoning
- Transferring relationships
- Creative problem solving

**Parameters**:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `a` | string | Yes | First term |
| `b` | string | Yes | Second term |
| `c` | string | Yes | Third term |

**Returns**: The fourth term completing the analogy.

**Example**:
```json
{
  "a": "KING",
  "b": "QUEEN",
  "c": "MAN"
}
```
→ Returns: "WOMAN"

---

## Verification System

### VerificationProof Structure

Every PPP operation generates a proof:

```typescript
interface VerificationProof {
  operationId: string;      // Unique operation ID
  timestamp: string;        // ISO 8601 timestamp
  auditHash: string;        // SHA-256 hash linking to audit chain
  previousHash: string;     // Previous hash for chain verification
  operation: string;        // Operation type
  inputFingerprint: string; // Hash of inputs
  outputFingerprint: string;// Hash of outputs
  confidence: number;       // Confidence score
  chainPosition: number;    // Position in audit chain
}
```

### Verification Flow

```typescript
// 1. LLM executes tool
const response = await ppp.executeTool('ppp_classify', { subject: 'DOG' });

// 2. Extract proof from response
const proof = response.proof;

// 3. Verify proof
const verification = ppp.verifyProof(proof);

if (verification.valid) {
  console.log('Result is trustworthy');
} else {
  console.error('UNTRUSTED:', verification.reason);
  // Possible reasons:
  // - "Proof hash mismatch - possible fabrication"
  // - "Audit chain is invalid"
  // - "Proof timestamp is in the future"
}
```

### Why This Prevents Lying

1. **Hash chaining**: Each operation's hash depends on the previous hash
2. **Input/output fingerprints**: The proof includes hashes of actual inputs and outputs
3. **External verification**: Any system can verify proofs against the audit chain
4. **Temporal ordering**: Timestamps and chain positions prevent replay attacks

**If an LLM fabricates a result**:
- It cannot generate a valid `auditHash` (would need to compute SHA-256)
- The `inputFingerprint` won't match the actual operation
- The `chainPosition` won't exist in the audit chain
- Verification WILL fail

---

## Integration Guide

### OpenAI Integration

```typescript
import OpenAI from 'openai';
import { PPPIntegration } from './core/llm-integration';

const openai = new OpenAI();
const ppp = new PPPIntegration();

// Get function definitions
const functions = ppp.getOpenAIFunctions();

// Make a chat completion with tools
const completion = await openai.chat.completions.create({
  model: 'gpt-4',
  messages: [
    { role: 'system', content: PPP_SYSTEM_PROMPT },
    { role: 'user', content: 'Is a poodle a mammal?' }
  ],
  functions: functions,
  function_call: 'auto'
});

// Handle function calls
if (completion.choices[0].message.function_call) {
  const { name, arguments: args } = completion.choices[0].message.function_call;
  const result = await ppp.executeTool(name, JSON.parse(args));

  // Verify the result
  const verification = ppp.verifyProof(result.proof);
  if (!verification.valid) {
    throw new Error('Verification failed');
  }
}
```

### Anthropic Integration

```typescript
import Anthropic from '@anthropic-ai/sdk';
import { PPPIntegration, PPP_SYSTEM_PROMPT } from './core/llm-integration';

const anthropic = new Anthropic();
const ppp = new PPPIntegration();

// Get tool definitions
const tools = ppp.getAnthropicTools();

// Make a message with tools
const message = await anthropic.messages.create({
  model: 'claude-sonnet-4-20250514',
  max_tokens: 1024,
  system: PPP_SYSTEM_PROMPT,
  tools: tools,
  messages: [
    { role: 'user', content: 'Is a poodle a mammal?' }
  ]
});

// Handle tool use
for (const block of message.content) {
  if (block.type === 'tool_use') {
    const result = await ppp.executeTool(block.name, block.input);

    // Verify before trusting
    const verification = ppp.verifyProof(result.proof);
    console.log('Verified:', verification.valid);
  }
}
```

---

## Response Format

### Required Format for LLM Responses

LLMs using PPP MUST structure responses as:

```
[REASONING]
{Step-by-step reasoning using PPP tools}

[GROUNDED CLAIMS]
1. {Claim} [PROOF: {proofHash}] (confidence: {X}%)
2. {Claim} [PROOF: {proofHash}] (confidence: {X}%)

[ANSWER]
{Final answer with aggregated confidence}

[VERIFICATION]
All claims can be verified using proof hashes against audit chain.
Audit chain head: {headHash}
```

### Example Response

```
[REASONING]
1. Invoked ppp_classify("POODLE") → DOG [PROOF: a3f2c1] (89%)
2. Invoked ppp_infer("DOG", "IS_A") → ANIMAL [PROOF: b4e3d2] (94%)
3. Invoked ppp_verify("DOG", "IS_A", "MAMMAL") → TRUE [PROOF: c5f4e3] (91%)

[GROUNDED CLAIMS]
1. A poodle is a type of dog [PROOF: a3f2c1] (confidence: 89%)
2. Dogs are animals [PROOF: b4e3d2] (confidence: 94%)
3. Dogs are mammals [PROOF: c5f4e3] (confidence: 91%)

[ANSWER]
Yes, a poodle is a mammal. Poodles are dogs, and dogs are mammals.
(Combined confidence: 76%)

[VERIFICATION]
Verify proofs a3f2c1, b4e3d2, c5f4e3 against the PPP audit chain.
Audit chain head: d6g5f4
```

---

## Error Handling

### Error Codes

| Code | Description | Resolution |
|------|-------------|------------|
| `CONCEPT_EXISTS` | Concept already exists | Use different name |
| `CONCEPT_NOT_FOUND` | Concept doesn't exist | Define concept first |
| `UNKNOWN_RULE` | Rule doesn't exist | Define rule first |
| `INVALID_SUPERCONCEPT` | Parent concept missing | Define parent first |
| `CHAIN_BROKEN` | Inference chain invalid | Check rule connectivity |
| `NO_CONCEPTS` | No concepts defined | Define concepts |
| `NO_APPLICABLE_RULES` | No rules can apply | Define more rules |
| `VERIFICATION_FAILED` | Proof verification failed | Do not trust result |

### Error Response Format

```json
{
  "success": false,
  "action": "ppp_infer",
  "timestamp": "2024-01-15T10:30:00Z",
  "error": {
    "code": "UNKNOWN_RULE",
    "message": "Rule 'FLIES' does not exist",
    "details": {
      "availableRules": ["IS_A", "HAS_PART", "CAUSES"]
    }
  }
}
```

---

## Best Practices

### For Tool Designers

1. **Clear descriptions**: Make tool purposes unambiguous
2. **Explicit parameters**: Document all parameters with types and constraints
3. **Usage guidance**: Include `whenToUse` and `whenNotToUse`
4. **Error documentation**: List all possible errors and resolutions
5. **Examples**: Provide input/output examples

### For LLM Integration

1. **Always verify**: Never trust unverified results
2. **Include proofs**: Always include proof hashes in responses
3. **Chain operations**: Use `ppp_chain_inference` for multi-step reasoning
4. **Check confidence**: Present low-confidence results with uncertainty
5. **Cite operations**: Ground every claim in a specific PPP operation

### For External Verification

1. **Verify all proofs**: Check every proof hash before trusting
2. **Check timestamps**: Reject proofs with future timestamps
3. **Audit chain integrity**: Periodically validate the audit chain
4. **Log verification failures**: Track and investigate failures

---

## Examples

### Example 1: Building a Taxonomy

```typescript
// Define concepts
await ppp.executeTool('ppp_define_concept', { conceptName: 'ANIMAL' });
await ppp.executeTool('ppp_define_concept', {
  conceptName: 'MAMMAL',
  superConcepts: ['ANIMAL']
});
await ppp.executeTool('ppp_define_concept', {
  conceptName: 'DOG',
  superConcepts: ['MAMMAL']
});

// Define rules
await ppp.executeTool('ppp_define_rule', {
  ruleName: 'IS_MAMMAL',
  description: 'Is a mammal',
  fromConcept: 'DOG',
  toConcept: 'MAMMAL'
});
```

### Example 2: Verified Reasoning

```typescript
// Classify
const classifyResult = await ppp.executeTool('ppp_classify', {
  subject: 'POODLE'
});
console.log(`Classified as: ${classifyResult.response.data.result.answer}`);
console.log(`Proof hash: ${classifyResult.proof.auditHash}`);

// Verify
const verification = ppp.verifyProof(classifyResult.proof);
if (verification.valid) {
  console.log('Result is verified and trustworthy');
}
```

### Example 3: Multi-Step Inference

```typescript
const result = await ppp.executeTool('ppp_chain_inference', {
  subject: 'SOCRATES',
  ruleNames: ['IS_HUMAN', 'HUMANS_MORTAL']
});

// Result includes proof for entire chain
console.log(`Final answer: ${result.response.data.result.answer}`);
console.log(`Chain confidence: ${result.response.data.result.confidence}`);
console.log(`Proof: ${result.proof.auditHash}`);
```

---

## API Reference

### PPPIntegration

```typescript
class PPPIntegration {
  constructor(dimension?: number);

  // Get tool definitions
  getToolDefinitions(): ToolDefinition[];
  getOpenAIFunctions(): OpenAIFunction[];
  getAnthropicTools(): AnthropicTool[];

  // Execute tools
  executeTool(toolName: string, params: object): Promise<VerifiedResponse>;

  // Verification
  verifyProof(proof: VerificationProof): { valid: boolean; reason?: string };
  getAuditChain(): { entries: number; headHash: string; isValid: boolean };

  // State management
  exportVerifiedState(): VerifiedState;
}
```

### VerifiedResponse

```typescript
interface VerifiedResponse {
  response: MachineAPIResponse;  // The actual result
  proof: VerificationProof;      // Cryptographic proof
  reasoningTrace: string[];      // Step-by-step trace
  citations: GroundingCitation[]; // Grounded claims
  isVerifiable: boolean;         // Can be verified
}
```

### VerificationProof

```typescript
interface VerificationProof {
  operationId: string;
  timestamp: string;
  auditHash: string;
  previousHash: string;
  operation: string;
  inputFingerprint: string;
  outputFingerprint: string;
  confidence: number;
  chainPosition: number;
}
```

---

## Conclusion

The PPP Reasoning Engine provides:

1. **Geometric reasoning**: Concepts as polytopes, rules as rotations
2. **Verifiable proofs**: Every operation generates a cryptographic proof
3. **Anti-lying mechanism**: Fabricated results fail verification
4. **Standard tool interface**: Compatible with OpenAI and Anthropic

**Trust comes from verification, not assertion.**

Every claim made using PPP can be independently verified. If an LLM lies, the proof won't match. This is the foundation of trustworthy AI reasoning.
