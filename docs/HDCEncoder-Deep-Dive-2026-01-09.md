# HDCEncoder Deep Dive: Neural-Geometric Bridge Analysis
**Date:** 2026-01-09
**Module:** `lib/encoding/HDCEncoder.ts`
**Version:** 2.0.0

> **v2.0 Update:** This document has been updated to reflect the enhanced HDCEncoder with:
> - Real embedding API integration (**Gemini**, **Anthropic/Voyage**, OpenAI, Cohere, Custom)
> - Positional encoding for word order preservation
> - Multi-head self-attention aggregation
> - Configurable/domain-specific archetypes
> - Improved tokenization with subword support
> - Preset domain archetypes (Medical, Legal, Software)

---

## Executive Summary

The HDCEncoder is the **neural-geometric bridge** of the Chronomorphic Polytopal Engine (CPE). It converts semantic input (text or neural embeddings) into 4D force vectors that drive the physics simulation. This module answers a critical question: *How do we translate meaning into geometry?*

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [The Encoding Pipeline](#2-the-encoding-pipeline)
3. [Core Components Explained](#3-core-components-explained)
4. [Mathematical Foundations](#4-mathematical-foundations)
5. [The 24 Concept Archetypes](#5-the-24-concept-archetypes)
6. [Code Walkthrough](#6-code-walkthrough)
7. [Limitations & Improvement Opportunities](#7-limitations--improvement-opportunities)
8. [Integration Guide](#8-integration-guide)

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        HDCEncoder Pipeline                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   INPUT                                                             │
│   ├── Text: "reasoning about causality"                            │
│   └── Embedding: Float32Array[1536]                                │
│            │                                                        │
│            ▼                                                        │
│   ┌─────────────────┐                                              │
│   │   Tokenization   │  ← Whitespace split, lowercase, filter      │
│   │   ["reasoning",  │                                              │
│   │    "about",      │                                              │
│   │    "causality"]  │                                              │
│   └────────┬────────┘                                              │
│            │                                                        │
│            ▼                                                        │
│   ┌─────────────────┐                                              │
│   │ Token Embedding  │  ← Hash-based deterministic vectors         │
│   │ 1536D per token  │                                              │
│   └────────┬────────┘                                              │
│            │                                                        │
│            ▼                                                        │
│   ┌─────────────────┐                                              │
│   │ Weighted Sum     │  ← TF-style: weight = min(1, len/10)        │
│   │ + Normalize      │                                              │
│   └────────┬────────┘                                              │
│            │                                                        │
│            ▼                                                        │
│   ┌─────────────────────────────────────────────────────────┐      │
│   │              Johnson-Lindenstrauss Projection            │      │
│   │                                                          │      │
│   │   1536D ──► 4D Linear Component (position force)        │      │
│   │   1536D ──► 6D Rotational Component (bivector force)    │      │
│   │                                                          │      │
│   │   Matrix: Gaussian random × scale (1/√dim)              │      │
│   └────────┬────────────────────────────────────────────────┘      │
│            │                                                        │
│            ▼                                                        │
│   ┌─────────────────┐                                              │
│   │ Concept Matching │  ← Distance to 24 archetype embeddings      │
│   │ + Softmax        │  ← Temperature-controlled weighting          │
│   └────────┬────────┘                                              │
│            │                                                        │
│            ▼                                                        │
│   OUTPUT: Force { linear: Vector4D, rotational: Bivector4D }       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. The Encoding Pipeline

### Step-by-Step Breakdown

#### Step 1: Tokenization (lines 595-611)

```typescript
_tokenize(text: string): TokenizationResult {
    const tokens = text
        .toLowerCase()                    // Normalize case
        .replace(/[^\w\s]/g, ' ')         // Remove punctuation
        .split(/\s+/)                     // Split on whitespace
        .filter(t => t.length > 1);       // Remove single chars

    // Weight by word length (longer = more important)
    const weights = tokens.map(t => Math.min(1, t.length / 10));
}
```

**What it does:** Takes raw text and breaks it into processable units.

**How it weighs:** Longer words get higher weights (maxes at 1.0 for 10+ character words). The intuition: "causality" (9 chars → 0.9 weight) carries more semantic content than "is" (2 chars → 0.2 weight).

#### Step 2: Token to Embedding (lines 616-656)

```typescript
_getTokenEmbedding(token: string): Float32Array {
    // Check cache first
    const cached = this._vocabulary.get(token);
    if (cached) return cached;

    // Generate deterministic embedding from hash
    const embedding = this._hashToEmbedding(token);
    this._vocabulary.set(token, embedding);
    return embedding;
}

_hashToEmbedding(str: string): Float32Array {
    // Hash the string to a seed
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
        const char = str.charCodeAt(i);
        hash = ((hash << 5) - hash) + char;
        hash = hash & hash;
    }

    // Use hash as PRNG seed → 1536D Gaussian vector
    const rng = mulberry32(hash);
    const embedding = new Float32Array(1536);
    for (let i = 0; i < 1536; i++) {
        embedding[i] = randomGaussian(rng);  // Box-Muller transform
    }
    // Normalize to unit sphere
}
```

**What it does:** Converts each token to a unique 1536-dimensional vector.

**Key insight:** The same token *always* produces the same embedding (deterministic via hash). This is critical for reproducibility.

#### Step 3: Weighted Aggregation (lines 565-590)

```typescript
_textToEmbedding(text: string): Float32Array {
    const tokens = this._tokenize(text);
    const embedding = new Float32Array(1536);

    for (const token of tokens.tokens) {
        const tokenEmb = this._getTokenEmbedding(token);
        const weight = tokens.weights[count];

        // Accumulate weighted embeddings
        for (let i = 0; i < 1536; i++) {
            embedding[i] += tokenEmb[i] * weight;
        }
    }

    // Normalize to unit sphere
    const norm = this._embeddingNorm(embedding);
    for (let i = 0; i < embedding.length; i++) {
        embedding[i] /= norm;
    }
}
```

**What it does:** Combines token embeddings into a single document embedding.

**The math:** `E_doc = normalize(Σ(w_i × E_token_i))`

#### Step 4: Random Projection (lines 422-468)

```typescript
_createProjectionMatrix(outputDim: number): Float32Array {
    const matrix = new Float32Array(inputDim * outputDim);
    const scale = 1 / Math.sqrt(outputDim);

    for (let i = 0; i < matrix.length; i++) {
        matrix[i] = randomGaussian(this._rng) * scale;
    }
}

_projectTo4D(embedding: Float32Array): Vector4D {
    const result: Vector4D = [0, 0, 0, 0];

    for (let j = 0; j < 4; j++) {
        let sum = 0;
        for (let i = 0; i < 1536; i++) {
            sum += embedding[i] * this._projectionMatrix[i * 4 + j];
        }
        result[j] = sum;
    }
    return result;
}
```

**What it does:** Projects 1536D → 4D using Johnson-Lindenstrauss random projection.

**Why it works:** The JL lemma guarantees that random projections approximately preserve distances between points. Two similar embeddings in 1536D will remain similar in 4D.

#### Step 5: Concept Activation (lines 473-501)

```typescript
_computeConceptActivations(embedding: Float32Array) {
    // Distance to each of 24 archetypes
    const distances = archetypeEmbeddings.map(arch =>
        euclideanDistance(embedding, arch)
    );

    // Convert to similarities (inverse distance)
    const similarities = distances.map(d => 1 / (d + ε));

    // Softmax with temperature
    const maxSim = Math.max(...similarities);
    const expSims = similarities.map(s =>
        Math.exp((s - maxSim) / temperature)
    );
    const weights = expSims.map(e => e / sumExp);

    // Return top 5 activations
    return weights
        .map((weight, index) => ({ index, weight }))
        .sort((a, b) => b.weight - a.weight)
        .slice(0, 5);
}
```

**What it does:** Determines which of the 24 concept archetypes the input is closest to.

**Temperature effect:**
- Low temperature (0.1): Sharp distribution, winner-take-all
- High temperature (2.0): Soft distribution, many concepts activated
- Default (1.0): Balanced

---

## 3. Core Components Explained

### 3.1 Mulberry32 PRNG (lines 157-164)

```typescript
function mulberry32(seed: number): () => number {
    return function() {
        let t = seed += 0x6D2B79F5;
        t = Math.imul(t ^ t >>> 15, t | 1);
        t ^= t + Math.imul(t ^ t >>> 7, t | 61);
        return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
}
```

**Purpose:** Deterministic random number generation for reproducible projections.

**Why Mulberry32:** Fast, compact, good statistical properties, and JavaScript's `Math.random()` isn't seedable.

### 3.2 Box-Muller Transform (lines 169-173)

```typescript
function randomGaussian(rng: () => number): number {
    const u1 = rng();
    const u2 = rng();
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}
```

**Purpose:** Convert uniform random [0,1] → Gaussian N(0,1).

**Why Gaussian:** JL lemma requires Gaussian random matrices for optimal distance preservation.

### 3.3 Force Output Structure (lines 407-419)

```typescript
const force: Force = {
    linear: [x, y, z, w],      // 4D position force
    rotational: [xy, xz, xw, yz, yw, zw],  // 6D bivector
    magnitude: ‖linear‖,
    source: 'hdc_encoder'
};
```

**Linear component:** Pushes the cognitive state in a direction.

**Rotational component:** Applies torque, causing rotation in specific planes.

---

## 4. Mathematical Foundations

### 4.1 Johnson-Lindenstrauss Lemma

**Statement:** For any ε ∈ (0,1) and n points in ℝᵈ, there exists a linear map f: ℝᵈ → ℝᵏ where k = O(log(n)/ε²) such that:

```
(1-ε)‖u-v‖² ≤ ‖f(u)-f(v)‖² ≤ (1+ε)‖u-v‖²
```

**Practical meaning:** Random projection from 1536D to 4D preserves relative distances (with some distortion). Similar concepts stay similar; different concepts stay different.

**Implementation:** The projection matrix has entries sampled from N(0, 1/√k) where k is the output dimension.

### 4.2 Softmax Temperature

```
softmax(x_i) = exp(x_i / T) / Σ exp(x_j / T)
```

| Temperature | Effect |
|-------------|--------|
| T → 0 | argmax (one-hot) |
| T = 1 | Standard softmax |
| T → ∞ | Uniform distribution |

### 4.3 TF-Style Weighting

The current implementation uses a simplified term frequency approach:

```
weight(token) = min(1, length(token) / 10)
```

This is a proxy for "semantic importance" but differs from true TF-IDF.

---

## 5. The 24 Concept Archetypes

Each vertex of the 24-Cell represents a fundamental reasoning concept:

| Index | Archetype | Keywords | Geometric Interpretation |
|-------|-----------|----------|-------------------------|
| 0 | **causation** | cause, effect, because | Direct causal relationship |
| 1 | **correlation** | correlate, associate, link | Statistical association |
| 2 | **inference** | infer, conclude, deduce | Drawing conclusions |
| 3 | **deduction** | derive, logical, proof | Top-down reasoning |
| 4 | **induction** | generalize, pattern, observe | Bottom-up reasoning |
| 5 | **abduction** | explain, hypothesis, theory | Best explanation inference |
| 6 | **analogy** | similar, like, compare | Structural mapping |
| 7 | **similarity** | same, alike, resemble | Feature matching |
| 8 | **difference** | differ, unlike, distinct | Discrimination |
| 9 | **contrast** | oppose, versus, against | Opposition |
| 10 | **sequence** | order, series, chain | Temporal ordering |
| 11 | **parallel** | simultaneous, concurrent | Co-occurrence |
| 12 | **hierarchy** | level, rank, order | Vertical structure |
| 13 | **network** | connect, graph, web | Horizontal structure |
| 14 | **boundary** | limit, edge, border | Constraint |
| 15 | **transition** | shift, move, transform | State change |
| 16 | **stability** | constant, steady, fixed | Equilibrium |
| 17 | **change** | vary, alter, modify | Dynamics |
| 18 | **growth** | increase, expand, rise | Positive change |
| 19 | **decay** | decrease, shrink, fall | Negative change |
| 20 | **emergence** | appear, arise, develop | Bottom-up creation |
| 21 | **reduction** | simplify, minimize, compress | Complexity decrease |
| 22 | **integration** | combine, merge, unify | Synthesis |
| 23 | **differentiation** | specialize, divide, branch | Analysis |

### Archetype Geometry

The archetypes map to 24-Cell vertices, which are permutations of (±1, ±1, 0, 0):

```
Vertex 0:  (+1, +1, 0, 0)  →  causation
Vertex 1:  (+1, -1, 0, 0)  →  correlation
Vertex 2:  (-1, +1, 0, 0)  →  inference
...
Vertex 23: (0, 0, -1, -1)  →  differentiation
```

**Geometric insight:** Archetypes that are "opposite" concepts map to antipodal vertices. For example, "growth" and "decay" are geometrically opposite on the 24-Cell.

---

## 6. Code Walkthrough

### Main Entry Points

```typescript
// Simple text → force
const force = encoder.textToForce("reasoning about causality");

// With full metadata
const result = encoder.encodeText("reasoning about causality");
// result.force           → Force object
// result.activatedConcepts → [{index: 0, weight: 0.4}, ...]
// result.confidence      → 0.4 (sharpness of activation)

// From neural embedding (e.g., OpenAI)
const embedding = await openai.createEmbedding("text");
const force = encoder.embeddingToForce(embedding.data[0].embedding);

// Concept lookup
const vertexId = encoder.conceptToVertex("causality");  // → 0
```

### Configuration

```typescript
const encoder = new HDCEncoder({
    inputDimension: 1536,    // Match your embedding model
    seed: 42,                // For reproducibility
    forceMagnitude: 1.0,     // Scale output forces
    rotationalWeight: 0.3,   // How much rotation vs translation
    numArchetypes: 24,       // Match 24-Cell vertices
    temperature: 1.0,        // Softmax sharpness
    normalizeForce: true     // Unit force vectors
});
```

---

## 7. v2.0 Enhancements (Implemented)

The following improvements have been implemented in v2.0:

### 7.1 Real Embedding API Integration ✅

**Solution:** Added async methods with multi-provider support, prioritizing **Gemini** and **Anthropic**.

```typescript
// Google Gemini embeddings (768 dimensions)
const geminiEncoder = createGeminiEncoder(process.env.GOOGLE_API_KEY);
const force = await geminiEncoder.textToForceAsync("reasoning about causality");

// Anthropic-recommended embeddings via Voyage AI (1024 dimensions)
const anthropicEncoder = createAnthropicEncoder(process.env.VOYAGE_API_KEY);
const force = await anthropicEncoder.textToForceAsync("reasoning about causality");

// Generic API encoder (also supports 'openai', 'cohere', 'local', 'custom')
const encoder = createAPIEncoder('gemini', process.env.GOOGLE_API_KEY);
```

**Supported providers (priority order):**
| Provider | Model | Dimensions | Notes |
|----------|-------|------------|-------|
| `gemini` | text-embedding-004 | 768 | Google's latest |
| `anthropic` | voyage-3 | 1024 | Anthropic-recommended (Voyage AI) |
| `voyage` | voyage-3 | 1024 | Direct Voyage AI |
| `openai` | text-embedding-3-small | 1536 | OpenAI |
| `cohere` | embed-english-v3.0 | 1024 | Cohere |
| `local` | - | varies | Local endpoint |
| `custom` | - | varies | Custom endpoint |

**Benefits achieved:**
- True semantic similarity ("dog" ≈ "canine")
- Contextual understanding
- Embedding cache for repeated queries

### 7.2 Positional Encoding ✅

**Solution:** Transformer-style sinusoidal positional encoding.

```typescript
// Enabled by default
const encoder = new HDCEncoder({
    usePositionalEncoding: true,
    positionalDimension: 64
});

// Now "dog bites man" ≠ "man bites dog"
```

**Implementation:**
```
PE(pos, 2i) = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```

### 7.3 Multi-Head Self-Attention ✅

**Solution:** 4-head attention mechanism for intelligent token weighting.

```typescript
const encoder = new HDCEncoder({
    useAttention: true,
    attentionHeads: 4,
    temperature: 1.0
});

// Get attention visualization
const viz = encoder.getAttentionVisualization("the quick brown fox");
// viz.tokens: ["the", "quick", "brown", "fox"]
// viz.contributions: [0.15, 0.28, 0.29, 0.28]  // "the" weighted down
```

**How it works:**
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

### 7.4 Configurable Domain Archetypes ✅

**Solution:** Custom archetype definitions + preset domain packages.

```typescript
// Use medical domain archetypes
import { MEDICAL_ARCHETYPES, createDomainEncoder } from './HDCEncoder';
const medicalEncoder = createDomainEncoder(MEDICAL_ARCHETYPES);

// Or define your own
const customEncoder = new HDCEncoder({
    customArchetypes: [
        { label: 'my-concept', keywords: ['word1', 'word2'] },
        // ...
    ]
});
```

**Preset domains included:**
- `MEDICAL_ARCHETYPES` (24 concepts: diagnosis, treatment, symptom, etc.)
- `LEGAL_ARCHETYPES` (24 concepts: statute, precedent, contract, etc.)
- `SOFTWARE_ARCHETYPES` (24 concepts: architecture, algorithm, api, etc.)

### 7.5 Improved Tokenization ✅

**Solution:** Subword tokenization + stop word penalties + frequency weighting.

```typescript
// "understanding" → ["understand", "ing"]
// Morphological awareness for better semantic capture

// Stop words automatically down-weighted:
// "the" → 0.3 weight
// "causality" → 0.9 weight
```

---

## 8. Remaining Improvement Opportunities

### 8.A Learned Projection Matrix

**Status:** Not yet implemented

**Future improvement:** Train projection matrix on domain-specific data:

```typescript
this._projectionMatrix = await trainProjection(
    trainingEmbeddings,
    targetForces,
    { epochs: 100, lr: 0.001 }
);
```

### 8.B Dynamic Archetype Discovery

**Status:** Not yet implemented

**Future improvement:** K-means clustering to discover natural concepts:

```typescript
const clusters = kMeans(corpusEmbeddings, k=24);
this._archetypeEmbeddings = clusters.centroids;
```

### 8.C Streaming/Incremental Encoding

**Status:** Not yet implemented

**Future improvement:** Real-time token-by-token encoding for streaming applications.

---

## 9. Integration Guide

### Basic Usage

```typescript
import { HDCEncoder, textToForce } from '@clear-seas/cpe';
import { createEngine } from '@clear-seas/cpe';

// Create engine and encoder
const engine = createEngine();
const encoder = new HDCEncoder();

// Process text input
const userInput = "I think the economic downturn caused unemployment";
const force = encoder.textToForce(userInput);

// Apply to engine
engine.applyForce(force);
engine.update(1/60);  // 60fps tick

// Check result
const state = engine.getState();
console.log('Position:', state.position);
console.log('Coherence:', engine.getCoherence());
```

### With External Embeddings (Recommended)

```typescript
// Get real embedding from API
const response = await openai.embeddings.create({
    model: 'text-embedding-3-small',
    input: userInput
});
const embedding = new Float32Array(response.data[0].embedding);

// Project to 4D
const force = encoder.embeddingToForce(embedding);
engine.applyForce(force);
```

### WebGL Integration

```typescript
import { initializeCPEIntegration } from './CPERendererBridge';

const { engine, encoder, bridge } = initializeCPEIntegration(gl, shaderProgram);

// Text input drives visualization
document.getElementById('input').addEventListener('keyup', (e) => {
    const force = encoder.textToForce(e.target.value);
    engine.applyForce(force);
});
```

---

## Summary

The HDCEncoder serves as the **semantic-to-geometric translator** for the CPE system. It takes human-readable text or neural embeddings and converts them into 4D force vectors that push and rotate the cognitive state within the 24-Cell boundary.

**Strengths:**
- Deterministic and reproducible (seeded PRNG)
- Fast (no external API calls in default mode)
- Geometrically grounded (maps to 24-Cell vertices)
- Configurable (temperature, magnitude, rotation weight)

**Weaknesses:**
- Hash-based embeddings lack semantic understanding
- Simple tokenization misses linguistic nuance
- No word order preservation
- Fixed concept taxonomy

**Best used with:** Real neural embeddings from OpenAI/Anthropic/Cohere APIs for production applications, falling back to hash-based encoding for offline/testing scenarios.

---

*Document generated for the Chronomorphic Polytopal Engine project*
*Clear Seas Solutions LLC - 2026*
