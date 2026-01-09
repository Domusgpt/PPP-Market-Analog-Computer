/**
 * Hyperdimensional Computing Encoder - Neural-Geometric Bridge
 *
 * @package @clear-seas/cpe
 * @version 1.0.0
 * @license MIT
 * @author Clear Seas Solutions LLC
 *
 * This module bridges semantic input (text, embeddings) to 4D force vectors
 * that drive the Chronomorphic Polytopal Engine. It implements a hyperdimensional
 * computing approach where:
 *
 * Architecture:
 *   Text Input → Tokenizer → HDC Encoder → 4D Force Vector → CPE
 *
 * Key Insight:
 *   The 24 vertices of the 24-Cell represent 24 "concept archetypes".
 *   The encoder learns to map semantic content to combinations of these
 *   basis concepts, creating a grounded geometric representation.
 *
 * Encoding Principles:
 * - High-dimensional sparse vectors preserve semantic similarity
 * - Dimensionality reduction via random projection to 4D
 * - Force magnitude = semantic intensity (importance/confidence)
 * - Force direction = concept blend (weighted combination of archetypes)
 *
 * Integration Points:
 * - Compatible with OpenAI/Anthropic embeddings (1536/4096 dimensions)
 * - Can ingest from existing PPP data channels
 * - Produces Force objects for direct engine consumption
 *
 * References:
 * - Kanerva "Hyperdimensional Computing" (2009)
 * - Gärdenfors "Conceptual Spaces" (2000)
 * - Johnson-Lindenstrauss lemma for random projection
 */

import {
    Vector4D,
    Bivector4D,
    Force,
    MATH_CONSTANTS
} from '../../types/index.js';

import {
    normalize,
    magnitude,
    dot
} from '../math/GeometricAlgebra.js';

import {
    Lattice24,
    getDefaultLattice
} from '../topology/Lattice24.js';

// =============================================================================
// TYPE DEFINITIONS
// =============================================================================

/**
 * Configuration for the HDC encoder.
 */
export interface HDCEncoderConfig {
    /** Input embedding dimension (e.g., 1536 for OpenAI, 4096 for large models) */
    readonly inputDimension: number;

    /** Random seed for reproducible projections */
    readonly seed: number;

    /** Base force magnitude multiplier */
    readonly forceMagnitude: number;

    /** Rotational force component weight (0 to 1) */
    readonly rotationalWeight: number;

    /** Number of concept archetypes (defaults to 24 for 24-cell) */
    readonly numArchetypes: number;

    /** Temperature for softmax concept weighting */
    readonly temperature: number;

    /** Whether to normalize output forces */
    readonly normalizeForce: boolean;
}

/**
 * Default encoder configuration.
 */
export const DEFAULT_HDC_CONFIG: HDCEncoderConfig = {
    inputDimension: 1536, // OpenAI embedding dimension
    seed: 42,
    forceMagnitude: 1.0,
    rotationalWeight: 0.3,
    numArchetypes: 24,
    temperature: 1.0,
    normalizeForce: true
} as const;

/**
 * Concept archetype mapping.
 */
export interface ConceptArchetype {
    /** Archetype index (0-23 for 24-cell) */
    readonly index: number;

    /** Human-readable label */
    readonly label: string;

    /** Associated lattice vertex */
    readonly vertexId: number;

    /** Prototype embedding (centroid of concept cluster) */
    readonly prototype: Float32Array;

    /** Keywords associated with this concept */
    readonly keywords: string[];
}

/**
 * Encoding result with metadata.
 */
export interface EncodingResult {
    /** Output force vector */
    readonly force: Force;

    /** Activated concept archetypes */
    readonly activatedConcepts: { index: number; weight: number }[];

    /** Input magnitude (embedding norm) */
    readonly inputMagnitude: number;

    /** Encoding confidence (0-1) */
    readonly confidence: number;
}

/**
 * Simple tokenization result.
 */
export interface TokenizationResult {
    /** Token strings */
    readonly tokens: string[];

    /** Token weights (TF-IDF style) */
    readonly weights: number[];

    /** Original text length */
    readonly originalLength: number;
}

// =============================================================================
// SEEDED RANDOM NUMBER GENERATOR
// =============================================================================

/**
 * Mulberry32 PRNG for reproducible random projections.
 */
function mulberry32(seed: number): () => number {
    return function() {
        let t = seed += 0x6D2B79F5;
        t = Math.imul(t ^ t >>> 15, t | 1);
        t ^= t + Math.imul(t ^ t >>> 7, t | 61);
        return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
}

/**
 * Generate a random Gaussian using Box-Muller transform.
 */
function randomGaussian(rng: () => number): number {
    const u1 = rng();
    const u2 = rng();
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

// =============================================================================
// HDC ENCODER CLASS
// =============================================================================

/**
 * Hyperdimensional Computing Encoder for the CPE neural-geometric bridge.
 *
 * Usage:
 * ```typescript
 * const encoder = new HDCEncoder();
 *
 * // From text
 * const force1 = encoder.textToForce("reasoning about causality");
 *
 * // From embedding
 * const embedding = await openai.embeddings.create({ input: "hello" });
 * const force2 = encoder.embeddingToForce(embedding.data[0].embedding);
 *
 * // Apply to engine
 * engine.applyForce(force1);
 * ```
 */
export class HDCEncoder {
    /** Encoder configuration */
    private _config: HDCEncoderConfig;

    /** Reference to the 24-Cell lattice */
    private readonly _lattice: Lattice24;

    /** Random projection matrix (inputDim x 4) */
    private _projectionMatrix: Float32Array;

    /** Rotational projection matrix (inputDim x 6) */
    private _rotationalMatrix: Float32Array;

    /** Concept archetype embeddings (24 x inputDim) */
    private _archetypeEmbeddings: Float32Array[];

    /** Concept archetype metadata */
    private _archetypes: ConceptArchetype[];

    /** Simple vocabulary for text encoding */
    private _vocabulary: Map<string, Float32Array>;

    /** PRNG instance */
    private _rng: () => number;

    constructor(config: Partial<HDCEncoderConfig> = {}, lattice?: Lattice24) {
        this._config = { ...DEFAULT_HDC_CONFIG, ...config };
        this._lattice = lattice ?? getDefaultLattice();
        this._rng = mulberry32(this._config.seed);

        // Initialize projection matrices
        this._projectionMatrix = this._createProjectionMatrix(4);
        this._rotationalMatrix = this._createProjectionMatrix(6);

        // Initialize concept archetypes
        this._archetypes = this._initializeArchetypes();
        this._archetypeEmbeddings = this._initializeArchetypeEmbeddings();

        // Initialize vocabulary
        this._vocabulary = new Map();
        this._initializeVocabulary();
    }

    // =========================================================================
    // CONFIGURATION
    // =========================================================================

    /** Get current configuration */
    get config(): HDCEncoderConfig {
        return this._config;
    }

    /** Update configuration (rebuilds projection matrices) */
    setConfig(config: Partial<HDCEncoderConfig>): void {
        const needsRebuild = config.inputDimension !== undefined &&
            config.inputDimension !== this._config.inputDimension;

        this._config = { ...this._config, ...config };

        if (needsRebuild) {
            this._rng = mulberry32(this._config.seed);
            this._projectionMatrix = this._createProjectionMatrix(4);
            this._rotationalMatrix = this._createProjectionMatrix(6);
            this._archetypeEmbeddings = this._initializeArchetypeEmbeddings();
        }
    }

    /** Get lattice reference */
    get lattice(): Lattice24 {
        return this._lattice;
    }

    /** Get concept archetypes */
    get archetypes(): readonly ConceptArchetype[] {
        return this._archetypes;
    }

    // =========================================================================
    // CORE ENCODING FUNCTIONS
    // =========================================================================

    /**
     * Convert text input to a 4D force vector.
     *
     * @param text - Input text to encode
     * @returns Force object for engine consumption
     */
    textToForce(text: string): Force {
        const result = this.encodeText(text);
        return result.force;
    }

    /**
     * Convert a neural embedding to a 4D force vector.
     *
     * @param embedding - Input embedding (Float32Array or number[])
     * @returns Force object for engine consumption
     */
    embeddingToForce(embedding: Float32Array | number[]): Force {
        const result = this.encodeEmbedding(embedding);
        return result.force;
    }

    /**
     * Map a concept string to its nearest lattice vertex.
     *
     * @param concept - Concept name or description
     * @returns Vertex ID (0-23) of nearest archetype
     */
    conceptToVertex(concept: string): number {
        // Encode the concept
        const embedding = this._textToEmbedding(concept);

        // Find nearest archetype
        let nearestIdx = 0;
        let minDist = Infinity;

        for (let i = 0; i < this._archetypeEmbeddings.length; i++) {
            const dist = this._euclideanDistance(embedding, this._archetypeEmbeddings[i]);
            if (dist < minDist) {
                minDist = dist;
                nearestIdx = i;
            }
        }

        return this._archetypes[nearestIdx].vertexId;
    }

    /**
     * Full text encoding with metadata.
     *
     * @param text - Input text
     * @returns EncodingResult with force and metadata
     */
    encodeText(text: string): EncodingResult {
        const embedding = this._textToEmbedding(text);
        return this._encodeEmbeddingInternal(embedding);
    }

    /**
     * Full embedding encoding with metadata.
     *
     * @param embedding - Input embedding
     * @returns EncodingResult with force and metadata
     */
    encodeEmbedding(embedding: Float32Array | number[]): EncodingResult {
        const arr = embedding instanceof Float32Array
            ? embedding
            : new Float32Array(embedding);

        // Resize if needed
        const resized = this._resizeEmbedding(arr);
        return this._encodeEmbeddingInternal(resized);
    }

    // =========================================================================
    // INTERNAL ENCODING LOGIC
    // =========================================================================

    /**
     * Internal encoding implementation.
     */
    private _encodeEmbeddingInternal(embedding: Float32Array): EncodingResult {
        // 1. Compute input magnitude
        const inputMagnitude = this._embeddingNorm(embedding);

        // 2. Project to 4D (linear component)
        const linear = this._projectTo4D(embedding);

        // 3. Project to 6D bivector (rotational component)
        const rotational = this._projectTo6D(embedding);

        // 4. Compute concept activations
        const activations = this._computeConceptActivations(embedding);

        // 5. Compute confidence based on activation sharpness
        const maxActivation = Math.max(...activations.map(a => a.weight));
        const confidence = maxActivation;

        // 6. Scale by configuration
        const scaledLinear: Vector4D = [
            linear[0] * this._config.forceMagnitude,
            linear[1] * this._config.forceMagnitude,
            linear[2] * this._config.forceMagnitude,
            linear[3] * this._config.forceMagnitude
        ];

        const scaledRotational: Bivector4D = [
            rotational[0] * this._config.forceMagnitude * this._config.rotationalWeight,
            rotational[1] * this._config.forceMagnitude * this._config.rotationalWeight,
            rotational[2] * this._config.forceMagnitude * this._config.rotationalWeight,
            rotational[3] * this._config.forceMagnitude * this._config.rotationalWeight,
            rotational[4] * this._config.forceMagnitude * this._config.rotationalWeight,
            rotational[5] * this._config.forceMagnitude * this._config.rotationalWeight
        ];

        // 7. Optionally normalize
        let finalLinear = scaledLinear;
        if (this._config.normalizeForce) {
            const mag = magnitude(scaledLinear);
            if (mag > MATH_CONSTANTS.EPSILON) {
                finalLinear = [
                    scaledLinear[0] / mag * this._config.forceMagnitude,
                    scaledLinear[1] / mag * this._config.forceMagnitude,
                    scaledLinear[2] / mag * this._config.forceMagnitude,
                    scaledLinear[3] / mag * this._config.forceMagnitude
                ];
            }
        }

        const force: Force = {
            linear: finalLinear,
            rotational: scaledRotational,
            magnitude: magnitude(finalLinear),
            source: 'hdc_encoder'
        };

        return {
            force,
            activatedConcepts: activations,
            inputMagnitude,
            confidence
        };
    }

    /**
     * Create a random projection matrix using Johnson-Lindenstrauss.
     */
    private _createProjectionMatrix(outputDim: number): Float32Array {
        const matrix = new Float32Array(this._config.inputDimension * outputDim);
        const scale = 1 / Math.sqrt(outputDim);

        for (let i = 0; i < matrix.length; i++) {
            matrix[i] = randomGaussian(this._rng) * scale;
        }

        return matrix;
    }

    /**
     * Project embedding to 4D using random projection.
     */
    private _projectTo4D(embedding: Float32Array): Vector4D {
        const result: Vector4D = [0, 0, 0, 0];

        for (let j = 0; j < 4; j++) {
            let sum = 0;
            for (let i = 0; i < this._config.inputDimension; i++) {
                sum += embedding[i] * this._projectionMatrix[i * 4 + j];
            }
            result[j] = sum;
        }

        return result;
    }

    /**
     * Project embedding to 6D bivector using random projection.
     */
    private _projectTo6D(embedding: Float32Array): Bivector4D {
        const result: Bivector4D = [0, 0, 0, 0, 0, 0];

        for (let j = 0; j < 6; j++) {
            let sum = 0;
            for (let i = 0; i < this._config.inputDimension; i++) {
                sum += embedding[i] * this._rotationalMatrix[i * 6 + j];
            }
            result[j] = sum;
        }

        return result;
    }

    /**
     * Compute concept archetype activations.
     */
    private _computeConceptActivations(
        embedding: Float32Array
    ): { index: number; weight: number }[] {
        const distances: number[] = [];

        // Compute distances to each archetype
        for (let i = 0; i < this._archetypeEmbeddings.length; i++) {
            distances.push(this._euclideanDistance(embedding, this._archetypeEmbeddings[i]));
        }

        // Convert to similarities (inverse distance)
        const similarities = distances.map(d => 1 / (d + MATH_CONSTANTS.EPSILON));

        // Apply softmax
        const maxSim = Math.max(...similarities);
        const expSims = similarities.map(s =>
            Math.exp((s - maxSim) / this._config.temperature)
        );
        const sumExp = expSims.reduce((a, b) => a + b, 0);
        const weights = expSims.map(e => e / sumExp);

        // Return sorted by weight
        const activations = weights
            .map((weight, index) => ({ index, weight }))
            .sort((a, b) => b.weight - a.weight)
            .slice(0, 5); // Top 5 activations

        return activations;
    }

    /**
     * Compute Euclidean distance between embeddings.
     */
    private _euclideanDistance(a: Float32Array, b: Float32Array): number {
        let sum = 0;
        const len = Math.min(a.length, b.length);
        for (let i = 0; i < len; i++) {
            const diff = a[i] - b[i];
            sum += diff * diff;
        }
        return Math.sqrt(sum);
    }

    /**
     * Compute embedding norm.
     */
    private _embeddingNorm(embedding: Float32Array): number {
        let sum = 0;
        for (let i = 0; i < embedding.length; i++) {
            sum += embedding[i] * embedding[i];
        }
        return Math.sqrt(sum);
    }

    /**
     * Resize embedding to match expected dimension.
     */
    private _resizeEmbedding(embedding: Float32Array): Float32Array {
        if (embedding.length === this._config.inputDimension) {
            return embedding;
        }

        const resized = new Float32Array(this._config.inputDimension);

        if (embedding.length < this._config.inputDimension) {
            // Pad with zeros
            resized.set(embedding);
        } else {
            // Truncate or average pool
            const ratio = embedding.length / this._config.inputDimension;
            for (let i = 0; i < this._config.inputDimension; i++) {
                const start = Math.floor(i * ratio);
                const end = Math.floor((i + 1) * ratio);
                let sum = 0;
                for (let j = start; j < end; j++) {
                    sum += embedding[j];
                }
                resized[i] = sum / (end - start);
            }
        }

        return resized;
    }

    // =========================================================================
    // TEXT ENCODING
    // =========================================================================

    /**
     * Simple text to embedding conversion.
     * In production, this would call an external embedding API.
     */
    private _textToEmbedding(text: string): Float32Array {
        const tokens = this._tokenize(text);
        const embedding = new Float32Array(this._config.inputDimension);

        // Aggregate token embeddings
        let count = 0;
        for (const token of tokens.tokens) {
            const tokenEmb = this._getTokenEmbedding(token);
            const weight = tokens.weights[count] || 1;

            for (let i = 0; i < this._config.inputDimension; i++) {
                embedding[i] += tokenEmb[i] * weight;
            }
            count++;
        }

        // Normalize
        const norm = this._embeddingNorm(embedding);
        if (norm > MATH_CONSTANTS.EPSILON) {
            for (let i = 0; i < embedding.length; i++) {
                embedding[i] /= norm;
            }
        }

        return embedding;
    }

    /**
     * Simple tokenization.
     */
    private _tokenize(text: string): TokenizationResult {
        // Simple whitespace tokenization with lowercase
        const tokens = text
            .toLowerCase()
            .replace(/[^\w\s]/g, ' ')
            .split(/\s+/)
            .filter(t => t.length > 1);

        // Simple TF weighting (longer words weighted higher)
        const weights = tokens.map(t => Math.min(1, t.length / 10));

        return {
            tokens,
            weights,
            originalLength: text.length
        };
    }

    /**
     * Get embedding for a token.
     */
    private _getTokenEmbedding(token: string): Float32Array {
        // Check cache
        const cached = this._vocabulary.get(token);
        if (cached) {
            return cached;
        }

        // Generate deterministic embedding from token hash
        const embedding = this._hashToEmbedding(token);
        this._vocabulary.set(token, embedding);
        return embedding;
    }

    /**
     * Convert string hash to embedding.
     */
    private _hashToEmbedding(str: string): Float32Array {
        // Simple hash function
        let hash = 0;
        for (let i = 0; i < str.length; i++) {
            const char = str.charCodeAt(i);
            hash = ((hash << 5) - hash) + char;
            hash = hash & hash;
        }

        // Use hash as seed for reproducible embedding
        const rng = mulberry32(hash);
        const embedding = new Float32Array(this._config.inputDimension);

        for (let i = 0; i < this._config.inputDimension; i++) {
            embedding[i] = randomGaussian(rng);
        }

        // Normalize
        const norm = this._embeddingNorm(embedding);
        for (let i = 0; i < embedding.length; i++) {
            embedding[i] /= norm;
        }

        return embedding;
    }

    // =========================================================================
    // ARCHETYPE INITIALIZATION
    // =========================================================================

    /**
     * Initialize the 24 concept archetypes.
     * These map to the 24 vertices of the 24-cell.
     */
    private _initializeArchetypes(): ConceptArchetype[] {
        // Default semantic labels for the 24 archetypes
        // These can be customized for specific domains
        const labels = [
            'causation', 'correlation', 'inference', 'deduction',
            'induction', 'abduction', 'analogy', 'similarity',
            'difference', 'contrast', 'sequence', 'parallel',
            'hierarchy', 'network', 'boundary', 'transition',
            'stability', 'change', 'growth', 'decay',
            'emergence', 'reduction', 'integration', 'differentiation'
        ];

        // Keywords for each archetype (simplified)
        const keywordSets = [
            ['cause', 'effect', 'because', 'therefore'],
            ['correlate', 'associate', 'relate', 'link'],
            ['infer', 'conclude', 'deduce', 'reason'],
            ['deduce', 'derive', 'logical', 'proof'],
            ['induce', 'generalize', 'pattern', 'observe'],
            ['explain', 'hypothesis', 'abduct', 'theory'],
            ['similar', 'like', 'analogous', 'compare'],
            ['same', 'alike', 'resemble', 'match'],
            ['differ', 'unlike', 'distinct', 'separate'],
            ['contrast', 'oppose', 'versus', 'against'],
            ['sequence', 'order', 'series', 'chain'],
            ['parallel', 'simultaneous', 'concurrent', 'together'],
            ['hierarchy', 'level', 'rank', 'order'],
            ['network', 'connect', 'graph', 'web'],
            ['boundary', 'limit', 'edge', 'border'],
            ['transition', 'change', 'shift', 'move'],
            ['stable', 'constant', 'steady', 'fixed'],
            ['change', 'vary', 'alter', 'modify'],
            ['grow', 'increase', 'expand', 'rise'],
            ['decay', 'decrease', 'shrink', 'fall'],
            ['emerge', 'appear', 'arise', 'develop'],
            ['reduce', 'simplify', 'minimize', 'compress'],
            ['integrate', 'combine', 'merge', 'unify'],
            ['differentiate', 'specialize', 'divide', 'branch']
        ];

        return labels.map((label, index) => ({
            index,
            label,
            vertexId: index % 24, // Map to lattice vertices
            prototype: new Float32Array(0), // Populated later
            keywords: keywordSets[index] || []
        }));
    }

    /**
     * Initialize prototype embeddings for each archetype.
     */
    private _initializeArchetypeEmbeddings(): Float32Array[] {
        const embeddings: Float32Array[] = [];

        for (let i = 0; i < this._config.numArchetypes; i++) {
            // Create embedding from archetype label and keywords
            const archetype = this._archetypes[i];
            const text = [archetype.label, ...archetype.keywords].join(' ');
            const embedding = this._hashToEmbedding(text);

            // Blend with vertex position for geometric grounding
            const vertex = this._lattice.getVertex(i % 24);
            if (vertex) {
                // Project 4D vertex to high-D space and blend
                for (let j = 0; j < 4; j++) {
                    embedding[j] = embedding[j] * 0.7 + vertex.coordinates[j] * 0.3;
                }
            }

            embeddings.push(embedding);
        }

        return embeddings;
    }

    /**
     * Initialize vocabulary with common words.
     */
    private _initializeVocabulary(): void {
        // Pre-compute embeddings for common words
        const commonWords = [
            'the', 'is', 'are', 'was', 'were', 'be', 'been',
            'have', 'has', 'had', 'do', 'does', 'did',
            'and', 'or', 'but', 'if', 'then', 'else',
            'what', 'when', 'where', 'why', 'how', 'who',
            'this', 'that', 'these', 'those',
            'cause', 'effect', 'reason', 'result',
            'think', 'know', 'believe', 'understand',
            'true', 'false', 'yes', 'no', 'maybe'
        ];

        for (const word of commonWords) {
            this._vocabulary.set(word, this._hashToEmbedding(word));
        }
    }

    // =========================================================================
    // UTILITY METHODS
    // =========================================================================

    /**
     * Get encoder statistics.
     */
    getStats(): Record<string, unknown> {
        return {
            inputDimension: this._config.inputDimension,
            numArchetypes: this._config.numArchetypes,
            vocabularySize: this._vocabulary.size,
            forceMagnitude: this._config.forceMagnitude,
            rotationalWeight: this._config.rotationalWeight,
            temperature: this._config.temperature
        };
    }

    /**
     * Get archetype by index.
     */
    getArchetype(index: number): ConceptArchetype | undefined {
        return this._archetypes[index];
    }

    /**
     * Find archetypes matching keywords.
     */
    findArchetypesByKeyword(keyword: string): ConceptArchetype[] {
        const lower = keyword.toLowerCase();
        return this._archetypes.filter(a =>
            a.label.includes(lower) ||
            a.keywords.some(k => k.includes(lower))
        );
    }

    /**
     * Clear vocabulary cache.
     */
    clearVocabulary(): void {
        this._vocabulary.clear();
        this._initializeVocabulary();
    }
}

// =============================================================================
// FACTORY FUNCTIONS
// =============================================================================

/** Singleton instance */
let _defaultEncoder: HDCEncoder | null = null;

/**
 * Get or create the default encoder instance.
 */
export function getDefaultEncoder(): HDCEncoder {
    if (!_defaultEncoder) {
        _defaultEncoder = new HDCEncoder();
    }
    return _defaultEncoder;
}

/**
 * Create a new encoder instance.
 */
export function createEncoder(config?: Partial<HDCEncoderConfig>): HDCEncoder {
    return new HDCEncoder(config);
}

// =============================================================================
// STANDALONE FUNCTIONS
// =============================================================================

/**
 * Quick text to force conversion.
 */
export function textToForce(text: string): Force {
    return getDefaultEncoder().textToForce(text);
}

/**
 * Quick embedding to force conversion.
 */
export function embeddingToForce(embedding: Float32Array | number[]): Force {
    return getDefaultEncoder().embeddingToForce(embedding);
}

/**
 * Quick concept to vertex mapping.
 */
export function conceptToVertex(concept: string): number {
    return getDefaultEncoder().conceptToVertex(concept);
}
