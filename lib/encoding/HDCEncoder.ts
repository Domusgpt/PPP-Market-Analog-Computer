/**
 * Hyperdimensional Computing Encoder - Neural-Geometric Bridge
 *
 * @package @clear-seas/cpe
 * @version 2.0.0
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
 * v2.0 Enhancements:
 * - Real embedding API integration (OpenAI/Anthropic/Cohere compatible)
 * - Positional encoding for word order preservation
 * - Self-attention based token aggregation
 * - Configurable/dynamic concept archetypes
 * - Improved tokenization with subword support
 *
 * References:
 * - Kanerva "Hyperdimensional Computing" (2009)
 * - Gärdenfors "Conceptual Spaces" (2000)
 * - Johnson-Lindenstrauss lemma for random projection
 * - Vaswani et al. "Attention Is All You Need" (2017)
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
 * Embedding provider types supported out of the box.
 */
export type EmbeddingProvider = 'gemini' | 'anthropic' | 'openai' | 'cohere' | 'voyage' | 'local' | 'custom';

/**
 * Configuration for external embedding API.
 */
export interface EmbeddingAPIConfig {
    /** API provider */
    readonly provider: EmbeddingProvider;

    /** API endpoint URL */
    readonly endpoint?: string;

    /** API key (or use environment variable) */
    readonly apiKey?: string;

    /** Model identifier */
    readonly model?: string;

    /** Request timeout in milliseconds */
    readonly timeout?: number;

    /** Custom fetch function for environments without native fetch */
    readonly fetchFn?: typeof fetch;
}

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

    /** Enable positional encoding for word order */
    readonly usePositionalEncoding: boolean;

    /** Positional encoding dimension */
    readonly positionalDimension: number;

    /** Enable self-attention aggregation */
    readonly useAttention: boolean;

    /** Number of attention heads */
    readonly attentionHeads: number;

    /** External embedding API configuration */
    readonly embeddingAPI?: EmbeddingAPIConfig;

    /** Custom archetype definitions */
    readonly customArchetypes?: ArchetypeDefinition[];
}

/**
 * Custom archetype definition for domain-specific concepts.
 */
export interface ArchetypeDefinition {
    /** Human-readable label */
    readonly label: string;

    /** Keywords associated with this concept */
    readonly keywords: string[];

    /** Optional prototype embedding (will be computed if not provided) */
    readonly prototype?: number[];
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
    normalizeForce: true,
    usePositionalEncoding: true,
    positionalDimension: 64,
    useAttention: true,
    attentionHeads: 4
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

    /** Attention weights (if attention enabled) */
    readonly attentionWeights?: number[][];

    /** Token contributions (if attention enabled) */
    readonly tokenContributions?: { token: string; weight: number }[];
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

    /** Subword tokens (if subword tokenization enabled) */
    readonly subwords?: string[];
}

/**
 * Embedding API response.
 */
export interface EmbeddingResponse {
    /** The embedding vector */
    readonly embedding: number[];

    /** Token count used */
    readonly tokenCount?: number;

    /** Model used */
    readonly model?: string;
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
    return Math.sqrt(-2 * Math.log(u1 + 1e-10)) * Math.cos(2 * Math.PI * u2);
}

// =============================================================================
// POSITIONAL ENCODING
// =============================================================================

/**
 * Generate sinusoidal positional encoding (Transformer-style).
 *
 * PE(pos, 2i) = sin(pos / 10000^(2i/d))
 * PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
 */
function generatePositionalEncoding(
    position: number,
    dimension: number,
    maxLength: number = 512
): Float32Array {
    const encoding = new Float32Array(dimension);

    for (let i = 0; i < dimension / 2; i++) {
        const frequency = 1 / Math.pow(10000, (2 * i) / dimension);
        encoding[2 * i] = Math.sin(position * frequency);
        encoding[2 * i + 1] = Math.cos(position * frequency);
    }

    return encoding;
}

/**
 * Apply positional encoding to an embedding.
 */
function applyPositionalEncoding(
    embedding: Float32Array,
    position: number,
    positionalDim: number
): Float32Array {
    const posEnc = generatePositionalEncoding(position, positionalDim);
    const result = new Float32Array(embedding.length);

    // Copy original embedding
    result.set(embedding);

    // Add positional encoding to first positionalDim dimensions
    const applyDim = Math.min(positionalDim, embedding.length);
    for (let i = 0; i < applyDim; i++) {
        result[i] += posEnc[i] * 0.1; // Scale down positional contribution
    }

    return result;
}

// =============================================================================
// SELF-ATTENTION MECHANISM
// =============================================================================

/**
 * Compute scaled dot-product attention.
 *
 * Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
 */
function scaledDotProductAttention(
    queries: Float32Array[],
    keys: Float32Array[],
    values: Float32Array[],
    temperature: number = 1.0
): { output: Float32Array[]; weights: number[][] } {
    const n = queries.length;
    const dk = queries[0]?.length || 1;
    const scale = Math.sqrt(dk);

    // Compute attention scores
    const scores: number[][] = [];
    for (let i = 0; i < n; i++) {
        scores[i] = [];
        for (let j = 0; j < n; j++) {
            let score = 0;
            for (let k = 0; k < dk; k++) {
                score += queries[i][k] * keys[j][k];
            }
            scores[i][j] = score / scale / temperature;
        }
    }

    // Apply softmax row-wise
    const weights: number[][] = [];
    for (let i = 0; i < n; i++) {
        const maxScore = Math.max(...scores[i]);
        const expScores = scores[i].map(s => Math.exp(s - maxScore));
        const sumExp = expScores.reduce((a, b) => a + b, 0);
        weights[i] = expScores.map(e => e / sumExp);
    }

    // Compute weighted values
    const output: Float32Array[] = [];
    const valueDim = values[0]?.length || 0;

    for (let i = 0; i < n; i++) {
        const out = new Float32Array(valueDim);
        for (let j = 0; j < n; j++) {
            for (let k = 0; k < valueDim; k++) {
                out[k] += weights[i][j] * values[j][k];
            }
        }
        output.push(out);
    }

    return { output, weights };
}

/**
 * Multi-head attention aggregation.
 */
function multiHeadAttention(
    embeddings: Float32Array[],
    numHeads: number,
    temperature: number
): { aggregated: Float32Array; weights: number[][]; contributions: number[] } {
    if (embeddings.length === 0) {
        return {
            aggregated: new Float32Array(0),
            weights: [],
            contributions: []
        };
    }

    const dim = embeddings[0].length;
    const headDim = Math.floor(dim / numHeads);

    // For simplicity, use same embeddings for Q, K, V
    // In full implementation, these would be learned projections
    const allHeadOutputs: Float32Array[][] = [];
    const allWeights: number[][][] = [];

    for (let h = 0; h < numHeads; h++) {
        const start = h * headDim;
        const end = start + headDim;

        // Extract head-specific slices
        const queries = embeddings.map(e => e.slice(start, end) as unknown as Float32Array);
        const keys = embeddings.map(e => e.slice(start, end) as unknown as Float32Array);
        const values = embeddings.map(e => e.slice(start, end) as unknown as Float32Array);

        // Convert slices to Float32Array
        const q = queries.map(q => new Float32Array(q));
        const k = keys.map(k => new Float32Array(k));
        const v = values.map(v => new Float32Array(v));

        const { output, weights } = scaledDotProductAttention(q, k, v, temperature);
        allHeadOutputs.push(output);
        allWeights.push(weights);
    }

    // Average attention weights across heads
    const avgWeights: number[][] = [];
    for (let i = 0; i < embeddings.length; i++) {
        avgWeights[i] = [];
        for (let j = 0; j < embeddings.length; j++) {
            let sum = 0;
            for (let h = 0; h < numHeads; h++) {
                sum += allWeights[h][i][j];
            }
            avgWeights[i][j] = sum / numHeads;
        }
    }

    // Compute token contributions (sum of incoming attention weights)
    const contributions = new Array(embeddings.length).fill(0);
    for (let j = 0; j < embeddings.length; j++) {
        for (let i = 0; i < embeddings.length; i++) {
            contributions[j] += avgWeights[i][j];
        }
        contributions[j] /= embeddings.length;
    }

    // Aggregate using attention-weighted combination
    const aggregated = new Float32Array(dim);
    for (let i = 0; i < embeddings.length; i++) {
        const weight = contributions[i];
        for (let j = 0; j < dim; j++) {
            aggregated[j] += embeddings[i][j] * weight;
        }
    }

    // Normalize
    let norm = 0;
    for (let i = 0; i < dim; i++) {
        norm += aggregated[i] * aggregated[i];
    }
    norm = Math.sqrt(norm);
    if (norm > MATH_CONSTANTS.EPSILON) {
        for (let i = 0; i < dim; i++) {
            aggregated[i] /= norm;
        }
    }

    return { aggregated, weights: avgWeights, contributions };
}

// =============================================================================
// SUBWORD TOKENIZATION
// =============================================================================

/**
 * Simple BPE-style subword tokenization.
 * This is a simplified version - production would use a trained vocabulary.
 */
function subwordTokenize(word: string): string[] {
    // Common prefixes
    const prefixes = ['un', 're', 'pre', 'dis', 'mis', 'non', 'over', 'under', 'out', 'sub'];
    // Common suffixes
    const suffixes = ['ing', 'ed', 'er', 'est', 'ly', 'tion', 'sion', 'ness', 'ment', 'able', 'ible', 'ful', 'less', 'ous', 'ive', 'al', 'ial'];

    const subwords: string[] = [];
    let remaining = word.toLowerCase();

    // Extract prefix
    for (const prefix of prefixes) {
        if (remaining.startsWith(prefix) && remaining.length > prefix.length + 2) {
            subwords.push(prefix);
            remaining = remaining.slice(prefix.length);
            break;
        }
    }

    // Extract suffix
    let suffix = '';
    for (const suf of suffixes) {
        if (remaining.endsWith(suf) && remaining.length > suf.length + 2) {
            suffix = suf;
            remaining = remaining.slice(0, -suf.length);
            break;
        }
    }

    // Add root
    if (remaining.length > 0) {
        subwords.push(remaining);
    }

    // Add suffix
    if (suffix) {
        subwords.push(suffix);
    }

    return subwords.length > 0 ? subwords : [word];
}

// =============================================================================
// EMBEDDING API CLIENT
// =============================================================================

/**
 * Fetch embedding from external API.
 */
async function fetchEmbedding(
    text: string,
    config: EmbeddingAPIConfig
): Promise<EmbeddingResponse> {
    const fetchFn = config.fetchFn || fetch;

    let endpoint: string;
    let headers: Record<string, string>;
    let body: string;

    switch (config.provider) {
        case 'gemini':
            // Google Gemini/Vertex AI Embeddings
            // Model: text-embedding-004 (768 dims) or text-embedding-005
            const geminiModel = config.model || 'text-embedding-004';
            endpoint = config.endpoint ||
                `https://generativelanguage.googleapis.com/v1beta/models/${geminiModel}:embedContent?key=${config.apiKey}`;
            headers = {
                'Content-Type': 'application/json'
            };
            body = JSON.stringify({
                model: `models/${geminiModel}`,
                content: {
                    parts: [{ text }]
                },
                taskType: 'RETRIEVAL_DOCUMENT'
            });
            break;

        case 'anthropic':
            // Anthropic recommends Voyage AI for embeddings
            // Using Voyage AI as the Anthropic-recommended solution
            endpoint = config.endpoint || 'https://api.voyageai.com/v1/embeddings';
            headers = {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${config.apiKey}`
            };
            body = JSON.stringify({
                model: config.model || 'voyage-3',  // Anthropic-recommended model
                input: text,
                input_type: 'document'
            });
            break;

        case 'voyage':
            // Voyage AI (standalone, also used by Anthropic)
            endpoint = config.endpoint || 'https://api.voyageai.com/v1/embeddings';
            headers = {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${config.apiKey}`
            };
            body = JSON.stringify({
                model: config.model || 'voyage-3',
                input: text,
                input_type: 'document'
            });
            break;

        case 'openai':
            endpoint = config.endpoint || 'https://api.openai.com/v1/embeddings';
            headers = {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${config.apiKey}`
            };
            body = JSON.stringify({
                model: config.model || 'text-embedding-3-small',
                input: text
            });
            break;

        case 'cohere':
            endpoint = config.endpoint || 'https://api.cohere.ai/v1/embed';
            headers = {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${config.apiKey}`
            };
            body = JSON.stringify({
                model: config.model || 'embed-english-v3.0',
                texts: [text],
                input_type: 'search_document'
            });
            break;

        case 'local':
            endpoint = config.endpoint || 'http://localhost:8080/embed';
            headers = { 'Content-Type': 'application/json' };
            body = JSON.stringify({ text });
            break;

        case 'custom':
            if (!config.endpoint) {
                throw new Error('Custom provider requires endpoint');
            }
            endpoint = config.endpoint;
            headers = {
                'Content-Type': 'application/json',
                ...(config.apiKey ? { 'Authorization': `Bearer ${config.apiKey}` } : {})
            };
            body = JSON.stringify({ text, model: config.model });
            break;

        default:
            throw new Error(`Unknown embedding provider: ${config.provider}`);
    }

    const controller = new AbortController();
    const timeout = setTimeout(
        () => controller.abort(),
        config.timeout || 30000
    );

    try {
        const response = await fetchFn(endpoint, {
            method: 'POST',
            headers,
            body,
            signal: controller.signal
        });

        if (!response.ok) {
            throw new Error(`Embedding API error: ${response.status} ${response.statusText}`);
        }

        const data = await response.json();

        // Parse response based on provider
        let embedding: number[];
        let tokenCount: number | undefined;

        switch (config.provider) {
            case 'gemini':
                embedding = data.embedding?.values || data.embedding;
                break;
            case 'anthropic':
            case 'voyage':
                embedding = data.data[0].embedding;
                tokenCount = data.usage?.total_tokens;
                break;
            case 'openai':
                embedding = data.data[0].embedding;
                tokenCount = data.usage?.total_tokens;
                break;
            case 'cohere':
                embedding = data.embeddings[0];
                break;
            default:
                embedding = data.embedding || data.embeddings?.[0] || data;
        }

        return {
            embedding,
            tokenCount,
            model: config.model
        };
    } finally {
        clearTimeout(timeout);
    }
}

// =============================================================================
// HDC ENCODER CLASS
// =============================================================================

/**
 * Hyperdimensional Computing Encoder for the CPE neural-geometric bridge.
 *
 * v2.0 Features:
 * - Real embedding API integration
 * - Positional encoding for word order
 * - Self-attention aggregation
 * - Configurable archetypes
 *
 * Usage:
 * ```typescript
 * // Basic usage (hash-based, fast, offline)
 * const encoder = new HDCEncoder();
 * const force = encoder.textToForce("reasoning about causality");
 *
 * // With real embeddings (semantic, requires API)
 * const encoder = new HDCEncoder({
 *     embeddingAPI: {
 *         provider: 'openai',
 *         apiKey: process.env.OPENAI_API_KEY
 *     }
 * });
 * const force = await encoder.textToForceAsync("reasoning about causality");
 *
 * // Custom archetypes for domain-specific use
 * const medicalEncoder = new HDCEncoder({
 *     customArchetypes: [
 *         { label: 'diagnosis', keywords: ['diagnose', 'condition', 'disease'] },
 *         { label: 'treatment', keywords: ['treat', 'therapy', 'medication'] },
 *         // ... more domain concepts
 *     ]
 * });
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

    /** Embedding cache for API responses */
    private _embeddingCache: Map<string, Float32Array>;

    /** PRNG instance */
    private _rng: () => number;

    constructor(config: Partial<HDCEncoderConfig> = {}, lattice?: Lattice24) {
        this._config = { ...DEFAULT_HDC_CONFIG, ...config };
        this._lattice = lattice ?? getDefaultLattice();
        this._rng = mulberry32(this._config.seed);

        // Initialize projection matrices
        this._projectionMatrix = this._createProjectionMatrix(4);
        this._rotationalMatrix = this._createProjectionMatrix(6);

        // Initialize concept archetypes (custom or default)
        this._archetypes = this._initializeArchetypes();
        this._archetypeEmbeddings = this._initializeArchetypeEmbeddings();

        // Initialize vocabulary and cache
        this._vocabulary = new Map();
        this._embeddingCache = new Map();
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

        const needsArchetypeRebuild = config.customArchetypes !== undefined;

        this._config = { ...this._config, ...config };

        if (needsRebuild) {
            this._rng = mulberry32(this._config.seed);
            this._projectionMatrix = this._createProjectionMatrix(4);
            this._rotationalMatrix = this._createProjectionMatrix(6);
            this._archetypeEmbeddings = this._initializeArchetypeEmbeddings();
        }

        if (needsArchetypeRebuild) {
            this._archetypes = this._initializeArchetypes();
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

    /** Check if API embeddings are configured */
    get hasAPIEmbeddings(): boolean {
        return this._config.embeddingAPI !== undefined;
    }

    // =========================================================================
    // CORE ENCODING FUNCTIONS (SYNCHRONOUS - Hash-based)
    // =========================================================================

    /**
     * Convert text input to a 4D force vector (synchronous, hash-based).
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
     * Full text encoding with metadata (synchronous).
     *
     * @param text - Input text
     * @returns EncodingResult with force and metadata
     */
    encodeText(text: string): EncodingResult {
        const tokens = this._tokenize(text);
        const tokenEmbeddings: Float32Array[] = [];

        // Get embeddings for each token
        for (let i = 0; i < tokens.tokens.length; i++) {
            let embedding = this._getTokenEmbedding(tokens.tokens[i]);

            // Apply positional encoding if enabled
            if (this._config.usePositionalEncoding) {
                embedding = applyPositionalEncoding(
                    embedding,
                    i,
                    this._config.positionalDimension
                );
            }

            tokenEmbeddings.push(embedding);
        }

        // Aggregate embeddings
        let finalEmbedding: Float32Array;
        let attentionWeights: number[][] | undefined;
        let tokenContributions: { token: string; weight: number }[] | undefined;

        if (this._config.useAttention && tokenEmbeddings.length > 1) {
            const attention = multiHeadAttention(
                tokenEmbeddings,
                this._config.attentionHeads,
                this._config.temperature
            );
            finalEmbedding = attention.aggregated;
            attentionWeights = attention.weights;
            tokenContributions = tokens.tokens.map((token, i) => ({
                token,
                weight: attention.contributions[i]
            }));
        } else {
            // Fallback to weighted sum
            finalEmbedding = this._weightedSum(tokenEmbeddings, tokens.weights);
        }

        return this._encodeEmbeddingInternal(finalEmbedding, attentionWeights, tokenContributions);
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
    // ASYNC ENCODING FUNCTIONS (API-based)
    // =========================================================================

    /**
     * Convert text to force using external embedding API (async).
     *
     * @param text - Input text
     * @returns Promise resolving to Force object
     */
    async textToForceAsync(text: string): Promise<Force> {
        const result = await this.encodeTextAsync(text);
        return result.force;
    }

    /**
     * Full text encoding with real embeddings (async).
     *
     * @param text - Input text
     * @returns Promise resolving to EncodingResult
     */
    async encodeTextAsync(text: string): Promise<EncodingResult> {
        if (!this._config.embeddingAPI) {
            // Fall back to synchronous hash-based encoding
            return this.encodeText(text);
        }

        // Check cache
        const cacheKey = `${this._config.embeddingAPI.provider}:${text}`;
        let embedding = this._embeddingCache.get(cacheKey);

        if (!embedding) {
            // Fetch from API
            const response = await fetchEmbedding(text, this._config.embeddingAPI);
            embedding = new Float32Array(response.embedding);

            // Resize if needed
            embedding = this._resizeEmbedding(embedding);

            // Cache for future use
            this._embeddingCache.set(cacheKey, embedding);
        }

        return this._encodeEmbeddingInternal(embedding);
    }

    /**
     * Batch encode multiple texts (async, optimized for API calls).
     *
     * @param texts - Array of input texts
     * @returns Promise resolving to array of EncodingResults
     */
    async encodeTextBatchAsync(texts: string[]): Promise<EncodingResult[]> {
        // For now, encode sequentially
        // Future optimization: batch API calls where supported
        const results: EncodingResult[] = [];
        for (const text of texts) {
            results.push(await this.encodeTextAsync(text));
        }
        return results;
    }

    // =========================================================================
    // INTERNAL ENCODING LOGIC
    // =========================================================================

    /**
     * Internal encoding implementation.
     */
    private _encodeEmbeddingInternal(
        embedding: Float32Array,
        attentionWeights?: number[][],
        tokenContributions?: { token: string; weight: number }[]
    ): EncodingResult {
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
            confidence,
            attentionWeights,
            tokenContributions
        };
    }

    /**
     * Weighted sum aggregation (fallback when attention disabled).
     */
    private _weightedSum(embeddings: Float32Array[], weights: number[]): Float32Array {
        if (embeddings.length === 0) {
            return new Float32Array(this._config.inputDimension);
        }

        const dim = embeddings[0].length;
        const result = new Float32Array(dim);

        for (let i = 0; i < embeddings.length; i++) {
            const weight = weights[i] || 1;
            for (let j = 0; j < dim; j++) {
                result[j] += embeddings[i][j] * weight;
            }
        }

        // Normalize
        const norm = this._embeddingNorm(result);
        if (norm > MATH_CONSTANTS.EPSILON) {
            for (let i = 0; i < dim; i++) {
                result[i] /= norm;
            }
        }

        return result;
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
     * Text to embedding conversion (hash-based, synchronous).
     */
    private _textToEmbedding(text: string): Float32Array {
        const tokens = this._tokenize(text);
        const tokenEmbeddings: Float32Array[] = [];

        // Get embeddings for each token with positional encoding
        for (let i = 0; i < tokens.tokens.length; i++) {
            let embedding = this._getTokenEmbedding(tokens.tokens[i]);

            if (this._config.usePositionalEncoding) {
                embedding = applyPositionalEncoding(
                    embedding,
                    i,
                    this._config.positionalDimension
                );
            }

            tokenEmbeddings.push(embedding);
        }

        // Use attention or weighted sum
        if (this._config.useAttention && tokenEmbeddings.length > 1) {
            return multiHeadAttention(
                tokenEmbeddings,
                this._config.attentionHeads,
                this._config.temperature
            ).aggregated;
        }

        return this._weightedSum(tokenEmbeddings, tokens.weights);
    }

    /**
     * Improved tokenization with subword support.
     */
    private _tokenize(text: string): TokenizationResult {
        // Simple whitespace tokenization with lowercase
        const rawTokens = text
            .toLowerCase()
            .replace(/[^\w\s-]/g, ' ')
            .split(/\s+/)
            .filter(t => t.length > 1);

        const tokens: string[] = [];
        const subwords: string[] = [];

        // Apply subword tokenization for longer words
        for (const token of rawTokens) {
            tokens.push(token);

            if (token.length > 6) {
                const subs = subwordTokenize(token);
                subwords.push(...subs);
            } else {
                subwords.push(token);
            }
        }

        // Improved TF weighting:
        // - Length factor: longer words carry more meaning
        // - Frequency penalty: repeated words weighted down
        // - Stop word penalty: common words weighted down
        const stopWords = new Set(['the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'may', 'might', 'must', 'shall', 'can', 'need', 'dare', 'ought', 'used',
            'a', 'an', 'and', 'or', 'but', 'if', 'then', 'else', 'when', 'at', 'by',
            'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out',
            'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
            'there', 'all', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
            'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
            'just', 'also', 'now', 'of', 'this', 'that', 'these', 'those', 'it', 'its']);

        const tokenCounts = new Map<string, number>();
        for (const t of tokens) {
            tokenCounts.set(t, (tokenCounts.get(t) || 0) + 1);
        }

        const weights = tokens.map(t => {
            let weight = Math.min(1, t.length / 10); // Length factor

            // Stop word penalty
            if (stopWords.has(t)) {
                weight *= 0.3;
            }

            // Frequency penalty (inverse document frequency approximation)
            const count = tokenCounts.get(t) || 1;
            weight *= 1 / Math.sqrt(count);

            return weight;
        });

        return {
            tokens,
            weights,
            originalLength: text.length,
            subwords
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
        // Improved hash function (djb2)
        let hash = 5381;
        for (let i = 0; i < str.length; i++) {
            hash = ((hash << 5) + hash) ^ str.charCodeAt(i);
        }

        // Use hash as seed for reproducible embedding
        const rng = mulberry32(Math.abs(hash));
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
     * Initialize concept archetypes (custom or default).
     */
    private _initializeArchetypes(): ConceptArchetype[] {
        // Use custom archetypes if provided
        if (this._config.customArchetypes && this._config.customArchetypes.length > 0) {
            return this._config.customArchetypes.map((def, index) => ({
                index,
                label: def.label,
                vertexId: index % 24,
                prototype: def.prototype
                    ? new Float32Array(def.prototype)
                    : new Float32Array(0),
                keywords: def.keywords
            }));
        }

        // Default semantic labels for the 24 archetypes
        const labels = [
            'causation', 'correlation', 'inference', 'deduction',
            'induction', 'abduction', 'analogy', 'similarity',
            'difference', 'contrast', 'sequence', 'parallel',
            'hierarchy', 'network', 'boundary', 'transition',
            'stability', 'change', 'growth', 'decay',
            'emergence', 'reduction', 'integration', 'differentiation'
        ];

        // Keywords for each archetype
        const keywordSets = [
            ['cause', 'effect', 'because', 'therefore', 'result', 'consequence'],
            ['correlate', 'associate', 'relate', 'link', 'connection', 'relationship'],
            ['infer', 'conclude', 'deduce', 'reason', 'imply', 'suggest'],
            ['deduce', 'derive', 'logical', 'proof', 'theorem', 'axiom'],
            ['induce', 'generalize', 'pattern', 'observe', 'empirical', 'data'],
            ['explain', 'hypothesis', 'abduct', 'theory', 'model', 'account'],
            ['similar', 'like', 'analogous', 'compare', 'metaphor', 'parallel'],
            ['same', 'alike', 'resemble', 'match', 'equivalent', 'identical'],
            ['differ', 'unlike', 'distinct', 'separate', 'diverge', 'deviate'],
            ['contrast', 'oppose', 'versus', 'against', 'contrary', 'opposite'],
            ['sequence', 'order', 'series', 'chain', 'progression', 'timeline'],
            ['parallel', 'simultaneous', 'concurrent', 'together', 'sync', 'aligned'],
            ['hierarchy', 'level', 'rank', 'order', 'tier', 'structure'],
            ['network', 'connect', 'graph', 'web', 'mesh', 'distributed'],
            ['boundary', 'limit', 'edge', 'border', 'constraint', 'threshold'],
            ['transition', 'change', 'shift', 'move', 'transform', 'evolve'],
            ['stable', 'constant', 'steady', 'fixed', 'equilibrium', 'balanced'],
            ['change', 'vary', 'alter', 'modify', 'adapt', 'mutate'],
            ['grow', 'increase', 'expand', 'rise', 'scale', 'amplify'],
            ['decay', 'decrease', 'shrink', 'fall', 'diminish', 'wane'],
            ['emerge', 'appear', 'arise', 'develop', 'manifest', 'originate'],
            ['reduce', 'simplify', 'minimize', 'compress', 'condense', 'abstract'],
            ['integrate', 'combine', 'merge', 'unify', 'synthesize', 'consolidate'],
            ['differentiate', 'specialize', 'divide', 'branch', 'separate', 'distinguish']
        ];

        return labels.map((label, index) => ({
            index,
            label,
            vertexId: index % 24,
            prototype: new Float32Array(0),
            keywords: keywordSets[index] || []
        }));
    }

    /**
     * Initialize prototype embeddings for each archetype.
     */
    private _initializeArchetypeEmbeddings(): Float32Array[] {
        const embeddings: Float32Array[] = [];

        for (let i = 0; i < this._archetypes.length; i++) {
            const archetype = this._archetypes[i];

            // If custom prototype provided, use it
            if (archetype.prototype.length > 0) {
                embeddings.push(this._resizeEmbedding(archetype.prototype));
                continue;
            }

            // Create embedding from archetype label and keywords
            const text = [archetype.label, ...archetype.keywords].join(' ');
            const embedding = this._hashToEmbedding(text);

            // Blend with vertex position for geometric grounding
            const vertex = this._lattice.getVertex(archetype.vertexId);
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
            'true', 'false', 'yes', 'no', 'maybe',
            'because', 'therefore', 'however', 'although',
            'first', 'second', 'third', 'last', 'next',
            'similar', 'different', 'same', 'other',
            'increase', 'decrease', 'change', 'remain'
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
            numArchetypes: this._archetypes.length,
            vocabularySize: this._vocabulary.size,
            embeddingCacheSize: this._embeddingCache.size,
            forceMagnitude: this._config.forceMagnitude,
            rotationalWeight: this._config.rotationalWeight,
            temperature: this._config.temperature,
            usePositionalEncoding: this._config.usePositionalEncoding,
            useAttention: this._config.useAttention,
            attentionHeads: this._config.attentionHeads,
            hasAPIEmbeddings: this.hasAPIEmbeddings
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

    /**
     * Clear embedding cache (API responses).
     */
    clearEmbeddingCache(): void {
        this._embeddingCache.clear();
    }

    /**
     * Set custom archetypes for domain-specific use.
     */
    setCustomArchetypes(archetypes: ArchetypeDefinition[]): void {
        this.setConfig({ customArchetypes: archetypes });
    }

    /**
     * Get attention visualization data for debugging.
     */
    getAttentionVisualization(text: string): {
        tokens: string[];
        weights: number[][];
        contributions: number[];
    } | null {
        if (!this._config.useAttention) {
            return null;
        }

        const tokens = this._tokenize(text);
        const tokenEmbeddings: Float32Array[] = [];

        for (let i = 0; i < tokens.tokens.length; i++) {
            let embedding = this._getTokenEmbedding(tokens.tokens[i]);
            if (this._config.usePositionalEncoding) {
                embedding = applyPositionalEncoding(
                    embedding,
                    i,
                    this._config.positionalDimension
                );
            }
            tokenEmbeddings.push(embedding);
        }

        if (tokenEmbeddings.length < 2) {
            return null;
        }

        const attention = multiHeadAttention(
            tokenEmbeddings,
            this._config.attentionHeads,
            this._config.temperature
        );

        return {
            tokens: tokens.tokens,
            weights: attention.weights,
            contributions: attention.contributions
        };
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

/**
 * Create an encoder with API embeddings.
 */
export function createAPIEncoder(
    provider: EmbeddingProvider,
    apiKey: string,
    config?: Partial<HDCEncoderConfig>
): HDCEncoder {
    return new HDCEncoder({
        ...config,
        embeddingAPI: {
            provider,
            apiKey
        }
    });
}

/**
 * Create an encoder with Google Gemini embeddings.
 * Uses text-embedding-004 (768 dimensions) by default.
 */
export function createGeminiEncoder(
    apiKey: string,
    config?: Partial<HDCEncoderConfig>
): HDCEncoder {
    return new HDCEncoder({
        inputDimension: 768,  // Gemini text-embedding-004 output dimension
        ...config,
        embeddingAPI: {
            provider: 'gemini',
            apiKey,
            model: config?.embeddingAPI?.model || 'text-embedding-004'
        }
    });
}

/**
 * Create an encoder with Anthropic-recommended embeddings (Voyage AI).
 * Uses voyage-3 (1024 dimensions) by default.
 */
export function createAnthropicEncoder(
    voyageApiKey: string,
    config?: Partial<HDCEncoderConfig>
): HDCEncoder {
    return new HDCEncoder({
        inputDimension: 1024,  // Voyage-3 output dimension
        ...config,
        embeddingAPI: {
            provider: 'anthropic',
            apiKey: voyageApiKey,
            model: config?.embeddingAPI?.model || 'voyage-3'
        }
    });
}

/**
 * Create a domain-specific encoder with custom archetypes.
 */
export function createDomainEncoder(
    archetypes: ArchetypeDefinition[],
    config?: Partial<HDCEncoderConfig>
): HDCEncoder {
    return new HDCEncoder({
        ...config,
        customArchetypes: archetypes,
        numArchetypes: archetypes.length
    });
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

/**
 * Async text to force with API embeddings.
 */
export async function textToForceAsync(
    text: string,
    apiConfig: EmbeddingAPIConfig
): Promise<Force> {
    const encoder = new HDCEncoder({ embeddingAPI: apiConfig });
    return encoder.textToForceAsync(text);
}

// =============================================================================
// PRESET DOMAIN ARCHETYPES
// =============================================================================

/**
 * Medical domain archetypes.
 */
export const MEDICAL_ARCHETYPES: ArchetypeDefinition[] = [
    { label: 'diagnosis', keywords: ['diagnose', 'condition', 'disease', 'disorder', 'syndrome'] },
    { label: 'treatment', keywords: ['treat', 'therapy', 'medication', 'intervention', 'procedure'] },
    { label: 'symptom', keywords: ['symptom', 'sign', 'manifestation', 'presentation', 'complaint'] },
    { label: 'prognosis', keywords: ['prognosis', 'outcome', 'recovery', 'survival', 'course'] },
    { label: 'etiology', keywords: ['cause', 'origin', 'pathogenesis', 'mechanism', 'factor'] },
    { label: 'prevention', keywords: ['prevent', 'prophylaxis', 'screening', 'vaccination', 'risk'] },
    { label: 'anatomy', keywords: ['organ', 'tissue', 'structure', 'system', 'region'] },
    { label: 'physiology', keywords: ['function', 'process', 'regulation', 'homeostasis', 'metabolism'] },
    { label: 'pathology', keywords: ['abnormal', 'lesion', 'damage', 'dysfunction', 'failure'] },
    { label: 'pharmacology', keywords: ['drug', 'dose', 'effect', 'interaction', 'contraindication'] },
    { label: 'epidemiology', keywords: ['prevalence', 'incidence', 'population', 'transmission', 'outbreak'] },
    { label: 'genetics', keywords: ['gene', 'mutation', 'hereditary', 'chromosome', 'expression'] },
    { label: 'immunology', keywords: ['immune', 'antibody', 'antigen', 'inflammation', 'allergy'] },
    { label: 'microbiology', keywords: ['bacteria', 'virus', 'infection', 'pathogen', 'culture'] },
    { label: 'radiology', keywords: ['imaging', 'scan', 'xray', 'mri', 'ultrasound'] },
    { label: 'surgery', keywords: ['surgical', 'operation', 'incision', 'resection', 'transplant'] },
    { label: 'emergency', keywords: ['acute', 'critical', 'urgent', 'trauma', 'resuscitation'] },
    { label: 'chronic', keywords: ['chronic', 'longterm', 'progressive', 'degenerative', 'persistent'] },
    { label: 'pediatric', keywords: ['child', 'infant', 'developmental', 'congenital', 'growth'] },
    { label: 'geriatric', keywords: ['elderly', 'aging', 'frailty', 'dementia', 'palliative'] },
    { label: 'mental', keywords: ['psychiatric', 'psychological', 'behavioral', 'cognitive', 'mood'] },
    { label: 'nutrition', keywords: ['diet', 'nutrient', 'deficiency', 'supplement', 'calorie'] },
    { label: 'rehabilitation', keywords: ['rehab', 'physical', 'occupational', 'speech', 'mobility'] },
    { label: 'laboratory', keywords: ['test', 'assay', 'marker', 'level', 'result'] }
];

/**
 * Legal domain archetypes.
 */
export const LEGAL_ARCHETYPES: ArchetypeDefinition[] = [
    { label: 'statute', keywords: ['law', 'code', 'regulation', 'provision', 'section'] },
    { label: 'precedent', keywords: ['case', 'ruling', 'decision', 'holding', 'opinion'] },
    { label: 'contract', keywords: ['agreement', 'clause', 'term', 'party', 'breach'] },
    { label: 'liability', keywords: ['liable', 'responsible', 'duty', 'negligence', 'damages'] },
    { label: 'rights', keywords: ['right', 'entitlement', 'protection', 'constitutional', 'civil'] },
    { label: 'obligation', keywords: ['duty', 'requirement', 'compliance', 'mandate', 'must'] },
    { label: 'evidence', keywords: ['proof', 'testimony', 'exhibit', 'witness', 'admissible'] },
    { label: 'procedure', keywords: ['process', 'filing', 'motion', 'hearing', 'appeal'] },
    { label: 'jurisdiction', keywords: ['court', 'venue', 'authority', 'federal', 'state'] },
    { label: 'remedy', keywords: ['relief', 'injunction', 'compensation', 'restitution', 'specific'] },
    { label: 'criminal', keywords: ['crime', 'offense', 'prosecution', 'defendant', 'sentence'] },
    { label: 'civil', keywords: ['tort', 'plaintiff', 'lawsuit', 'claim', 'settlement'] },
    { label: 'property', keywords: ['ownership', 'title', 'deed', 'easement', 'lien'] },
    { label: 'corporate', keywords: ['company', 'shareholder', 'board', 'merger', 'fiduciary'] },
    { label: 'employment', keywords: ['worker', 'employer', 'discrimination', 'termination', 'wage'] },
    { label: 'intellectual', keywords: ['patent', 'copyright', 'trademark', 'trade secret', 'license'] },
    { label: 'regulatory', keywords: ['agency', 'compliance', 'enforcement', 'permit', 'inspection'] },
    { label: 'constitutional', keywords: ['amendment', 'fundamental', 'due process', 'equal', 'free'] },
    { label: 'international', keywords: ['treaty', 'convention', 'sovereign', 'extradition', 'trade'] },
    { label: 'dispute', keywords: ['conflict', 'arbitration', 'mediation', 'resolution', 'negotiation'] },
    { label: 'defense', keywords: ['defense', 'immunity', 'privilege', 'exception', 'excuse'] },
    { label: 'interpretation', keywords: ['construe', 'meaning', 'intent', 'ambiguous', 'plain'] },
    { label: 'standard', keywords: ['test', 'burden', 'threshold', 'reasonable', 'preponderance'] },
    { label: 'doctrine', keywords: ['principle', 'theory', 'rule', 'maxim', 'canon'] }
];

/**
 * Software engineering domain archetypes.
 */
export const SOFTWARE_ARCHETYPES: ArchetypeDefinition[] = [
    { label: 'architecture', keywords: ['design', 'pattern', 'structure', 'component', 'module'] },
    { label: 'algorithm', keywords: ['complexity', 'optimization', 'sort', 'search', 'graph'] },
    { label: 'database', keywords: ['sql', 'query', 'schema', 'index', 'transaction'] },
    { label: 'api', keywords: ['endpoint', 'request', 'response', 'rest', 'graphql'] },
    { label: 'security', keywords: ['authentication', 'authorization', 'encryption', 'vulnerability', 'attack'] },
    { label: 'testing', keywords: ['unit', 'integration', 'coverage', 'mock', 'assertion'] },
    { label: 'deployment', keywords: ['ci', 'cd', 'container', 'kubernetes', 'cloud'] },
    { label: 'performance', keywords: ['latency', 'throughput', 'memory', 'cpu', 'bottleneck'] },
    { label: 'debugging', keywords: ['bug', 'error', 'exception', 'trace', 'breakpoint'] },
    { label: 'refactoring', keywords: ['clean', 'technical debt', 'maintainability', 'readability', 'smell'] },
    { label: 'concurrency', keywords: ['thread', 'async', 'parallel', 'race', 'deadlock'] },
    { label: 'networking', keywords: ['tcp', 'http', 'socket', 'protocol', 'bandwidth'] },
    { label: 'frontend', keywords: ['ui', 'component', 'state', 'render', 'dom'] },
    { label: 'backend', keywords: ['server', 'service', 'middleware', 'queue', 'cache'] },
    { label: 'data', keywords: ['model', 'schema', 'migration', 'validation', 'serialization'] },
    { label: 'version', keywords: ['git', 'branch', 'merge', 'commit', 'release'] },
    { label: 'documentation', keywords: ['readme', 'comment', 'spec', 'api doc', 'tutorial'] },
    { label: 'dependency', keywords: ['package', 'library', 'import', 'version', 'conflict'] },
    { label: 'configuration', keywords: ['env', 'setting', 'flag', 'variable', 'secret'] },
    { label: 'monitoring', keywords: ['log', 'metric', 'alert', 'dashboard', 'trace'] },
    { label: 'scaling', keywords: ['horizontal', 'vertical', 'load', 'shard', 'replica'] },
    { label: 'microservice', keywords: ['service', 'mesh', 'discovery', 'gateway', 'saga'] },
    { label: 'devops', keywords: ['infrastructure', 'automation', 'pipeline', 'terraform', 'ansible'] },
    { label: 'agile', keywords: ['sprint', 'scrum', 'kanban', 'story', 'retrospective'] }
];
