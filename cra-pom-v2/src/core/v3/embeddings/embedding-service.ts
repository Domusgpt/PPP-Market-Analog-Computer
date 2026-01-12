/**
 * PPP v3 Embedding Service
 *
 * HONEST DOCUMENTATION:
 * This module handles semantic embeddings for concepts.
 *
 * EMBEDDING SOURCES (in order of preference):
 * 1. External API (if configured) - Real transformer-based embeddings
 * 2. Local model (if loaded) - Browser-based transformer
 * 3. Cached embeddings - Pre-computed from a real model
 * 4. Deterministic fallback - Hash-based, NOT semantic (clearly marked)
 *
 * WHAT REAL EMBEDDINGS PROVIDE:
 * - Semantic similarity (similar concepts are close)
 * - Analogical reasoning (king - man + woman ≈ queen)
 * - Zero-shot classification
 *
 * WHAT HASH-BASED FALLBACK PROVIDES:
 * - Deterministic vectors (same input → same output)
 * - Consistent dimension
 * - NOTHING SEMANTIC (vectors have no meaning relationship)
 *
 * The system MUST be honest about which mode it's using.
 */

// ============================================================================
// Types
// ============================================================================

export interface EmbeddingResult {
  /** The input text that was embedded */
  text: string;

  /** The embedding vector */
  vector: Float32Array;

  /** Where this embedding came from */
  source: EmbeddingSource;

  /** Model identifier if from a real model */
  model?: string;

  /** Metadata */
  meta: {
    dimension: number;
    normalized: boolean;
    timestamp: string;
    cached: boolean;
  };
}

export type EmbeddingSource =
  | 'external_api' // From an external embedding API
  | 'local_model' // From a browser-based transformer
  | 'cache' // Pre-computed and cached
  | 'deterministic_fallback'; // Hash-based (NOT semantic)

export interface EmbeddingConfig {
  /** External API configuration */
  api?: {
    url: string;
    apiKey?: string;
    model: string;
  };

  /** Default dimension for embeddings */
  dimension: number;

  /** Whether to normalize vectors to unit length */
  normalize: boolean;

  /** Cache configuration */
  cache?: {
    enabled: boolean;
    maxSize: number;
  };
}

export interface SimilarityResult {
  /** Text A */
  textA: string;

  /** Text B */
  textB: string;

  /** Cosine similarity (-1 to 1) */
  similarity: number;

  /** Whether this is meaningful (from real embeddings) */
  semanticallyMeaningful: boolean;

  /** Source of both embeddings */
  source: EmbeddingSource;
}

// ============================================================================
// Constants
// ============================================================================

const DEFAULT_DIMENSION = 384; // Common for small models like all-MiniLM-L6-v2
const DEFAULT_CONFIG: EmbeddingConfig = {
  dimension: DEFAULT_DIMENSION,
  normalize: true,
  cache: {
    enabled: true,
    maxSize: 1000,
  },
};

// ============================================================================
// Implementation
// ============================================================================

export class EmbeddingService {
  private config: EmbeddingConfig;
  private cache: Map<string, EmbeddingResult> = new Map();
  private localModelLoaded = false;
  private currentSource: EmbeddingSource = 'deterministic_fallback';

  constructor(config: Partial<EmbeddingConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  /**
   * Get the current embedding source
   */
  getSource(): EmbeddingSource {
    return this.currentSource;
  }

  /**
   * Is the current source semantically meaningful?
   */
  isSemantic(): boolean {
    return this.currentSource !== 'deterministic_fallback';
  }

  /**
   * Configure external API
   */
  configureApi(url: string, apiKey: string, model: string): void {
    this.config.api = { url, apiKey, model };
    this.currentSource = 'external_api';
  }

  /**
   * Embed a single text
   */
  async embed(text: string): Promise<EmbeddingResult> {
    const normalizedText = text.trim().toLowerCase();

    // Check cache first
    if (this.config.cache?.enabled && this.cache.has(normalizedText)) {
      const cached = this.cache.get(normalizedText)!;
      return { ...cached, meta: { ...cached.meta, cached: true } };
    }

    let result: EmbeddingResult;

    // Try sources in order of preference
    if (this.config.api) {
      result = await this.embedViaApi(normalizedText);
    } else if (this.localModelLoaded) {
      result = await this.embedViaLocalModel(normalizedText);
    } else {
      result = this.embedViaDeterministicFallback(normalizedText);
    }

    // Cache the result
    if (this.config.cache?.enabled) {
      if (this.cache.size >= (this.config.cache.maxSize || 1000)) {
        // Simple LRU: delete oldest
        const firstKey = this.cache.keys().next().value;
        if (firstKey) this.cache.delete(firstKey);
      }
      this.cache.set(normalizedText, result);
    }

    return result;
  }

  /**
   * Embed multiple texts
   */
  async embedBatch(texts: string[]): Promise<EmbeddingResult[]> {
    // For API sources, batch requests would be more efficient
    // For now, we process individually
    return Promise.all(texts.map((t) => this.embed(t)));
  }

  /**
   * Compute similarity between two texts
   */
  async similarity(textA: string, textB: string): Promise<SimilarityResult> {
    const [embA, embB] = await Promise.all([this.embed(textA), this.embed(textB)]);

    const similarity = this.cosineSimilarity(embA.vector, embB.vector);

    return {
      textA,
      textB,
      similarity,
      semanticallyMeaningful: embA.source !== 'deterministic_fallback',
      source: embA.source,
    };
  }

  /**
   * Find most similar from a set of candidates
   */
  async findMostSimilar(
    query: string,
    candidates: string[],
    topK = 5
  ): Promise<Array<{ text: string; similarity: number }>> {
    const queryEmb = await this.embed(query);
    const candidateEmbs = await this.embedBatch(candidates);

    const similarities = candidateEmbs.map((emb, i) => ({
      text: candidates[i],
      similarity: this.cosineSimilarity(queryEmb.vector, emb.vector),
    }));

    return similarities.sort((a, b) => b.similarity - a.similarity).slice(0, topK);
  }

  /**
   * Get embedding dimension
   */
  getDimension(): number {
    return this.config.dimension;
  }

  /**
   * Get stats about the service
   */
  getStats(): {
    source: EmbeddingSource;
    semantic: boolean;
    cacheSize: number;
    dimension: number;
  } {
    return {
      source: this.currentSource,
      semantic: this.isSemantic(),
      cacheSize: this.cache.size,
      dimension: this.config.dimension,
    };
  }

  // ============================================================================
  // Embedding Sources
  // ============================================================================

  /**
   * Embed via external API
   *
   * This produces REAL semantic embeddings.
   */
  private async embedViaApi(text: string): Promise<EmbeddingResult> {
    if (!this.config.api) {
      throw new Error('API not configured');
    }

    try {
      const response = await fetch(this.config.api.url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(this.config.api.apiKey && {
            Authorization: `Bearer ${this.config.api.apiKey}`,
          }),
        },
        body: JSON.stringify({
          input: text,
          model: this.config.api.model,
        }),
      });

      if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
      }

      const data = await response.json();

      // Handle OpenAI-style response format
      const embedding = data.data?.[0]?.embedding || data.embedding;

      if (!embedding || !Array.isArray(embedding)) {
        throw new Error('Invalid API response format');
      }

      const vector = new Float32Array(embedding);
      const normalized = this.config.normalize ? this.normalize(vector) : vector;

      this.currentSource = 'external_api';

      return {
        text,
        vector: normalized,
        source: 'external_api',
        model: this.config.api.model,
        meta: {
          dimension: normalized.length,
          normalized: this.config.normalize,
          timestamp: new Date().toISOString(),
          cached: false,
        },
      };
    } catch (err) {
      console.warn('API embedding failed, falling back:', err);
      // Fall back to deterministic
      return this.embedViaDeterministicFallback(text);
    }
  }

  /**
   * Embed via local model
   *
   * This would use Transformers.js or similar for browser-based inference.
   * Currently a placeholder - produces REAL semantic embeddings when implemented.
   */
  private async embedViaLocalModel(text: string): Promise<EmbeddingResult> {
    // TODO: Implement with Transformers.js
    // For now, fall back to deterministic
    console.warn('Local model not implemented, using deterministic fallback');
    return this.embedViaDeterministicFallback(text);
  }

  /**
   * Deterministic fallback embedding
   *
   * WARNING: This produces vectors that are:
   * - Deterministic (same input → same output)
   * - High-dimensional
   * - NOT SEMANTIC (no meaning relationship)
   *
   * This should only be used when real embeddings are unavailable.
   * The system MUST clearly indicate when using this mode.
   */
  private embedViaDeterministicFallback(text: string): EmbeddingResult {
    this.currentSource = 'deterministic_fallback';

    const vector = this.deterministicHash(text, this.config.dimension);
    const normalized = this.config.normalize ? this.normalize(vector) : vector;

    return {
      text,
      vector: normalized,
      source: 'deterministic_fallback',
      model: undefined,
      meta: {
        dimension: this.config.dimension,
        normalized: this.config.normalize,
        timestamp: new Date().toISOString(),
        cached: false,
      },
    };
  }

  // ============================================================================
  // Vector Operations
  // ============================================================================

  /**
   * Cosine similarity between two vectors
   */
  private cosineSimilarity(a: Float32Array, b: Float32Array): number {
    if (a.length !== b.length) {
      throw new Error('Vector dimension mismatch');
    }

    let dotProduct = 0;
    let normA = 0;
    let normB = 0;

    for (let i = 0; i < a.length; i++) {
      dotProduct += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }

    const denominator = Math.sqrt(normA) * Math.sqrt(normB);
    if (denominator === 0) return 0;

    return dotProduct / denominator;
  }

  /**
   * Normalize a vector to unit length
   */
  private normalize(vector: Float32Array): Float32Array {
    let norm = 0;
    for (let i = 0; i < vector.length; i++) {
      norm += vector[i] * vector[i];
    }
    norm = Math.sqrt(norm);

    if (norm === 0) return vector;

    const normalized = new Float32Array(vector.length);
    for (let i = 0; i < vector.length; i++) {
      normalized[i] = vector[i] / norm;
    }

    return normalized;
  }

  /**
   * Deterministic hash to vector
   *
   * Uses a seeded PRNG initialized from the text hash.
   * This is NOT semantic - it's just deterministic noise.
   */
  private deterministicHash(text: string, dimension: number): Float32Array {
    // Simple hash function
    let h1 = 0xdeadbeef;
    let h2 = 0x41c6ce57;

    for (let i = 0; i < text.length; i++) {
      const ch = text.charCodeAt(i);
      h1 = Math.imul(h1 ^ ch, 2654435761);
      h2 = Math.imul(h2 ^ ch, 1597334677);
    }

    h1 = Math.imul(h1 ^ (h1 >>> 16), 2246822507) ^ Math.imul(h2 ^ (h2 >>> 13), 3266489909);
    h2 = Math.imul(h2 ^ (h2 >>> 16), 2246822507) ^ Math.imul(h1 ^ (h1 >>> 13), 3266489909);

    // Seeded PRNG (mulberry32)
    let seed = (h1 >>> 0) + (h2 >>> 0);

    const vector = new Float32Array(dimension);
    for (let i = 0; i < dimension; i++) {
      seed = (seed + 0x6d2b79f5) | 0;
      let t = seed;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      const random = ((t ^ (t >>> 14)) >>> 0) / 4294967296;

      // Box-Muller for Gaussian distribution
      if (i + 1 < dimension) {
        seed = (seed + 0x6d2b79f5) | 0;
        t = seed;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        const random2 = ((t ^ (t >>> 14)) >>> 0) / 4294967296;

        const r = Math.sqrt(-2 * Math.log(random || 1e-10));
        const theta = 2 * Math.PI * random2;

        vector[i] = r * Math.cos(theta);
        vector[i + 1] = r * Math.sin(theta);
        i++; // Skip next iteration
      } else {
        vector[i] = random * 2 - 1;
      }
    }

    return vector;
  }
}

// ============================================================================
// Singleton
// ============================================================================

let globalEmbeddingService: EmbeddingService | null = null;

export function getEmbeddingService(): EmbeddingService {
  if (!globalEmbeddingService) {
    globalEmbeddingService = new EmbeddingService();
  }
  return globalEmbeddingService;
}

export function resetEmbeddingService(): void {
  globalEmbeddingService = null;
}

/**
 * Configure the global embedding service with an API
 */
export function configureEmbeddings(
  url: string,
  apiKey: string,
  model: string
): void {
  const service = getEmbeddingService();
  service.configureApi(url, apiKey, model);
}
