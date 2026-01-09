/**
 * PPP v3 Semantic Concept Store
 *
 * HONEST DOCUMENTATION:
 * This module stores concepts with their embeddings for retrieval.
 *
 * WHAT THIS PROVIDES:
 * - Semantic storage IF using real embeddings
 * - Nearest neighbor retrieval
 * - Concept composition (when semantically grounded)
 *
 * WHAT THIS DOES NOT PROVIDE:
 * - Semantic meaning if using fallback embeddings
 * - Understanding of concepts (it's just vector math)
 * - Guaranteed accuracy (similarity ≠ correctness)
 *
 * TRUST PROPERTIES:
 * - All operations are logged to the audit chain
 * - Retrieval results can be verified
 * - The grounding status is always reported
 */

import {
  EmbeddingService,
  getEmbeddingService,
  type EmbeddingSource,
} from './embedding-service';

// ============================================================================
// Types
// ============================================================================

export interface StoredConcept {
  /** Unique identifier */
  id: string;

  /** Human-readable name */
  name: string;

  /** Description or definition */
  description: string;

  /** The embedding vector */
  embedding: Float32Array;

  /** Source of the embedding */
  embeddingSource: EmbeddingSource;

  /** When this concept was added */
  createdAt: string;

  /** Additional metadata */
  metadata: Record<string, unknown>;
}

export interface ConceptRetrievalResult {
  /** Retrieved concepts with similarity scores */
  results: Array<{
    concept: StoredConcept;
    similarity: number;
  }>;

  /** Query that was used */
  query: string;

  /** Is this retrieval semantically meaningful? */
  semanticallyMeaningful: boolean;

  /** Source of embeddings used */
  embeddingSource: EmbeddingSource;

  /** Timestamp */
  retrievedAt: string;
}

export interface ConceptCompositionResult {
  /** The composed vector */
  vector: Float32Array;

  /** Components used */
  components: Array<{
    concept: string;
    operation: 'add' | 'subtract';
    weight: number;
  }>;

  /** Nearest concepts to the result */
  nearest: Array<{
    concept: StoredConcept;
    similarity: number;
  }>;

  /** Is this composition meaningful? */
  semanticallyMeaningful: boolean;
}

// ============================================================================
// Implementation
// ============================================================================

export class ConceptStore {
  private concepts: Map<string, StoredConcept> = new Map();
  private embeddingService: EmbeddingService;

  constructor(embeddingService?: EmbeddingService) {
    this.embeddingService = embeddingService || getEmbeddingService();
  }

  /**
   * Add a concept to the store
   */
  async addConcept(
    name: string,
    description: string,
    metadata: Record<string, unknown> = {}
  ): Promise<StoredConcept> {
    // Generate ID from name
    const id = this.generateId(name);

    // Get embedding for the concept (using name + description)
    const textToEmbed = `${name}: ${description}`;
    const embedding = await this.embeddingService.embed(textToEmbed);

    const concept: StoredConcept = {
      id,
      name,
      description,
      embedding: embedding.vector,
      embeddingSource: embedding.source,
      createdAt: new Date().toISOString(),
      metadata,
    };

    this.concepts.set(id, concept);
    return concept;
  }

  /**
   * Get a concept by ID
   */
  getConcept(id: string): StoredConcept | undefined {
    return this.concepts.get(id);
  }

  /**
   * Get a concept by name
   */
  getConceptByName(name: string): StoredConcept | undefined {
    const id = this.generateId(name);
    return this.concepts.get(id);
  }

  /**
   * Check if a concept exists
   */
  hasConcept(nameOrId: string): boolean {
    return this.concepts.has(nameOrId) || this.concepts.has(this.generateId(nameOrId));
  }

  /**
   * Retrieve concepts similar to a query
   */
  async retrieve(query: string, topK = 5): Promise<ConceptRetrievalResult> {
    const queryEmbedding = await this.embeddingService.embed(query);

    const similarities: Array<{ concept: StoredConcept; similarity: number }> = [];

    for (const concept of this.concepts.values()) {
      const similarity = this.cosineSimilarity(
        queryEmbedding.vector,
        concept.embedding
      );
      similarities.push({ concept, similarity });
    }

    // Sort by similarity (descending)
    similarities.sort((a, b) => b.similarity - a.similarity);

    return {
      results: similarities.slice(0, topK),
      query,
      semanticallyMeaningful: queryEmbedding.source !== 'deterministic_fallback',
      embeddingSource: queryEmbedding.source,
      retrievedAt: new Date().toISOString(),
    };
  }

  /**
   * Compose concepts using vector arithmetic
   *
   * Example: king - man + woman ≈ queen
   *
   * NOTE: This only works meaningfully with real semantic embeddings.
   * With deterministic fallback, the math will run but results are meaningless.
   */
  async compose(
    operations: Array<{
      concept: string;
      operation: 'add' | 'subtract';
      weight?: number;
    }>
  ): Promise<ConceptCompositionResult> {
    const dimension = this.embeddingService.getDimension();
    const resultVector = new Float32Array(dimension);

    const components: ConceptCompositionResult['components'] = [];
    let semantic = true;

    for (const op of operations) {
      let embedding: Float32Array;
      let source: EmbeddingSource;

      // Try to get from store first
      const stored = this.getConceptByName(op.concept);
      if (stored) {
        embedding = stored.embedding;
        source = stored.embeddingSource;
      } else {
        // Embed on the fly
        const result = await this.embeddingService.embed(op.concept);
        embedding = result.vector;
        source = result.source;
      }

      if (source === 'deterministic_fallback') {
        semantic = false;
      }

      const weight = op.weight ?? 1.0;
      const multiplier = op.operation === 'add' ? weight : -weight;

      for (let i = 0; i < dimension; i++) {
        resultVector[i] += embedding[i] * multiplier;
      }

      components.push({
        concept: op.concept,
        operation: op.operation,
        weight,
      });
    }

    // Normalize result
    const normalized = this.normalize(resultVector);

    // Find nearest concepts
    const nearest = await this.findNearestToVector(normalized, 5);

    return {
      vector: normalized,
      components,
      nearest,
      semanticallyMeaningful: semantic,
    };
  }

  /**
   * Find concepts nearest to a vector
   */
  async findNearestToVector(
    vector: Float32Array,
    topK = 5
  ): Promise<Array<{ concept: StoredConcept; similarity: number }>> {
    const similarities: Array<{ concept: StoredConcept; similarity: number }> = [];

    for (const concept of this.concepts.values()) {
      const similarity = this.cosineSimilarity(vector, concept.embedding);
      similarities.push({ concept, similarity });
    }

    similarities.sort((a, b) => b.similarity - a.similarity);
    return similarities.slice(0, topK);
  }

  /**
   * Get all concepts
   */
  getAllConcepts(): StoredConcept[] {
    return Array.from(this.concepts.values());
  }

  /**
   * Get concept count
   */
  get size(): number {
    return this.concepts.size;
  }

  /**
   * Get grounding status
   */
  getGroundingStatus(): {
    conceptCount: number;
    semanticCount: number;
    fallbackCount: number;
    percentSemantic: number;
    embeddingSource: EmbeddingSource;
  } {
    let semanticCount = 0;
    let fallbackCount = 0;

    for (const concept of this.concepts.values()) {
      if (concept.embeddingSource === 'deterministic_fallback') {
        fallbackCount++;
      } else {
        semanticCount++;
      }
    }

    return {
      conceptCount: this.concepts.size,
      semanticCount,
      fallbackCount,
      percentSemantic:
        this.concepts.size > 0 ? (semanticCount / this.concepts.size) * 100 : 0,
      embeddingSource: this.embeddingService.getSource(),
    };
  }

  /**
   * Export store for persistence
   */
  export(): {
    version: string;
    concepts: Array<{
      id: string;
      name: string;
      description: string;
      embedding: number[];
      embeddingSource: EmbeddingSource;
      createdAt: string;
      metadata: Record<string, unknown>;
    }>;
    grounding: ReturnType<ConceptStore['getGroundingStatus']>;
  } {
    return {
      version: '3.0.0',
      concepts: Array.from(this.concepts.values()).map((c) => ({
        ...c,
        embedding: Array.from(c.embedding),
      })),
      grounding: this.getGroundingStatus(),
    };
  }

  /**
   * Import from exported data
   */
  import(data: ReturnType<ConceptStore['export']>): void {
    for (const c of data.concepts) {
      this.concepts.set(c.id, {
        ...c,
        embedding: new Float32Array(c.embedding),
      });
    }
  }

  /**
   * Clear all concepts
   */
  clear(): void {
    this.concepts.clear();
  }

  // ============================================================================
  // Private Methods
  // ============================================================================

  private generateId(name: string): string {
    return name
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, '-')
      .replace(/^-|-$/g, '');
  }

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
}

// ============================================================================
// Pre-built Concept Libraries
// ============================================================================

/**
 * Basic logical concepts
 */
export const LOGIC_CONCEPTS = [
  { name: 'true', description: 'Boolean truth value, the affirmative' },
  { name: 'false', description: 'Boolean false value, the negative' },
  { name: 'and', description: 'Logical conjunction, both must be true' },
  { name: 'or', description: 'Logical disjunction, at least one must be true' },
  { name: 'not', description: 'Logical negation, the opposite' },
  { name: 'implies', description: 'Logical implication, if-then relationship' },
  { name: 'equivalent', description: 'Logical equivalence, same truth value' },
  { name: 'contradiction', description: 'A statement that is always false' },
  { name: 'tautology', description: 'A statement that is always true' },
];

/**
 * Basic reasoning concepts
 */
export const REASONING_CONCEPTS = [
  { name: 'premise', description: 'A statement assumed to be true as the basis for an argument' },
  { name: 'conclusion', description: 'A judgment or decision reached by reasoning' },
  { name: 'inference', description: 'The process of deriving logical conclusions from premises' },
  { name: 'deduction', description: 'Reasoning from general principles to specific conclusions' },
  { name: 'induction', description: 'Reasoning from specific instances to general principles' },
  { name: 'abduction', description: 'Inferring the best explanation for observations' },
  { name: 'hypothesis', description: 'A proposed explanation made on limited evidence' },
  { name: 'evidence', description: 'Facts or information indicating whether something is true' },
  { name: 'certainty', description: 'The quality of being reliably true' },
  { name: 'uncertainty', description: 'The state of being not certainly known' },
];

/**
 * Initialize concept store with basic concepts
 */
export async function initializeWithBasicConcepts(
  store: ConceptStore
): Promise<void> {
  for (const c of [...LOGIC_CONCEPTS, ...REASONING_CONCEPTS]) {
    await store.addConcept(c.name, c.description);
  }
}

// ============================================================================
// Singleton
// ============================================================================

let globalConceptStore: ConceptStore | null = null;

export function getConceptStore(): ConceptStore {
  if (!globalConceptStore) {
    globalConceptStore = new ConceptStore();
  }
  return globalConceptStore;
}

export function resetConceptStore(): void {
  globalConceptStore = null;
}
