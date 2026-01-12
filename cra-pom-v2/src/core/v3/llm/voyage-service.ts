/**
 * PPP v3 Voyage AI Integration
 *
 * Provides high-quality semantic embeddings via Voyage AI (Anthropic).
 *
 * Models:
 * - voyage-3-large: Best quality, 1024 dimensions
 * - voyage-3.5: Latest general purpose
 * - voyage-3.5-lite: Faster, cost-optimized
 * - voyage-code-3: Optimized for code
 *
 * API Docs: https://docs.voyageai.com/docs/embeddings
 */

// ============================================================================
// Types
// ============================================================================

export interface VoyageConfig {
  apiKey: string;
  model?: VoyageModel;
  outputDimension?: number;
}

export type VoyageModel =
  | 'voyage-3-large'
  | 'voyage-3.5'
  | 'voyage-3.5-lite'
  | 'voyage-code-3'
  | 'voyage-finance-2'
  | 'voyage-law-2';

export type VoyageInputType = 'query' | 'document';

export interface VoyageEmbeddingRequest {
  input: string | string[];
  model: string;
  input_type?: VoyageInputType;
  output_dimension?: number;
  truncation?: boolean;
}

export interface VoyageEmbeddingResponse {
  object: string;
  data: Array<{
    object: string;
    embedding: number[];
    index: number;
  }>;
  model: string;
  usage: {
    total_tokens: number;
  };
}

export interface VoyageEmbeddingResult {
  vector: Float32Array;
  dimension: number;
  model: string;
  tokens: number;
  inputType?: VoyageInputType;
}

// ============================================================================
// Constants
// ============================================================================

const VOYAGE_API_URL = 'https://api.voyageai.com/v1/embeddings';
const DEFAULT_MODEL: VoyageModel = 'voyage-3-large';
const DEFAULT_DIMENSION = 1024;

// ============================================================================
// Voyage Service
// ============================================================================

export class VoyageService {
  private apiKey: string;
  private model: VoyageModel;
  private outputDimension: number;

  constructor(config: VoyageConfig) {
    this.apiKey = config.apiKey;
    this.model = config.model || DEFAULT_MODEL;
    this.outputDimension = config.outputDimension || DEFAULT_DIMENSION;
  }

  /**
   * Get embedding for a single text
   */
  async embed(
    text: string,
    inputType?: VoyageInputType
  ): Promise<VoyageEmbeddingResult> {
    const results = await this.embedBatch([text], inputType);
    return results[0];
  }

  /**
   * Get embeddings for multiple texts (batch)
   */
  async embedBatch(
    texts: string[],
    inputType?: VoyageInputType
  ): Promise<VoyageEmbeddingResult[]> {
    const request: VoyageEmbeddingRequest = {
      input: texts,
      model: this.model,
      truncation: true,
    };

    if (inputType) {
      request.input_type = inputType;
    }

    if (this.outputDimension !== DEFAULT_DIMENSION) {
      request.output_dimension = this.outputDimension;
    }

    const response = await fetch(VOYAGE_API_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${this.apiKey}`,
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`Voyage API error: ${response.status} - ${error}`);
    }

    const data: VoyageEmbeddingResponse = await response.json();

    // Sort by index to maintain order
    const sorted = [...data.data].sort((a, b) => a.index - b.index);

    return sorted.map((item) => ({
      vector: new Float32Array(item.embedding),
      dimension: item.embedding.length,
      model: data.model,
      tokens: Math.round(data.usage.total_tokens / texts.length),
      inputType,
    }));
  }

  /**
   * Embed a query (optimized for retrieval queries)
   */
  async embedQuery(query: string): Promise<VoyageEmbeddingResult> {
    return this.embed(query, 'query');
  }

  /**
   * Embed a document (optimized for document storage)
   */
  async embedDocument(document: string): Promise<VoyageEmbeddingResult> {
    return this.embed(document, 'document');
  }

  /**
   * Embed multiple documents
   */
  async embedDocuments(documents: string[]): Promise<VoyageEmbeddingResult[]> {
    return this.embedBatch(documents, 'document');
  }

  /**
   * Compute cosine similarity between two texts
   */
  async similarity(textA: string, textB: string): Promise<{
    similarity: number;
    textA: string;
    textB: string;
  }> {
    const [embA, embB] = await this.embedBatch([textA, textB]);
    const similarity = this.cosineSimilarity(embA.vector, embB.vector);

    return {
      similarity,
      textA,
      textB,
    };
  }

  /**
   * Find most similar texts from candidates
   */
  async findMostSimilar(
    query: string,
    candidates: string[],
    topK = 5
  ): Promise<Array<{ text: string; similarity: number; index: number }>> {
    const queryEmb = await this.embedQuery(query);
    const candidateEmbs = await this.embedDocuments(candidates);

    const similarities = candidateEmbs.map((emb, i) => ({
      text: candidates[i],
      similarity: this.cosineSimilarity(queryEmb.vector, emb.vector),
      index: i,
    }));

    return similarities
      .sort((a, b) => b.similarity - a.similarity)
      .slice(0, topK);
  }

  /**
   * Get current model
   */
  getModel(): VoyageModel {
    return this.model;
  }

  /**
   * Get output dimension
   */
  getDimension(): number {
    return this.outputDimension;
  }

  // ==========================================================================
  // Private Methods
  // ==========================================================================

  private cosineSimilarity(a: Float32Array, b: Float32Array): number {
    if (a.length !== b.length) {
      throw new Error('Vector dimension mismatch');
    }

    let dot = 0;
    let normA = 0;
    let normB = 0;

    for (let i = 0; i < a.length; i++) {
      dot += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }

    const denominator = Math.sqrt(normA) * Math.sqrt(normB);
    if (denominator === 0) return 0;

    return dot / denominator;
  }
}

// ============================================================================
// Singleton
// ============================================================================

let globalVoyageService: VoyageService | null = null;

export function initializeVoyage(apiKey: string, model?: VoyageModel): VoyageService {
  globalVoyageService = new VoyageService({ apiKey, model });
  return globalVoyageService;
}

export function getVoyageService(): VoyageService {
  if (!globalVoyageService) {
    throw new Error('Voyage service not initialized. Call initializeVoyage(apiKey) first.');
  }
  return globalVoyageService;
}

export function resetVoyageService(): void {
  globalVoyageService = null;
}
