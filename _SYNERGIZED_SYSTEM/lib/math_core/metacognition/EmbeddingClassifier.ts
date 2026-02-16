/**
 * Embedding-Based State Classifier
 *
 * Uses external embedding services (OpenAI, Cohere, or local models)
 * to classify engine states based on semantic similarity.
 *
 * Ported from CPE repo (Domusgpt/-Chronomorphic-Polytopal-Engine)
 * with import paths adapted for PPP math_core layout.
 */

import type { Vector4D, TrinityAxis, BettiProfile } from '../geometric_algebra/types';
import { StateClassifier, type StateCategory, type StateClassification } from './StateClassifier';

// =============================================================================
// TYPES
// =============================================================================

/** Embedding vector */
export type Embedding = number[];

/** Embedding provider options */
export type EmbeddingProvider = 'openai' | 'cohere' | 'local' | 'custom';

/** Configuration for embedding classifier */
export interface EmbeddingClassifierConfig {
  provider: EmbeddingProvider;
  apiKey?: string;
  model?: string;
  baseUrl?: string;
  dimension: number;
  examplesPerCategory: number;
  similarityThreshold: number;
  customEmbedder?: (text: string) => Promise<Embedding>;
}

/** Labeled example for few-shot learning */
export interface LabeledExample {
  readonly category: StateCategory;
  readonly description: string;
  embedding?: Embedding;
}

/** Embedding classification result */
export interface EmbeddingClassificationResult extends StateClassification {
  readonly categorySimilarities: Record<StateCategory, number>;
  readonly nearestExamples: Array<{
    category: StateCategory;
    description: string;
    similarity: number;
  }>;
  readonly stateEmbedding?: Embedding;
}

// =============================================================================
// DEFAULT EXAMPLES
// =============================================================================

export const DEFAULT_EXAMPLES: LabeledExample[] = [
  { category: 'COHERENT', description: 'Stable harmonic state centered on a single axis with high coherence and low tension' },
  { category: 'COHERENT', description: 'Clear tonal center with consistent harmonic motion and predictable trajectory' },
  { category: 'COHERENT', description: 'Engine operating normally with smooth rotations and stable vertex proximity' },
  { category: 'TRANSITIONING', description: 'Phase shift in progress, moving between Trinity axes with rising tension' },
  { category: 'TRANSITIONING', description: 'Modulating from one key to another, passing through transitional harmonic space' },
  { category: 'TRANSITIONING', description: 'Active state change with increasing phase progress and cross-axis movement' },
  { category: 'AMBIGUOUS', description: 'Topological voids detected indicating missing harmonics and unstable structure' },
  { category: 'AMBIGUOUS', description: 'High beta-2 homology with multiple cavities in the activation pattern' },
  { category: 'AMBIGUOUS', description: 'Ghost frequencies present, harmonic content incomplete with structural gaps' },
  { category: 'POLYTONAL', description: 'Multiple Trinity axes active simultaneously in balanced superposition' },
  { category: 'POLYTONAL', description: 'Bitonality or polytonality with competing harmonic centers' },
  { category: 'POLYTONAL', description: 'Balanced weight distribution across alpha, beta, and gamma axes' },
  { category: 'STUCK', description: 'Motion stalled with near-zero velocity and oscillation around fixed point' },
  { category: 'STUCK', description: 'Engine locked in repetitive pattern unable to progress to new states' },
  { category: 'STUCK', description: 'Harmonic stagnation with no forward momentum or key changes' },
  { category: 'INVALID', description: 'State outside valid convex hull representing non-physical configuration' },
  { category: 'INVALID', description: 'Invalid polytope position beyond the 24-cell boundary' },
  { category: 'INVALID', description: 'Error state requiring reset to return to valid harmonic space' }
];

// =============================================================================
// EMBEDDING UTILITIES
// =============================================================================

export function cosineSimilarity(a: Embedding, b: Embedding): number {
  if (a.length !== b.length) {
    throw new Error('Embedding dimensions must match');
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
  return denominator > 0 ? dotProduct / denominator : 0;
}

export function averageEmbeddings(embeddings: Embedding[]): Embedding {
  if (embeddings.length === 0) return [];

  const dim = embeddings[0].length;
  const result = new Array(dim).fill(0);

  for (const emb of embeddings) {
    for (let i = 0; i < dim; i++) {
      result[i] += emb[i];
    }
  }

  const n = embeddings.length;
  return result.map(v => v / n);
}

// =============================================================================
// EMBEDDING CLASSIFIER
// =============================================================================

export class EmbeddingClassifier {
  private _config: EmbeddingClassifierConfig;
  private _examples: LabeledExample[];
  private _categoryEmbeddings: Map<StateCategory, Embedding[]>;
  private _categoryCentroids: Map<StateCategory, Embedding>;
  private _fallbackClassifier: StateClassifier;
  private _initialized: boolean;

  constructor(config: Partial<EmbeddingClassifierConfig> = {}) {
    this._config = {
      provider: 'custom',
      dimension: 1536,
      examplesPerCategory: 3,
      similarityThreshold: 0.7,
      ...config
    };

    this._examples = [...DEFAULT_EXAMPLES];
    this._categoryEmbeddings = new Map();
    this._categoryCentroids = new Map();
    this._fallbackClassifier = new StateClassifier();
    this._initialized = false;
  }

  async initialize(): Promise<void> {
    if (this._initialized) return;

    for (const example of this._examples) {
      if (!example.embedding) {
        example.embedding = await this._embed(example.description);
      }

      if (!this._categoryEmbeddings.has(example.category)) {
        this._categoryEmbeddings.set(example.category, []);
      }
      this._categoryEmbeddings.get(example.category)!.push(example.embedding);
    }

    for (const [category, embeddings] of this._categoryEmbeddings) {
      this._categoryCentroids.set(category, averageEmbeddings(embeddings));
    }

    this._initialized = true;
  }

  async classify(
    position: Vector4D,
    velocity: number,
    trinityState: { activeAxis: TrinityAxis; weights: [number, number, number]; tension: number },
    betti: BettiProfile | null,
    coherence: number,
    isInsideHull: boolean,
    nearestVertexDistance: number
  ): Promise<EmbeddingClassificationResult> {
    const ruleBasedResult = this._fallbackClassifier.classify(
      position,
      velocity,
      { ...trinityState, phaseProgress: 0 },
      betti,
      coherence,
      isInsideHull,
      nearestVertexDistance
    );

    if (!this._initialized || !this._config.customEmbedder) {
      return {
        ...ruleBasedResult,
        categorySimilarities: {} as Record<StateCategory, number>,
        nearestExamples: []
      };
    }

    const stateDescription = this._generateStateDescription(
      position, velocity, trinityState, betti, coherence, isInsideHull
    );

    const stateEmbedding = await this._embed(stateDescription);

    const categorySimilarities: Record<StateCategory, number> = {} as Record<StateCategory, number>;
    for (const [category, centroid] of this._categoryCentroids) {
      categorySimilarities[category] = cosineSimilarity(stateEmbedding, centroid);
    }

    const sortedCategories = Object.entries(categorySimilarities)
      .sort(([, a], [, b]) => b - a) as [StateCategory, number][];

    const [bestCategory, bestSimilarity] = sortedCategories[0];

    const nearestExamples = this._examples
      .filter(ex => ex.embedding)
      .map(ex => ({
        category: ex.category,
        description: ex.description,
        similarity: cosineSimilarity(stateEmbedding, ex.embedding!)
      }))
      .sort((a, b) => b.similarity - a.similarity)
      .slice(0, 3);

    let finalCategory = bestCategory;
    let score = bestSimilarity;

    if (bestSimilarity < this._config.similarityThreshold) {
      if (ruleBasedResult.score > bestSimilarity) {
        finalCategory = ruleBasedResult.category;
        score = ruleBasedResult.score;
      }
    }

    const confidence = score > 0.8 ? 'high' : score > 0.6 ? 'medium' : 'low';

    return {
      category: finalCategory,
      confidence,
      score,
      alternatives: sortedCategories.slice(1, 3).map(([cat]) => cat),
      description: ruleBasedResult.description,
      suggestions: ruleBasedResult.suggestions,
      features: ruleBasedResult.features,
      categorySimilarities,
      nearestExamples,
      stateEmbedding
    };
  }

  async addExample(category: StateCategory, description: string): Promise<void> {
    const embedding = await this._embed(description);

    const example: LabeledExample = { category, description, embedding };
    this._examples.push(example);

    if (!this._categoryEmbeddings.has(category)) {
      this._categoryEmbeddings.set(category, []);
    }
    this._categoryEmbeddings.get(category)!.push(embedding);

    this._categoryCentroids.set(
      category,
      averageEmbeddings(this._categoryEmbeddings.get(category)!)
    );
  }

  private _generateStateDescription(
    position: Vector4D,
    velocity: number,
    trinityState: { activeAxis: TrinityAxis; weights: [number, number, number]; tension: number },
    betti: BettiProfile | null,
    coherence: number,
    isInsideHull: boolean
  ): string {
    const parts: string[] = [];

    const dist = Math.sqrt(position.reduce((s, v) => s + v*v, 0));
    parts.push(`Position at distance ${dist.toFixed(2)} from origin`);

    if (velocity < 0.01) parts.push('nearly stationary');
    else if (velocity < 0.1) parts.push('slow movement');
    else if (velocity < 0.5) parts.push('moderate velocity');
    else parts.push('high velocity motion');

    const [a, b, g] = trinityState.weights;
    if (a > 0.6) parts.push('dominant alpha axis');
    else if (b > 0.6) parts.push('dominant beta axis');
    else if (g > 0.6) parts.push('dominant gamma axis');
    else parts.push('balanced multi-axis state');

    parts.push(`tension at ${(trinityState.tension * 100).toFixed(0)}%`);
    parts.push(`coherence at ${(coherence * 100).toFixed(0)}%`);

    if (betti) {
      if (betti.beta2 > 0) parts.push(`${betti.beta2} topological void(s)`);
      if (betti.beta1 > 2) parts.push('complex cyclic structure');
    }

    if (!isInsideHull) parts.push('outside convex hull (invalid)');

    return parts.join(', ');
  }

  private async _embed(text: string): Promise<Embedding> {
    if (this._config.customEmbedder) {
      return this._config.customEmbedder(text);
    }

    if (this._config.provider === 'openai' && this._config.apiKey) {
      return this._embedOpenAI(text);
    }

    if (this._config.provider === 'cohere' && this._config.apiKey) {
      return this._embedCohere(text);
    }

    return this._pseudoEmbed(text);
  }

  private async _embedOpenAI(text: string): Promise<Embedding> {
    const response = await fetch(
      this._config.baseUrl || 'https://api.openai.com/v1/embeddings',
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${this._config.apiKey}`
        },
        body: JSON.stringify({
          model: this._config.model || 'text-embedding-ada-002',
          input: text
        })
      }
    );

    const data = await response.json();
    return data.data[0].embedding;
  }

  private async _embedCohere(text: string): Promise<Embedding> {
    const response = await fetch(
      this._config.baseUrl || 'https://api.cohere.ai/v1/embed',
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${this._config.apiKey}`
        },
        body: JSON.stringify({
          model: this._config.model || 'embed-english-v3.0',
          texts: [text],
          input_type: 'classification'
        })
      }
    );

    const data = await response.json();
    return data.embeddings[0];
  }

  private _pseudoEmbed(text: string): Embedding {
    const dim = this._config.dimension;
    const embedding = new Array(dim).fill(0);

    const words = text.toLowerCase().split(/\s+/);

    for (let i = 0; i < words.length; i++) {
      const word = words[i];
      for (let j = 0; j < word.length; j++) {
        const charCode = word.charCodeAt(j);
        const idx = (charCode * 31 + i * 17 + j * 7) % dim;
        embedding[idx] += 1 / (1 + j);
      }
    }

    const norm = Math.sqrt(embedding.reduce((s, v) => s + v*v, 0)) || 1;
    return embedding.map(v => v / norm);
  }

  setEmbedder(embedder: (text: string) => Promise<Embedding>): void {
    this._config.customEmbedder = embedder;
  }

  exportExamples(): LabeledExample[] {
    return [...this._examples];
  }

  async importExamples(examples: LabeledExample[]): Promise<void> {
    this._examples = examples;
    this._initialized = false;
    await this.initialize();
  }
}

export default EmbeddingClassifier;
