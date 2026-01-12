/**
 * PPP v3 Embeddings Module
 *
 * Provides semantic grounding for concepts through embeddings.
 *
 * HONEST STATUS:
 * - With external API or local model: REAL semantic embeddings
 * - Without: Deterministic fallback (NOT semantic)
 *
 * The system always reports which mode it's using.
 */

export {
  EmbeddingService,
  getEmbeddingService,
  resetEmbeddingService,
  configureEmbeddings,
  type EmbeddingResult,
  type EmbeddingSource,
  type EmbeddingConfig,
  type SimilarityResult,
} from './embedding-service';

export {
  ConceptStore,
  getConceptStore,
  resetConceptStore,
  initializeWithBasicConcepts,
  LOGIC_CONCEPTS,
  REASONING_CONCEPTS,
  type StoredConcept,
  type ConceptRetrievalResult,
  type ConceptCompositionResult,
} from './concept-store';
