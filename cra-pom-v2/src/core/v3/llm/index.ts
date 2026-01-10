/**
 * PPP v3 LLM Integration
 */

// Gemini (Google) - Embeddings + Chat
export {
  GeminiService,
  initializeGemini,
  getGeminiService,
  resetGeminiService,
  type GeminiConfig,
  type ChatCompletionResult,
} from './gemini-service';

// Voyage AI (Anthropic) - High-quality Embeddings
export {
  VoyageService,
  initializeVoyage,
  getVoyageService,
  resetVoyageService,
  type VoyageConfig,
  type VoyageModel,
  type VoyageInputType,
  type VoyageEmbeddingResult,
} from './voyage-service';
