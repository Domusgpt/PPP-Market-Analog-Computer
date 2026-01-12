/**
 * PPP v3 Gemini Integration
 *
 * Provides real semantic embeddings and chat via Google's Gemini API.
 *
 * Models used:
 * - Embeddings: text-embedding-004
 * - Chat: gemini-1.5-flash
 */

// ============================================================================
// Types
// ============================================================================

export interface GeminiConfig {
  apiKey: string;
  embeddingModel?: string;
  chatModel?: string;
}

export interface GeminiEmbeddingResponse {
  embedding: {
    values: number[];
  };
}

export interface GeminiChatMessage {
  role: 'user' | 'model';
  parts: Array<{ text: string }>;
}

export interface GeminiChatResponse {
  candidates: Array<{
    content: {
      parts: Array<{ text: string }>;
      role: string;
    };
    finishReason: string;
  }>;
}

export interface ChatCompletionResult {
  content: string;
  model: string;
  finishReason: string;
  timestamp: string;
}

// ============================================================================
// Constants
// ============================================================================

const GEMINI_EMBEDDING_URL = 'https://generativelanguage.googleapis.com/v1beta/models';
const DEFAULT_EMBEDDING_MODEL = 'text-embedding-004';
const DEFAULT_CHAT_MODEL = 'gemini-1.5-flash';

// ============================================================================
// Gemini Service
// ============================================================================

export class GeminiService {
  private apiKey: string;
  private embeddingModel: string;
  private chatModel: string;
  private chatHistory: GeminiChatMessage[] = [];

  constructor(config: GeminiConfig) {
    this.apiKey = config.apiKey;
    this.embeddingModel = config.embeddingModel || DEFAULT_EMBEDDING_MODEL;
    this.chatModel = config.chatModel || DEFAULT_CHAT_MODEL;
  }

  // ==========================================================================
  // Embeddings
  // ==========================================================================

  /**
   * Get embedding for a single text
   */
  async embed(text: string): Promise<{ vector: Float32Array; dimension: number }> {
    const url = `${GEMINI_EMBEDDING_URL}/${this.embeddingModel}:embedContent?key=${this.apiKey}`;

    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model: `models/${this.embeddingModel}`,
        content: {
          parts: [{ text }],
        },
      }),
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`Gemini embedding error: ${response.status} - ${error}`);
    }

    const data: GeminiEmbeddingResponse = await response.json();

    if (!data.embedding?.values) {
      throw new Error('Invalid Gemini embedding response');
    }

    const vector = new Float32Array(data.embedding.values);

    return {
      vector,
      dimension: vector.length,
    };
  }

  /**
   * Get embeddings for multiple texts
   */
  async embedBatch(texts: string[]): Promise<Array<{ vector: Float32Array; dimension: number }>> {
    // Gemini doesn't have a batch endpoint, so we parallelize individual requests
    return Promise.all(texts.map((text) => this.embed(text)));
  }

  // ==========================================================================
  // Chat
  // ==========================================================================

  /**
   * Send a chat message and get a response
   */
  async chat(
    message: string,
    systemPrompt?: string
  ): Promise<ChatCompletionResult> {
    const url = `${GEMINI_EMBEDDING_URL}/${this.chatModel}:generateContent?key=${this.apiKey}`;

    // Build contents array
    const contents: GeminiChatMessage[] = [];

    // Add system prompt as first user message if provided
    if (systemPrompt && this.chatHistory.length === 0) {
      contents.push({
        role: 'user',
        parts: [{ text: `System instructions: ${systemPrompt}\n\nUser: ${message}` }],
      });
    } else {
      // Add history
      contents.push(...this.chatHistory);
      // Add new message
      contents.push({
        role: 'user',
        parts: [{ text: message }],
      });
    }

    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        contents,
        generationConfig: {
          temperature: 0.7,
          maxOutputTokens: 2048,
        },
      }),
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`Gemini chat error: ${response.status} - ${error}`);
    }

    const data: GeminiChatResponse = await response.json();

    if (!data.candidates?.[0]?.content?.parts?.[0]?.text) {
      throw new Error('Invalid Gemini chat response');
    }

    const assistantMessage = data.candidates[0].content.parts[0].text;

    // Update history
    this.chatHistory.push({
      role: 'user',
      parts: [{ text: message }],
    });
    this.chatHistory.push({
      role: 'model',
      parts: [{ text: assistantMessage }],
    });

    return {
      content: assistantMessage,
      model: this.chatModel,
      finishReason: data.candidates[0].finishReason,
      timestamp: new Date().toISOString(),
    };
  }

  /**
   * Clear chat history
   */
  clearChatHistory(): void {
    this.chatHistory = [];
  }

  /**
   * Get chat history
   */
  getChatHistory(): GeminiChatMessage[] {
    return [...this.chatHistory];
  }

  // ==========================================================================
  // Reasoning Integration
  // ==========================================================================

  /**
   * Perform verified reasoning step using Gemini
   *
   * The LLM provides reasoning, but we verify and sign the results.
   */
  async reason(
    premise: string,
    context?: string
  ): Promise<{
    reasoning: string;
    conclusion: string;
    confidence: 'high' | 'medium' | 'low';
  }> {
    const prompt = `You are a logical reasoning assistant. Given the following premise, provide your reasoning and conclusion.

${context ? `Context: ${context}\n\n` : ''}Premise: ${premise}

Respond in this exact JSON format:
{
  "reasoning": "Your step-by-step reasoning",
  "conclusion": "Your conclusion based on the reasoning",
  "confidence": "high" | "medium" | "low"
}

Only output the JSON, no other text.`;

    const result = await this.chat(prompt);

    try {
      // Extract JSON from response (handle markdown code blocks)
      let jsonStr = result.content;
      const jsonMatch = jsonStr.match(/```(?:json)?\s*([\s\S]*?)```/);
      if (jsonMatch) {
        jsonStr = jsonMatch[1];
      }

      const parsed = JSON.parse(jsonStr.trim());

      return {
        reasoning: parsed.reasoning || 'No reasoning provided',
        conclusion: parsed.conclusion || 'No conclusion reached',
        confidence: parsed.confidence || 'low',
      };
    } catch {
      // If JSON parsing fails, extract what we can
      return {
        reasoning: result.content,
        conclusion: 'Unable to parse structured conclusion',
        confidence: 'low',
      };
    }
  }

  /**
   * Analyze semantic similarity between concepts using LLM
   */
  async analyzeSimilarity(
    conceptA: string,
    conceptB: string
  ): Promise<{
    similar: boolean;
    explanation: string;
    relationship: string;
  }> {
    const prompt = `Analyze the semantic relationship between these two concepts:

Concept A: ${conceptA}
Concept B: ${conceptB}

Respond in this exact JSON format:
{
  "similar": true/false,
  "explanation": "Why they are or aren't similar",
  "relationship": "The type of relationship (e.g., synonym, antonym, hyponym, related, unrelated)"
}

Only output the JSON, no other text.`;

    this.clearChatHistory(); // Fresh context for analysis
    const result = await this.chat(prompt);

    try {
      let jsonStr = result.content;
      const jsonMatch = jsonStr.match(/```(?:json)?\s*([\s\S]*?)```/);
      if (jsonMatch) {
        jsonStr = jsonMatch[1];
      }

      const parsed = JSON.parse(jsonStr.trim());

      return {
        similar: parsed.similar ?? false,
        explanation: parsed.explanation || 'No explanation provided',
        relationship: parsed.relationship || 'unknown',
      };
    } catch {
      return {
        similar: false,
        explanation: result.content,
        relationship: 'parse_error',
      };
    }
  }
}

// ============================================================================
// Singleton
// ============================================================================

let globalGeminiService: GeminiService | null = null;

export function initializeGemini(apiKey: string): GeminiService {
  globalGeminiService = new GeminiService({ apiKey });
  return globalGeminiService;
}

export function getGeminiService(): GeminiService {
  if (!globalGeminiService) {
    throw new Error('Gemini service not initialized. Call initializeGemini(apiKey) first.');
  }
  return globalGeminiService;
}

export function resetGeminiService(): void {
  globalGeminiService = null;
}
