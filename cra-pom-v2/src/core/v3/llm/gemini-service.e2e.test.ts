/**
 * PPP v3 End-to-End Tests with Gemini API
 *
 * These tests use real API calls to verify:
 * - Semantic embeddings work correctly
 * - Chat integration works
 * - Similarity scores are meaningful
 * - Full reasoning flow integrates properly
 *
 * Run with: npm run test:e2e
 */

import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { GeminiService, initializeGemini, resetGeminiService } from './gemini-service';

// API Key from environment variable or fallback for local testing
const GEMINI_API_KEY = process.env.GEMINI_API_KEY || 'AIzaSyDuB37AAz-NvNO54CKbgEeFj9CTOalyvgI';

describe('Gemini E2E Tests', () => {
  let gemini: GeminiService;

  beforeAll(() => {
    gemini = initializeGemini(GEMINI_API_KEY);
  });

  afterAll(() => {
    resetGeminiService();
  });

  describe('Embeddings', () => {
    it('should generate real embeddings', async () => {
      const result = await gemini.embed('hello world');

      expect(result.vector).toBeDefined();
      expect(result.vector.length).toBeGreaterThan(0);
      expect(result.dimension).toBe(result.vector.length);

      console.log(`Embedding dimension: ${result.dimension}`);
    }, 30000);

    it('should produce different embeddings for different texts', async () => {
      const [emb1, emb2] = await Promise.all([
        gemini.embed('the cat sat on the mat'),
        gemini.embed('quantum physics equations'),
      ]);

      // Check they're not identical
      let identical = true;
      for (let i = 0; i < Math.min(emb1.vector.length, emb2.vector.length); i++) {
        if (Math.abs(emb1.vector[i] - emb2.vector[i]) > 0.0001) {
          identical = false;
          break;
        }
      }

      expect(identical).toBe(false);
    }, 30000);

    it('should show high similarity for semantically similar texts', async () => {
      const [emb1, emb2, emb3] = await Promise.all([
        gemini.embed('dog'),
        gemini.embed('puppy'),
        gemini.embed('quantum mechanics'),
      ]);

      const cosineSim = (a: Float32Array, b: Float32Array): number => {
        let dot = 0, normA = 0, normB = 0;
        for (let i = 0; i < a.length; i++) {
          dot += a[i] * b[i];
          normA += a[i] * a[i];
          normB += b[i] * b[i];
        }
        return dot / (Math.sqrt(normA) * Math.sqrt(normB));
      };

      const dogPuppySim = cosineSim(emb1.vector, emb2.vector);
      const dogQuantumSim = cosineSim(emb1.vector, emb3.vector);

      console.log(`dog-puppy similarity: ${dogPuppySim.toFixed(4)}`);
      console.log(`dog-quantum similarity: ${dogQuantumSim.toFixed(4)}`);

      // Dog and puppy should be more similar than dog and quantum
      expect(dogPuppySim).toBeGreaterThan(dogQuantumSim);
    }, 30000);

    it('should batch embed multiple texts', async () => {
      const texts = ['apple', 'banana', 'orange', 'car'];
      const embeddings = await gemini.embedBatch(texts);

      expect(embeddings.length).toBe(4);
      embeddings.forEach((emb) => {
        expect(emb.vector.length).toBeGreaterThan(0);
      });
    }, 60000);
  });

  describe('Chat', () => {
    beforeAll(() => {
      gemini.clearChatHistory();
    });

    it('should respond to a simple message', async () => {
      const result = await gemini.chat('What is 2 + 2? Reply with just the number.');

      expect(result.content).toBeDefined();
      expect(result.content.length).toBeGreaterThan(0);
      expect(result.model).toBe('gemini-1.5-flash');

      console.log(`Chat response: ${result.content}`);
    }, 30000);

    it('should maintain conversation history', async () => {
      gemini.clearChatHistory();

      await gemini.chat('My favorite color is blue. Remember this.');
      const result = await gemini.chat('What is my favorite color?');

      expect(result.content.toLowerCase()).toContain('blue');
    }, 60000);

    it('should follow a system prompt', async () => {
      gemini.clearChatHistory();

      const result = await gemini.chat(
        'Hello!',
        'You are a pirate. Always respond like a pirate would.'
      );

      // Check for pirate-like language
      const pirateWords = ['arr', 'matey', 'ahoy', 'ye', 'aye', 'ship', 'sea', 'captain'];
      const hasPirateWord = pirateWords.some((word) =>
        result.content.toLowerCase().includes(word)
      );

      console.log(`Pirate response: ${result.content}`);
      expect(hasPirateWord).toBe(true);
    }, 30000);
  });

  describe('Reasoning Integration', () => {
    beforeAll(() => {
      gemini.clearChatHistory();
    });

    it('should perform structured reasoning', async () => {
      const result = await gemini.reason(
        'All mammals are warm-blooded. Dogs are mammals.',
        'Basic syllogistic logic'
      );

      expect(result.reasoning).toBeDefined();
      expect(result.conclusion).toBeDefined();
      expect(['high', 'medium', 'low']).toContain(result.confidence);

      console.log(`Reasoning: ${result.reasoning}`);
      console.log(`Conclusion: ${result.conclusion}`);
      console.log(`Confidence: ${result.confidence}`);

      // The conclusion should mention dogs being warm-blooded
      expect(
        result.conclusion.toLowerCase().includes('warm') ||
        result.conclusion.toLowerCase().includes('dog') ||
        result.reasoning.toLowerCase().includes('warm-blooded')
      ).toBe(true);
    }, 30000);

    it('should analyze semantic similarity', async () => {
      const result = await gemini.analyzeSimilarity('happy', 'joyful');

      expect(result.similar).toBe(true);
      expect(result.explanation).toBeDefined();
      expect(result.relationship).toBeDefined();

      console.log(`Similar: ${result.similar}`);
      console.log(`Explanation: ${result.explanation}`);
      console.log(`Relationship: ${result.relationship}`);
    }, 30000);

    it('should detect dissimilar concepts', async () => {
      const result = await gemini.analyzeSimilarity('bicycle', 'philosophy');

      console.log(`Similar: ${result.similar}`);
      console.log(`Explanation: ${result.explanation}`);
      console.log(`Relationship: ${result.relationship}`);

      // These should be recognized as not similar
      expect(result.similar).toBe(false);
    }, 30000);
  });

  describe('Full E2E Flow', () => {
    it('should complete a full reasoning workflow', async () => {
      gemini.clearChatHistory();

      // Step 1: Get embeddings for concepts
      console.log('\n=== Step 1: Embedding Concepts ===');
      const concepts = ['logic', 'reasoning', 'mathematics', 'creativity'];
      const embeddings = await gemini.embedBatch(concepts);

      console.log(`Embedded ${embeddings.length} concepts`);
      embeddings.forEach((emb, i) => {
        console.log(`  ${concepts[i]}: ${emb.dimension} dimensions`);
      });

      // Step 2: Find most similar pair
      console.log('\n=== Step 2: Finding Similar Concepts ===');
      const cosineSim = (a: Float32Array, b: Float32Array): number => {
        let dot = 0, normA = 0, normB = 0;
        for (let i = 0; i < a.length; i++) {
          dot += a[i] * b[i];
          normA += a[i] * a[i];
          normB += b[i] * b[i];
        }
        return dot / (Math.sqrt(normA) * Math.sqrt(normB));
      };

      let maxSim = -1;
      let mostSimilar: [string, string] = ['', ''];

      for (let i = 0; i < concepts.length; i++) {
        for (let j = i + 1; j < concepts.length; j++) {
          const sim = cosineSim(embeddings[i].vector, embeddings[j].vector);
          console.log(`  ${concepts[i]} <-> ${concepts[j]}: ${sim.toFixed(4)}`);
          if (sim > maxSim) {
            maxSim = sim;
            mostSimilar = [concepts[i], concepts[j]];
          }
        }
      }

      console.log(`Most similar: ${mostSimilar[0]} and ${mostSimilar[1]} (${maxSim.toFixed(4)})`);

      // Step 3: Use LLM to reason about the relationship
      console.log('\n=== Step 3: LLM Reasoning ===');
      const reasoning = await gemini.reason(
        `${mostSimilar[0]} and ${mostSimilar[1]} are semantically similar concepts.`,
        'Explain why these concepts are related and what they have in common.'
      );

      console.log(`Reasoning: ${reasoning.reasoning}`);
      console.log(`Conclusion: ${reasoning.conclusion}`);
      console.log(`Confidence: ${reasoning.confidence}`);

      // Step 4: Verify with similarity analysis
      console.log('\n=== Step 4: Verification ===');
      const analysis = await gemini.analyzeSimilarity(mostSimilar[0], mostSimilar[1]);

      console.log(`LLM confirms similar: ${analysis.similar}`);
      console.log(`Relationship type: ${analysis.relationship}`);

      // Assertions
      expect(embeddings.length).toBe(4);
      expect(maxSim).toBeGreaterThan(0);
      expect(reasoning.conclusion).toBeDefined();
      expect(analysis.explanation).toBeDefined();

      console.log('\n=== E2E Flow Complete ===\n');
    }, 120000);
  });
});
