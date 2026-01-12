/**
 * PPP v3 Voyage AI End-to-End Tests
 *
 * Tests real semantic embeddings via Voyage AI API.
 *
 * Run with: npm run test:e2e:voyage
 */

import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { VoyageService, initializeVoyage, resetVoyageService } from './voyage-service';

// API Key from environment variable
const VOYAGE_API_KEY = process.env.VOYAGE_API_KEY || '';

// Skip tests if no API key
const describeWithKey = VOYAGE_API_KEY ? describe : describe.skip;

describeWithKey('Voyage AI E2E Tests', () => {
  let voyage: VoyageService;

  beforeAll(() => {
    voyage = initializeVoyage(VOYAGE_API_KEY);
  });

  afterAll(() => {
    resetVoyageService();
  });

  describe('Embeddings', () => {
    it('should generate embeddings', async () => {
      const result = await voyage.embed('Hello, world!');

      expect(result.vector).toBeDefined();
      expect(result.vector.length).toBe(1024); // Default dimension
      expect(result.model).toContain('voyage');
      expect(result.tokens).toBeGreaterThan(0);

      console.log(`Embedding dimension: ${result.dimension}`);
      console.log(`Model: ${result.model}`);
      console.log(`Tokens used: ${result.tokens}`);
    }, 30000);

    it('should batch embed multiple texts', async () => {
      const texts = ['apple', 'banana', 'orange', 'computer'];
      const results = await voyage.embedBatch(texts);

      expect(results.length).toBe(4);
      results.forEach((result, i) => {
        expect(result.vector.length).toBe(1024);
        console.log(`"${texts[i]}" - ${result.tokens} tokens`);
      });
    }, 30000);

    it('should differentiate query vs document embeddings', async () => {
      const text = 'machine learning algorithms';

      const queryEmb = await voyage.embedQuery(text);
      const docEmb = await voyage.embedDocument(text);

      expect(queryEmb.inputType).toBe('query');
      expect(docEmb.inputType).toBe('document');

      // They should be similar but not identical
      const similarity = cosineSimilarity(queryEmb.vector, docEmb.vector);
      console.log(`Query vs Document similarity: ${similarity.toFixed(4)}`);

      expect(similarity).toBeGreaterThan(0.9); // Should be very similar
      expect(similarity).toBeLessThan(1.0); // But not identical
    }, 30000);
  });

  describe('Semantic Similarity', () => {
    it('should show high similarity for related concepts', async () => {
      const result = await voyage.similarity('dog', 'puppy');

      console.log(`dog <-> puppy: ${result.similarity.toFixed(4)}`);
      expect(result.similarity).toBeGreaterThan(0.7);
    }, 30000);

    it('should show low similarity for unrelated concepts', async () => {
      const result = await voyage.similarity('banana', 'quantum physics');

      console.log(`banana <-> quantum physics: ${result.similarity.toFixed(4)}`);
      expect(result.similarity).toBeLessThan(0.5);
    }, 30000);

    it('should rank similar concepts higher', async () => {
      const [catDog, catCar, catQuantum] = await Promise.all([
        voyage.similarity('cat', 'dog'),
        voyage.similarity('cat', 'car'),
        voyage.similarity('cat', 'quantum mechanics'),
      ]);

      console.log(`cat <-> dog: ${catDog.similarity.toFixed(4)}`);
      console.log(`cat <-> car: ${catCar.similarity.toFixed(4)}`);
      console.log(`cat <-> quantum: ${catQuantum.similarity.toFixed(4)}`);

      // Cat should be most similar to dog
      expect(catDog.similarity).toBeGreaterThan(catCar.similarity);
      expect(catDog.similarity).toBeGreaterThan(catQuantum.similarity);
    }, 60000);
  });

  describe('Retrieval', () => {
    it('should find most similar texts', async () => {
      const query = 'programming language';
      const candidates = [
        'Python is a popular programming language',
        'The weather is nice today',
        'JavaScript runs in browsers',
        'I like pizza',
        'Rust is a systems programming language',
        'The cat sat on the mat',
      ];

      const results = await voyage.findMostSimilar(query, candidates, 3);

      console.log('\nQuery: "programming language"');
      console.log('Top 3 matches:');
      results.forEach((r, i) => {
        console.log(`  ${i + 1}. "${r.text}" (${r.similarity.toFixed(4)})`);
      });

      // Programming-related texts should rank highest
      const topTexts = results.map((r) => r.text.toLowerCase());
      const hasProgLang = topTexts.some(
        (t) => t.includes('programming') || t.includes('python') || t.includes('javascript') || t.includes('rust')
      );

      expect(hasProgLang).toBe(true);
    }, 60000);
  });

  describe('Full E2E Flow', () => {
    it('should complete semantic search workflow', async () => {
      console.log('\n=== Semantic Search Workflow ===\n');

      // Step 1: Index some documents
      const documents = [
        'Machine learning is a subset of artificial intelligence',
        'Neural networks are inspired by biological brains',
        'Deep learning uses multiple layers of neurons',
        'The stock market closed higher today',
        'Weather forecast predicts rain tomorrow',
      ];

      console.log('Step 1: Indexing documents...');
      const docEmbeddings = await voyage.embedDocuments(documents);
      console.log(`Indexed ${docEmbeddings.length} documents`);

      // Step 2: Search with a query
      const query = 'AI and machine intelligence';
      console.log(`\nStep 2: Searching for "${query}"...`);

      const queryEmb = await voyage.embedQuery(query);

      const scores = docEmbeddings.map((docEmb, i) => ({
        doc: documents[i],
        score: cosineSimilarity(queryEmb.vector, docEmb.vector),
      }));

      scores.sort((a, b) => b.score - a.score);

      console.log('\nResults:');
      scores.forEach((s, i) => {
        console.log(`  ${i + 1}. [${s.score.toFixed(4)}] ${s.doc}`);
      });

      // AI-related docs should rank highest
      expect(scores[0].doc).toContain('intelligence');
      expect(scores[0].score).toBeGreaterThan(0.5);

      console.log('\n=== Workflow Complete ===\n');
    }, 90000);
  });
});

// Helper function
function cosineSimilarity(a: Float32Array, b: Float32Array): number {
  let dot = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }

  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}
