/**
 * PPP v3 Concept Store Tests
 */

import { describe, it, expect, beforeEach } from 'vitest';
import {
  ConceptStore,
  getConceptStore,
  resetConceptStore,
  initializeWithBasicConcepts,
  LOGIC_CONCEPTS,
  REASONING_CONCEPTS,
} from './concept-store';

describe('ConceptStore', () => {
  let store: ConceptStore;

  beforeEach(() => {
    store = new ConceptStore();
  });

  describe('addConcept', () => {
    it('should add a concept', async () => {
      const concept = await store.addConcept('democracy', 'A system of government by the people');
      expect(concept.name).toBe('democracy');
      expect(concept.description).toBe('A system of government by the people');
      expect(concept.id).toBe('democracy');
    });

    it('should generate embedding for concept', async () => {
      const concept = await store.addConcept('freedom', 'The state of being free');
      expect(concept.embedding).toBeDefined();
      expect(concept.embedding.length).toBeGreaterThan(0);
    });

    it('should track embedding source', async () => {
      const concept = await store.addConcept('test', 'A test concept');
      expect(concept.embeddingSource).toBe('deterministic_fallback');
    });

    it('should store metadata', async () => {
      const concept = await store.addConcept('tagged', 'A tagged concept', {
        category: 'test',
        priority: 1,
      });
      expect(concept.metadata.category).toBe('test');
      expect(concept.metadata.priority).toBe(1);
    });
  });

  describe('getConcept', () => {
    it('should retrieve concept by ID', async () => {
      await store.addConcept('justice', 'Fair treatment');
      const retrieved = store.getConcept('justice');
      expect(retrieved).toBeDefined();
      expect(retrieved!.name).toBe('justice');
    });

    it('should return undefined for non-existent concept', () => {
      const retrieved = store.getConcept('nonexistent');
      expect(retrieved).toBeUndefined();
    });

    it('should retrieve concept by name', async () => {
      await store.addConcept('equality', 'Equal rights');
      const retrieved = store.getConceptByName('equality');
      expect(retrieved).toBeDefined();
      expect(retrieved!.name).toBe('equality');
    });
  });

  describe('hasConcept', () => {
    it('should return true for existing concept', async () => {
      await store.addConcept('liberty', 'Freedom from oppression');
      expect(store.hasConcept('liberty')).toBe(true);
    });

    it('should return false for non-existent concept', () => {
      expect(store.hasConcept('nonexistent')).toBe(false);
    });
  });

  describe('retrieve', () => {
    beforeEach(async () => {
      await store.addConcept('dog', 'A domesticated canine');
      await store.addConcept('cat', 'A domesticated feline');
      await store.addConcept('bird', 'A feathered flying animal');
      await store.addConcept('fish', 'An aquatic animal with gills');
    });

    it('should retrieve concepts by similarity', async () => {
      const result = await store.retrieve('pet animal', 3);
      expect(result.results.length).toBeLessThanOrEqual(3);
      expect(result.results[0].similarity).toBeGreaterThan(0);
    });

    it('should return results sorted by similarity', async () => {
      const result = await store.retrieve('animal', 4);
      for (let i = 0; i < result.results.length - 1; i++) {
        expect(result.results[i].similarity).toBeGreaterThanOrEqual(
          result.results[i + 1].similarity
        );
      }
    });

    it('should include query in result', async () => {
      const result = await store.retrieve('cute pet', 2);
      expect(result.query).toBe('cute pet');
    });

    it('should indicate semantic status', async () => {
      const result = await store.retrieve('test', 1);
      expect(result.semanticallyMeaningful).toBe(false); // Using fallback
    });
  });

  describe('compose', () => {
    beforeEach(async () => {
      await store.addConcept('king', 'A male monarch');
      await store.addConcept('queen', 'A female monarch');
      await store.addConcept('man', 'An adult male human');
      await store.addConcept('woman', 'An adult female human');
    });

    it('should compose vectors with add operation', async () => {
      const result = await store.compose([
        { concept: 'king', operation: 'add' },
        { concept: 'woman', operation: 'add' },
      ]);
      expect(result.vector).toBeDefined();
      expect(result.components.length).toBe(2);
    });

    it('should compose vectors with subtract operation', async () => {
      const result = await store.compose([
        { concept: 'king', operation: 'add' },
        { concept: 'man', operation: 'subtract' },
        { concept: 'woman', operation: 'add' },
      ]);
      expect(result.components).toHaveLength(3);
      expect(result.nearest).toBeDefined();
    });

    it('should support weights', async () => {
      const result = await store.compose([
        { concept: 'king', operation: 'add', weight: 2.0 },
        { concept: 'man', operation: 'subtract', weight: 0.5 },
      ]);
      expect(result.components[0].weight).toBe(2.0);
      expect(result.components[1].weight).toBe(0.5);
    });

    it('should embed unknown concepts on the fly', async () => {
      const result = await store.compose([
        { concept: 'unknown_concept', operation: 'add' },
      ]);
      expect(result.vector).toBeDefined();
    });
  });

  describe('size and getAllConcepts', () => {
    it('should track concept count', async () => {
      expect(store.size).toBe(0);
      await store.addConcept('a', 'First');
      expect(store.size).toBe(1);
      await store.addConcept('b', 'Second');
      expect(store.size).toBe(2);
    });

    it('should return all concepts', async () => {
      await store.addConcept('x', 'X');
      await store.addConcept('y', 'Y');
      const all = store.getAllConcepts();
      expect(all.length).toBe(2);
    });
  });

  describe('grounding status', () => {
    it('should report grounding status', async () => {
      await store.addConcept('test1', 'Test 1');
      await store.addConcept('test2', 'Test 2');

      const status = store.getGroundingStatus();
      expect(status.conceptCount).toBe(2);
      expect(status.fallbackCount).toBe(2); // All using fallback
      expect(status.percentSemantic).toBe(0);
    });

    it('should report empty store status', () => {
      const status = store.getGroundingStatus();
      expect(status.conceptCount).toBe(0);
      expect(status.percentSemantic).toBe(0);
    });
  });

  describe('export and import', () => {
    it('should export store', async () => {
      await store.addConcept('export-test', 'Test for export');
      const exported = store.export();
      expect(exported.version).toBe('3.0.0');
      expect(exported.concepts.length).toBe(1);
      expect(exported.concepts[0].name).toBe('export-test');
    });

    it('should import store', async () => {
      await store.addConcept('original', 'Original concept');
      const exported = store.export();

      const newStore = new ConceptStore();
      newStore.import(exported);
      expect(newStore.size).toBe(1);
      expect(newStore.getConcept('original')).toBeDefined();
    });
  });

  describe('clear', () => {
    it('should clear all concepts', async () => {
      await store.addConcept('a', 'A');
      await store.addConcept('b', 'B');
      expect(store.size).toBe(2);

      store.clear();
      expect(store.size).toBe(0);
    });
  });
});

describe('initializeWithBasicConcepts', () => {
  it('should load logic and reasoning concepts', async () => {
    const store = new ConceptStore();
    await initializeWithBasicConcepts(store);

    const expectedCount = LOGIC_CONCEPTS.length + REASONING_CONCEPTS.length;
    expect(store.size).toBe(expectedCount);
  });

  it('should include specific logic concepts', async () => {
    const store = new ConceptStore();
    await initializeWithBasicConcepts(store);

    expect(store.hasConcept('true')).toBe(true);
    expect(store.hasConcept('false')).toBe(true);
    expect(store.hasConcept('and')).toBe(true);
    expect(store.hasConcept('or')).toBe(true);
  });

  it('should include specific reasoning concepts', async () => {
    const store = new ConceptStore();
    await initializeWithBasicConcepts(store);

    expect(store.hasConcept('premise')).toBe(true);
    expect(store.hasConcept('conclusion')).toBe(true);
    expect(store.hasConcept('inference')).toBe(true);
  });
});

describe('getConceptStore singleton', () => {
  beforeEach(() => {
    resetConceptStore();
  });

  it('should return same instance', () => {
    const store1 = getConceptStore();
    const store2 = getConceptStore();
    expect(store1).toBe(store2);
  });

  it('should reset properly', () => {
    const store1 = getConceptStore();
    resetConceptStore();
    const store2 = getConceptStore();
    expect(store1).not.toBe(store2);
  });
});
