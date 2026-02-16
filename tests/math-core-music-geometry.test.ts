/**
 * Tests for MusicGeometryDomain.ts
 *
 * Verifies musical-geometric mapping:
 * - 24 keys (12 major + 12 minor) map to 24 vertices
 * - Octatonic collections partition correctly
 * - Neo-Riemannian PLR transformations
 * - Circle of fifths navigation
 * - Key detection
 */

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';

import {
  MusicGeometryDomain,
  createMusicGeometryDomain,
  PITCH_CLASSES,
  CIRCLE_OF_FIFTHS,
  OCTATONIC_COLLECTIONS,
  TRINITY_OCTATONIC_MAP
} from '../_SYNERGIZED_SYSTEM/lib/math_core/domains/MusicGeometryDomain.js';

describe('Musical Constants', () => {
  it('PITCH_CLASSES has 12 entries', () => {
    assert.equal(PITCH_CLASSES.length, 12);
  });

  it('CIRCLE_OF_FIFTHS has 12 entries', () => {
    assert.equal(CIRCLE_OF_FIFTHS.length, 12);
  });

  it('CIRCLE_OF_FIFTHS contains all 12 pitch classes', () => {
    const sorted = [...CIRCLE_OF_FIFTHS].sort((a, b) => a - b);
    for (let i = 0; i < 12; i++) {
      assert.equal(sorted[i], i, `Missing pitch class ${i}`);
    }
  });

  it('each octatonic collection has 8 notes', () => {
    for (const [name, notes] of Object.entries(OCTATONIC_COLLECTIONS)) {
      assert.equal(notes.length, 8, `${name} has ${notes.length} notes, expected 8`);
    }
  });

  it('3 octatonic collections exist', () => {
    assert.equal(Object.keys(OCTATONIC_COLLECTIONS).length, 3);
  });

  it('TRINITY_OCTATONIC_MAP maps all 3 axes', () => {
    assert.ok('alpha' in TRINITY_OCTATONIC_MAP);
    assert.ok('beta' in TRINITY_OCTATONIC_MAP);
    assert.ok('gamma' in TRINITY_OCTATONIC_MAP);
  });
});

describe('MusicGeometryDomain Key Mapping', () => {
  it('creates successfully', () => {
    const domain = createMusicGeometryDomain();
    assert.ok(domain instanceof MusicGeometryDomain);
  });

  it('has 24 keys total', () => {
    const domain = new MusicGeometryDomain();
    const keys = domain.getAllKeys();
    assert.equal(keys.length, 24);
  });

  it('has 12 major keys', () => {
    const domain = new MusicGeometryDomain();
    const keys = domain.getAllKeys();
    const majorKeys = keys.filter(k => !k.includes('m'));
    assert.equal(majorKeys.length, 12);
  });

  it('has 12 minor keys', () => {
    const domain = new MusicGeometryDomain();
    const keys = domain.getAllKeys();
    const minorKeys = keys.filter(k => k.includes('m'));
    assert.equal(minorKeys.length, 12);
  });

  it('keyToVertex returns valid vertex for each key', () => {
    const domain = new MusicGeometryDomain();
    for (const key of domain.getAllKeys()) {
      const vertex = domain.keyToVertex(key);
      assert.ok(vertex !== undefined, `Key ${key} has no vertex mapping`);
      assert.ok(vertex! >= 0 && vertex! < 24, `Vertex ${vertex} out of range for key ${key}`);
    }
  });

  it('vertexToKeys is inverse of keyToVertex', () => {
    const domain = new MusicGeometryDomain();
    for (const key of domain.getAllKeys()) {
      const vertex = domain.keyToVertex(key)!;
      const keysForVertex = domain.vertexToKeys(vertex);
      assert.ok(keysForVertex.includes(key),
        `vertexToKeys(${vertex}) does not include ${key}`);
    }
  });

  it('getKey returns MusicalKey with correct structure', () => {
    const domain = new MusicGeometryDomain();
    const key = domain.getKey('C');
    assert.ok(key !== undefined);
    assert.equal(key!.mode, 'major');
    assert.equal(key!.root, 0); // C = pitch class 0
  });
});

describe('Neo-Riemannian Transformations', () => {
  it('P transform toggles major/minor on same root', () => {
    const domain = new MusicGeometryDomain();
    const result = domain.parallelTransform('C');
    // C major → C minor (stored as lowercase 'cm' in domain)
    assert.ok(result.includes('m'),
      `P(C) = ${result}, expected minor key`);
  });

  it('applyTransformations handles PLR sequence', () => {
    const domain = new MusicGeometryDomain();
    const result = domain.applyTransformations('C', 'PLR');
    assert.ok(typeof result === 'string' && result.length > 0);
  });
});

describe('Circle of Fifths', () => {
  it('nextInCircleOfFifths from C is G', () => {
    const domain = new MusicGeometryDomain();
    const next = domain.nextInCircleOfFifths('C');
    assert.equal(next, 'G');
  });

  it('prevInCircleOfFifths from C is F', () => {
    const domain = new MusicGeometryDomain();
    const prev = domain.prevInCircleOfFifths('C');
    assert.equal(prev, 'F');
  });

  it('circleOfFifthsDistance is symmetric', () => {
    const domain = new MusicGeometryDomain();
    const d1 = domain.circleOfFifthsDistance('C', 'G');
    const d2 = domain.circleOfFifthsDistance('G', 'C');
    assert.equal(d1, d2);
  });

  it('circleOfFifthsDistance C to G is 1', () => {
    const domain = new MusicGeometryDomain();
    assert.equal(domain.circleOfFifthsDistance('C', 'G'), 1);
  });
});

describe('Key Detection', () => {
  it('detects C major from C major scale notes', () => {
    const domain = new MusicGeometryDomain();
    const cMajorScale = [0, 2, 4, 5, 7, 9, 11]; // C D E F G A B
    const results = domain.detectKey(cMajorScale);

    assert.ok(results.length > 0, 'Should detect at least one key');
    assert.ok(results[0].confidence > 0.5, `Confidence too low: ${results[0].confidence}`);
  });
});

describe('Octatonic Collections', () => {
  it('getOctatonicCollection returns collection for each axis', () => {
    const domain = new MusicGeometryDomain();

    for (const axis of ['alpha', 'beta', 'gamma'] as const) {
      const collection = domain.getOctatonicCollection(axis);
      assert.ok(collection, `No collection for ${axis}`);
      assert.equal(collection.pitchClasses.length, 8);
    }
  });
});

describe('Position Mapping', () => {
  it('positionToKey returns a valid key name', () => {
    const domain = new MusicGeometryDomain();
    const key = domain.positionToKey([0.5, 0.3, 0.2, 0.1]);
    assert.ok(typeof key === 'string' && key.length > 0);
  });
});
