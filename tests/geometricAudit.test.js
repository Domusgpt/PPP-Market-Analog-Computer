import assert from 'node:assert/strict';
import test from 'node:test';

import {
    appendGeometricEvidence,
    buildMerkleProof,
    buildMerkleRoot,
    canonicalizePolytopalState,
    computeConstellationHash,
    computePolytopalFingerprint,
    createGeometricEvidence,
    verifyMerkleProof,
    stableStringify
} from '../scripts/geometricAudit.js';

test('canonicalization sorts vertices by centroid distance and normalizes topology', () => {
    const state = {
        vertices: [
            [2, 0, 0],
            [0, 0, 0],
            [1, 1, 0]
        ],
        topologySignatures: [
            [1.2, 4],
            [0.9, 3]
        ],
        quaternion: [0, 0, 0.707, 0.707],
        metadata: { region: 'licensed' },
        mode: 'audit'
    };

    const canonical = canonicalizePolytopalState(state);
    assert.deepEqual(canonical.vertices, [
        [1, 1, 0],
        [0, 0, 0],
        [2, 0, 0]
    ]);
    assert.deepEqual(canonical.topologySignatures, [
        [0.9, 3],
        [1.2, 4]
    ]);
    assert.deepEqual(canonical.quaternion, [0, 0, 0.707, 0.707]);
    assert.equal(canonical.mode, 'audit');
    assert.equal(canonical.metadata.region, 'licensed');
});

test('fingerprints derive hashes from canonicalized state', () => {
    const state = {
        vertices: [
            [0, 0, 0],
            [1, 0, 0]
        ],
        topologySignatures: [[0.5, 1.0]],
        quaternion: [1, 0, 0, 0]
    };

    const fingerprint = computePolytopalFingerprint(state);
    assert.match(fingerprint.constellation, /^[a-f0-9]{64}$/);
    assert.match(fingerprint.topology, /^[a-f0-9]{64}$/);
    assert.match(fingerprint.quaternion, /^[a-f0-9]{64}$/);

    const reordered = {
        vertices: state.vertices.slice().reverse(),
        topologySignatures: state.topologySignatures.slice().reverse(),
        quaternion: state.quaternion.slice()
    };
    const reorderedFingerprint = computePolytopalFingerprint(reordered);
    assert.equal(fingerprint.constellation, reorderedFingerprint.constellation);
    assert.equal(fingerprint.topology, reorderedFingerprint.topology);
    assert.equal(fingerprint.quaternion, reorderedFingerprint.quaternion);
});

test('geometric evidence links hashes and preserves canonical payloads', () => {
    const chain = appendGeometricEvidence([], {
        eventType: 'GEOMETRIC_STATE',
        polytopalState: { vertices: [[0, 0], [1, 0]] },
        metadata: { policy: 'beta' },
        timestamp: 1_700_000_000_000
    });
    const extendedChain = appendGeometricEvidence(chain, {
        eventType: 'COHERENCE_CHECK',
        polytopalState: { vertices: [[0, 0], [0, 1]] },
        metadata: { policy: 'beta', result: 'ok' },
        timestamp: 1_700_000_000_500
    });

    assert.equal(chain[0].previousHash, null);
    assert.equal(extendedChain[1].previousHash, chain[0].eventHash);
    assert.equal(typeof extendedChain[1].eventHash, 'string');
    assert.equal(extendedChain[0].polytopalFingerprint.constellation.length, 64);
    assert.equal(extendedChain[1].polytopalFingerprint.constellation.length, 64);
    assert.deepEqual(extendedChain[0].payload.vertices, [[0, 0], [1, 0]]);
});

test('merkle proofs verify event inclusion', () => {
    const events = [
        createGeometricEvidence({ polytopalState: { vertices: [[0]] }, timestamp: 1000 }),
        createGeometricEvidence({ polytopalState: { vertices: [[1]] }, timestamp: 2000 }),
        createGeometricEvidence({ polytopalState: { vertices: [[2]] }, timestamp: 3000 }),
        createGeometricEvidence({ polytopalState: { vertices: [[3]] }, timestamp: 4000 })
    ];
    const hashes = events.map((event) => event.eventHash);
    const { proof, root } = buildMerkleProof(hashes, 2);

    assert.equal(proof.length, 2);
    assert.equal(root, buildMerkleRoot(hashes));
    assert.equal(verifyMerkleProof(hashes[2], proof, root), true);
});

test('stableStringify produces deterministic output for nested objects', () => {
    const input = { b: 2, a: { d: 4, c: 3 } };
    const first = stableStringify(input);
    const second = stableStringify({ a: { c: 3, d: 4 }, b: 2 });

    assert.equal(first, second);
    assert.equal(first, '{"a":{"c":3,"d":4},"b":2}');
});
