import assert from 'node:assert/strict';
import test from 'node:test';

import { buildPolytopalStateFromConstellation, createGeometricAuditBridge } from '../scripts/geometricAuditBridge.js';

test('buildPolytopalStateFromConstellation maps constellation fields into audit state', () => {
    const constellation = {
        progress: 0.42,
        transport: { playing: true, mode: 'stream' },
        orientation: {
            centroid: [0.1, 0.2, 0.3, 0.4],
            bridge: [0.5, 0.6, 0.7, 0.8],
            hopf: [1, 0, 0, 0]
        },
        nodes: [
            { orientation: [0.2, 0.1, 0, -0.1] },
            { orientation: [0.4, 0.3, 0.2, 0.1] }
        ],
        sequence: { bitDensity: 0.3, bitEntropy: 1.2 },
        spinor: { coherence: 0.9, braid: 0.4 },
        energy: { carrier: 0.8, grid: 0.6 }
    };

    const state = buildPolytopalStateFromConstellation(constellation);
    assert.equal(state.mode, 'constellation');
    assert.equal(state.vertices.length, 4);
    assert.deepEqual(state.vertices[0], [0.2, 0.1, 0, -0.1]);
    assert.deepEqual(state.vertices[2], [0.1, 0.2, 0.3, 0.4]);
    assert.deepEqual(state.quaternion, [0.5, 0.6, 0.7, 0.8]);
    assert.deepEqual(state.topologySignatures, [
        [0.3, 1.2],
        [0.9, 0.4],
        [0.8, 0.6]
    ]);
    assert.equal(state.metadata.progress, 0.42);
    assert.equal(state.metadata.transport.mode, 'stream');
});

test('createGeometricAuditBridge ingests constellation snapshots into evidence chain', () => {
    const bridge = createGeometricAuditBridge({ batchSize: 2 });
    const first = bridge.ingestConstellation({ nodes: [{ orientation: [0.1, 0.2] }] }, { timestamp: 10 });
    assert.ok(first.evidence.eventHash);
    assert.equal(first.evidence.eventType, 'CONSTELLATION_SNAPSHOT');

    const second = bridge.ingestConstellation({ nodes: [{ orientation: [0.3, 0.4] }] }, { timestamp: 20 });
    assert.ok(second.sealedBatch);
    assert.equal(bridge.verifyChainIntegrity(), true);
});
