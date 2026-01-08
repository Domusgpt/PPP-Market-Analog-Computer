import assert from 'node:assert/strict';
import test from 'node:test';

import { buildMerkleRoot, verifyMerkleProof } from '../scripts/geometricAudit.js';
import { createGeometricAuditSession } from '../scripts/geometricAuditPipeline.js';

test('seals batches at configured size and produces verifiable proofs', () => {
    const session = createGeometricAuditSession({ batchSize: 2 });
    const first = session.appendTelemetry({ polytopalState: { vertices: [[0, 0]] }, timestamp: 1000 });
    assert.equal(first.sealedBatch, null);
    const second = session.appendTelemetry({ polytopalState: { vertices: [[1, 0]] }, timestamp: 2000 });
    assert.ok(second.sealedBatch);
    assert.equal(session.verifyChainIntegrity(), true);

    const { batches } = session.getState();
    assert.equal(batches.length, 1);
    const batch = batches[0];
    const hashes = batch.events.map((event) => event.eventHash);
    assert.equal(batch.root, buildMerkleRoot(hashes));
    batch.proofs.forEach(({ hash, proof, root }) => {
        assert.equal(verifyMerkleProof(hash, proof, root), true);
    });
});

test('allows manual sealing, anchoring, and integrity verification', () => {
    const session = createGeometricAuditSession({ batchSize: 5 });
    session.appendTelemetry({ eventType: 'GEOMETRIC_STATE', polytopalState: { vertices: [[0, 1]] }, timestamp: 10 });
    session.appendTelemetry({ eventType: 'COHERENCE_CHECK', polytopalState: { vertices: [[1, 1]] }, timestamp: 20 });

    const batch = session.sealPendingBatch();
    assert.ok(batch);
    assert.equal(session.verifyBatchIntegrity(0), true);

    const anchored = session.anchorBatch(0, 'trace-anchor-hash', 50);
    assert.ok(anchored.anchored);
    assert.equal(anchored.anchored.hash, 'trace-anchor-hash');
    assert.equal(anchored.anchored.anchoredAt, 50);
    assert.equal(session.verifyChainIntegrity(), true);
});

test('summarizes batches with windows, event mix, and fingerprint coverage', () => {
    const session = createGeometricAuditSession({ batchSize: 3 });
    session.appendTelemetry({
        eventType: 'GEOMETRIC_STATE',
        polytopalState: { vertices: [[0, 0]], topologySignatures: [[0.1, 1.2]] },
        timestamp: 1
    });
    session.appendTelemetry({
        eventType: 'COHERENCE_CHECK',
        polytopalState: { vertices: [[1, 0]] },
        timestamp: 3
    });
    const sealed = session.sealPendingBatch();

    assert.equal(sealed.summary.count, 2);
    assert.equal(sealed.summary.eventTypes.GEOMETRIC_STATE, 1);
    assert.equal(sealed.summary.eventTypes.COHERENCE_CHECK, 1);
    assert.equal(sealed.summary.window.start, 1);
    assert.equal(sealed.summary.window.end, 3);
    assert.equal(sealed.summary.missingFingerprints, 0);
});
