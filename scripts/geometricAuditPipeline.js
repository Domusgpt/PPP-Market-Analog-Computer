import {
    buildMerkleProof,
    buildMerkleRoot,
    createGeometricEvidence,
    verifyEvidenceChain,
    verifyMerkleProof
} from './geometricAudit.js';

const isFiniteNumber = (value) => Number.isFinite(value);

const normalizeBatchSize = (value) => {
    if (Number.isInteger(value) && value > 0) {
        return value;
    }
    return 10;
};

const normalizeOptionalLimit = (value) => {
    if (Number.isInteger(value) && value > 0) {
        return value;
    }
    return null;
};

const summarizeBatch = (events = []) => {
    const summary = {
        count: events.length,
        eventTypes: {},
        window: { start: null, end: null },
        missingFingerprints: 0
    };
    events.forEach((event) => {
        const type = event.eventType || 'unknown';
        summary.eventTypes[type] = (summary.eventTypes[type] || 0) + 1;
        const timestamp = isFiniteNumber(event.timestamp) ? event.timestamp : null;
        if (timestamp !== null) {
            summary.window.start = summary.window.start === null ? timestamp : Math.min(summary.window.start, timestamp);
            summary.window.end = summary.window.end === null ? timestamp : Math.max(summary.window.end, timestamp);
        }
        const fingerprint = event.polytopalFingerprint || {};
        if (!fingerprint.constellation && !fingerprint.topology && !fingerprint.quaternion) {
            summary.missingFingerprints += 1;
        }
    });
    return summary;
};

const buildBatch = (events, index) => {
    const hashes = events.map((event) => event.eventHash);
    const root = buildMerkleRoot(hashes);
    const proofs = hashes.map((hash, proofIndex) => ({ hash, ...buildMerkleProof(hashes, proofIndex) }));
    return {
        index,
        root,
        events,
        proofs,
        summary: summarizeBatch(events),
        anchored: null
    };
};

export const createGeometricAuditSession = (options = {}) => {
    const batchSize = normalizeBatchSize(options.batchSize);
    const maxChainLength = normalizeOptionalLimit(options.maxChainLength);
    const maxBatches = normalizeOptionalLimit(options.maxBatches);
    const session = {
        chain: [],
        pending: [],
        batches: [],
        chainHeadHash: null,
        batchOffset: 0
    };

    const trimChain = () => {
        if (!maxChainLength) {
            return;
        }
        while (session.chain.length > maxChainLength) {
            const removed = session.chain.shift();
            if (removed && removed.eventHash) {
                session.chainHeadHash = removed.eventHash;
            }
        }
    };

    const trimBatches = () => {
        if (!maxBatches) {
            return;
        }
        while (session.batches.length > maxBatches) {
            session.batches.shift();
            session.batchOffset += 1;
        }
    };

    const findBatch = (batchIndex) => (
        session.batches.find((batch) => batch.index === batchIndex) || null
    );

    const appendTelemetry = (input = {}) => {
        const previousHash = session.chain.length ? session.chain[session.chain.length - 1].eventHash : null;
        const evidence = createGeometricEvidence({ ...input, previousHash });
        session.chain.push(evidence);
        trimChain();
        session.pending.push(evidence);
        let sealedBatch = null;
        if (session.pending.length >= batchSize) {
            sealedBatch = sealPendingBatch();
        }
        return { evidence, sealedBatch };
    };

    const sealPendingBatch = () => {
        if (!session.pending.length) {
            return null;
        }
        const batchIndex = session.batchOffset + session.batches.length;
        const batch = buildBatch(session.pending, batchIndex);
        session.batches.push(batch);
        trimBatches();
        session.pending = [];
        return batch;
    };

    const anchorBatch = (batchIndex, anchorHash, anchoredAt = Date.now()) => {
        if (!Number.isInteger(batchIndex) || batchIndex < 0) {
            return null;
        }
        if (typeof anchorHash !== 'string' || !anchorHash.length) {
            return null;
        }
        const batch = findBatch(batchIndex);
        if (!batch) {
            return null;
        }
        batch.anchored = { hash: anchorHash, anchoredAt };
        return batch;
    };

    const verifyBatchIntegrity = (batchIndex) => {
        if (!Number.isInteger(batchIndex) || batchIndex < 0) {
            return false;
        }
        const batch = findBatch(batchIndex);
        if (!batch) {
            return false;
        }
        return batch.proofs.every(({ hash, proof, root }) => verifyMerkleProof(hash, proof, root));
    };

    return {
        appendTelemetry,
        sealPendingBatch,
        anchorBatch,
        verifyBatchIntegrity,
        verifyChainIntegrity: () => verifyEvidenceChain(session.chain, { headHash: session.chainHeadHash }),
        getState: () => ({
            chain: session.chain.slice(),
            pending: session.pending.slice(),
            batches: session.batches.slice(),
            batchSize,
            maxChainLength,
            maxBatches,
            chainHeadHash: session.chainHeadHash,
            batchOffset: session.batchOffset
        })
    };
};
