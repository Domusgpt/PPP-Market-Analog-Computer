import { createGeometricAuditSession } from './geometricAuditPipeline.js';

const isFiniteNumber = (value) => Number.isFinite(value);

const normalizeVector = (value) => (
    Array.isArray(value)
        ? value.map((component) => (Number.isFinite(component) ? Number(component) : null)).filter((component) => component !== null)
        : []
);

const buildVertices = (constellation) => {
    if (!constellation || typeof constellation !== 'object') {
        return [];
    }
    const nodes = Array.isArray(constellation.nodes) ? constellation.nodes : [];
    const vertices = nodes
        .map((node) => normalizeVector(node?.orientation))
        .filter((vector) => vector.length > 0);
    const centroid = normalizeVector(constellation?.orientation?.centroid);
    if (centroid.length) {
        vertices.push(centroid);
    }
    const bridge = normalizeVector(constellation?.orientation?.bridge);
    if (bridge.length) {
        vertices.push(bridge);
    }
    return vertices;
};

const buildTopologySignatures = (constellation) => {
    if (!constellation || typeof constellation !== 'object') {
        return [];
    }
    const sequence = constellation.sequence || {};
    const spinor = constellation.spinor || {};
    const energy = constellation.energy || {};
    const pairs = [
        [sequence.bitDensity, sequence.bitEntropy],
        [spinor.coherence, spinor.braid],
        [energy.carrier, energy.grid]
    ];
    return pairs
        .map(([first, second]) => (isFiniteNumber(first) && isFiniteNumber(second) ? [Number(first), Number(second)] : null))
        .filter((pair) => pair);
};

const pickQuaternion = (constellation) => {
    const candidates = [
        normalizeVector(constellation?.orientation?.bridge),
        normalizeVector(constellation?.orientation?.hopf),
        normalizeVector(constellation?.orientation?.centroid)
    ];
    const match = candidates.find((candidate) => candidate.length >= 4);
    return match ? match.slice(0, 4) : null;
};

export const buildPolytopalStateFromConstellation = (constellation = {}) => ({
    vertices: buildVertices(constellation),
    topologySignatures: buildTopologySignatures(constellation),
    quaternion: pickQuaternion(constellation),
    metadata: {
        transport: constellation?.transport || {},
        progress: isFiniteNumber(constellation?.progress) ? constellation.progress : 0,
        energy: constellation?.energy || {},
        spinor: constellation?.spinor || {}
    },
    mode: 'constellation'
});

export const createGeometricAuditBridge = (options = {}) => {
    const session = createGeometricAuditSession(options);

    const ingestConstellation = (constellation, overrides = {}) => {
        const polytopalState = buildPolytopalStateFromConstellation(constellation);
        return session.appendTelemetry({
            eventType: 'CONSTELLATION_SNAPSHOT',
            polytopalState,
            metadata: overrides.metadata || {},
            timestamp: overrides.timestamp
        });
    };

    return {
        ingestConstellation,
        sealPendingBatch: session.sealPendingBatch,
        anchorBatch: session.anchorBatch,
        verifyBatchIntegrity: session.verifyBatchIntegrity,
        verifyChainIntegrity: session.verifyChainIntegrity,
        getState: session.getState
    };
};
