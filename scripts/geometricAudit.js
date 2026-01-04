import crypto from 'node:crypto';

const stableStringify = (value) => {
    if (value === null || value === undefined) {
        return 'null';
    }
    if (typeof value !== 'object') {
        return JSON.stringify(value);
    }
    if (Array.isArray(value)) {
        return `[${value.map(stableStringify).join(',')}]`;
    }
    const keys = Object.keys(value).sort();
    const entries = keys.map((key) => `${JSON.stringify(key)}:${stableStringify(value[key])}`);
    return `{${entries.join(',')}}`;
};

const sha256Hex = (data) => crypto.createHash('sha256').update(data).digest('hex');

const normalizeVertex = (vertex = []) => {
    if (!Array.isArray(vertex)) {
        return null;
    }
    const components = vertex.map((component) => (Number.isFinite(component) ? Number(component) : null));
    return components.every((component) => component !== null) ? components : null;
};

const centroidOfVertices = (vertices) => {
    if (!Array.isArray(vertices) || vertices.length === 0) {
        return null;
    }
    const dimension = Array.isArray(vertices[0]) ? vertices[0].length : 0;
    if (dimension === 0) {
        return null;
    }
    const sums = new Array(dimension).fill(0);
    let count = 0;
    vertices.forEach((vertex) => {
        if (!Array.isArray(vertex) || vertex.length !== dimension) {
            return;
        }
        vertex.forEach((component, idx) => {
            sums[idx] += component;
        });
        count += 1;
    });
    return count === 0 ? null : sums.map((value) => value / count);
};

const distanceToCentroid = (vertex, centroid) => {
    if (!Array.isArray(vertex) || !Array.isArray(centroid) || vertex.length !== centroid.length) {
        return Number.POSITIVE_INFINITY;
    }
    const sumSquares = vertex.reduce((sum, component, idx) => sum + (component - centroid[idx]) ** 2, 0);
    return Math.sqrt(sumSquares);
};

export const canonicalizePolytopalState = (state = {}) => {
    const vertices = Array.isArray(state.vertices)
        ? state.vertices.map(normalizeVertex).filter((vertex) => Array.isArray(vertex))
        : [];
    const centroid = centroidOfVertices(vertices);
    const sortedVertices = centroid
        ? vertices
            .slice()
            .sort((a, b) => {
                const distanceA = distanceToCentroid(a, centroid);
                const distanceB = distanceToCentroid(b, centroid);
                return distanceA === distanceB ? stableStringify(a).localeCompare(stableStringify(b)) : distanceA - distanceB;
            })
        : vertices;

    const topologySignatures = Array.isArray(state.topologySignatures)
        ? state.topologySignatures
            .map((pair) => (Array.isArray(pair) && pair.length === 2
                ? pair.map((value) => (Number.isFinite(value) ? Number(value) : null))
                : null))
            .filter((pair) => pair && pair.every((value) => value !== null))
            .sort((a, b) => (a[0] === b[0] ? a[1] - b[1] : a[0] - b[0]))
        : [];

    const quaternion = Array.isArray(state.quaternion)
        ? state.quaternion
            .slice(0, 4)
            .map((component) => (Number.isFinite(component) ? Number(component) : null))
            .every((component) => component !== null)
                ? state.quaternion.slice(0, 4).map((component) => Number(component))
                : null
        : null;

    const canonical = {};
    if (sortedVertices.length) {
        canonical.vertices = sortedVertices;
    }
    if (topologySignatures.length) {
        canonical.topologySignatures = topologySignatures;
    }
    if (quaternion) {
        canonical.quaternion = quaternion;
    }
    if (state.metadata && typeof state.metadata === 'object') {
        canonical.metadata = state.metadata;
    }
    if (typeof state.mode === 'string' && state.mode.length) {
        canonical.mode = state.mode;
    }
    return canonical;
};

export const computeConstellationHash = (state = {}) => {
    const canonical = canonicalizePolytopalState(state);
    if (!Array.isArray(canonical.vertices) || canonical.vertices.length === 0) {
        return null;
    }
    return sha256Hex(stableStringify(canonical.vertices));
};

export const computeTopologyHash = (state = {}) => {
    const canonical = canonicalizePolytopalState(state);
    if (!Array.isArray(canonical.topologySignatures) || canonical.topologySignatures.length === 0) {
        return null;
    }
    return sha256Hex(stableStringify(canonical.topologySignatures));
};

export const computeQuaternionCommitment = (state = {}) => {
    const canonical = canonicalizePolytopalState(state);
    if (!Array.isArray(canonical.quaternion) || canonical.quaternion.length !== 4) {
        return null;
    }
    return sha256Hex(stableStringify(canonical.quaternion));
};

export const computePolytopalFingerprint = (state = {}) => ({
    constellation: computeConstellationHash(state),
    topology: computeTopologyHash(state),
    quaternion: computeQuaternionCommitment(state)
});

export const createGeometricEvidence = (input = {}) => {
    const timestamp = Number.isFinite(input.timestamp) ? input.timestamp : Date.now();
    const canonicalState = canonicalizePolytopalState(input.polytopalState || {});
    const fingerprint = computePolytopalFingerprint(input.polytopalState || {});
    const evidence = {
        eventType: input.eventType || 'GEOMETRIC_STATE',
        timestamp,
        previousHash: input.previousHash ?? null,
        polytopalFingerprint: fingerprint,
        payload: canonicalState,
        metadata: input.metadata && typeof input.metadata === 'object' ? input.metadata : {}
    };
    evidence.eventHash = sha256Hex(stableStringify(evidence));
    return evidence;
};

export const appendGeometricEvidence = (chain = [], input = {}) => {
    const previousHash = chain.length ? chain[chain.length - 1].eventHash : null;
    const evidence = createGeometricEvidence({ ...input, previousHash });
    return [...chain, evidence];
};

const merklePairHash = (a, b) => sha256Hex(a + b);

export const buildMerkleRoot = (hashes = []) => {
    if (!Array.isArray(hashes) || hashes.length === 0) {
        return null;
    }
    let layer = hashes.slice();
    while (layer.length > 1) {
        const nextLayer = [];
        for (let i = 0; i < layer.length; i += 2) {
            const left = layer[i];
            const right = layer[i + 1] ?? left;
            nextLayer.push(merklePairHash(left, right));
        }
        layer = nextLayer;
    }
    return layer[0];
};

export const buildMerkleProof = (hashes = [], index = 0) => {
    if (!Array.isArray(hashes) || hashes.length === 0 || index < 0 || index >= hashes.length) {
        return { proof: [], root: null };
    }
    let layer = hashes.slice();
    let idx = index;
    const proof = [];
    while (layer.length > 1) {
        const isRightNode = idx % 2 === 1;
        const siblingIndex = isRightNode ? idx - 1 : idx + 1;
        const siblingHash = layer[siblingIndex] ?? layer[idx];
        proof.push({ position: isRightNode ? 'left' : 'right', hash: siblingHash });
        const nextLayer = [];
        for (let i = 0; i < layer.length; i += 2) {
            const left = layer[i];
            const right = layer[i + 1] ?? left;
            nextLayer.push(merklePairHash(left, right));
        }
        idx = Math.floor(idx / 2);
        layer = nextLayer;
    }
    return { proof, root: layer[0] };
};

export const verifyMerkleProof = (leafHash, proof, root) => {
    if (!leafHash || !root) {
        return false;
    }
    if (!Array.isArray(proof)) {
        return false;
    }
    let computed = leafHash;
    for (const step of proof) {
        if (!step || typeof step.hash !== 'string' || (step.position !== 'left' && step.position !== 'right')) {
            return false;
        }
        computed = step.position === 'left'
            ? merklePairHash(step.hash, computed)
            : merklePairHash(computed, step.hash);
    }
    return computed === root;
};

export { stableStringify };
