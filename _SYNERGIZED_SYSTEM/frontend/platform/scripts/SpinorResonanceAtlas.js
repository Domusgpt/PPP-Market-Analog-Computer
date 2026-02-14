const clamp = (value, min, max) => {
    if (!Number.isFinite(value)) {
        return min;
    }
    if (value < min) {
        return min;
    }
    if (value > max) {
        return max;
    }
    return value;
};

const multiplyMatrixVector = (matrix, vector) => {
    if (!Array.isArray(matrix) || !Array.isArray(vector)) {
        return [];
    }
    return matrix.map((row) => {
        if (!Array.isArray(row)) {
            return 0;
        }
        let sum = 0;
        for (let index = 0; index < row.length; index += 1) {
            sum += row[index] * (vector[index] || 0);
        }
        return sum;
    });
};

const vectorMagnitude = (vector) => {
    if (!Array.isArray(vector) || !vector.length) {
        return 0;
    }
    return Math.sqrt(vector.reduce((sum, value) => sum + value * value, 0));
};

const normalizeVector = (vector) => {
    const magnitude = vectorMagnitude(vector);
    if (magnitude <= 0) {
        return null;
    }
    return vector.map((value) => value / magnitude);
};

const dotProduct = (a, b) => {
    if (!Array.isArray(a) || !Array.isArray(b) || !a.length || !b.length) {
        return 0;
    }
    const length = Math.min(a.length, b.length);
    let sum = 0;
    for (let index = 0; index < length; index += 1) {
        sum += a[index] * b[index];
    }
    return sum;
};

const rotateVector = (matrix, vector) => {
    const result = multiplyMatrixVector(matrix, vector);
    if (!result.length) {
        return [0, 0, 0, 0];
    }
    while (result.length < 4) {
        result.push(0);
    }
    return result.slice(0, 4);
};

const buildVoiceAtlas = ({
    voice,
    matrix,
    bridge,
    hopfFiber
}) => {
    if (!voice) {
        return null;
    }
    const gate = voice.modulation
        ? (voice.modulation.binary ? 1 : clamp(Number(voice.modulation.gate) || 0, 0, 1))
        : 0;
    const harmonic = clamp(Number(voice.harmonicCoordinate) || 0, 0, 1);
    const spinorPhase = voice.spinor ? Number(voice.spinor.phase) || 0 : 0;
    const spinorRatio = voice.spinor ? Number(voice.spinor.ratio) || 1 : 1;
    const spinorPhaseRadians = spinorPhase * Math.PI * 2;
    const source = [
        harmonic,
        gate,
        Math.cos(spinorPhaseRadians),
        Math.sin(spinorPhaseRadians)
    ];
    const rotated = rotateVector(matrix, source);
    const normalized = normalizeVector(rotated);
    const projection = bridge.normalized.length
        ? dotProduct(bridge.normalized, rotated)
        : 0;
    const hopfProjection = hopfFiber && hopfFiber.length
        ? dotProduct(hopfFiber, rotated)
        : 0;
    const magnitude = vectorMagnitude(rotated);

    const carriers = Array.isArray(voice.carriers)
        ? voice.carriers.map((carrier) => {
            const multiple = Number.isFinite(carrier.multiple) ? carrier.multiple : 1;
            const carrierPhase = spinorPhaseRadians * multiple;
            const carrierSource = [
                clamp(Number(carrier.energy) || 0, 0, 1),
                clamp(Number(carrier.amplitude) || 0, 0, Number.MAX_SAFE_INTEGER),
                Math.cos(carrierPhase),
                Math.sin(carrierPhase)
            ];
            const carrierRotated = rotateVector(matrix, carrierSource);
            return {
                band: carrier.band || '',
                multiple,
                frequency: Number(carrier.frequency) || 0,
                energy: carrierSource[0],
                amplitude: carrierSource[1],
                source: carrierSource,
                rotated: carrierRotated,
                magnitude: vectorMagnitude(carrierRotated)
            };
        })
        : [];

    return {
        index: voice.index,
        frequency: Number(voice.frequency) || 0,
        pan: clamp(Number(voice.pan) || 0, -1, 1),
        gate,
        harmonic,
        ratio: spinorRatio,
        phase: spinorPhase,
        source,
        rotated,
        normalized,
        magnitude,
        projection,
        hopfProjection,
        carriers
    };
};

const aggregateVoices = (voices, bridge) => {
    if (!voices.length) {
        return {
            centroid: [0, 0, 0, 0],
            magnitude: 0,
            bridgeProjection: 0,
            hopfProjection: 0,
            spinorRatio: 0,
            gateMean: 0,
            gateVariance: 0,
            panMean: 0,
            panVariance: 0,
            phaseVector: [0, 0],
            carrierCentroid: [0, 0, 0, 0],
            carrierMagnitude: 0,
            spinorCoherence: bridge.coherence,
            braidDensity: bridge.braidDensity
        };
    }
    const centroid = [0, 0, 0, 0];
    const phaseVector = [0, 0];
    const carrierCentroid = [0, 0, 0, 0];
    let carrierCount = 0;
    let magnitude = 0;
    let bridgeProjection = 0;
    let hopfProjection = 0;
    let ratioSum = 0;
    let gateMean = 0;
    let panMean = 0;
    let carrierMagnitude = 0;

    voices.forEach((voice) => {
        for (let index = 0; index < centroid.length; index += 1) {
            centroid[index] += voice.rotated[index] || 0;
        }
        phaseVector[0] += voice.source[2] || 0;
        phaseVector[1] += voice.source[3] || 0;
        magnitude += voice.magnitude;
        bridgeProjection += voice.projection;
        hopfProjection += voice.hopfProjection;
        ratioSum += voice.ratio;
        gateMean += voice.gate;
        panMean += voice.pan;
        if (Array.isArray(voice.carriers)) {
            voice.carriers.forEach((carrier) => {
                for (let index = 0; index < carrierCentroid.length; index += 1) {
                    carrierCentroid[index] += carrier.rotated[index] || 0;
                }
                carrierMagnitude += carrier.magnitude;
                carrierCount += 1;
            });
        }
    });

    const count = voices.length;
    const normalizedCentroid = centroid.map((value) => value / count);
    const normalizedPhase = phaseVector.map((value) => value / count);
    const averageMagnitude = magnitude / count;
    const averageBridgeProjection = bridgeProjection / count;
    const averageHopfProjection = hopfProjection / count;
    const meanRatio = ratioSum / count;
    gateMean /= count;
    panMean /= count;
    const averageCarrierCentroid = carrierCount
        ? carrierCentroid.map((value) => value / carrierCount)
        : carrierCentroid;
    const averageCarrierMagnitude = carrierCount ? (carrierMagnitude / carrierCount) : 0;

    let gateVariance = 0;
    let panVariance = 0;
    voices.forEach((voice) => {
        const gateDiff = voice.gate - gateMean;
        const panDiff = voice.pan - panMean;
        gateVariance += gateDiff * gateDiff;
        panVariance += panDiff * panDiff;
    });
    gateVariance /= count;
    panVariance /= count;

    return {
        centroid: normalizedCentroid,
        magnitude: averageMagnitude,
        bridgeProjection: averageBridgeProjection,
        hopfProjection: averageHopfProjection,
        spinorRatio: meanRatio,
        gateMean,
        gateVariance,
        panMean,
        panVariance,
        phaseVector: normalizedPhase,
        carrierCentroid: averageCarrierCentroid,
        carrierMagnitude: averageCarrierMagnitude,
        spinorCoherence: bridge.coherence,
        braidDensity: bridge.braidDensity
    };
};

export const buildSpinorResonanceAtlas = ({
    rotationMatrix,
    bridgeVector,
    normalizedBridge,
    hopfFiber,
    spinor,
    voices,
    timelineProgress = 0
} = {}) => {
    if (!Array.isArray(rotationMatrix) || rotationMatrix.length !== 4) {
        return null;
    }
    const rows = rotationMatrix.map((row) => (
        Array.isArray(row) ? row.slice(0, 4) : [0, 0, 0, 0]
    ));
    if (!Array.isArray(voices) || !voices.length) {
        return null;
    }
    const resolvedBridgeVector = Array.isArray(bridgeVector) ? bridgeVector.slice(0, 4) : [0, 0, 0, 0];
    const bridgeMagnitude = vectorMagnitude(resolvedBridgeVector);
    const bridge = {
        vector: resolvedBridgeVector,
        normalized: Array.isArray(normalizedBridge) ? normalizedBridge.slice(0, 4) : [0, 0, 0, 0],
        magnitude: bridgeMagnitude,
        coherence: spinor && Number.isFinite(spinor.coherence) ? spinor.coherence : 0,
        braidDensity: spinor && Number.isFinite(spinor.braidDensity) ? spinor.braidDensity : 0
    };
    const hopf = Array.isArray(hopfFiber) ? hopfFiber.slice(0, 4) : null;
    const axes = rows.map((row) => normalizeVector(row) || row.map(() => 0));
    const voiceAtlases = voices
        .map((voice) => buildVoiceAtlas({
            voice,
            matrix: rows,
            bridge,
            hopfFiber: hopf
        }))
        .filter(Boolean);
    if (!voiceAtlases.length) {
        return null;
    }

    const aggregate = aggregateVoices(voiceAtlases, bridge);

    return {
        timeline: clamp(Number(timelineProgress) || 0, 0, 1),
        matrix: rows.map((row) => row.slice()),
        axes: axes.map((axis) => axis.slice()),
        bridge,
        hopf,
        voices: voiceAtlases,
        aggregate
    };
};

export const cloneSpinorResonanceAtlas = (atlas) => {
    if (!atlas || typeof atlas !== 'object') {
        return null;
    }
    return {
        timeline: atlas.timeline,
        matrix: Array.isArray(atlas.matrix)
            ? atlas.matrix.map((row) => (Array.isArray(row) ? row.slice() : []))
            : [],
        axes: Array.isArray(atlas.axes)
            ? atlas.axes.map((axis) => (Array.isArray(axis) ? axis.slice() : []))
            : [],
        bridge: atlas.bridge
            ? {
                vector: Array.isArray(atlas.bridge.vector) ? atlas.bridge.vector.slice() : [],
                normalized: Array.isArray(atlas.bridge.normalized) ? atlas.bridge.normalized.slice() : [],
                magnitude: atlas.bridge.magnitude,
                coherence: atlas.bridge.coherence,
                braidDensity: atlas.bridge.braidDensity
            }
            : null,
        hopf: Array.isArray(atlas.hopf) ? atlas.hopf.slice() : null,
        voices: Array.isArray(atlas.voices)
            ? atlas.voices.map((voice) => ({
                ...voice,
                source: Array.isArray(voice.source) ? voice.source.slice() : [],
                rotated: Array.isArray(voice.rotated) ? voice.rotated.slice() : [],
                normalized: Array.isArray(voice.normalized) ? voice.normalized.slice() : null,
                carriers: Array.isArray(voice.carriers)
                    ? voice.carriers.map((carrier) => ({
                        ...carrier,
                        source: Array.isArray(carrier.source) ? carrier.source.slice() : [],
                        rotated: Array.isArray(carrier.rotated) ? carrier.rotated.slice() : []
                    }))
                    : []
            }))
            : [],
        aggregate: atlas.aggregate ? { ...atlas.aggregate } : null
    };
};
